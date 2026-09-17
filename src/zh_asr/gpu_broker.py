from __future__ import annotations

import json
from contextvars import ContextVar
import os
import threading
import urllib.error
import urllib.request
from typing import Callable


class GpuBrokerError(RuntimeError):
    pass


class GpuBrokerConflict(GpuBrokerError):
    def __init__(self, message: str, *, owner: str = "unknown", reason: str = "gpu_conflict"):
        super().__init__(message)
        self.owner = owner
        self.reason = reason


class GpuBrokerLeaseLost(GpuBrokerError):
    def __init__(self, message: str, *, reason: str = ""):
        super().__init__(message)
        self.reason = reason


Transport = Callable[[str, dict], dict]
GPU_BROKER_CHILD_TOKEN_ENV = "ZH_ASR_GPU_BROKER_CHILD_TOKEN"
GPU_BROKER_OWNERS = frozenset({"chineseasr", "chineseasr-cli"})
_VERIFIED_WORKER_TOKEN: ContextVar[str] = ContextVar("asr_verified_worker_token", default="")


def _default_transport(base_url: str) -> Transport:
    def send(action: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{base_url}/_gpu_broker/{action}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                raise GpuBrokerError(f"GPU broker HTTP {exc.code}") from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise GpuBrokerError(f"GPU broker unavailable: {type(exc).__name__}: {exc}") from exc

    return send


def verify_inherited_gpu_lease(
    token: str,
    *,
    base_url: str | None = None,
    ttl_seconds: int = 120,
    transport: Transport | None = None,
) -> str:
    """Prove that a supervised worker inherited a live ChineseASR lease.

    A plain environment marker is not evidence of serialization. The worker
    must possess the opaque token for the currently live machine-wide lease;
    the broker validates that token by renewing it and returns the authoritative
    owner.
    """
    inherited_token = str(token or "").strip()
    if not inherited_token:
        raise GpuBrokerError("Inherited GPU broker lease token is missing.")
    endpoint = (
        base_url
        or os.environ.get("LOCAL_GPU_BROKER_URL")
        or "http://127.0.0.1:32100"
    ).rstrip("/")
    send = transport or _default_transport(endpoint)
    result = send(
        "renew",
        {
            "token": inherited_token,
            "ttl_seconds": ttl_seconds,
            "owner_pid": os.getpid(),
        },
    )
    if not result.get("ok"):
        reason = result.get("reason") or "renew_rejected"
        raise GpuBrokerLeaseLost(
            f"Inherited GPU broker lease is not live: {reason}"
        )
    owner = str(result.get("owner") or "").strip()
    if owner not in GPU_BROKER_OWNERS:
        raise GpuBrokerError(
            "Inherited GPU broker lease is not owned by ChineseASR."
        )
    _VERIFIED_WORKER_TOKEN.set(inherited_token)
    return owner


def require_worker_gpu_lease(device: str) -> None:
    """Reuse the verified worker context, rechecking the live broker before loading."""
    if not str(device).lower().startswith(("cuda", "gpu")):
        return
    token = _VERIFIED_WORKER_TOKEN.get() or os.environ.get(GPU_BROKER_CHILD_TOKEN_ENV, "")
    verify_inherited_gpu_lease(token)


class GpuBrokerLease:
    def __init__(
        self,
        owner: str,
        *,
        base_url: str | None = None,
        ttl_seconds: int = 120,
        renew_interval_seconds: int = 20,
        transport: Transport | None = None,
    ) -> None:
        self.owner = owner
        self.base_url = (base_url or os.environ.get("LOCAL_GPU_BROKER_URL") or "http://127.0.0.1:32100").rstrip("/")
        self.ttl_seconds = ttl_seconds
        self.renew_interval_seconds = renew_interval_seconds
        self.transport = transport or _default_transport(self.base_url)
        self.token = ""
        self._stop = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._failure_lock = threading.Lock()
        self._lost_error: GpuBrokerLeaseLost | None = None
        self._on_lost: Callable[[GpuBrokerLeaseLost], None] | None = None

    def __enter__(self):
        result = self.transport(
            "acquire", {"owner": self.owner, "ttl_seconds": self.ttl_seconds, "owner_pid": os.getpid()}
        )
        if not result.get("ok"):
            active_owner = result.get("owner") or "unknown"
            reason = result.get("reason") or "gpu_conflict"
            raise GpuBrokerConflict(
                f"GPU broker blocked {self.owner}: {reason}; active={active_owner}",
                owner=active_owner, reason=reason,
            )
        self.token = str(result.get("token") or "")
        if not self.token:
            raise GpuBrokerError("GPU broker returned no lease token")
        if self.renew_interval_seconds > 0:
            self._renew_thread = threading.Thread(target=self._renew_loop, daemon=True)
            self._renew_thread.start()
        return self

    @property
    def lost(self) -> bool:
        with self._failure_lock:
            return self._lost_error is not None

    @property
    def loss_error(self) -> GpuBrokerLeaseLost | None:
        with self._failure_lock:
            return self._lost_error

    def set_on_lost(
        self,
        callback: Callable[[GpuBrokerLeaseLost], None],
    ) -> None:
        with self._failure_lock:
            self._on_lost = callback
            existing = self._lost_error
        if existing is not None:
            callback(existing)

    def complete_worker(self) -> None:
        """A caller proved its supervised process exited successfully.

        Only that evidence makes process-exit reclamation a normal completion;
        transport failure or rejected renewal during live work remains an error.
        """
        self._worker_completed = True
        self._stop.set()
        if self._renew_thread:
            self._renew_thread.join(timeout=2)

    def raise_if_lost(self) -> None:
        error = self.loss_error
        if error is not None:
            if getattr(self, "_worker_completed", False) and error.reason == "lease_not_found":
                return
            raise error

    def _mark_lost(self, error: GpuBrokerLeaseLost) -> None:
        callback = None
        with self._failure_lock:
            if self._lost_error is not None:
                return
            self._lost_error = error
            callback = self._on_lost
        if callback is not None:
            try:
                callback(error)
            except Exception:
                # The original lease loss remains authoritative. The service
                # will also observe it when the context exits.
                pass

    def _renew_loop(self) -> None:
        while not self._stop.wait(self.renew_interval_seconds):
            try:
                result = self.transport(
                    "renew", {"token": self.token, "ttl_seconds": self.ttl_seconds}
                )
                if not result.get("ok"):
                    reason = result.get("reason") or "renew_rejected"
                    active_owner = result.get("owner") or "unknown"
                    self._mark_lost(
                        GpuBrokerLeaseLost(
                            f"GPU broker lease renewal failed for {self.owner}: "
                            f"{reason}; active={active_owner}", reason=reason,
                        )
                    )
                    return
            except Exception as exc:
                self._mark_lost(
                    GpuBrokerLeaseLost(
                        f"GPU broker lease renewal failed for {self.owner}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )
                return

    def __exit__(self, exc_type, _exc, _traceback):
        self._stop.set()
        if self._renew_thread:
            self._renew_thread.join(timeout=2)
        release_error: BaseException | None = None
        if self.token:
            try:
                result = self.transport("release", {"token": self.token})
                if not result.get("ok"):
                    reason = result.get("reason") or "release_rejected"
                    if not (reason == "lease_not_found" and getattr(self, "_worker_completed", False)):
                        raise GpuBrokerError(
                            f"GPU broker lease release failed for {self.owner}: {reason}"
                        )
            except BaseException as error:
                release_error = error
            finally:
                self.token = ""
        if exc_type is not None:
            return False
        self.raise_if_lost()
        if release_error is not None:
            raise release_error
        return False

