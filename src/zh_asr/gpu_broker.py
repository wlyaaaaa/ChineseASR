from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Callable


class GpuBrokerError(RuntimeError):
    pass


class GpuBrokerConflict(GpuBrokerError):
    pass


Transport = Callable[[str, dict], dict]


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


class GpuBrokerLease:
    def __init__(
        self,
        owner: str,
        *,
        base_url: str | None = None,
        ttl_seconds: int = 21_600,
        renew_interval_seconds: int = 60,
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

    def __enter__(self):
        result = self.transport(
            "acquire", {"owner": self.owner, "ttl_seconds": self.ttl_seconds}
        )
        if not result.get("ok"):
            active_owner = result.get("owner") or "unknown"
            reason = result.get("reason") or "gpu_conflict"
            raise GpuBrokerConflict(f"GPU broker blocked {self.owner}: {reason}; active={active_owner}")
        self.token = str(result.get("token") or "")
        if not self.token:
            raise GpuBrokerError("GPU broker returned no lease token")
        if self.renew_interval_seconds > 0:
            self._renew_thread = threading.Thread(target=self._renew_loop, daemon=True)
            self._renew_thread.start()
        return self

    def _renew_loop(self) -> None:
        while not self._stop.wait(self.renew_interval_seconds):
            try:
                result = self.transport(
                    "renew", {"token": self.token, "ttl_seconds": self.ttl_seconds}
                )
                if not result.get("ok"):
                    return
            except Exception:
                return

    def __exit__(self, *_args):
        self._stop.set()
        if self._renew_thread:
            self._renew_thread.join(timeout=2)
        if self.token:
            try:
                self.transport("release", {"token": self.token})
            finally:
                self.token = ""

