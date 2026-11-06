"""Direct DashScope HTTP/duplex transport; the shared DPAPI copy supplies the key."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

KEY_MODULE = Path(r"E:\PCConfig\tools\vendor_key_copy.py")
ORIGINS = {"http": "https://dashscope.aliyuncs.com", "websocket": "wss://dashscope.aliyuncs.com"}
MAX_RESPONSE = 16 * 1024 * 1024


class CloudSendError(RuntimeError):
    def __init__(self, code, *, uncertain=False, credential_result="Scope-Error", provider_code="", provider_message=""):
        super().__init__(code)
        self.code, self.uncertain = code, uncertain
        self.fields = dict(error_code=code, credential_result=credential_result,
                           provider_error_code=provider_code, provider_error_message=provider_message)

    @property
    def invalid_key(self):
        return self.fields["credential_result"] == "Invalid" or self.fields["provider_error_code"].lower().startswith("invalidapikey")


def key_copy():
    spec = importlib.util.spec_from_file_location("zh_asr_vendor_key_copy", KEY_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scrub(value, keys):
    if isinstance(value, str):
        for key in keys:
            if key: value = value.replace(key, "[redacted]")
        return value
    if isinstance(value, dict): return {scrub(k, keys): scrub(v, keys) for k, v in value.items()}
    if isinstance(value, list): return [scrub(v, keys) for v in value]
    return value


def deadline(value):
    if value and datetime.now(timezone.utc) >= datetime.fromisoformat(value.replace("Z", "+00:00")):
        raise CloudSendError("request_send_deadline_passed")


def read_blob(spec, root):
    path = Path(spec["path"]).resolve()
    if not path.is_relative_to(root.resolve()): raise CloudSendError("request_path_escape")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]: raise CloudSendError("request_file_changed")
    return raw


def provider_error(code, message, status=0):
    category = {401: "Invalid", 403: "Permission-Denied", 429: "Rate-Limited"}.get(
        status, "Provider-5xx" if status >= 500 else "Scope-Error")
    return CloudSendError("http_" + str(status) if status else "provider_task_failed",
        credential_result=category, provider_code=str(code or "")[:200], provider_message=str(message or "")[:500])


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None


def http_transport(url, operation, key, root):
    payload = json.loads(read_blob(operation["body_file"], root))
    request = Request(url, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=operation.get("timeout_seconds", 180)) as response:
            data, status = response.read(MAX_RESPONSE + 1), response.status
    except HTTPError as exc:
        try:
            body = json.loads(exc.read(256 * 1024))
            details = body.get("error", body.get("detail", body)) if isinstance(body, dict) else {}
            if not isinstance(details, dict): details = {}
        except (ValueError, OSError): details = {}
        finally: exc.close()
        raise provider_error(details.get("code"), details.get("message"), exc.code) from None
    except (URLError, OSError, TimeoutError):
        raise CloudSendError("network_failure", uncertain=True, credential_result="Network-Failure") from None
    try:
        body = json.loads(data)
        if len(data) > MAX_RESPONSE or not isinstance(body, dict): raise ValueError()
    except (ValueError, UnicodeError): raise CloudSendError("provider_response_invalid", uncertain=True) from None
    if body.get("code"): raise provider_error(body.get("code"), body.get("message"))
    return dict(http_status=status, raw_response=body, provider_request_id=body.get("request_id"), usage=body.get("usage"))


def websocket_transport(url, operation, key, root):
    from websockets.sync.client import connect
    from websockets.exceptions import InvalidStatus
    events, response_bytes = [], 0
    end = time.monotonic() + operation.get("timeout_seconds", 180)

    def field(event, names):
        for name in names: event = event.get(name) if isinstance(event, dict) else None
        return event

    def remaining():
        seconds = end - time.monotonic()
        if seconds <= 0: raise CloudSendError("provider_task_timeout", uncertain=True, credential_result="Timeout")
        return seconds

    def collect(raw):
        nonlocal response_bytes
        try: event = json.loads(raw)
        except (ValueError, UnicodeError): raise CloudSendError("provider_event_invalid", uncertain=True) from None
        if not isinstance(event, dict): raise CloudSendError("provider_event_invalid", uncertain=True)
        if any(field(event, b["field"]) != b["value"] for b in operation.get("event_bindings", [])):
            raise CloudSendError("provider_event_binding_mismatch", uncertain=True)
        response_bytes += len(raw)
        if response_bytes > MAX_RESPONSE: raise CloudSendError("provider_response_too_large", uncertain=True)
        events.append(event)
        error = operation.get("event_error", {})
        if error and field(event, error["field"]) in error["values"]:
            raise provider_error(field(event, error["code_field"]), field(event, error["message_field"]))
        return event

    try:
        with connect(url, additional_headers={"Authorization": "Bearer " + key}, proxy=None,
                open_timeout=min(30, remaining()), close_timeout=5, max_size=MAX_RESPONSE) as socket:
            for frame in operation["frames"]:
                remaining()
                if "json" in frame:
                    socket.send(json.dumps(frame["json"], ensure_ascii=False, allow_nan=False))
                elif "binary_file" in frame:
                    raw = read_blob(frame["binary_file"], root)
                    size, pace = frame.get("chunk_bytes", 3200), frame.get("interval_ms", 0) / 1000
                    for pos in range(0, len(raw), size):
                        remaining(); socket.send(raw[pos:pos + size])
                        if pace: time.sleep(min(pace, remaining()))
                        if frame.get("drain_available"):
                            while True:
                                try: event_raw = socket.recv(timeout=0)
                                except TimeoutError: break
                                collect(event_raw)
                else:
                    rule = frame["receive_until"]
                    for _ in range(10000):
                        event = collect(socket.recv(timeout=remaining()))
                        if field(event, rule["field"]) in rule.get("errors", []):
                            raise provider_error(field(event, rule["error_code_field"]), field(event, rule["error_message_field"]))
                        if field(event, rule["field"]) == rule["equals"]: break
                    else: raise CloudSendError("provider_event_limit", uncertain=True)
    except CloudSendError: raise
    except InvalidStatus as exc:
        raise provider_error("", "", exc.response.status_code) from None
    except TimeoutError: raise CloudSendError("provider_task_timeout", uncertain=True, credential_result="Timeout") from None
    except Exception: raise CloudSendError("websocket_failure", uncertain=True, credential_result="Network-Failure") from None
    return {"raw_events": events}


def transmit(operation, root, cutoff, store, keys):
    """Only an explicit authentication rejection permits one key refresh/retry."""
    def once(key):
        deadline(cutoff)
        try:
            return (http_transport if operation["protocol"] == "http" else websocket_transport)(
                ORIGINS[operation["protocol"]] + operation["path"], operation, key, root)
        except CloudSendError as exc:
            if "ASR_RESPONSE_HAVE_NO_WORDS" in (exc.fields["provider_error_code"], exc.fields["provider_error_message"]):
                return dict(no_speech=True, **exc.fields)
            raise
    try: return once(keys[-1])
    except CloudSendError as exc:
        if exc.uncertain or not exc.invalid_key: raise
    response, failed = [], []
    def validate(candidate):
        keys.append(candidate)
        try: response.append(once(candidate)); return True
        except CloudSendError as exc: failed.append(exc); return False
    try: state = store.sync_key("qwen", validate=validate)
    except Exception: raise CloudSendError("cloud_key_sync_failed") from None
    if response:
        return dict(response[0], key_copy_sync={k: state[k] for k in ("status", "reason") if k in state})
    if state.get("status") != "ok":
        if failed:
            error = failed[0]
            error.code = "cloud_key_refresh_" + error.code
            error.fields["error_code"] = error.code
            raise error
        raise CloudSendError("cloud_key_sync_" + str(state.get("reason") or "failed"))
    raise CloudSendError("cloud_key_sync_validation_missing")
