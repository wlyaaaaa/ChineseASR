from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from collections.abc import Iterable, Mapping
from typing import Any


PROCESS_TOKEN_ENV = "ZH_ASR_PROCESS_TOKEN"
_TOKEN_PATTERN = re.compile(r"\A[A-Za-z0-9._-]{1,200}\Z")
_WSL_SIGNAL_SCRIPT = r"""
token=$1
requested_signal=$2
for environment in /proc/[0-9]*/environ; do
  [ -r "$environment" ] || continue
  if tr '\000' '\n' < "$environment" 2>/dev/null |
      grep -Fqx -- "ZH_ASR_PROCESS_TOKEN=$token"; then
    pid=${environment#/proc/}
    pid=${pid%/environ}
    [ "$pid" = "$$" ] || kill "-$requested_signal" "$pid" 2>/dev/null || true
  fi
done
"""


def tagged_process_env(
    token: str,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    _validate_token(token)
    env = dict(os.environ if base is None else base)
    env[PROCESS_TOKEN_ENV] = token
    wslenv_entries = [entry for entry in env.get("WSLENV", "").split(":") if entry]
    if not any(entry.split("/", 1)[0] == PROCESS_TOKEN_ENV for entry in wslenv_entries):
        wslenv_entries.append(PROCESS_TOKEN_ENV)
    env["WSLENV"] = ":".join(wslenv_entries)
    return env


def managed_popen_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            )
        }
    return {"start_new_session": True}


def terminate_process_tree(process: subprocess.Popen, wait_sec: float = 5.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, wait_sec),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        process.wait(timeout=wait_sec)
        return
    except (AttributeError, subprocess.TimeoutExpired):
        pass

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        try:
            process.kill()
        except OSError:
            return
    try:
        process.wait(timeout=wait_sec)
    except (AttributeError, subprocess.TimeoutExpired):
        return


def terminate_wsl_processes(
    distributions: Iterable[str],
    token: str,
    *,
    grace_sec: float = 0.5,
) -> None:
    _validate_token(token)
    names = tuple(dict.fromkeys(name.strip() for name in distributions if name.strip()))
    if not names:
        return
    for distribution in names:
        _signal_wsl_token(distribution, token, "TERM")
    if grace_sec > 0:
        time.sleep(grace_sec)
    for distribution in names:
        _signal_wsl_token(distribution, token, "KILL")


def _signal_wsl_token(distribution: str, token: str, requested_signal: str) -> None:
    try:
        subprocess.run(
            [
                "wsl.exe",
                "-d",
                distribution,
                "--exec",
                "sh",
                "-c",
                _WSL_SIGNAL_SCRIPT,
                "chineseasr-process-cleanup",
                token,
                requested_signal,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _validate_token(token: str) -> None:
    if not _TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Process token contains unsupported characters.")
