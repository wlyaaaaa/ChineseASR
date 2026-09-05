"""Audio-device selection for the Windows dictation controller.

The controller owns stream lifetime. open_microphone is called only between
recordings, after the previous stream has been closed.
"""
from __future__ import annotations

from typing import Any, Callable

import sounddevice as sd


class MicrophoneOpenError(RuntimeError):
    """A named input device could not be refreshed, selected, or started."""


_HOST_API_ORDER = ("wasapi", "mme", "directsound", "wdm")
_BLOCK_MS = 20


def _host_api_rank(name: str) -> int:
    lowered = str(name).casefold()
    for rank, marker in enumerate(_HOST_API_ORDER):
        if marker in lowered:
            return rank
    return len(_HOST_API_ORDER)


def _refresh_portaudio() -> None:
    """Refresh PortAudio between recordings.

    sounddevice 0.5.6 exposes only the private _terminate/_initialize pair
    for re-enumeration. This narrow use is safe only while this process has no
    active stream; the caller closes the previous recording first.
    """

    try:
        if getattr(sd, "_initialized", 1) > 0:
            sd._terminate()
        sd._initialize()
    except Exception as exc:
        raise MicrophoneOpenError(
            f"无法刷新麦克风设备枚举（PortAudio）：{type(exc).__name__}"
        ) from exc


def _host_api_names() -> list[str]:
    try:
        return [str(item.get("name", "")) for item in sd.query_hostapis()]
    except Exception:
        return []


def _candidate_devices(requested: str | int | None) -> list[dict[str, Any]]:
    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        raise MicrophoneOpenError(
            f"无法枚举麦克风设备：{type(exc).__name__}"
        ) from exc

    if isinstance(requested, int) and not isinstance(requested, bool):
        candidates = [
            dict(info)
            for info in devices
            if int(info.get("index", -1)) == requested
            and int(info.get("max_input_channels", 0)) > 0
        ]
    elif requested is None:
        try:
            default_index = int(sd.default.device[0])
        except Exception as exc:
            raise MicrophoneOpenError(
                "未指定输入设备，且当前 Windows 默认输入设备不可用"
            ) from exc
        candidates = [
            dict(info)
            for info in devices
            if int(info.get("index", -1)) == default_index
            and int(info.get("max_input_channels", 0)) > 0
        ]
    else:
        needle = str(requested).strip().casefold()
        candidates = [
            dict(info)
            for info in devices
            if needle
            and needle in str(info.get("name", "")).casefold()
            and int(info.get("max_input_channels", 0)) > 0
        ]

    host_names = _host_api_names()
    for info in candidates:
        hostapi = int(info.get("hostapi", -1))
        info["_host_name"] = (
            host_names[hostapi] if 0 <= hostapi < len(host_names) else ""
        )
        info["_host_rank"] = _host_api_rank(info["_host_name"])
    return sorted(
        candidates, key=lambda item: (item["_host_rank"], int(item["index"]))
    )


def list_microphones(refresh: bool = True) -> list[dict[str, str | None]]:
    """Names are stable preferences; endpoint indices are resolved at open time."""
    if refresh:
        _refresh_portaudio()
    options = [{"value": None, "label": "Windows 默认麦克风"}]
    seen = set()
    host_names = _host_api_names()
    devices = sorted(sd.query_devices(), key=lambda d: _host_api_rank(
        host_names[int(d["hostapi"])] if 0 <= int(d["hostapi"]) < len(host_names) else ""))
    for info in devices:
        name = str(info["name"]).strip()
        if not info["max_input_channels"] or "声音映射器" in name or "主声音捕获" in name:
            continue
        host_index = int(info["hostapi"])
        host_name = host_names[host_index] if 0 <= host_index < len(host_names) else ""
        if _host_api_rank(host_name) == 3 and "DJI Mic Mini" not in name:
            continue  # Do not crowd a small menu with raw driver pins and orphan endpoints.
        if "DJI Mic Mini" in name:
            name = "DJI Mic Mini"
        # Prefer the ordinary Windows endpoint over raw WDM device-resource names.
        if name.startswith("@") or "@System32" in name:
            if "DJI Mic Mini" in name:
                name = "DJI Mic Mini"
            else:
                continue
        key = name.rstrip(" )").casefold()
        if key in seen:
            continue
        seen.add(key)
        options.append({"value": name, "label": name})
    return options


def open_microphone(settings: Any, callback: Callable[..., Any]):
    """Open and start the configured input device at 16 kHz mono float32.

    A configured device name is matched as a case-insensitive substring so
    Windows endpoint suffixes such as DJI Mic Mini-FB7E6B remain reconnectable.
    Only matching input endpoints are tried; another microphone is never
    selected as a fallback.
    """

    _refresh_portaudio()
    requested = getattr(settings, "input_device", None)
    candidates = _candidate_devices(requested)
    if not candidates:
        label = "Windows 默认输入设备" if requested is None else repr(requested)
        raise MicrophoneOpenError(f"指定输入设备不存在或没有输入通道：{label}")

    sample_rate = int(getattr(settings, "sample_rate", 16000))
    blocksize = round(sample_rate * _BLOCK_MS / 1000)
    failures: list[str] = []
    for info in candidates:
        index = int(info["index"])
        label = f"{info.get('name', '')} [{info.get('_host_name', '')}]"
        try:
            sd.check_input_settings(
                device=index,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
        except Exception as exc:
            failures.append(f"{label}: 16kHz 不支持（{type(exc).__name__}）")
            continue

        stream = None
        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                device=index,
                callback=callback,
            )
            stream.start()
            return stream
        except Exception as exc:
            failures.append(f"{label}: 启动失败（{type(exc).__name__}）")
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    label = "Windows 默认输入设备" if requested is None else repr(requested)
    detail = "; ".join(failures) or "没有可用的输入端点"
    raise MicrophoneOpenError(f"指定输入设备无法打开：{label}；{detail}")
