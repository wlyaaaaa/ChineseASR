from __future__ import annotations

import os
from collections.abc import Mapping


PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

NO_PROXY_VALUE = ",".join(
    [
        "localhost",
        "127.0.0.1",
        "::1",
        "aliyun.com",
        "*.aliyun.com",
        "aliyuncs.com",
        "*.aliyuncs.com",
        "modelscope.cn",
        "*.modelscope.cn",
    ]
)


def sanitized_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for name in PROXY_ENV_NAMES:
        env.pop(name, None)
    env["NO_PROXY"] = NO_PROXY_VALUE
    env["no_proxy"] = NO_PROXY_VALUE
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def sanitize_current_process_env() -> None:
    for name in PROXY_ENV_NAMES:
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = NO_PROXY_VALUE
    os.environ["no_proxy"] = NO_PROXY_VALUE

