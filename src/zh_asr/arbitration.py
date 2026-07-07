from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ArbitrationConfig:
    enabled: bool = False
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen-main-v1:latest"
    fallback_model: str = "qwen3.6-27b-256k:latest"
    mode: str = "uncertain_only"
    temperature: float = 0.1
    keep_alive: int | str = 0
    timeout_sec: int = 120

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "ArbitrationConfig":
        data = dict(mapping or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            provider=str(data.get("provider", "ollama")).strip().lower(),
            base_url=str(data.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
            model=str(data.get("model", "qwen-main-v1:latest")).strip(),
            fallback_model=str(data.get("fallback_model", "qwen3.6-27b-256k:latest")).strip(),
            mode=str(data.get("mode", "uncertain_only")).strip().lower(),
            temperature=float(data.get("temperature", 0.1)),
            keep_alive=data.get("keep_alive", 0),
            timeout_sec=int(data.get("timeout_sec", 120)),
        )


@dataclass(frozen=True)
class ArbitrationEvidence:
    chunk_id: str
    time_range: str
    primary_text: str = ""
    secondary_text: str = ""
    previous_context: str = ""
    next_context: str = ""
    similarity: float = 0.0
    flags: list[str] = field(default_factory=list)
    rule_hits: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_audit(cls, chunk_id: str, time_range: str, audit: dict[str, Any]) -> "ArbitrationEvidence":
        return cls(
            chunk_id=chunk_id,
            time_range=time_range,
            primary_text=str(audit.get("primary_text", "")),
            secondary_text=str(audit.get("secondary_text", "")),
            similarity=float(audit.get("similarity", 0.0) or 0.0),
            flags=list(audit.get("flags", []) or []),
            rule_hits=list(audit.get("rule_hits", []) or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArbitrationDecision:
    final_text: str
    confidence: float
    decisions: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NullArbiter:
    def arbitrate(self, evidence: ArbitrationEvidence | dict[str, Any]) -> None:
        return None


PostJson = Callable[[str, dict[str, Any], int], dict[str, Any]]


class OllamaArbiter:
    def __init__(self, config: ArbitrationConfig, post_json: PostJson | None = None) -> None:
        self.config = config
        self._post_json = post_json or _post_json

    def arbitrate(self, evidence: ArbitrationEvidence | dict[str, Any]) -> ArbitrationDecision:
        evidence_obj = evidence if isinstance(evidence, ArbitrationEvidence) else _evidence_from_raw(evidence)
        payload = self._build_payload(evidence_obj)
        try:
            response = self._post_json(f"{self.config.base_url}/api/chat", payload, self.config.timeout_sec)
            content = str(response.get("message", {}).get("content", ""))
            data = json.loads(content)
            return ArbitrationDecision(
                final_text=str(data.get("final_text", "")),
                confidence=float(data.get("confidence", 0.0) or 0.0),
                decisions=list(data.get("decisions", []) or []),
                unresolved=list(data.get("unresolved", []) or []),
            )
        except Exception as exc:
            return ArbitrationDecision(
                final_text="",
                confidence=0.0,
                unresolved=[{"label": "arbitration_failed", "reason": f"{type(exc).__name__}: {exc}"}],
                error=f"{type(exc).__name__}: {exc}",
            )

    def _build_payload(self, evidence: ArbitrationEvidence) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "stream": False,
            "format": "json",
            "keep_alive": self.config.keep_alive,
            "options": {"temperature": self.config.temperature},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是中文ASR仲裁器。只根据给定证据判断，不要编造。"
                        "输出JSON，字段为 final_text, confidence, decisions, unresolved。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2),
                },
            ],
        }


def make_arbiter(config: ArbitrationConfig):
    if not config.enabled:
        return NullArbiter()
    if config.provider != "ollama":
        raise ValueError(f"Unsupported arbitration provider: {config.provider}")
    return OllamaArbiter(config)


def load_arbitration_config(path: Path) -> ArbitrationConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read llm_arbitration config.") from exc
    if not path.exists():
        return ArbitrationConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return ArbitrationConfig()
    section = data.get("llm_arbitration", {})
    return ArbitrationConfig.from_mapping(section if isinstance(section, dict) else {})


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _evidence_from_raw(raw: dict[str, Any]) -> ArbitrationEvidence:
    return ArbitrationEvidence(
        chunk_id=str(raw.get("chunk_id", "")),
        time_range=str(raw.get("time_range", "")),
        primary_text=str(raw.get("primary_text", "")),
        secondary_text=str(raw.get("secondary_text", "")),
        similarity=float(raw.get("similarity", 0.0) or 0.0),
        flags=list(raw.get("flags", []) or []),
        rule_hits=list(raw.get("rule_hits", []) or []),
    )
