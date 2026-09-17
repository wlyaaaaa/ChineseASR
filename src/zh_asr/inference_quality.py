"""Small inference helpers shared by the file ASR adapters, not dictation UI."""
from __future__ import annotations
from typing import Any, Callable


class BatchItemFailure:
    """An isolated input failure; never treat its placeholder as successful ASR."""
    def __init__(self, error: Exception):
        self.error = error


def isolated_batch(
    inputs: list[str], batch: Callable[[list[str]], list[Any]],
    single: Callable[[str], Any],
) -> list[Any]:
    if not inputs:
        return []
    try:
        results = batch(inputs)
        if not isinstance(results, list) or len(results) != len(inputs):
            raise RuntimeError("ASR batch result cardinality does not match input cardinality")
        return results
    except Exception:
        # The batch path may fail on one corrupt clip or on its memory footprint.
        # At most one independent retry per item preserves successful neighbours.
        results = []
        for value in inputs:
            try:
                results.append(single(value))
            except Exception as error:
                results.append(BatchItemFailure(error))
        return results


def generation_diagnostics(model: Any, text: str) -> dict[str, Any]:
    """Report estimates honestly: the upstream wrapper does not expose finish_reason."""
    limit = getattr(model, "max_new_tokens", None)
    tokenizer = getattr(getattr(model, "processor", None), "tokenizer", None)
    count = None
    if tokenizer is not None:
        try:
            count = len(tokenizer.encode(text, add_special_tokens=False))
        except (AttributeError, TypeError, ValueError):
            pass
    suspected = bool(isinstance(limit, int) and count is not None and count >= max(1, limit - 8))
    return {
        "output_token_estimate": count,
        "max_new_tokens": limit if isinstance(limit, int) else None,
        "finish_reason": "not_exposed_by_runtime",
        "truncation_suspected": suspected,
    }


def collect_generation_warnings(result: Any) -> list[str]:
    warnings = set()
    def walk(value):
        if isinstance(value, dict):
            generation = value.get("generation")
            if isinstance(generation, dict) and generation.get("truncation_suspected"):
                warnings.add("truncation_suspected")
            for nested in value.values():
                if isinstance(nested, (list, dict)):
                    walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
    walk(result)
    return sorted(warnings)
