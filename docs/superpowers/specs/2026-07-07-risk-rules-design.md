# Risk Rules Design

## Goal

Add a deterministic hallucination risk rule library for strict ASR audits. The library flags silent-audio text, stock phrases, abnormal repetition, large dual-model disagreement, traditional Chinese residue, and long unpunctuated output.

## Scope

The first implementation covers rule detection and audit serialization. It does not add an LLM judge, UI, or configurable YAML thresholds. Thresholds stay in code until real benchmark results justify moving them to config.

## Architecture

Create `zh_asr.risk_rules` as the only module that owns hallucination rule logic. It returns structured `RuleHit` objects with `id`, `severity`, `message`, and `evidence`.

`audit.py` uses the rules when building `AuditReport`. `strict_writer.py` writes rule hits into `*.strict.audit.json` and `*.strict.audit.md`. `eval_pack.py` re-runs the same deterministic rules with manifest context such as `expect_empty`, merges rule hits into metrics, and prints them in `review.md`.

## Rules

- `empty_audio_hallucination`: high severity when `expect_empty=true` and the final text contains substantive characters.
- `suspicious_stock_phrase`: high severity when output contains phrases like `谢谢观看`, `感谢观看`, `字幕组`, `点赞`, or `订阅`.
- `abnormal_repetition`: medium severity when a normalized text span repeats abnormally.
- `model_conflict`: medium severity below `0.95` similarity and high severity below `0.80`, only when both ASR engines produced substantive text.
- `traditional_residue`: medium severity when the final text still differs after Simplified Chinese conversion.
- `long_unpunctuated_text`: medium severity when a long Chinese text has too little punctuation.

## Output Contract

`AuditReport` gains `rule_hits`. `flags` remains a tuple of rule ids for backward compatibility.

`metrics.json` case objects gain `rule_hits`. `risk_flags` remains present and includes rule ids plus existing metric-only flags such as `high_cer`.

`review.md` lists rule ids and short evidence for any case requiring review.

## Privacy

Rule evidence should be short and local. It may include a short phrase or character counts, but it must not copy long transcript spans into evidence.

## Testing

Tests must cover all six rule ids, strict audit JSON/Markdown serialization, eval metrics/review propagation, and no-model fake strict paths.
