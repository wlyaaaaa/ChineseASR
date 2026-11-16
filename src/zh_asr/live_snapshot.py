"""On-demand, metadata-only inventory; never loads models or source recordings."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

from .model_lifecycle import write_json_atomic
from .retained_results import read_latest_complete_results
STATE_TEXT = dict(execution="执行", coverage="覆盖", quality="质量", completed="完成", unsupported="不支持", failed="失败", corrupt="损坏", complete="完整", partial="部分", unknown="未知", sufficient="有把握", low_confidence="把握较低")

SAMPLE_DIRS = ("acceptance", "batch-smoke", "benchmark-smoke", "closeout", "cloud-smoke",
               "e2e", "maintenance", "quality-acceptance", "smoke", "strict-", "a4-demo", "cloud-jobs")
SAMPLE_SOURCES = ("/eval/corpus/public-smoke-20260917/", "/asr-cloudfirst-acceptance/synthetic-zh.wav",
                  "/speech-check/wechat-warm-fixture-02.wav")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def _sample(source):
    return any(part in str(source).replace("\\", "/").lower() for part in SAMPLE_SOURCES)


def covered_ms(chunks):
    """Union overlapping chunk intervals; gaps and repeat calls are not duration."""
    end, total = 0, 0
    for start, stop in sorted((c["start_ms"], c["end_ms"]) for c in chunks):
        if start < 0 or stop < start:
            raise ValueError("invalid chunk interval")
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def collect(data_root, code_root):
    out = data_root / "outputs"
    if not out.is_dir():
        raise FileNotFoundError("retained output directory is unavailable")
    local, cloud = [], []
    states = {key: Counter(dict.fromkeys(values, 0)) for key, values in (
        ("execution", ("completed", "unsupported", "failed")), ("coverage", ("complete", "partial", "unknown")),
        ("quality", ("sufficient", "low_confidence", "unknown")))}

    def normalize(path, data):
        is_cloud = path.name.endswith(".result.json")
        if is_cloud:
            intent = _json(path.with_name(path.name.replace(".result.json", ".intent.json")))
            source = intent.get("audio_path", "")
            if _sample(source) or data.get("status") != "succeeded":
                return None
            transcript = path.with_name(path.name.replace(".result.json", ".transcript.txt"))
            row = dict(source_id=data.get("source_audio_sha256"), at=data["completed_utc"],
                       complete=transcript.is_file() and transcript.stat().st_size > 0,
                       duration_seconds=covered_ms(data.get("chunks", [])) / 1000, route="cloud")
            cloud.append(row)
        else:
            source = data.get("analysis", {}).get("source_path", "")
            if _sample(source):
                return None
            for key, counter in states.items():
                counter[data.get(key, {}).get("status") or "unknown"] += 1
            stem = path.name.removesuffix(".objective-result.json")
            candidates = ([path.parent / "transcript.md"] if path.name == "objective-result.json" else
                          [path.with_name(stem + ".strict.md"), path.with_name(stem + ".md")])
            transcript = next((p for p in candidates if p.is_file()), None)
            refs = data.get("raw_artifacts", [])
            complete = (data.get("execution", {}).get("status") == "completed"
                        and data.get("coverage", {}).get("status") == "complete"
                        and data.get("transcript_available") is True and bool(refs)
                        and all((path.parent / ref["path"]).is_file() for ref in refs))
            duration = data.get("coverage", {}).get("end_ms") or data.get("analysis", {}).get("duration_ms")
            row = dict(source_id=data.get("audio", {}).get("raw_sha256"),
                       at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                       complete=complete, duration_seconds=duration / 1000 if duration else None, route="local")
            local.append(row)
        row.update(source_name=Path(source).name if source else "未记名", transcript=transcript)
        return row

    paths = [p for p in out.rglob("*objective-result.json")
             if not p.relative_to(out).parts[0].startswith(SAMPLE_DIRS)]
    latest = read_latest_complete_results(paths + list((out / "cloud-jobs").glob("*.result.json")), normalize)
    local_ids = {r["source_id"] for r in local if r["source_id"]}
    complete = [r for r in local if r["complete"] and r["source_id"]]
    lc, cc = {r["source_id"] for r in complete}, {r["source_id"] for r in cloud if r["complete"] and r["source_id"]}
    durations = {identity: max(r["duration_seconds"] for r in cloud if r["source_id"] == identity and r["complete"])
                 for identity in cc}
    metrics = dict(complete_audio=len(lc | cc), local_complete=len(lc), cloud_complete=len(cc),
                   overlap=len(lc & cc), local_audio=len(local_ids), local_receipts=len(local),
                   local_complete_receipts=len(complete), cloud_results=sum(r["complete"] for r in cloud),
                   cloud_hours=round(sum(durations.values()) / 3600, 4),
                   cloud_latest=max((r["at"] for r in cloud if r["complete"] and r["at"]), default=None),
                   cloud_undated=sum(r["complete"] and not r["at"] for r in cloud),
                   cloud_unnamed=sum(r["complete"] and r["source_name"] == "未记名" for r in cloud),
                   local_missing_duration=sum(r["duration_seconds"] is None for r in complete),
                   local_missing_duration_audio=len({r["source_id"] for r in complete if r["duration_seconds"] is None}),
                   local_latest=max((r["at"] for r in complete), default=None))
    index = out / "api/file-jobs"
    metrics["file_job_success"] = sum(_json(p).get("status") == "succeeded" for p in index.glob("*.json") if re.fullmatch("[0-9a-f]{64}", p.stem)) if index.is_dir() else None
    jobs = out / "api/jobs.json"
    metrics["api_retained_jobs"] = len(_json(jobs)["jobs"]) if jobs.is_file() else None
    metrics["receipt_states"] = dict(text=f"{len(local)} 份回执；分三组统计，不是准确率", denominator=len(local),
                                    **{key: dict(value) for key, value in states.items()})
    receipt_rows = [dict(dimension=key, state=state, count=count, denominator=len(local),
                         text=f"{STATE_TEXT[key]} · {STATE_TEXT.get(state, state)}：{count}/{len(local)}", state_text=STATE_TEXT.get(state, state))
                    for key, values in states.items() for state, count in sorted(values.items())]
    models = data_root / "models"
    components = list((models / "modelscope").glob("*/*")) + list((models / "aligner").glob("*"))
    components += [p for p in (models / "firered").glob("*") if (p / "model.pth.tar").is_file()]
    metrics.update(model_components=sum(p.is_dir() for p in components) if models.is_dir() else None,
                   models_bytes=sum(p.stat().st_size for p in models.rglob("*") if p.is_file()) if models.is_dir() else None)
    metrics["model_list"] = [dict(text=p.name, state_text="已下载组件") for p in sorted(components) if p.is_dir()]
    log = out / "dictation/runtime.log"
    samples = []
    if log.is_file():
        for line in log.read_text(encoding="utf-8-sig").splitlines():
            match = re.search(r"model[^\n]*?operation=transcribe[^\n]*?seconds=([\d.]+)", line)
            if match:
                instant = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone(timedelta(hours=-7)))
                samples.append((instant.astimezone(timezone(timedelta(hours=8))).isoformat(), float(match[1])))
    metrics["dictation_history"] = (dict(text=f"历史模型返回中位 {statistics.median(v for _, v in samples):.3f} 秒；{len(samples)} 条；截至 {datetime.fromisoformat(samples[-1][0]):%m-%d %H:%M}",
                                        median_seconds=statistics.median(v for _, v in samples), samples=len(samples), through=samples[-1][0]) if samples else None)
    files = subprocess.check_output(["git", "-C", str(code_root), "ls-files", "-z"]).decode("utf-8").split("\0")
    size = dict(docs_chars=0, code_lines=0)
    for name in filter(None, files):
        path = code_root / name
        if path.suffix.lower() not in {".md", ".py", ".ps1", ".vbs", ".sh", ".gs"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        size["docs_chars" if path.suffix.lower() == ".md" else "code_lines"] += (
            sum(not c.isspace() for c in text) if path.suffix.lower() == ".md" else len(text.splitlines()))
    recent, named, artifacts, aliases = [], [], [], {"asr.transcripts": dict(path=str(out.resolve()), format="exists")}
    ordered = sorted(latest.values(), key=lambda r: datetime.fromisoformat(r["at"]).timestamp(), reverse=True)[:20]
    for index, row in enumerate(ordered, 1):
        route = "云端" if row["route"] == "cloud" else "本地"
        duration = f"{row['duration_seconds']:g} 秒" if row["duration_seconds"] is not None else "时长未记录"
        record = dict(at=row["at"], route=row["route"], duration_seconds=row["duration_seconds"], state_text="留存正文完整")
        recent.append(dict(record, text=f"稿件 {index:02d} · {route} · {duration}"))
        named.append(dict(record, text=row["source_name"]))
        path = row["transcript"]
        if path and path.is_file():
            artifacts.append(dict(at=row["at"], kind="transcript", title=f"{row['source_name']} · {route} · {duration}",
                                  status="completed", public="owner", open=dict(source="asr.transcripts", path=path.relative_to(out).as_posix())))
    metrics["recent"] = recent
    return metrics, receipt_rows, size, dict(metrics=dict(recent_named=named), artifacts=artifacts), aliases


def write_snapshot(data_root, code_root, output, owner_output, sources_output):
    output, owner_output, sources_output = (p.resolve() for p in (output, owner_output, sources_output))
    metrics, receipts, size, owner, aliases = collect(data_root, code_root)
    previous, now = _json(output), datetime.now(timezone.utc).isoformat()
    times = {key: previous.get("times", {}).get(key, now) if previous.get("metrics", {}).get(key) == value else now
             for key, value in metrics.items()}
    inventory = ("complete_audio", "local_complete", "cloud_complete", "overlap", "local_audio",
                 "local_receipts", "local_complete_receipts", "local_latest", "receipt_states")
    times["inventory"] = previous.get("times", {}).get("inventory", now) if all(
        previous.get("metrics", {}).get(key) == metrics[key] for key in inventory) else now
    size["at"] = previous.get("size", {}).get("at", now) if all(previous.get("size", {}).get(k) == v for k, v in size.items()) else now
    public = dict(v=1, at=now, checked_at=now, scope="排除明确样例的现存结果；不是全部历史或人工准确认证", metrics=metrics,
                  times=times, receipt_records=receipts, size=size, artifacts=[])
    public["at"] = previous.get("at", now) if all(previous.get(k) == v for k, v in public.items() if k not in {"at", "checked_at"}) else now
    owner.update(v=1, at=public["at"], scope="同原音合并后的最近二十份留存完整正文；文件名和正文仅本人")
    for path, payload in ((output, public), (owner_output, owner)):
        if _json(path) != payload:
            write_json_atomic(path, payload, newline="\n")
    registration = dict(sources={"asr.snapshot": dict(path=str(output)), "asr.owner": dict(path=str(owner_output)), **aliases})
    if owner_output.with_name("live.contract.json").is_file():
        registration["contract"] = dict(path=str(owner_output.with_name("live.contract.json")))
    write_json_atomic(sources_output, registration, newline="\n")
    manifest_path = owner_output.parent.parent.parent / "manifest.json"
    manifest = _json(manifest_path)
    for item in manifest.get("files", []):
        if (manifest_path.parent / item["target_relative_path"]).resolve() == owner_output:
            data = owner_output.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if item["sha256"] != digest:
                item.update(bytes=len(data), sha256=digest)
                manifest["backed_up_at_utc"] = now
    if manifest and _json(manifest_path) != manifest:
        write_json_atomic(manifest_path, manifest, newline="\n")
    return public


def refresh_after_change(data_root):
    from .process_control import managed_popen_kwargs, terminate_process_tree
    output = data_root / 'outputs'
    try:
        process = subprocess.Popen([sys.executable, '-B', '-m', 'zh_asr.live_snapshot', '--data-root', str(data_root),
            '--output', str(output/'live.snapshot.json'), '--owner-output', str(output/'live.owner.snapshot.json'),
            '--sources-output', str(output/'live.sources.json')], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **managed_popen_kwargs())
        try:
            if process.wait(timeout=120): raise subprocess.CalledProcessError(process.returncode, 'live_snapshot')
        finally: terminate_process_tree(process)
    except Exception as error:
        import logging
        logging.getLogger(__name__).warning('live_refresh_failed: %s', type(error).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner-output", type=Path, required=True)
    parser.add_argument("--sources-output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_snapshot(args.data_root, args.code_root, args.output, args.owner_output, args.sources_output)
    print(json.dumps(dict(at=result["at"], complete_audio=result["metrics"]["complete_audio"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
