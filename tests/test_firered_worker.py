from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _write_wav(
    path: Path,
    *,
    duration_sec: float = 0.25,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    frame_count = int(duration_sec * sample_rate)
    frame = b"\0" * sample_width * channels
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(frame * frame_count)


def _engine_spec(model_dir: Path, **options: object):
    from zh_asr.config import EngineSpec

    defaults: dict[str, object] = {
        "worker_command": [sys.executable],
        "timeout_sec": 12,
    }
    defaults.update(options)
    return EngineSpec(
        name="fireredasr2-llm",
        adapter="firered-worker",
        role="primary",
        model=str(model_dir),
        language="Chinese",
        options=defaults,
    )


def _load_runtime_worker():
    path = Path(__file__).resolve().parents[1] / "runtime" / "firered_worker.py"
    spec = importlib.util.spec_from_file_location("test_runtime_firered_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load worker module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_valid_model_receipt(
    worker,
    model_dir: Path,
    *,
    revision: str = "expected-revision",
) -> Path:
    records = []
    for index, relative_path in enumerate(worker.MODEL_REQUIRED_FILES):
        target = model_dir / Path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = f"artifact-{index}:{relative_path}".encode("utf-8")
        target.write_bytes(payload)
        records.append(
            {
                "path": relative_path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    receipt = {
        "schema": worker.MODEL_RECEIPT_SCHEMA,
        "repository": worker.MODEL_REPOSITORY,
        "revision": revision,
        "created_utc": "2026-07-30T00:00:00+00:00",
        "files": records,
    }
    receipt_path = model_dir / "MODEL_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return receipt_path


def _git(source_dir: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_dir), *args],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed.stdout.strip()


def _fake_firered_runtime_modules(
    *,
    llm_module: ModuleType,
    fire_red_asr: type,
    fire_red_config: type,
    bfloat16: object,
    float16: object,
    bf16_supported: bool,
) -> dict[str, ModuleType]:
    torch_module = ModuleType("torch")
    torch_module.bfloat16 = bfloat16
    torch_module.float16 = float16
    torch_module.cuda = SimpleNamespace(
        is_bf16_supported=lambda: bf16_supported,
    )

    root_package = ModuleType("fireredasr2s")
    root_package.__path__ = []
    api_package = ModuleType("fireredasr2s.fireredasr2")
    api_package.__path__ = []
    api_package.FireRedAsr2 = fire_red_asr
    api_package.FireRedAsr2Config = fire_red_config
    models_package = ModuleType("fireredasr2s.fireredasr2.models")
    models_package.__path__ = []
    models_package.fireredasr_llm = llm_module
    api_package.models = models_package
    root_package.fireredasr2 = api_package

    return {
        "torch": torch_module,
        "fireredasr2s": root_package,
        "fireredasr2s.fireredasr2": api_package,
        "fireredasr2s.fireredasr2.models": models_package,
        "fireredasr2s.fireredasr2.models.fireredasr_llm": llm_module,
    }


def _meminfo_text(
    *,
    memory_gib: int,
    swap_gib: int,
    memory_available_gib: int | None = None,
    swap_free_gib: int | None = None,
) -> str:
    kib_per_gib = 1024 * 1024
    if memory_available_gib is None:
        memory_available_gib = memory_gib
    if swap_free_gib is None:
        swap_free_gib = swap_gib
    return "\n".join(
        [
            f"MemTotal:       {memory_gib * kib_per_gib} kB",
            f"MemAvailable:   {memory_available_gib * kib_per_gib} kB",
            f"SwapTotal:      {swap_gib * kib_per_gib} kB",
            f"SwapFree:       {swap_free_gib * kib_per_gib} kB",
            "",
        ]
    )


class FireRedWaveContractTests(unittest.TestCase):
    def test_accepts_16khz_16bit_mono_pcm_at_limit(self):
        from zh_asr.adapters.firered_worker import inspect_firered_wav

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "valid.wav"
            _write_wav(audio, duration_sec=40.0)

            info = inspect_firered_wav(audio)

        self.assertEqual(info.sample_rate, 16_000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.sample_width_bytes, 2)
        self.assertEqual(info.duration_sec, 40.0)

    def test_rejects_audio_outside_contract(self):
        from zh_asr.adapters.firered_worker import InvalidFireRedAudio, inspect_firered_wav

        cases = [
            ({"sample_rate": 8_000}, "16000 Hz"),
            ({"channels": 2}, "mono"),
            ({"sample_width": 1}, "16-bit"),
            ({"duration_sec": 40.01}, "40"),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as tmp:
                    audio = Path(tmp) / "invalid.wav"
                    _write_wav(audio, **overrides)
                    with self.assertRaisesRegex(InvalidFireRedAudio, expected):
                        inspect_firered_wav(audio)

    def test_configured_maximum_can_only_tighten_official_limit(self):
        from zh_asr.adapters.firered_worker import InvalidFireRedAudio, inspect_firered_wav

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "ten-seconds.wav"
            _write_wav(audio, duration_sec=10.0)
            with self.assertRaisesRegex(InvalidFireRedAudio, "5"):
                inspect_firered_wav(audio, max_audio_sec=5)

            too_long = Path(tmp) / "forty-one-seconds.wav"
            _write_wav(too_long, duration_sec=41.0)
            with self.assertRaisesRegex(InvalidFireRedAudio, "40"):
                inspect_firered_wav(too_long, max_audio_sec=60)

    def test_non_finite_duration_limit_is_rejected(self):
        from zh_asr.adapters.firered_worker import inspect_firered_wav

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            _write_wav(audio)
            with self.assertRaisesRegex(ValueError, "finite"):
                inspect_firered_wav(audio, max_audio_sec=float("nan"))

    def test_truncated_pcm_payload_is_rejected_before_worker_start(self):
        from zh_asr.adapters.firered_worker import InvalidFireRedAudio, inspect_firered_wav

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "truncated.wav"
            _write_wav(audio)
            audio.write_bytes(audio.read_bytes()[:-10])
            with self.assertRaisesRegex(InvalidFireRedAudio, "truncated"):
                inspect_firered_wav(audio)


class FireRedAdapterTests(unittest.TestCase):
    def test_adapter_is_registered(self):
        from zh_asr.adapters import get_adapter

        self.assertEqual(get_adapter("firered-worker").name, "firered-worker")

    def test_worker_invocation_uses_json_protocol_and_normalizes_text(self):
        from zh_asr.adapters.firered_worker import (
            FireRedWorkerAdapter,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            audio = root / "sample.wav"
            _write_wav(audio)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "zh_asr.firered_worker.response.v1",
                        "ok": True,
                        "result": [{"text": "可以來一樓換票。", "rtf": "0.1000"}],
                    },
                    ensure_ascii=False,
                ),
                stderr="model diagnostics",
            )

            with patch(
                "zh_asr.adapters.firered_worker.subprocess.run",
                return_value=completed,
            ) as run:
                wrapper = FireRedWorkerAdapter().build_model(
                    _engine_spec(model_dir, beam_size=1),
                    "cuda:0",
                    root / "cache",
                    {},
                )
                result = wrapper.generate(input=str(audio), batch_size_s=300)

            command = run.call_args.args[0]
            request = json.loads(run.call_args.kwargs["input"])

        self.assertEqual(command[0], sys.executable)
        self.assertTrue(command[-1].replace("\\", "/").endswith("/runtime/firered_worker.py"))
        self.assertEqual(request["schema"], "zh_asr.firered_worker.request.v1")
        self.assertEqual(request["audio_path"], str(audio))
        self.assertEqual(request["model_dir"], str(model_dir))
        self.assertEqual(request["device"], "cuda:0")
        self.assertEqual(request["options"]["beam_size"], 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 12.0)
        self.assertEqual(
            result,
            [{"text": "可以来一楼换票。", "rtf": "0.1000", "original_text": "可以來一樓換票。"}],
        )

    def test_wsl_path_style_translates_windows_request_paths(self):
        from zh_asr.adapters.firered_worker import FireRedWorkerAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            audio = root / "sample.wav"
            _write_wav(audio)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "zh_asr.firered_worker.response.v1",
                        "ok": True,
                        "result": [{"text": "测试"}],
                    }
                ),
                stderr="",
            )
            spec = _engine_spec(
                model_dir,
                worker_command=["wsl.exe", "-d", "Ubuntu-22.04", "--", "python3"],
                path_style="wsl",
            )

            with patch(
                "zh_asr.adapters.firered_worker.subprocess.run",
                return_value=completed,
            ) as run:
                wrapper = FireRedWorkerAdapter().build_model(spec, "cuda:0", root / "cache", {})
                wrapper.generate(input=str(audio))

            command = run.call_args.args[0]
            request = json.loads(run.call_args.kwargs["input"])

        self.assertEqual(command[:6], ["wsl.exe", "-d", "Ubuntu-22.04", "--", "python3", command[5]])
        self.assertTrue(command[-1].startswith("/mnt/"))
        self.assertTrue(request["audio_path"].startswith("/mnt/"))
        self.assertTrue(request["model_dir"].startswith("/mnt/"))
        self.assertEqual(run.call_args.kwargs["env"]["WSL_UTF8"], "1")

    def test_registry_wsl_shorthand_builds_isolated_command(self):
        from zh_asr.adapters.firered_worker import FireRedWorkerAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            source_dir = root / "source"
            source_dir.mkdir()
            spec = _engine_spec(
                model_dir,
                worker_command=None,
                runtime="wsl",
                wsl_distribution="Ubuntu",
                python_path="/opt/chineseasr/firered/.venv/bin/python",
                source_dir=str(source_dir),
            )

            wrapper = FireRedWorkerAdapter().build_model(spec, "cuda:0", root / "cache", {})

        self.assertEqual(
            wrapper.command[:5],
            [
                "wsl.exe",
                "-d",
                "Ubuntu",
                "--",
                "/opt/chineseasr/firered/.venv/bin/python",
            ],
        )
        self.assertEqual(wrapper.path_style, "wsl")
        self.assertTrue(wrapper.source_dir.startswith("/mnt/"))

    def test_timeout_and_worker_failure_are_actionable(self):
        from zh_asr.adapters.firered_worker import (
            FireRedWorkerAdapter,
            FireRedWorkerError,
            FireRedWorkerTimeout,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            audio = root / "sample.wav"
            _write_wav(audio)
            wrapper = FireRedWorkerAdapter().build_model(
                _engine_spec(model_dir, timeout_sec=7),
                "cuda:0",
                root / "cache",
                {},
            )

            with patch(
                "zh_asr.adapters.firered_worker.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=7),
            ):
                with self.assertRaisesRegex(FireRedWorkerTimeout, "7.*sample.wav"):
                    wrapper.generate(input=str(audio))

            failed = subprocess.CompletedProcess(
                args=[],
                returncode=3,
                stdout="",
                stderr="CUDA out of memory",
            )
            with patch("zh_asr.adapters.firered_worker.subprocess.run", return_value=failed):
                with self.assertRaisesRegex(FireRedWorkerError, "exit code 3.*CUDA out of memory"):
                    wrapper.generate(input=str(audio))

    def test_wsl_timeout_cleans_token_bound_descendants(self):
        from zh_asr.adapters.firered_worker import FireRedWorkerAdapter, FireRedWorkerTimeout
        from zh_asr.process_control import PROCESS_TOKEN_ENV

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            audio = root / "sample.wav"
            _write_wav(audio)
            spec = _engine_spec(
                model_dir,
                worker_command=["wsl.exe", "-d", "Ubuntu-22.04", "--", "python3"],
                path_style="wsl",
                timeout_sec=7,
            )
            wrapper = FireRedWorkerAdapter().build_model(
                spec,
                "cuda:0",
                root / "cache",
                {},
            )
            with (
                patch.dict(os.environ, {PROCESS_TOKEN_ENV: "chineseasr-timeout-test"}),
                patch(
                    "zh_asr.adapters.firered_worker.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd=["wsl.exe"], timeout=7),
                ),
                patch(
                    "zh_asr.adapters.firered_worker.terminate_wsl_processes"
                ) as cleanup,
            ):
                with self.assertRaisesRegex(FireRedWorkerTimeout, "7.*sample.wav"):
                    wrapper.generate(input=str(audio))

        cleanup.assert_called_once_with(("Ubuntu-22.04",), "chineseasr-timeout-test")

    def test_generate_many_uses_one_worker_process_for_multiple_batch_one_inputs(self):
        from zh_asr.adapters.firered_worker import FireRedWorkerAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            first = root / "first.wav"
            second = root / "second.wav"
            _write_wav(first)
            _write_wav(second)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "zh_asr.firered_worker.response.v1",
                        "ok": True,
                        "result": [
                            {"uttid": "first", "text": "第一段"},
                            {"uttid": "second", "text": "第二段"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                stderr="",
            )
            with patch(
                "zh_asr.adapters.firered_worker.subprocess.run",
                return_value=completed,
            ) as run:
                wrapper = FireRedWorkerAdapter().build_model(
                    _engine_spec(model_dir),
                    "cuda:0",
                    root / "cache",
                    {},
                )
                result = wrapper.generate_many([str(first), str(second)])

            request = json.loads(run.call_args.kwargs["input"])

        self.assertEqual(run.call_count, 1)
        self.assertEqual(request["audio_paths"], [str(first), str(second)])
        self.assertEqual([item["text"] for item in result], ["第一段", "第二段"])

    def test_protocol_requires_boolean_success(self):
        from zh_asr.adapters.firered_worker import FireRedWorkerAdapter, FireRedWorkerError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            audio = root / "sample.wav"
            _write_wav(audio)
            malformed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "zh_asr.firered_worker.response.v1",
                        "ok": "true",
                        "result": [{"text": "不应接受"}],
                    }
                ),
                stderr="",
            )
            with patch("zh_asr.adapters.firered_worker.subprocess.run", return_value=malformed):
                wrapper = FireRedWorkerAdapter().build_model(
                    _engine_spec(model_dir),
                    "cuda:0",
                    root / "cache",
                    {},
                )
                with self.assertRaisesRegex(FireRedWorkerError, "boolean"):
                    wrapper.generate(input=str(audio))


class FireRedRuntimeWorkerTests(unittest.TestCase):
    def test_runtime_worker_parses_proc_meminfo_kib_values(self):
        worker = _load_runtime_worker()

        parsed = worker._parse_proc_meminfo(
            "\n".join(
                [
                    "MemTotal:       32862328 kB",
                    "MemAvailable:   31326148 kB",
                    "SwapTotal:       8388608 kB",
                    "SwapFree:        8388608 kB",
                ]
            )
        )

        self.assertEqual(parsed["MemTotal"], 32862328 * 1024)
        self.assertEqual(parsed["MemAvailable"], 31326148 * 1024)
        self.assertEqual(parsed["SwapTotal"], 8388608 * 1024)
        self.assertEqual(parsed["SwapFree"], 8388608 * 1024)
        with self.assertRaisesRegex(ValueError, "MemTotal"):
            worker._parse_proc_meminfo("SwapTotal: 4194304 kB\n")
        with self.assertRaisesRegex(ValueError, "MemAvailable"):
            worker._parse_proc_meminfo(
                "MemTotal: 33554432 kB\n"
                "SwapTotal: 8388608 kB\n"
                "SwapFree: 8388608 kB\n"
            )
        with self.assertRaisesRegex(ValueError, "unit"):
            worker._parse_proc_meminfo("MemTotal: 16 GB\nSwapTotal: 4 GB\n")

    def test_runtime_worker_wsl_memory_preflight_rejects_16_plus_4_before_hashing(self):
        worker = _load_runtime_worker()

        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text(
                _meminfo_text(memory_gib=16, swap_gib=4),
                encoding="utf-8",
            )
            with (
                patch.object(worker.sys, "platform", "linux"),
                self.assertRaisesRegex(
                    RuntimeError,
                    r"16\.0 GiB.*4\.0 GiB.*memory=32GB.*swap=8GB.*wsl --shutdown",
                ),
            ):
                worker._preflight_wsl_memory(
                    device="cuda:0",
                    use_half=True,
                    meminfo_path=meminfo,
                )

        with (
            patch.object(
                worker,
                "_preflight_wsl_memory",
                side_effect=RuntimeError("insufficient WSL memory"),
            ),
            patch.object(worker, "_verify_model_revision") as verify_model,
            self.assertRaisesRegex(RuntimeError, "insufficient WSL memory"),
        ):
            worker._load_model("model", "cuda:0", {"use_half": True}, "source")

        verify_model.assert_not_called()

    def test_runtime_worker_wsl_memory_preflight_allows_32_plus_8_half_load(self):
        worker = _load_runtime_worker()

        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text(
                _meminfo_text(memory_gib=32, swap_gib=8),
                encoding="utf-8",
            )
            with patch.object(worker.sys, "platform", "linux"):
                worker._preflight_wsl_memory(
                    device="cuda:0",
                    use_half=True,
                    meminfo_path=meminfo,
                )

    def test_runtime_worker_wsl_memory_preflight_skips_non_linux_but_fails_closed_on_missing_proc(
        self,
    ):
        worker = _load_runtime_worker()

        with patch.object(worker.sys, "platform", "win32"):
            worker._preflight_wsl_memory(
                device="cuda:0",
                use_half=True,
                meminfo_path=Path("missing"),
            )
        with (
            patch.object(worker.sys, "platform", "linux"),
            self.assertRaisesRegex(
                RuntimeError,
                r"could not read.*currently available.*free -h",
            ),
        ):
            worker._preflight_wsl_memory(
                device="cuda:0",
                use_half=True,
                meminfo_path=Path("definitely-missing-meminfo"),
            )

    def test_runtime_worker_wsl_memory_preflight_rejects_low_current_availability(
        self,
    ):
        worker = _load_runtime_worker()

        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text(
                _meminfo_text(
                    memory_gib=32,
                    swap_gib=8,
                    memory_available_gib=1,
                    swap_free_gib=0,
                ),
                encoding="utf-8",
            )
            with (
                patch.object(worker.sys, "platform", "linux"),
                self.assertRaisesRegex(
                    RuntimeError,
                    r"currently available capacity.*MemAvailable=1\.0 GiB.*"
                    r"SwapFree=0\.0 GiB.*configured capacity is sufficient.*"
                    r"Stop other WSL workloads",
                ),
            ):
                worker._preflight_wsl_memory(
                    device="cuda:0",
                    use_half=True,
                    meminfo_path=meminfo,
                )

    def test_runtime_worker_wsl_memory_preflight_enforces_each_availability_floor(
        self,
    ):
        worker = _load_runtime_worker()

        cases = (
            ("available-ram", 17, 8, r"MemAvailable=17\.0 GiB.*at least 18 GiB"),
            (
                "available-combined",
                18,
                3,
                r"currently available combined=21\.0 GiB.*22 GiB",
            ),
        )
        for name, memory_available_gib, swap_free_gib, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                meminfo = Path(tmp) / "meminfo"
                meminfo.write_text(
                    _meminfo_text(
                        memory_gib=32,
                        swap_gib=8,
                        memory_available_gib=memory_available_gib,
                        swap_free_gib=swap_free_gib,
                    ),
                    encoding="utf-8",
                )
                with (
                    patch.object(worker.sys, "platform", "linux"),
                    self.assertRaisesRegex(RuntimeError, expected),
                ):
                    worker._preflight_wsl_memory(
                        device="cuda:0",
                        use_half=True,
                        meminfo_path=meminfo,
                    )

    def test_runtime_worker_fp32_memory_preflight_is_stricter_than_half(self):
        worker = _load_runtime_worker()

        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text(
                _meminfo_text(memory_gib=32, swap_gib=8),
                encoding="utf-8",
            )
            with (
                patch.object(worker.sys, "platform", "linux"),
                self.assertRaisesRegex(RuntimeError, r"FP32.*use_half=true"),
            ):
                worker._preflight_wsl_memory(
                    device="cuda:0",
                    use_half=False,
                    meminfo_path=meminfo,
                )

    def test_runtime_worker_initial_llm_load_uses_requested_cuda_half_dtype(self):
        worker = _load_runtime_worker()

        for bf16_supported, expected_name in (
            (True, "bfloat16"),
            (False, "float16"),
        ):
            with self.subTest(bf16_supported=bf16_supported):
                load_calls: list[dict[str, object]] = []
                bfloat16 = object()
                float16 = object()
                expected_dtype = bfloat16 if bf16_supported else float16

                class OriginalAutoModelForCausalLM:
                    @classmethod
                    def from_pretrained(cls, *_args, **kwargs):
                        load_calls.append(dict(kwargs))
                        return object()

                llm_module = ModuleType(
                    "fireredasr2s.fireredasr2.models.fireredasr_llm"
                )
                llm_module.AutoModelForCausalLM = OriginalAutoModelForCausalLM

                class FakeFireRedAsr2Config:
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs

                class FakeFireRedAsr2:
                    @classmethod
                    def from_pretrained(cls, *_args):
                        llm_module.AutoModelForCausalLM.from_pretrained(
                            "qwen2",
                            torch_dtype="official-float32",
                        )
                        return SimpleNamespace()

                modules = _fake_firered_runtime_modules(
                    llm_module=llm_module,
                    fire_red_asr=FakeFireRedAsr2,
                    fire_red_config=FakeFireRedAsr2Config,
                    bfloat16=bfloat16,
                    float16=float16,
                    bf16_supported=bf16_supported,
                )
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(worker, "_verify_model_revision"),
                    patch.object(worker, "_verify_source_revision"),
                    patch.dict(sys.modules, modules),
                ):
                    root = Path(tmp)
                    model_dir = root / "model"
                    source_dir = root / "source"
                    model_dir.mkdir()
                    source_dir.mkdir()
                    model = worker._load_model(
                        str(model_dir),
                        "cuda:0",
                        {"use_half": True},
                        str(source_dir),
                    )

                self.assertEqual(len(load_calls), 1)
                self.assertIs(load_calls[0]["torch_dtype"], expected_dtype)
                self.assertIs(
                    llm_module.AutoModelForCausalLM,
                    OriginalAutoModelForCausalLM,
                )
                self.assertEqual(model._zh_asr_llm_load_dtype, expected_name)

    def test_runtime_worker_restores_official_loader_after_half_load_failure(self):
        worker = _load_runtime_worker()

        class OriginalAutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                raise RuntimeError("simulated load failure")

        llm_module = ModuleType(
            "fireredasr2s.fireredasr2.models.fireredasr_llm"
        )
        llm_module.AutoModelForCausalLM = OriginalAutoModelForCausalLM

        class FakeFireRedAsr2Config:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeFireRedAsr2:
            @classmethod
            def from_pretrained(cls, *_args):
                return llm_module.AutoModelForCausalLM.from_pretrained("qwen2")

        modules = _fake_firered_runtime_modules(
            llm_module=llm_module,
            fire_red_asr=FakeFireRedAsr2,
            fire_red_config=FakeFireRedAsr2Config,
            bfloat16=object(),
            float16=object(),
            bf16_supported=True,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(worker, "_verify_model_revision"),
            patch.object(worker, "_verify_source_revision"),
            patch.dict(sys.modules, modules),
        ):
            root = Path(tmp)
            model_dir = root / "model"
            source_dir = root / "source"
            model_dir.mkdir()
            source_dir.mkdir()
            with self.assertRaisesRegex(RuntimeError, "simulated load failure"):
                worker._load_model(
                    str(model_dir),
                    "cuda:0",
                    {"use_half": True},
                    str(source_dir),
                )

        self.assertIs(
            llm_module.AutoModelForCausalLM,
            OriginalAutoModelForCausalLM,
        )

    def test_runtime_worker_verifies_canonical_model_receipt_and_all_artifacts(self):
        worker = _load_runtime_worker()

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            receipt = _write_valid_model_receipt(worker, model_dir)

            worker._verify_model_revision(model_dir, "expected-revision")
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                worker._verify_model_revision(model_dir, "different-revision")
            receipt.unlink()
            with self.assertRaisesRegex(RuntimeError, "receipt is missing"):
                worker._verify_model_revision(model_dir, "expected-revision")

    def test_runtime_worker_rejects_wrong_receipt_schema_or_repository(self):
        worker = _load_runtime_worker()

        for field, value, expected in (
            ("schema", "unknown.schema", "schema mismatch"),
            ("repository", "somebody/other-model", "repository mismatch"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                model_dir = Path(tmp) / "model"
                model_dir.mkdir()
                receipt_path = _write_valid_model_receipt(worker, model_dir)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt[field] = value
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, expected):
                    worker._verify_model_revision(model_dir, "expected-revision")

    def test_runtime_worker_rejects_missing_duplicate_extra_or_escaping_receipt_paths(self):
        worker = _load_runtime_worker()

        mutations = {
            "missing": lambda files: files.pop(),
            "duplicate": lambda files: files.append(dict(files[0])),
            "extra": lambda files: files.append(
                {"path": "unexpected.bin", "bytes": 1, "sha256": "0" * 64}
            ),
            "escape": lambda files: files.__setitem__(
                0, {"path": "../escape.bin", "bytes": 1, "sha256": "0" * 64}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                model_dir = Path(tmp) / "model"
                model_dir.mkdir()
                receipt_path = _write_valid_model_receipt(worker, model_dir)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                mutate(receipt["files"])
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

                with self.assertRaisesRegex(
                    RuntimeError,
                    "duplicate|missing|required file list|unsafe",
                ):
                    worker._verify_model_revision(model_dir, "expected-revision")

    def test_runtime_worker_rejects_size_or_sha256_tampering(self):
        worker = _load_runtime_worker()

        for name, mutate, expected in (
            (
                "declared-size",
                lambda receipt, _model_dir: receipt["files"][0].__setitem__(
                    "bytes", receipt["files"][0]["bytes"] + 1
                ),
                "size mismatch",
            ),
            (
                "declared-sha",
                lambda receipt, _model_dir: receipt["files"][0].__setitem__(
                    "sha256", "0" * 64
                ),
                "SHA-256 mismatch",
            ),
            (
                "artifact-content",
                lambda receipt, model_dir: (
                    model_dir / receipt["files"][0]["path"]
                ).write_bytes(b"tampered"),
                "size mismatch|SHA-256 mismatch",
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                model_dir = Path(tmp) / "model"
                model_dir.mkdir()
                receipt_path = _write_valid_model_receipt(worker, model_dir)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                mutate(receipt, model_dir)
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, expected):
                    worker._verify_model_revision(model_dir, "expected-revision")

    def test_runtime_worker_requires_pinned_clean_source_but_allows_ignored_egg_info(self):
        worker = _load_runtime_worker()

        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            _git(source_dir, "init")
            _git(source_dir, "config", "user.email", "test@example.invalid")
            _git(source_dir, "config", "user.name", "Test")
            (source_dir / ".gitignore").write_text("*.egg-info/\n", encoding="utf-8")
            tracked = source_dir / "tracked.py"
            tracked.write_text("clean = True\n", encoding="utf-8")
            _git(source_dir, "add", ".gitignore", "tracked.py")
            _git(source_dir, "commit", "-m", "fixture")
            expected_revision = _git(source_dir, "rev-parse", "HEAD")

            ignored = source_dir / "fireredasr2s.egg-info" / "PKG-INFO"
            ignored.parent.mkdir()
            ignored.write_text("ignored build metadata\n", encoding="utf-8")
            worker._verify_source_revision(source_dir, expected_revision)

            tracked.write_text("clean = False\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "working tree is dirty.*tracked.py"):
                worker._verify_source_revision(source_dir, expected_revision)
            _git(source_dir, "checkout", "--", "tracked.py")

            (source_dir / "untracked.txt").write_text("not allowed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "working tree is dirty.*untracked.txt"):
                worker._verify_source_revision(source_dir, expected_revision)

    def test_runtime_worker_normalizes_windows_checkout_line_endings_for_git_checks(self):
        worker = _load_runtime_worker()

        with patch.object(worker.subprocess, "run") as run:
            run.side_effect = [
                SimpleNamespace(
                    returncode=0,
                    stdout="expected-revision\n",
                    stderr="",
                ),
                SimpleNamespace(returncode=0, stdout="", stderr=""),
            ]

            worker._verify_source_revision(Path("/tmp/source"), "expected-revision")

        self.assertEqual("git", run.call_args_list[0].args[0][0])
        self.assertEqual(
            ["-c", "core.autocrlf=true"],
            run.call_args_list[0].args[0][1:3],
        )
        self.assertEqual(
            ["-c", "core.autocrlf=true"],
            run.call_args_list[1].args[0][1:3],
        )

    def test_runtime_worker_uses_official_llm_api_shape(self):
        worker = _load_runtime_worker()

        class FakeModel:
            _zh_asr_llm_load_dtype = "bfloat16"

            def transcribe(self, uttids, wav_paths):
                self.uttids = uttids
                self.wav_paths = wav_paths
                return [{"uttid": uttids[0], "text": "可以去一楼换票", "rtf": "0.1"}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            audio = root / "sample.wav"
            _write_wav(audio)
            fake_model = FakeModel()

            response = worker.run_request(
                {
                    "schema": "zh_asr.firered_worker.request.v1",
                    "audio_path": str(audio),
                    "model_dir": str(model_dir),
                    "device": "cuda:0",
                    "options": {},
                },
                model_loader=lambda *_args, **_kwargs: fake_model,
            )

        self.assertTrue(response["ok"])
        self.assertEqual(response["result"][0]["text"], "可以去一楼换票")
        self.assertEqual(fake_model.wav_paths, [str(audio)])
        self.assertEqual(
            response["diagnostics"]["llm_initial_load_dtype"],
            "bfloat16",
        )
        self.assertEqual(
            response["result"][0]["_zh_asr_runtime"]["llm_initial_load_dtype"],
            "bfloat16",
        )

    def test_runtime_worker_loads_once_and_transcribes_multiple_inputs_as_batch_one(self):
        worker = _load_runtime_worker()

        class FakeModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, uttids, wav_paths):
                self.calls.append((list(uttids), list(wav_paths)))
                return [{"uttid": uttids[0], "text": Path(wav_paths[0]).stem}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            first = root / "first.wav"
            second = root / "second.wav"
            _write_wav(first)
            _write_wav(second)
            fake_model = FakeModel()
            load_count = 0

            def load_model(*_args, **_kwargs):
                nonlocal load_count
                load_count += 1
                return fake_model

            response = worker.run_request(
                {
                    "schema": "zh_asr.firered_worker.request.v1",
                    "audio_paths": [str(first), str(second)],
                    "model_dir": str(model_dir),
                    "device": "cuda:0",
                    "options": {"batch_size": 1},
                },
                model_loader=load_model,
            )

        self.assertEqual(load_count, 1)
        self.assertEqual(len(fake_model.calls), 2)
        self.assertEqual([item["text"] for item in response["result"]], ["first", "second"])

    def test_runtime_worker_json_protocol_runs_in_a_short_lived_process(self):
        worker = _load_runtime_worker()
        runtime = Path(__file__).resolve().parents[1] / "runtime" / "firered_worker.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            package_dir = source_dir / "fireredasr2s" / "fireredasr2"
            package_dir.mkdir(parents=True)
            (source_dir / "fireredasr2s" / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "__init__.py").write_text(
                "\n".join(
                    [
                        "class FireRedAsr2Config:",
                        "    def __init__(self, **kwargs): self.kwargs = kwargs",
                        "class _Model:",
                        "    def transcribe(self, uttids, wav_paths):",
                        "        return [{'uttid': uttids[0], 'text': '短命进程成功', 'wav': wav_paths[0]}]",
                        "class FireRedAsr2:",
                        "    @staticmethod",
                        "    def from_pretrained(kind, model_dir, config): return _Model()",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (source_dir / ".gitignore").write_text(
                "__pycache__/\n*.pyc\n*.egg-info/\n",
                encoding="utf-8",
            )
            _git(source_dir, "init")
            _git(source_dir, "config", "user.email", "test@example.invalid")
            _git(source_dir, "config", "user.name", "Test")
            _git(source_dir, "add", ".")
            _git(source_dir, "commit", "-m", "fixture")
            source_revision = _git(source_dir, "rev-parse", "HEAD")
            model_dir = root / "model"
            model_dir.mkdir()
            _write_valid_model_receipt(worker, model_dir)
            audio = root / "sample.wav"
            _write_wav(audio)
            request = {
                "schema": "zh_asr.firered_worker.request.v1",
                "audio_path": str(audio),
                "model_dir": str(model_dir),
                "source_dir": str(source_dir),
                "device": "cpu",
                "options": {
                    "model_revision": "expected-revision",
                    "source_revision": source_revision,
                },
            }

            completed = subprocess.run(
                [sys.executable, str(runtime)],
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=20,
                check=False,
            )
            response = json.loads(completed.stdout)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"][0]["text"], "短命进程成功")


if __name__ == "__main__":
    unittest.main()
