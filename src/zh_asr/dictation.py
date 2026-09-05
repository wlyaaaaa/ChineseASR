"""Local desktop dictation; no file-job queue, cloud upload or transcript archive."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

from .config import get_engine_spec, load_model_config, project_root
from .gpu_broker import GpuBrokerConflict, GpuBrokerLease

LOG = logging.getLogger("zh_asr.dictation")


@dataclass(frozen=True)
class DictationSettings:
    engine: str = "qwen3-asr-1.7b"
    sample_rate: int = 16000
    silence_ms: int = 600
    min_speech_ms: int = 240
    max_chunk_sec: float = 20
    input_device: str | int | None = None
    hotwords: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> "DictationSettings":
        import yaml
        source = path or project_root() / "configs" / "dictation.yaml"
        settings = cls(**(yaml.safe_load(source.read_text(encoding="utf-8")) or {}))
        if settings.sample_rate != 16000:
            raise ValueError("Dictation requires 16 kHz audio.")
        if not 200 <= settings.silence_ms <= 2000:
            raise ValueError("silence_ms must be between 200 and 2000.")
        if not 60 <= settings.min_speech_ms <= 1000:
            raise ValueError("min_speech_ms must be between 60 and 1000.")
        if not 3 <= settings.max_chunk_sec <= 30:
            raise ValueError("max_chunk_sec must be between 3 and 30.")
        return settings


class PauseSegmenter:
    """Keep short pre-roll and end a phrase at pauses, with a bounded long phrase.

    RMS is only a microphone activity gate, not evidence that audio is speech.
    Low-energy blocks are retained inside a phrase, including quiet word endings.
    """

    def __init__(self, settings: DictationSettings):
        self.settings = settings
        self.pre_roll: deque[np.ndarray] = deque(maxlen=10)  # 200 ms at 20 ms/block
        self.blocks: list[np.ndarray] = []
        self.voiced_samples = 0
        self.silent_samples = 0
        self.samples = 0
        self.noise = 0.00015

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        active = rms >= max(0.0008, min(self.noise * 3.5, 0.008))
        if not self.blocks and not active:
            self.noise = self.noise * 0.98 + rms * 0.02
            self.pre_roll.append(samples)
            return []
        if not self.blocks:
            self.blocks.extend(self.pre_roll)
            self.samples = sum(len(block) for block in self.blocks)
            self.pre_roll.clear()
        self.blocks.append(samples)
        self.samples += len(samples)
        self.voiced_samples += len(samples) if active else 0
        self.silent_samples = 0 if active else self.silent_samples + len(samples)
        sr = self.settings.sample_rate
        if self.silent_samples >= sr * self.settings.silence_ms / 1000:
            result = self.flush()
            return [result] if result is not None else []
        if self.samples >= sr * self.settings.max_chunk_sec:
            # Prefer the quietest 20 ms near the limit over cutting through a
            # syllable exactly at the clock boundary. Keep every following sample.
            first = max(1, len(self.blocks) - 50)
            cut = min(range(first, len(self.blocks)),
                      key=lambda i: float(np.mean(self.blocks[i] ** 2))) + 1
            head, tail = self.blocks[:cut], self.blocks[cut:]
            self.blocks = tail
            self.samples = sum(len(block) for block in tail)
            threshold = max(0.0008, min(self.noise * 3.5, 0.008))
            self.voiced_samples = sum(len(block) for block in tail
                if float(np.sqrt(np.mean(block ** 2))) >= threshold)
            self.silent_samples = 0
            return [np.concatenate(head)]
        return []

    def flush(self) -> np.ndarray | None:
        result = None
        if (self.blocks and self.voiced_samples >=
                self.settings.sample_rate * self.settings.min_speech_ms / 1000):
            result = np.concatenate(self.blocks)
        self.blocks.clear()
        self.pre_roll.clear()
        self.voiced_samples = self.silent_samples = self.samples = 0
        return result


class QwenDictationEngine:
    """Keep weights in RAM; hold the shared GPU lease only during a session.

    Parking on CPU avoids both repeated disk/model initialization and an idle
    tray app monopolizing the GPU used by other local tools.
    """

    def __init__(self, settings: DictationSettings):
        self.settings = settings
        self.wrapper = None
        self.lease: GpuBrokerLease | None = None

    def load(self) -> None:
        if self.wrapper is not None:
            return
        from .adapters import get_adapter
        from .pipeline import prepare_model_env
        config = load_model_config()
        spec = get_engine_spec(self.settings.engine, config)
        if spec.adapter != "qwen-asr":
            raise ValueError("The desktop adapter currently supports qwen-asr engines.")
        options = dict(spec.options or {})
        options.update(context=self.settings.hotwords,
                       max_inference_batch_size=1, max_new_tokens=384)
        spec = replace(spec, options=options)
        self.wrapper = get_adapter(spec.adapter).build_model(
            spec, "cpu", prepare_model_env(), config.model_aliases)
        self.wrapper.model.model.eval()

    def activate(self) -> None:
        self.load()
        if self.lease is not None:
            self.lease.raise_if_lost()
            return
        lease = GpuBrokerLease("chineseasr", ttl_seconds=120, renew_interval_seconds=20)
        lease.__enter__()
        self.lease = lease
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is unavailable.")
            self.wrapper.model.model.to("cuda:0")
        except BaseException:
            self.park()
            raise

    def transcribe(self, audio: np.ndarray) -> str:
        self.activate()
        self.lease.raise_if_lost()
        started = time.perf_counter()
        import torch
        from .text_normalizer import to_simplified
        with torch.inference_mode():
            results = self.wrapper.model.transcribe(
                audio=(audio, self.settings.sample_rate),
                language=self.wrapper.language, context=self.settings.hotwords)
        self.lease.raise_if_lost()
        text = "".join(str(getattr(item, "text", "")) for item in results).strip()
        LOG.info("transcribed audio_sec=%.3f elapsed_sec=%.3f chars=%d",
                 len(audio) / self.settings.sample_rate, time.perf_counter() - started, len(text))
        return to_simplified(text)

    def park(self) -> None:
        lease, self.lease = self.lease, None
        if lease is None:
            return
        try:
            import torch
            if self.wrapper is not None:
                try:
                    self.wrapper.model.model.to("cpu")
                except Exception:
                    # A failed transfer must not leave GPU weights behind a released lease.
                    self.wrapper = None
                    import gc
                    gc.collect()
                    raise
            torch.cuda.empty_cache()
        finally:
            lease.__exit__(None, None, None)


@dataclass
class Recording:
    target: object
    chunks: queue.Queue
    cancelled: threading.Event
    stopped: threading.Event
    text: str = ""
    insertion_failed: bool = False
    error: str = ""


class DictationController:
    def __init__(self, host, settings: DictationSettings, engine=None):
        self.host = host
        self.settings = settings
        self.engine = engine or QwenDictationEngine(settings)
        self.commands: queue.Queue = queue.Queue()
        self.recording: Recording | None = None
        self.stream = None
        self.segmenter: PauseSegmenter | None = None
        self.audio_lock = threading.Lock()
        self.quit_event = threading.Event()
        self.initialization_failed = False
        self.worker = threading.Thread(target=self._worker, name="dictation-asr", daemon=True)

    def start(self) -> None:
        self.host.show("正在准备语音输入", "首次加载模型；Win+H 可开始录音")
        self.worker.start()

    def toggle(self) -> None:
        if self.initialization_failed:
            self.host.show("语音模型尚未就绪", "从托盘退出后重启语音输入", error=True)
            return
        if self.recording:
            if not self.recording.stopped.is_set():
                self.stop()
            else:
                self.host.show("正在完成转写", "请稍候，Esc 可取消")
            return
        from .dictation_audio import open_microphone
        recording = Recording(self.host.capture_target(), queue.Queue(),
                              threading.Event(), threading.Event())
        self.segmenter = PauseSegmenter(self.settings)
        self.recording = recording
        self.host.set_busy(True)
        try:
            self.stream = open_microphone(self.settings, self._audio_callback)
        except Exception as exc:
            if self.stream:
                self.stream.close()
                self.stream = None
            self.recording = None
            self.host.set_busy(False)
            LOG.error("microphone failed: %s", type(exc).__name__)
            selected = self.settings.input_device or "Windows 默认麦克风"
            self.host.show("麦克风无法打开", f"请连接 {selected}，再按 Win+H 重试", error=True)
            return
        self.commands.put(recording)
        self.host.show("正在聆听", "Win+H 结束 · Esc 取消", recording=True)

    def _audio_callback(self, data, frames, time_info, status) -> None:
        with self.audio_lock:
            recording = self.recording
            if recording is None or recording.stopped.is_set():
                return
            if status:
                # Missing microphone samples must not be reported as a clean transcript.
                recording.cancelled.set()
                recording.error = "麦克风采集发生丢帧，请检查默认输入设备后重试"
                LOG.warning("microphone buffer status: %s", status)
                return
            for chunk in self.segmenter.feed(data[:, 0]):
                recording.chunks.put(chunk)

    def stop(self, cancel: bool = False) -> None:
        recording = self.recording
        if recording is None:
            return
        if cancel:
            recording.cancelled.set()
        try:
            self._close_stream()
        except Exception:
            recording.cancelled.set()
            recording.error = "麦克风连接已中断，请连接设备后重试"
            LOG.exception("microphone stop failed")
        with self.audio_lock:
            if not recording.stopped.is_set():
                recording.stopped.set()
                tail = self.segmenter.flush()
                if tail is not None and not cancel:
                    recording.chunks.put(tail)
                recording.chunks.put(None)
        self.host.show("正在结束" if cancel else "正在完成转写", "已停止麦克风", recording=False)

    def _close_stream(self) -> None:
        # UI cancellation and the inference worker can finish simultaneously.
        # Only one thread takes responsibility for closing the PortAudio stream.
        with self.audio_lock:
            stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    def cancel(self) -> None:
        self.stop(cancel=True)

    def close(self) -> None:
        self.stop(cancel=True)
        self.quit_event.set()
        self.commands.put(None)
        # The UI stays responsive; the worker parks the GPU before exiting.
        self.host.show("正在退出", "正在释放语音模型")
        threading.Thread(target=self._finish_close, daemon=True).start()

    def _finish_close(self) -> None:
        self.worker.join(timeout=30)
        self.host.close()

    def _worker(self) -> None:
        try:
            self.engine.load()
            try:
                self.engine.activate()
                self.engine.transcribe(np.zeros(self.settings.sample_rate // 2, dtype=np.float32))
            except GpuBrokerConflict:
                pass  # Prewarming is optional when another local task owns the GPU.
            except Exception:
                LOG.exception("optional GPU prewarm failed; dictation can retry")
            finally:
                try:
                    self.engine.park()
                except Exception:
                    LOG.exception("optional prewarm cleanup failed; dictation can retry")
            LOG.info("ready engine=%s; idle weights in RAM", self.settings.engine)
            if self.recording is None:
                self.host.show("语音输入已就绪", "在输入框按 Win+H 开始")
            while not self.quit_event.is_set():
                recording = self.commands.get()
                if recording is None:
                    break
                try:
                    self._recognize(recording)
                except Exception as exc:
                    LOG.exception("dictation session failed")
                    recording.cancelled.set()
                    self.host.show("语音输入暂时失败", "可重新按 Win+H；详情见本地运行日志", error=True)
                finally:
                    try:
                        self.engine.park()
                    except Exception:
                        LOG.exception("GPU park failed")
                    # Usually already stopped by the UI. Error/overflow can end early.
                    try:
                        self._close_stream()
                    except Exception:
                        LOG.exception("microphone cleanup failed")
                    self.recording = None
                    self.host.set_busy(False)
        except Exception:
            self.initialization_failed = True
            LOG.exception("dictation initialization failed")
            self.stop(cancel=True)
            self.recording = None
            self.host.set_busy(False)
            self.host.show("语音模型加载失败", "请从托盘退出后重启语音输入；详情见本地运行日志", error=True)
        finally:
            try:
                self.engine.park()
            except Exception:
                LOG.exception("GPU release failed")
            LOG.info("dictation worker stopped")

    def _recognize(self, recording: Recording) -> None:
        while not recording.cancelled.is_set() and not self.quit_event.is_set():
            try:
                self.engine.activate()
                break
            except GpuBrokerConflict:
                self.host.show("等待本地 GPU", "录音已保留在内存中；Esc 可取消",
                               recording=not recording.stopped.is_set())
                recording.cancelled.wait(0.3)
        while not recording.cancelled.is_set() and not self.quit_event.is_set():
            try:
                audio = recording.chunks.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio is None:
                break
            text = self.engine.transcribe(audio)
            if recording.cancelled.is_set():
                break
            if not text:
                continue
            recording.text += text
            self.host.set_last_text(recording.text)
            # Never inject late text into a newly focused app or input control.
            if not recording.insertion_failed:
                recording.insertion_failed = not self.host.insert_text(text, recording.target)
            if recording.insertion_failed:
                self.host.show("输入位置已改变", recording.text, error=True,
                               recording=not recording.stopped.is_set())
            elif not recording.stopped.is_set():
                self.host.show("正在聆听", "Win+H 结束 · Esc 取消", recording=True)
        if recording.error:
            self.host.show("录音未能完整采集", recording.error, error=True)
        elif recording.cancelled.is_set():
            self.host.show("已取消", "已输入的文字保留在输入框中")
        elif recording.insertion_failed:
            self.host.show("文字已识别，未继续输入", recording.text, error=True)
        elif recording.text:
            self.host.show("已输入", "Win+H 可继续；文字不会自动发送")
        else:
            self.host.show("没有识别到可输入的文字", "靠近麦克风后按 Win+H 重试")


def configure_logging() -> None:
    directory = project_root() / "outputs" / "dictation"
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(directory / "runtime.log", maxBytes=262144,
                                  backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.setLevel(logging.INFO)
    LOG.addHandler(handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ChineseASR Windows Win+H dictation")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--stop", action="store_true", help="Gracefully stop the running tray app")
    parser.add_argument("--transcribe", type=Path, help="Verify the dictation engine using one named audio file")
    args = parser.parse_args(argv)
    if args.stop:
        from .dictation_windows import request_existing_quit
        request_existing_quit()
        return 0
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    settings = DictationSettings.load(args.config)
    # pythonw has no console streams. Dependencies must not crash writing progress.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    temp_dir = Path("E:/Cache/Codex/Temp/chineseasr-dictation")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(temp_dir)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    configure_logging()
    if args.transcribe:
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        audio, sample_rate = sf.read(args.transcribe, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if sample_rate != settings.sample_rate:
            divisor = gcd(sample_rate, settings.sample_rate)
            mono = resample_poly(mono, settings.sample_rate // divisor, sample_rate // divisor)
        engine = QwenDictationEngine(settings)
        try:
            started = time.perf_counter()
            engine.load()
            loaded = time.perf_counter()
            text = engine.transcribe(mono)
            print(json.dumps({"text": text, "load_seconds": loaded - started,
                              "transcribe_seconds": time.perf_counter() - loaded}, ensure_ascii=False))
        finally:
            engine.park()
        return 0
    from .dictation_windows import WindowsHost
    controller = None
    host = WindowsHost(on_toggle=lambda: controller.toggle(),
                       on_cancel=lambda: controller.cancel(),
                       on_quit=lambda: controller.close())
    if not host.acquire_single_instance():
        host.close()
        return 0
    controller = DictationController(host, settings)
    controller.start()
    host.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
