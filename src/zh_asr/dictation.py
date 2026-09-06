"""Local desktop dictation; no file-job queue, cloud upload or transcript archive."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
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
from .dictation_vad import VoiceActivityDetector, contains_speech
from .gpu_broker import GpuBrokerConflict, GpuBrokerLease

LOG = logging.getLogger("zh_asr.dictation")
HOTKEY_LABEL = "Win+H / Ctrl+Win+H"


@dataclass(frozen=True)
class DictationSettings:
    engine: str = "qwen3-asr-1.7b"
    sample_rate: int = 16000
    silence_ms: int = 600
    min_speech_ms: int = 240
    max_chunk_sec: float = 20
    input_device: str | int | None = None
    hotwords: str = ""

    def with_preferences(self) -> "DictationSettings":
        try:
            value = json.loads(preferences_path().read_text(encoding="utf-8"))
            selected = value["input_device"]
            if selected is None or isinstance(selected, str):
                return replace(self, input_device=selected)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return self

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


def preferences_path() -> Path:
    return project_root() / "outputs" / "dictation" / "preferences.json"


def save_microphone_preference(value: str | None) -> None:
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps({"input_device": value}, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)


class PauseSegmenter:
    """Keep short pre-roll and end a phrase at pauses, with a bounded long phrase.

    VAD selects phrase boundaries; every sample inside a phrase is retained,
    including pre-roll, unvoiced consonants and quiet word endings.
    """

    def __init__(self, settings: DictationSettings, detector=None):
        self.settings = settings
        self.detector = detector or VoiceActivityDetector(settings.sample_rate)
        self.pre_roll: deque[np.ndarray] = deque(maxlen=10)  # 200 ms at 20 ms/block
        self.blocks: list[np.ndarray] = []
        self.speech_flags: list[bool] = []
        self.voiced_samples = 0
        self.silent_samples = 0
        self.samples = 0

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if not samples.size:
            return []
        active = self.detector.is_speech(samples)
        if not self.blocks and not active:
            self.pre_roll.append(samples)
            return []
        if not self.blocks:
            self.blocks.extend(self.pre_roll)
            self.speech_flags.extend(False for _ in self.pre_roll)
            self.samples = sum(len(block) for block in self.blocks)
            self.pre_roll.clear()
        self.blocks.append(samples)
        self.speech_flags.append(active)
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
            head_voiced = sum(len(block) for block, voiced in
                              zip(head, self.speech_flags[:cut]) if voiced)
            self.blocks = tail
            self.speech_flags = self.speech_flags[cut:]
            self.samples = sum(len(block) for block in tail)
            self.voiced_samples = sum(len(block) for block, voiced in
                                      zip(tail, self.speech_flags) if voiced)
            self.silent_samples = 0
            return ([np.concatenate(head)] if head_voiced >=
                    sr * self.settings.min_speech_ms / 1000 else [])
        return []

    def flush(self) -> np.ndarray | None:
        result = None
        if (self.blocks and self.voiced_samples >=
                self.settings.sample_rate * self.settings.min_speech_ms / 1000):
            result = np.concatenate(self.blocks)
        self.blocks.clear()
        self.speech_flags.clear()
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
        if not contains_speech(audio, self.settings.sample_rate, self.settings.min_speech_ms):
            LOG.info("dictation chunk skipped: no sustained speech detected")
            return ""
        return self._infer(audio)

    def _infer(self, audio: np.ndarray) -> str:
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
        text = "".join(str(getattr(item, "text", "") or "") for item in (results or [])).strip()
        LOG.info("transcribed audio_sec=%.3f elapsed_sec=%.3f chars=%d",
                 len(audio) / self.settings.sample_rate, time.perf_counter() - started, len(text))
        return to_simplified(text)

    def warmup(self) -> None:
        self.activate()
        limit = self.wrapper.model.max_new_tokens
        try:
            self.wrapper.model.max_new_tokens = 1
            # Warm the actual GPU kernels; the normal input gate rejects silence.
            self._infer(np.zeros(self.settings.sample_rate // 2, dtype=np.float32))
        finally:
            self.wrapper.model.max_new_tokens = limit

    def dispose(self) -> None:
        """Quit directly, without copying GPU weights back to RAM first."""
        lease, self.lease = self.lease, None
        had_model = self.wrapper is not None
        self.wrapper = None
        try:
            if had_model:
                import torch
                # CPU-resident weights need no full-heap GC before process exit.
                # A live GPU lease does: free any cycles before admitting another job.
                if lease is not None:
                    import gc
                    gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
        finally:
            if lease is not None:
                lease.__exit__(None, None, None)

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
    settings: DictationSettings | None = None
    segmenter: PauseSegmenter | None = None
    stream: object = None
    submitted: bool = False
    audio_lock: threading.Lock = field(default_factory=threading.Lock)
    capture_finished: threading.Event = field(default_factory=threading.Event)
    recognition_finished: threading.Event = field(default_factory=threading.Event)


class DictationController:
    """The UI never waits for a driver, model load, inference or GPU cleanup."""

    def __init__(self, host, settings: DictationSettings, engine=None):
        self.host = host
        self.settings = settings
        self.engine = engine or QwenDictationEngine(settings)
        self.commands: queue.Queue = queue.Queue()
        self.audio_commands: queue.Queue = queue.Queue()
        self.recording: Recording | None = None
        self.quit_event = threading.Event()
        self.initialization_failed = False
        self.pending_start = False
        self.pending_target = None
        self.worker = threading.Thread(target=self._worker, name="dictation-asr", daemon=True)
        self.audio_worker = threading.Thread(target=self._audio_worker, name="dictation-audio", daemon=True)

    def start(self) -> None:
        self.host.show("正在准备模型", "可以先录音，准备完成后自动转写")
        self.audio_worker.start()
        self.audio_commands.put(("devices", False))
        self.worker.start()

    def toggle(self) -> None:
        LOG.info("dictation toggle received")
        target = self.host.capture_target()  # Capture before any window can take focus.
        self.host.open_panel()
        if self.quit_event.is_set():
            return
        if self.initialization_failed:
            self.host.show("模型加载失败", "从托盘退出后重新启动", error=True)
            return
        if self.recording is not None:
            if not self.recording.stopped.is_set():
                self.pending_start = False
                self.stop()
            else:
                self.pending_start = not self.pending_start
                self.pending_target = target if self.pending_start else None
                self.host.show("准备继续录音" if self.pending_start else "已暂停", "正在完成上一段" if self.pending_start else "点击麦克风继续")
            return
        self._begin_recording(target)

    def toggle_visibility(self) -> None:
        """Hotkeys control visibility; the on-panel button controls recording."""
        if self.host.panel_visible:
            self.hide()
        else:
            self.toggle()

    def _begin_recording(self, target=None) -> None:
        if self.quit_event.is_set():
            return
        recording = Recording(target or self.host.capture_target(), queue.Queue(),
                              threading.Event(), threading.Event(), settings=self.settings,
                              segmenter=PauseSegmenter(self.settings))
        self.recording = recording
        self.host.set_busy(True)
        self.host.show("正在打开麦克风", "再次按快捷键或按钮可暂停")
        self.audio_commands.put(("open", recording))

    def _audio_worker(self) -> None:
        from .dictation_audio import list_microphones, open_microphone
        while True:
            command = self.audio_commands.get()
            if command is None:
                return
            action, value = command
            if action == "devices":
                try:
                    options = list_microphones(refresh=bool(value))
                    self.host.post_to_ui(lambda options=options: self.host.set_microphones(options, self.settings.input_device))
                except Exception:
                    LOG.exception("microphone enumeration failed")
                continue
            recording = value
            if action == "open":
                if recording.stopped.is_set() or self.quit_event.is_set():
                    self._close_capture(recording)
                    continue
                try:
                    recording.stream = open_microphone(
                        recording.settings,
                        lambda data, frames, timing, status, r=recording: self._audio_callback(r, data, status))
                    if recording.stopped.is_set() or self.quit_event.is_set():
                        self._close_capture(recording)
                        continue
                    recording.submitted = True
                    self.commands.put(recording)
                    self.host.post_to_ui(lambda r=recording: self._capture_started(r))
                except Exception:
                    LOG.exception("microphone open failed")
                    recording.error = "请连接所选麦克风，或右键选择其他设备"
                    recording.cancelled.set()
                    recording.stopped.set()
                    self._close_capture(recording)
            elif action == "close":
                self._close_capture(recording)

    def _capture_started(self, recording: Recording) -> None:
        if self.recording is recording and not recording.stopped.is_set() and not self.quit_event.is_set():
            self.host.show("正在聆听", "点击麦克风或快捷键暂停", recording=True)

    def _close_capture(self, recording: Recording) -> None:
        # All PortAudio open/close/enumeration runs on this one audio thread.
        stream, recording.stream = recording.stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                recording.error = "麦克风连接已中断，请重新连接或选择其他设备"
                recording.cancelled.set()
                LOG.exception("microphone stop failed")
            finally:
                try:
                    stream.close()
                except Exception:
                    LOG.exception("microphone close failed")
        recording.capture_finished.set()
        if not recording.submitted:
            recording.recognition_finished.set()
        self.host.post_to_ui(lambda r=recording: self._recording_finished(r))

    def _audio_callback(self, recording: Recording, data, status) -> None:
        with recording.audio_lock:
            if recording.stopped.is_set() or recording.cancelled.is_set():
                return
            if status:
                recording.cancelled.set()
                recording.error = "录音发生丢帧，请重新连接或选择其他麦克风"
                self.host.post_to_ui(lambda: self.stop(cancel=True))
                return
            for chunk in recording.segmenter.feed(data[:, 0]):
                recording.chunks.put(chunk)

    def stop(self, cancel: bool = False) -> None:
        recording = self.recording
        if recording is None:
            return
        if cancel:
            recording.cancelled.set()
        with recording.audio_lock:
            if not recording.stopped.is_set():
                recording.stopped.set()  # Stop accepting samples before touching a driver.
                tail = recording.segmenter.flush() if recording.segmenter is not None else None
                if tail is not None and not cancel:
                    recording.chunks.put(tail)
                recording.chunks.put(None)
                self.audio_commands.put(("close", recording))
        self.host.show("已暂停", "点击麦克风或快捷键继续")

    def cancel(self) -> None:
        self.pending_start = False
        self.stop(cancel=True)

    def hide(self) -> None:
        self.host.hide_panel()
        self.pending_start = False
        self.pending_target = None
        self.stop()

    def select_microphone(self, value: str | None) -> None:
        self.pending_start = False
        self.stop()  # Complete the old device's last phrase; apply only to the next session.
        self.settings = replace(self.settings, input_device=value)
        try:
            save_microphone_preference(value)
        except OSError:
            LOG.exception("microphone preference save failed")
        self.host.show("麦克风已切换", "点击麦克风开始录音")
        self.audio_commands.put(("devices", True))

    def refresh_devices(self) -> None:
        self.pending_start = False
        self.stop()
        self.audio_commands.put(("devices", True))

    def _recording_finished(self, recording: Recording) -> None:
        if not (recording.capture_finished.is_set() and recording.recognition_finished.is_set()):
            return
        if self.recording is not recording:
            return
        self.recording = None
        self.host.set_busy(False)
        if recording.error:
            self.host.show("暂时无法听写", recording.error, error=True)
        if self.pending_start and not self.quit_event.is_set():
            self.pending_start = False
            target, self.pending_target = self.pending_target, None
            self._begin_recording(target)

    def close(self) -> None:
        if self.quit_event.is_set():
            return
        self.host.hide_panel()  # Disappear now; cleanup must never hold a panel on screen.
        self.pending_start = False
        self.stop(cancel=True)
        self.quit_event.set()
        self.commands.put(None)
        self.audio_commands.put(None)
        threading.Thread(target=self._finish_close, name="dictation-close", daemon=True).start()

    def _finish_close(self) -> None:
        self.worker.join(timeout=20)
        self.audio_worker.join(timeout=3)
        self.host.close()

    def _worker(self) -> None:
        try:
            started = time.perf_counter()
            self.engine.load()
            LOG.info("model loaded seconds=%.3f", time.perf_counter() - started)
            if self.quit_event.is_set():
                return
            try:
                self.engine.warmup()
            except GpuBrokerConflict:
                pass
            except Exception:
                LOG.exception("optional prewarm failed")
            finally:
                if not self.quit_event.is_set():
                    try:
                        self.engine.park()
                    except Exception:
                        LOG.exception("optional prewarm cleanup failed")
            LOG.info("ready engine=%s; idle weights in RAM", self.settings.engine)
            if self.recording is None:
                self.host.show("准备就绪", "快捷键唤出即可录音")
            while not self.quit_event.is_set():
                recording = self.commands.get()
                if recording is None:
                    break
                try:
                    self._recognize(recording)
                except Exception:
                    LOG.exception("dictation session failed")
                    recording.cancelled.set()
                    recording.error = "转写暂时失败，可重新开始"
                finally:
                    if not recording.stopped.is_set():
                        self.host.post_to_ui(lambda: self.stop(cancel=True))
                    try:
                        if self.quit_event.is_set():
                            self.engine.dispose()
                        else:
                            self.engine.park()
                    except Exception:
                        LOG.exception("GPU cleanup failed")
                    recording.recognition_finished.set()
                    self.host.post_to_ui(lambda r=recording: self._recording_finished(r))
        except Exception:
            self.initialization_failed = True
            LOG.exception("dictation initialization failed")
            self.host.post_to_ui(self._model_failed)
        finally:
            started = time.perf_counter()
            try:
                self.engine.dispose()
            except Exception:
                LOG.exception("GPU release failed")
            LOG.info("dictation worker stopped; dispose_seconds=%.3f", time.perf_counter() - started)

    def _model_failed(self) -> None:
        self.cancel()
        if self.recording is not None:
            self.recording.recognition_finished.set()
            self._recording_finished(self.recording)
        self.host.set_busy(False)
        self.host.show("语音模型加载失败", "请从托盘退出后重启", error=True)

    def _recognize(self, recording: Recording) -> None:
        while not recording.cancelled.is_set() and not self.quit_event.is_set():
            try:
                self.engine.activate()
                break
            except GpuBrokerConflict:
                self.host.show("等待 GPU", "录音仍在内存中，Esc 可取消",
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
            if recording.cancelled.is_set() or self.quit_event.is_set():
                break
            if not text:
                continue
            recording.text += text
            self.host.set_last_text(recording.text)
            if not recording.insertion_failed:
                recording.insertion_failed = not self.host.insert_text(text, recording.target)
            if recording.insertion_failed:
                self.host.show("输入位置已改变", "可从托盘复制最近文字", error=True,
                               recording=not recording.stopped.is_set())
            elif not recording.stopped.is_set():
                self.host.show("正在聆听", "点击麦克风或快捷键暂停", recording=True)
        if recording.error:
            self.host.show("录音未能完整采集", recording.error, error=True)
        elif recording.cancelled.is_set():
            self.host.show("已暂停", "已输入的文字保留")
        elif recording.insertion_failed:
            self.host.show("已暂停", "可从托盘复制最近文字", error=True)
        else:
            self.host.show("已暂停", "点击麦克风或快捷键继续")


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
    settings = DictationSettings.load(args.config).with_preferences()
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
                       on_quit=lambda: controller.close(),
                       on_hide=lambda: controller.hide(),
                       on_hotkey=lambda: controller.toggle_visibility(),
                       on_device_change=lambda value: controller.select_microphone(value),
                       on_refresh_devices=lambda: controller.refresh_devices())
    if not host.acquire_single_instance():
        host.close()
        return 0
    controller = DictationController(host, settings)
    controller.start()
    host.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
