from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from zh_asr import dictation_audio


class FakeStream:
    instances = []
    fail_start = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        type(self).instances.append(self)

    def start(self):
        if type(self).fail_start:
            raise RuntimeError("device unavailable")
        self.started = True

    def close(self):
        self.closed = True


class FakeSoundDevice:
    def __init__(self, devices):
        self.devices = devices
        self.hostapis = [
            {"name": "MME"},
            {"name": "Windows DirectSound"},
            {"name": "Windows WASAPI"},
            {"name": "Windows WDM-KS"},
        ]
        self.default = SimpleNamespace(device=(0, 0))
        self._initialize = Mock()
        self._terminate = Mock()
        self.check_input_settings = Mock()
        self.InputStream = FakeStream

    def query_devices(self):
        return list(self.devices)

    def query_hostapis(self):
        return list(self.hostapis)


def device(name, index, hostapi, channels=1):
    return {
        "name": name,
        "index": index,
        "hostapi": hostapi,
        "max_input_channels": channels,
    }


class DictationAudioTests(unittest.TestCase):
    def setUp(self):
        FakeStream.instances = []
        FakeStream.fail_start = False
        self.settings = SimpleNamespace(input_device="DJI Mic Mini", sample_rate=16000)
        self.callback = Mock()

    def test_prefers_wasapi_for_same_named_input(self):
        fake = FakeSoundDevice([
            device("DJI Mic Mini", 10, 0),
            device("DJI Mic Mini", 11, 1),
            device("DJI Mic Mini-FB7E6B", 12, 2),
            device("DJI Mic Mini", 13, 3),
            device("Other microphone", 14, 2),
        ])
        with patch.object(dictation_audio, "sd", fake):
            stream = dictation_audio.open_microphone(self.settings, self.callback)

        self.assertTrue(stream.started)
        self.assertEqual(stream.kwargs["device"], 12)
        self.assertEqual(stream.kwargs["samplerate"], 16000)
        self.assertEqual(stream.kwargs["channels"], 1)
        self.assertEqual(stream.kwargs["dtype"], "float32")
        self.assertEqual(stream.kwargs["blocksize"], 320)
        self.assertIs(stream.kwargs["callback"], self.callback)
        fake._terminate.assert_called_once_with()
        fake._initialize.assert_called_once_with()

    def test_refreshes_enumeration_for_a_reconnected_device(self):
        fake = FakeSoundDevice([device("DJI Mic Mini-FB7E6B", 49, 3)])
        with patch.object(dictation_audio, "sd", fake):
            first = dictation_audio.open_microphone(self.settings, self.callback)
            first.close()
            fake.devices = [device("DJI Mic Mini-FB7E6B", 55, 2)]
            second = dictation_audio.open_microphone(self.settings, self.callback)

        self.assertEqual(first.kwargs["device"], 49)
        self.assertEqual(second.kwargs["device"], 55)
        self.assertEqual(fake._terminate.call_count, 2)
        self.assertEqual(fake._initialize.call_count, 2)

    def test_never_falls_back_to_another_microphone(self):
        fake = FakeSoundDevice([device("Laptop microphone", 7, 2)])
        with patch.object(dictation_audio, "sd", fake):
            with self.assertRaisesRegex(
                dictation_audio.MicrophoneOpenError, "DJI Mic Mini"
            ):
                dictation_audio.open_microphone(self.settings, self.callback)

        self.assertEqual(FakeStream.instances, [])

    def test_closes_stream_when_start_fails(self):
        fake = FakeSoundDevice([device("DJI Mic Mini-FB7E6B", 49, 3)])
        FakeStream.fail_start = True
        with patch.object(dictation_audio, "sd", fake):
            with self.assertRaisesRegex(dictation_audio.MicrophoneOpenError, "启动失败"):
                dictation_audio.open_microphone(self.settings, self.callback)

        self.assertEqual(len(FakeStream.instances), 1)
        self.assertTrue(FakeStream.instances[0].closed)

    def test_reports_16khz_when_named_device_does_not_support_it(self):
        fake = FakeSoundDevice([device("DJI Mic Mini-FB7E6B", 49, 3)])
        fake.check_input_settings.side_effect = ValueError("bad sample rate")
        with patch.object(dictation_audio, "sd", fake):
            with self.assertRaisesRegex(dictation_audio.MicrophoneOpenError, "16kHz"):
                dictation_audio.open_microphone(self.settings, self.callback)

        self.assertEqual(FakeStream.instances, [])


if __name__ == "__main__":
    unittest.main()
