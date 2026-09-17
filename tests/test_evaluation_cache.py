import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from zh_asr.config import load_model_config
from zh_asr.eval_pack import run_evaluation
from zh_asr.evaluation_cache import observe_case
from zh_asr.strict_writer import write_strict_bundle


class EvaluationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus, self.output = self.root / 'corpus', self.root / 'output'
        self.corpus.mkdir()
        self.audio = self.corpus / 'speech.wav'
        with wave.open(str(self.audio), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes(b'\0\0' * 1600)
        self.truth = self.corpus / 'speech.txt'
        self.truth.write_text('测试录音', encoding='utf-8')
        self.case = {'id': 'sample', 'kind': 'public', 'audio': 'speech.wav',
                     'truth': 'speech.txt', 'expect_empty': False}
        self.manifest = self.corpus / 'manifest.json'
        self.manifest.write_text(json.dumps({'cases': [self.case]}), encoding='utf-8')
        self.calls = 0
        self.config_path = self.root / 'models.yaml'
        self.config_path.write_bytes(load_model_config().path.read_bytes())
        self.config = load_model_config(self.config_path)
        self.artifacts = patch('zh_asr.long_audio._runtime_artifact_identity', return_value={})
        self.artifacts.start(); self.addCleanup(self.artifacts.stop)

    def transcribe(self, audio_path, *, primary_engine, secondary_engine, out_dir, **kwargs):
        self.calls += 1
        outputs = write_strict_bundle(audio_path=audio_path, out_dir=out_dir,
            primary_engine=primary_engine, primary_result={'text': '测试录音'},
            secondary_engine=secondary_engine, secondary_result={'text': '测试录音'})
        outputs['timing'] = {'total_sec': 9.5, 'primary_sec': 6.0, 'secondary_sec': 3.5}
        return outputs

    def evaluate(self, **kwargs):
        return run_evaluation(self.corpus, self.output, device='cpu', config=self.config,
            strict_fn=self.transcribe, **kwargs)

    def test_verified_cache_reuses_same_input_and_preserves_original_timing(self):
        self.evaluate()
        self.evaluate()
        self.assertEqual(self.calls, 1)
        metrics = json.loads((self.output / 'metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(metrics['cases'][0]['timing']['total_sec'], 9.5)
        self.assertEqual(len(metrics['cases'][0]['audio_sha256']), 64)
        self.assertEqual(len(metrics['cases'][0]['truth_sha256']), 64)

    def test_engine_pair_change_reprocesses_instead_of_relabeling_old_audit(self):
        self.evaluate()
        self.evaluate(primary_engine='fireredasr2-llm', secondary_engine='qwen3-asr-1.7b')
        self.assertEqual(self.calls, 2)
        metrics = json.loads((self.output / 'metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(metrics['cases'][0]['models']['primary'], 'fireredasr2-llm')

    def test_legacy_path_only_audit_is_not_a_cache_proof(self):
        self.evaluate()
        (self.output / 'cases/sample/evaluation-cache.json').unlink()
        self.evaluate()
        self.assertEqual(self.calls, 2)

    def test_reference_and_config_changes_invalidate_cache(self):
        self.evaluate()
        self.truth.write_text('不同的参考文本', encoding='utf-8')
        self.evaluate()
        self.assertEqual(self.calls, 2)
        self.config_path.write_text(self.config_path.read_text(encoding='utf-8') + '\n# compatibility update\n', encoding='utf-8')
        self.evaluate()
        self.assertEqual(self.calls, 3)

    def test_changed_source_bytes_invalidate_cache(self):
        self.evaluate()
        raw = bytearray(self.audio.read_bytes()); raw[-2:] = b'\x01\x00'
        self.audio.write_bytes(raw)
        self.evaluate()
        self.assertEqual(self.calls, 2)

    def test_changed_runtime_identity_cannot_reuse_old_case(self):
        self.evaluate()
        with patch('zh_asr.long_audio._runtime_code_identity', return_value={'sha256': 'new-test-runtime'}):
            self.evaluate()
        self.assertEqual(self.calls, 2)

    def test_corrupted_raw_output_reprocesses(self):
        self.evaluate()
        raw = next((self.output / 'cases/sample').glob('*.qwen3-asr-1.7b.raw.json'))
        raw.write_text('{"text":"changed"}', encoding='utf-8')
        self.evaluate()
        self.assertEqual(self.calls, 2)

    def test_manifest_hash_mismatch_is_not_reported_as_old_source(self):
        observed, _ = observe_case(self.corpus, self.case)
        self.manifest.write_text(json.dumps({'cases': [observed]}), encoding='utf-8')
        self.truth.write_text('参考已改变', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'truth_sha256'):
            self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_mid_run_reference_change_cannot_get_a_new_current_identity(self):
        original = self.transcribe
        def mutate(*args, **kwargs):
            outputs = original(*args, **kwargs)
            self.truth.write_text('正在运行时修改的参考', encoding='utf-8')
            return outputs
        self.transcribe = mutate
        with self.assertRaisesRegex(RuntimeError, 'changed during inference'):
            self.evaluate()
        self.assertFalse((self.output / 'metrics.json').exists())
        self.assertFalse((self.output / 'cases/sample/evaluation-cache.json').exists())

    def test_duplicate_case_ids_cannot_overwrite_each_other(self):
        self.manifest.write_text(json.dumps({'cases': [self.case, self.case]}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.evaluate()
        self.assertEqual(self.calls, 0)
