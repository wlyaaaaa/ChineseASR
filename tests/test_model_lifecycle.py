import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from zh_asr.config import load_model_config
from zh_asr.long_audio import _runtime_code_identity
from zh_asr.model_lifecycle import (
    ALIGNER_FILES, activate_profile, compare_evaluations, fetch_aligner,
    file_hash, rollback_profile, verify_aligner,
)


class ModelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original = load_model_config()
        self.path = self.root / 'configs' / 'models.yaml'
        self.path.parent.mkdir()
        self.path.write_bytes(original.path.read_bytes())
        self.config = load_model_config(self.path)

    def fixture_lock(self):
        data = {name: ('fixture:' + name).encode() for name in ALIGNER_FILES}
        alignment = dict(self.config.alignment, model_dir='models/fixture-aligner',
                         artifact_lock='configs/fixture-aligner.lock.json')
        config = replace(self.config, alignment=alignment)
        lock = {'schema': 'zh_asr.artifact_lock.v1', 'repository': alignment['model'],
                'revision': alignment['model_revision'], 'files': [
                    {'path': name, 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
                    for name, value in data.items()]}
        (self.root / alignment['artifact_lock']).write_text(json.dumps(lock), encoding='utf-8')
        return config, data

    def test_fresh_checkout_restores_exact_locked_files_then_reuses(self):
        config, data = self.fixture_lock()
        before = (self.root / config.alignment['artifact_lock']).read_bytes()
        def download(*, filename, local_dir, **kwargs):
            target = Path(local_dir) / filename
            target.write_bytes(data[filename])
            return str(target)
        with patch('huggingface_hub.hf_hub_download', side_effect=download) as fetch:
            result = fetch_aligner(config, self.root / 'downloads')
            self.assertEqual(result['status'], 'verified')
            self.assertEqual(fetch.call_count, len(ALIGNER_FILES))
            fetch_aligner(config, self.root / 'downloads')
            self.assertEqual(fetch.call_count, len(ALIGNER_FILES))
        self.assertEqual((self.root / config.alignment['artifact_lock']).read_bytes(), before)
        self.assertEqual(verify_aligner(config)['status'], 'verified')

    def test_corrupt_installed_file_is_not_rebaselined(self):
        config, _ = self.fixture_lock()
        directory = self.root / config.alignment['model_dir']
        directory.mkdir(parents=True)
        broken = directory / ALIGNER_FILES[0]
        broken.write_bytes(b'corrupt')
        with patch('huggingface_hub.hf_hub_download') as fetch, self.assertRaises(RuntimeError):
            fetch_aligner(config, self.root / 'downloads')
        fetch.assert_not_called()
        self.assertEqual(broken.read_bytes(), b'corrupt')

    def test_bad_download_does_not_enter_installed_model_directory(self):
        config, _ = self.fixture_lock()
        def bad_download(*, filename, local_dir, **kwargs):
            target = Path(local_dir) / filename
            target.write_bytes(b'unexpected bytes')
            return str(target)
        with patch('huggingface_hub.hf_hub_download', side_effect=bad_download), self.assertRaises(RuntimeError):
            fetch_aligner(config, self.root / 'downloads')
        self.assertFalse((self.root / config.alignment['model_dir']).exists())
        self.assertEqual(list((self.root / 'downloads').iterdir()), [])

    def reports(self, kind='public'):
        def report(primary, secondary, cer):
            return {'schema_version': 3, 'comparison_policy': 'fixture-v1',
                'model_config': {'sha256': file_hash(self.path)},
                'runtime_code': _runtime_code_identity(),
                'cases': [{'id': str(i), 'kind': kind, 'audio_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
                    'truth_sha256': 'b' * 64, 'cer': cer, 'critical_errors': [],
                    'audit_status': 'consistent', 'false_confident': False,
                    'audit_needs_review': True, 'models': {'primary': primary, 'secondary': secondary}}
                    for i in range(5)]}
        left = report(self.config.strict_primary_engine, self.config.strict_secondary_engine, 0.2)
        target = self.config.profiles['high_quality']
        right = report(target['primary_engine'], target['secondary_engine'], 0.1)
        paths = self.root / 'baseline.json', self.root / 'candidate.json'
        for path, value in zip(paths, (left, right)):
            path.write_text(json.dumps(value), encoding='utf-8')
        return paths

    def mutate(self, path, operation):
        value = json.loads(path.read_text(encoding='utf-8'))
        operation(value)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_eligible_matched_evaluation_does_not_change_defaults(self):
        paths = self.reports()
        before = self.path.read_bytes()
        result = compare_evaluations(*paths)
        self.assertTrue(result['eligible'])
        self.assertEqual(self.path.read_bytes(), before)

    def test_comparison_rejects_duplicates_nonfinite_or_source_drift(self):
        for mutation in (
            lambda x: x['cases'].append(copy.deepcopy(x['cases'][0])),
            lambda x: x['cases'][0].update(cer=float('nan')),
        ):
            with self.subTest(mutation=mutation):
                paths = self.reports(); self.mutate(paths[1], mutation)
                with self.assertRaises(ValueError):
                    compare_evaluations(*paths)
        paths = self.reports()
        self.mutate(paths[1], lambda x: x['cases'][0].update(audio_sha256='different'))
        self.assertFalse(compare_evaluations(*paths)['eligible'])

    def test_synthetic_only_cases_cannot_promote_defaults(self):
        paths = self.reports(kind='tts')
        result = compare_evaluations(*paths)
        self.assertFalse(result['eligible'])
        self.assertIn('synthetic_only_not_quality_promotion_evidence', result['reasons'])

    def test_activation_and_rollback_preserve_comments_and_unrelated_changes(self):
        paths = self.reports()
        before = self.path.read_text(encoding='utf-8')
        receipt = activate_profile(self.config, 'high_quality', *paths)
        self.assertEqual(receipt['status'], 'applied')
        current = load_model_config(self.path)
        self.assertEqual(current.strict_primary_engine, 'fireredasr2-llm')
        text = self.path.read_text(encoding='utf-8')
        text += '\n# unrelated annotation after activation\n'
        self.path.write_text(text, encoding='utf-8')
        restored = rollback_profile(current, Path(receipt['receipt']))
        self.assertEqual(restored['status'], 'applied')
        self.assertEqual(load_model_config(self.path).strict_primary_engine, self.config.strict_primary_engine)
        self.assertIn('# unrelated annotation after activation', self.path.read_text(encoding='utf-8'))
        self.assertEqual(self.path.read_text(encoding='utf-8').split('\n# unrelated annotation')[0].rstrip(), before.rstrip())

    def test_stale_config_or_runtime_evaluation_cannot_activate(self):
        for field in ('model_config', 'runtime_code'):
            paths = self.reports()
            self.mutate(paths[1], lambda x: x[field].update(sha256='stale'))
            before = self.path.read_bytes()
            with self.assertRaises(ValueError):
                activate_profile(self.config, 'high_quality', *paths)
            self.assertEqual(self.path.read_bytes(), before)
