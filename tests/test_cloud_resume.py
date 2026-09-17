import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from test_qwen_audio3_broker_worker import _load_worker, _write_request, _write_wav


class CloudResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker = _load_worker()
        self.audio = self.root / 'fixture.wav'
        _write_wav(self.audio, duration_sec=2.5)
        self.request = self.root / 'fixture.running.json'
        _write_request(self.request, self.audio)
        self.change_request(chunk_sec=1, overlap_sec=0)

    def change_request(self, **values):
        request = json.loads(self.request.read_text(encoding='utf-8'))
        request.update(values)
        self.request.write_text(json.dumps(request), encoding='utf-8')

    def test_successful_chunks_survive_failure_and_ambiguous_outcomes_need_explicit_retry(self):
        response = {'output': {'text': '第一段'}, 'request_id': 'fixture-1'}
        transport = Mock(side_effect=[response, TimeoutError('unknown response')])
        failed = self.worker.process_request_file(self.request, api_key='fixture-not-a-real-key', transport=transport)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(len(failed['chunks']), 1)
        self.assertEqual(failed['text'], '第一段')
        self.assertEqual(transport.call_count, 2)
        no_transport = Mock(side_effect=AssertionError('must not blindly resend'))
        blocked = self.worker.process_request_file(self.request, api_key='fixture-not-a-real-key', transport=no_transport)
        self.assertEqual(blocked['status'], 'blocked')
        self.assertEqual(blocked['error_code'], 'cloud_chunk_outcome_unknown_explicit_retry_required')
        self.assertEqual(blocked['reused_chunks'], 1)
        no_transport.assert_not_called()
        self.change_request(retry_uncertain_chunks=True)
        transport = Mock(return_value={'output': {'text': '同意'}, 'request_id': 'fixture-next'})
        completed = self.worker.process_request_file(self.request, api_key='fixture-not-a-real-key', transport=transport)
        self.assertEqual(completed['status'], 'succeeded')
        self.assertEqual(completed['completion'], 'complete')
        self.assertEqual(completed['reused_chunks'], 1)
        self.assertEqual(completed['new_requests'], 2)
        self.assertEqual(completed['text'], '第一段\n同意\n同意')
        transport.reset_mock()
        cached = self.worker.process_request_file(self.request, api_key='fixture-not-a-real-key', transport=transport)
        self.assertEqual(cached['status'], 'succeeded')
        self.assertEqual(cached['reused_chunks'], 3)
        self.assertEqual(cached['new_requests'], 0)
        self.assertFalse(cached['cloud_upload_performed'])
        transport.assert_not_called()

    def test_modified_checkpoint_is_not_reused_or_silently_reuploaded(self):
        transport = Mock(return_value={'output': {'text': '测试'}, 'request_id': 'fixture'})
        self.worker.process_request_file(self.request, api_key='fixture-not-a-real-key', transport=transport)
        checkpoint = next(self.root.glob('*.work/chunk-000001.result.json'))
        data = json.loads(checkpoint.read_text(encoding='utf-8'))
        data['result']['text'] = 'different text'
        checkpoint.write_text(json.dumps(data), encoding='utf-8')
        transport.reset_mock()
        blocked = self.worker.process_request_file(self.request, api_key='fixture-not-a-real-key', transport=transport)
        self.assertEqual(blocked['error_code'], 'cloud_checkpoint_invalid')
        transport.assert_not_called()

    def test_changed_source_does_not_reuse_previous_paid_results(self):
        transport = Mock(return_value={'output': {'text': '测试'}, 'request_id': 'fixture'})
        self.worker.process_request_file(self.request, api_key='fixture-not-a-real-key', transport=transport)
        _write_wav(self.audio, duration_sec=2.8)
        transport.reset_mock()
        blocked = self.worker.process_request_file(self.request, api_key='fixture-not-a-real-key', transport=transport)
        self.assertEqual(blocked['error_code'], 'resume_input_mismatch')
        transport.assert_not_called()
