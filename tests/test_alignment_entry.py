from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave
from zh_asr import alignment_entry as entry


class AlignmentEntryTests(unittest.TestCase):
    def fixture(self,root):
        audio=root/'original.wav';text=root/'original.txt';out=root/'result.json'
        with wave.open(str(audio),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*16000)
        text.write_text('测试',encoding='utf-8')
        return audio,text,out

    def response(self,*args,**kwargs):
        return [{'status':'succeeded','exact_text_coverage':True,
            'model_identity':{'model':'synthetic-unit-model'},
            'items':[{'text':'测','start_ms':0,'end_ms':400},{'text':'试','start_ms':400,'end_ms':800}]}]

    def patches(self,root):
        stack=ExitStack()
        stack.enter_context(mock.patch.object(entry,'alignment_contract',return_value=({'max_audio_sec':300},root,{})))
        return stack

    def test_shared_alignment_publishes_bound_seconds_without_rewriting_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audio,text,out=self.fixture(root);before=audio.read_bytes()
            with self.patches(root),mock.patch.object(entry,'align_many',side_effect=self.response) as align:
                value=entry.align_file(audio,text,out,device='cpu',config=object())
            self.assertEqual(before,audio.read_bytes())
            self.assertEqual('zh_asr.alignment-entry.v1',value['schema'])
            self.assertEqual(hashlib.sha256(before).hexdigest(),value['audio_sha256'])
            self.assertEqual(hashlib.sha256('测试'.encode()).hexdigest(),value['text_sha256'])
            self.assertEqual(0.8,value['words'][-1]['end'])
            self.assertFalse(value['lexical_truth_verified'])
            self.assertTrue(value['exact_text_coverage'])
            self.assertFalse(list(root.glob('.align-*')))
            self.assertEqual(1,align.call_count)

    def test_incomplete_alignment_keeps_previous_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audio,text,out=self.fixture(root);out.write_text('previous')
            with self.patches(root),mock.patch.object(entry,'align_many',return_value=[{'status':'succeeded','exact_text_coverage':False}]):
                with self.assertRaisesRegex(RuntimeError,'coverage'):
                    entry.align_file(audio,text,out,device='cpu',config=object())
            self.assertEqual('previous',out.read_text())

    def test_input_change_during_alignment_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audio,text,out=self.fixture(root)
            def changed(*args,**kwargs):
                text.write_text('修改',encoding='utf-8')
                return self.response()
            with self.patches(root),mock.patch.object(entry,'align_many',side_effect=changed):
                with self.assertRaisesRegex(RuntimeError,'inputs changed'):
                    entry.align_file(audio,text,out,device='cpu',config=object())
            self.assertFalse(out.exists())

    def test_invalid_alignment_times_are_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audio,text,out=self.fixture(root)
            result=self.response();result[0]['items'][0]['start_ms']=float('nan')
            with self.patches(root),mock.patch.object(entry,'align_many',return_value=result):
                with self.assertRaises(ValueError):entry.align_file(audio,text,out,device='cpu',config=object())
            self.assertFalse(out.exists())

    def test_overlong_audio_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audio,text,out=self.fixture(root)
            with mock.patch.object(entry,'alignment_contract',return_value=({'max_audio_sec':0.5},root,{})),mock.patch.object(entry,'align_many') as align:
                with self.assertRaisesRegex(ValueError,'at most'):
                    entry.align_file(audio,text,out,device='cpu',config=object())
                align.assert_not_called()

    def test_originals_cannot_be_output_destinations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audio,text,out=self.fixture(root)
            for target in (audio,text):
                with self.assertRaisesRegex(ValueError,'original'):
                    entry.align_file(audio,text,target,device='cpu',config=object())

    def test_alignment_info_does_not_run_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.patches(root),mock.patch.object(entry,'align_many') as align:
                value=entry.describe(object());align.assert_not_called()
            self.assertTrue(value['read_only'])
            self.assertFalse(value['weights_present'])

    def test_align_dispatch_uses_existing_bounded_supervisor(self):
        from zh_asr import __main__ as cli
        with mock.patch.dict('os.environ',{},clear=False),mock.patch.object(cli,'_supervise_gpu_cli',return_value=0) as supervisor:
            import os
            os.environ.pop('ZH_ASR_GPU_BROKER_CHILD_TOKEN',None)
            os.environ.pop('ZH_ASR_SUPERVISOR_PID',None)
            code=cli.main(['align','a.wav','--text-file','a.txt','--output','result.json','--timeout-sec','41'])
            self.assertEqual(0,code)
            self.assertEqual(41,supervisor.call_args.kwargs['timeout_seconds'])

    def test_cpu_alignment_uses_supervisor_without_gpu_reservation(self):
        from zh_asr import __main__ as cli
        with mock.patch.dict('os.environ',{},clear=False),mock.patch.object(cli,'_supervise_gpu_cli',return_value=0) as supervisor:
            import os
            os.environ.pop('ZH_ASR_SUPERVISOR_PID',None)
            self.assertEqual(0,cli.main(['align','a.wav','--text-file','a.txt','--output','result.json','--device','cpu']))
            self.assertFalse(supervisor.call_args.kwargs['gpu'])


if __name__=='__main__':unittest.main()
