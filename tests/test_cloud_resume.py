import copy
import unittest
from zh_asr.cloud_review import project_vendor_result, CloudReviewError


class CloudVendorResultTests(unittest.TestCase):
    def test_partial_and_resumed_counts_survive_projection(self):
        intent = {"job_id":"job", "model":"anything", "protocol":"http", "vendor_request_sha256":"bound",
                  "source_audio_sha256":"source", "chunk_bindings":[{"index":1,"start_ms":0,"end_ms":100,"audio_sha256":"audio"}]}
        result = {"schema":"passwordcenter.vendor-result.v1", "vendor":"qwen", "request_sha256":"bound",
                  "status":"succeeded", "results":[{"id":"1","raw_response":{"output":{"text":"测试"}}}],
                  "new_requests":0, "reused_requests":1}
        projected = project_vendor_result(result, intent)
        self.assertEqual(1, projected["reused_chunks"])
        self.assertEqual(0, projected["new_requests"])
        self.assertEqual("audio", projected["chunks"][0]["audio_sha256"])
        self.assertTrue(projected["cloud_upload_performed"])
        for field, value in (("request_sha256","other"),("vendor","other")):
            bad = copy.deepcopy(result); bad[field] = value
            with self.assertRaises(CloudReviewError): project_vendor_result(bad, intent)
        bad = copy.deepcopy(result); bad["results"][0]["id"]="other"
        with self.assertRaises(CloudReviewError): project_vendor_result(bad, intent)


if __name__ == "__main__":
    unittest.main()
