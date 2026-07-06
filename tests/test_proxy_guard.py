import os
import unittest
from unittest.mock import patch


class ProxyGuardTests(unittest.TestCase):
    def test_sanitized_env_removes_proxy_variables(self):
        from zh_asr.proxy_guard import PROXY_ENV_NAMES, sanitized_env

        dirty = {name: "http://127.0.0.1:7890" for name in PROXY_ENV_NAMES}

        with patch.dict(os.environ, dirty, clear=False):
            env = sanitized_env()

        for name in PROXY_ENV_NAMES:
            self.assertNotIn(name, env)

    def test_sanitized_env_keeps_local_and_aliyun_no_proxy(self):
        from zh_asr.proxy_guard import sanitized_env

        env = sanitized_env()

        self.assertIn("127.0.0.1", env["NO_PROXY"])
        self.assertIn("localhost", env["NO_PROXY"])
        self.assertIn("aliyun.com", env["NO_PROXY"])
        self.assertEqual(env["NO_PROXY"], env["no_proxy"])

    def test_sanitize_current_process_env_deletes_proxy_variables(self):
        from zh_asr.proxy_guard import PROXY_ENV_NAMES, sanitize_current_process_env

        dirty = {name: "http://127.0.0.1:7890" for name in PROXY_ENV_NAMES}

        with patch.dict(os.environ, dirty, clear=False):
            sanitize_current_process_env()
            for name in PROXY_ENV_NAMES:
                self.assertNotIn(name, os.environ)


if __name__ == "__main__":
    unittest.main()

