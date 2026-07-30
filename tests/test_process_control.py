import subprocess
import unittest
from unittest.mock import patch

from zh_asr.process_control import (
    PROCESS_TOKEN_ENV,
    _signal_wsl_token,
    tagged_process_env,
    terminate_process_tree,
    terminate_wsl_processes,
)


class ProcessControlTests(unittest.TestCase):
    def test_tagged_environment_preserves_wslenv_and_adds_token_once(self):
        env = tagged_process_env(
            "chineseasr-job-1",
            {"WSLENV": "EXISTING/p:ZH_ASR_PROCESS_TOKEN/w"},
        )

        self.assertEqual("chineseasr-job-1", env[PROCESS_TOKEN_ENV])
        self.assertEqual(
            ["EXISTING/p", "ZH_ASR_PROCESS_TOKEN/w"],
            env["WSLENV"].split(":"),
        )

    def test_windows_tree_termination_targets_one_exact_pid(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

            def wait(self, timeout):
                return 1

        with (
            patch("zh_asr.process_control.os.name", "nt"),
            patch("zh_asr.process_control.subprocess.run") as run,
        ):
            terminate_process_tree(FakeProcess())

        command = run.call_args.args[0]
        self.assertEqual(
            ["taskkill", "/PID", "12345", "/T", "/F"],
            command,
        )

    def test_wsl_cleanup_signals_only_named_distributions_and_exact_token(self):
        with (
            patch("zh_asr.process_control._signal_wsl_token") as signal_token,
            patch("zh_asr.process_control.time.sleep"),
        ):
            terminate_wsl_processes(
                ("Ubuntu", "Ubuntu", "Other"),
                "chineseasr-job-2",
            )

        self.assertEqual(
            [
                ("Ubuntu", "chineseasr-job-2", "TERM"),
                ("Other", "chineseasr-job-2", "TERM"),
                ("Ubuntu", "chineseasr-job-2", "KILL"),
                ("Other", "chineseasr-job-2", "KILL"),
            ],
            [call.args for call in signal_token.call_args_list],
        )

    def test_wsl_cleanup_uses_exec_and_passes_token_as_data(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("zh_asr.process_control.subprocess.run", return_value=completed) as run:
            _signal_wsl_token("Ubuntu", "chineseasr-job-3", "TERM")

        command = run.call_args.args[0]
        self.assertEqual(command[:5], ["wsl.exe", "-d", "Ubuntu", "--exec", "sh"])
        self.assertEqual(command[-2:], ["chineseasr-job-3", "TERM"])
        self.assertNotIn("chineseasr-job-3", command[6])


if __name__ == "__main__":
    unittest.main()
