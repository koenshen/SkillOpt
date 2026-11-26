import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.evolution_test import pipeline


class PipelineTests(unittest.TestCase):
    def test_exact_reported_run_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve() / "run with spaces"
            output.mkdir()
            summary = output / "summary.json"
            summary.write_text('{"status":"complete"}', encoding="utf-8")
            for args, marker in [(["--source", "source"], f"Output: {output}"),
                                 (["--resume", str(output)], f"Already complete: {summary}")]:
                with self.subTest(marker=marker), patch.object(pipeline.subprocess, "Popen") as spawn, \
                     patch.object(pipeline.subprocess, "run") as evaluate, redirect_stdout(io.StringIO()):
                    child = spawn.return_value.__enter__.return_value
                    child.stdout = io.StringIO(marker + "\n")
                    child.wait.return_value = 0
                    evaluate.return_value.returncode = 7
                    self.assertEqual(pipeline.main(args), 7)
                    self.assertEqual(spawn.call_args.args[0][-len(args):], args)
                    evaluate.assert_called_once_with([sys.executable, "-B", "-u", "-m",
                                                      "scripts.evolution_test.run", "--evolution-dir", str(output)])

    def test_failure_and_unfinished_result_never_start_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            (output / "summary.json").write_text('{"status":"running"}', encoding="utf-8")
            for code in (1, 0):
                with self.subTest(code=code), patch.object(pipeline.subprocess, "Popen") as spawn, \
                     patch.object(pipeline.subprocess, "run") as evaluate, redirect_stdout(io.StringIO()):
                    child = spawn.return_value.__enter__.return_value
                    child.stdout = io.StringIO(f"Output: {output}\n")
                    child.wait.return_value = code
                    if code:
                        self.assertEqual(pipeline.main(["--source", "source"]), code)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "no complete result"):
                            pipeline.main(["--source", "source"])
                    evaluate.assert_not_called()

    def test_inspection_only_forwards_to_evolution(self):
        with patch.object(pipeline.subprocess, "run", return_value=MagicMock(returncode=0)) as invoke:
            args = ["--source", "source", "--inspect"]
            self.assertEqual(pipeline.main(args), 0)
            invoke.assert_called_once_with([sys.executable, "-B", "-u", "-m", "skillopt_evolution.run", *args])
