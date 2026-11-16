"""Focused checks for EDA verification provenance and failure handling."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

from eda import verify_notebooks as verifier


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        (self.root / "eda").mkdir()
        (self.root / "input" / "train.csv").write_text("a\n1\n", encoding="utf-8")
        (self.root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (self.root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        self.notebook = self.root / "eda" / "sample.ipynb"
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('ok')")]), self.notebook)
        self.report = self.root / "output" / "report.json"

    def test_snapshot_detects_same_size_change_addition_and_removal(self):
        inputs = self.root / "input"
        (inputs / "unused_directory").mkdir()
        old = verifier.input_snapshot(inputs)
        (inputs / "train.csv").write_text("a\n2\n", encoding="utf-8")
        (inputs / "test.csv").write_text("x\n", encoding="utf-8")
        new = verifier.input_snapshot(inputs)
        self.assertEqual(verifier.changed_inputs(old, new), {"added": ["test.csv"], "removed": [], "changed": ["train.csv"]})
        (inputs / "train.csv").unlink()
        self.assertEqual(verifier.changed_inputs(new, verifier.input_snapshot(inputs))["removed"], ["train.csv"])

    def test_report_write_failure_preserves_existing_file(self):
        self.report.parent.mkdir()
        self.report.write_text("previous\n", encoding="utf-8")
        with patch.object(verifier.json, "dump", side_effect=ValueError("failed")):
            with self.assertRaisesRegex(ValueError, "failed"):
                verifier.write_report(self.report, {"ok": True})
        self.assertEqual(self.report.read_text(encoding="utf-8"), "previous\n")
        self.assertEqual(list(self.report.parent.iterdir()), [self.report])

    def run_main(self, mutate=None):
        class FakeClient:
            def __init__(self, notebook, **kwargs):
                self.notebook = notebook

            def execute(self):
                if mutate:
                    mutate()
                self.notebook.cells[0].execution_count = 1

        argv = ["verify_notebooks.py", str(self.notebook), "--report", str(self.report)]
        with patch.object(verifier, "ROOT", self.root), patch.object(verifier, "NotebookClient", FakeClient), \
             patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            verifier.main()

    def test_success_records_fingerprints_and_environment(self):
        self.run_main()
        result = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(result["input_files"]["train.csv"]["sha256"], verifier.sha256(self.root / "input" / "train.csv"))
        self.assertEqual(result["uv_lock_sha256"], verifier.sha256(self.root / "uv.lock"))
        self.assertEqual(result["pyproject_sha256"], verifier.sha256(self.root / "pyproject.toml"))
        self.assertEqual(result["notebooks"][0]["sha256"], verifier.sha256(self.notebook))
        self.assertIn("ipykernel", result["packages"])

    def test_input_change_aborts_and_preserves_prior_report(self):
        self.report.parent.mkdir()
        self.report.write_text("old result", encoding="utf-8")
        def change_input():
            (self.root / "input" / "train.csv").write_text("a\n2\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Input files changed"):
            self.run_main(change_input)
        self.assertEqual(self.report.read_text(encoding="utf-8"), "old result")

    def test_notebook_change_aborts_and_preserves_prior_report(self):
        self.report.parent.mkdir()
        self.report.write_text("old result", encoding="utf-8")
        def change_notebook():
            self.notebook.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Notebook changed"):
            self.run_main(change_notebook)
        self.assertEqual(self.report.read_text(encoding="utf-8"), "old result")

    def test_already_verified_notebook_is_rechecked(self):
        saved = [{"sha256": verifier.sha256(self.notebook)}]
        verifier.verify_notebook_hashes([self.notebook], saved)
        self.notebook.write_text("changed after execution", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed after verification"):
            verifier.verify_notebook_hashes([self.notebook], saved)

    def test_configuration_change_aborts_and_preserves_prior_report(self):
        self.report.parent.mkdir()
        self.report.write_text("old result", encoding="utf-8")
        def change_lock():
            (self.root / "uv.lock").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "configuration or verifier changed"):
            self.run_main(change_lock)
        self.assertEqual(self.report.read_text(encoding="utf-8"), "old result")

    def test_report_cannot_overwrite_input(self):
        target = self.root / "input" / "train.csv"
        before = target.read_bytes()
        argv = ["verify_notebooks.py", str(self.notebook), "--report", str(target)]
        with patch.object(verifier, "ROOT", self.root), patch.object(sys, "argv", argv), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_info:
                verifier.main()
        self.assertEqual(exit_info.exception.code, 2)
        self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
