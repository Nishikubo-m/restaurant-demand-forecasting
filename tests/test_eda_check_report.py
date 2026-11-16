"""Checks for detecting stale EDA verification reports."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eda import check_verification as checker
from eda.verify_notebooks import input_snapshot, sha256


class ReportCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        (self.root / "eda").mkdir()
        (self.root / "input" / "train.csv").write_text("a\n1\n")
        (self.root / "eda" / "one.ipynb").write_text("notebook")
        (self.root / "eda" / "verify_notebooks.py").write_text("verifier")
        (self.root / "pyproject.toml").write_text("project")
        (self.root / "uv.lock").write_text("lock")
        self.report = {
            "input_files": input_snapshot(self.root / "input"),
            "notebooks": [{"notebook": "eda/one.ipynb", "sha256": sha256(self.root / "eda" / "one.ipynb")}],
            "uv_lock_sha256": sha256(self.root / "uv.lock"),
            "pyproject_sha256": sha256(self.root / "pyproject.toml"),
            "verifier_sha256": sha256(self.root / "eda" / "verify_notebooks.py"),
            "python": sys.version.split()[0],
            "packages": {"example": "1.0"},
        }
        self.package_patch = patch.object(checker, "package_versions", return_value={"example": "1.0"})
        self.package_patch.start()
        self.addCleanup(self.package_patch.stop)

    def test_matching_report_has_no_discrepancies(self):
        self.assertEqual(checker.check_report(self.report, self.root), [])

    def test_detects_changed_inputs_notebooks_and_lock(self):
        (self.root / "input" / "train.csv").write_text("a\n2\n")
        (self.root / "eda" / "one.ipynb").write_text("changed")
        (self.root / "uv.lock").write_text("new lock")
        problems = checker.check_report(self.report, self.root)
        self.assertTrue(any("input changed: train.csv" in p for p in problems))
        self.assertTrue(any("notebook changed or is missing: eda/one.ipynb" in p for p in problems))
        self.assertTrue(any("uv.lock changed" in p for p in problems))

    def test_partial_coverage_is_explicit(self):
        (self.root / "eda" / "two.ipynb").write_text("new notebook")
        self.assertTrue(any("notebooks not covered" in p for p in checker.check_report(self.report, self.root)))
        self.assertEqual(checker.check_report(self.report, self.root, allow_partial=True), [])

    def test_detects_package_change(self):
        with patch.object(checker, "package_versions", return_value={"example": "2.0"}):
            self.assertTrue(any("package example changed" in p for p in checker.check_report(self.report, self.root)))

    def test_legacy_report_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lacks provenance fields"):
            checker.check_report({"notebooks": []}, self.root)


if __name__ == "__main__":
    unittest.main()
