"""共通評価・リーク防止・公式変換の境界条件。"""
import contextlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp002_availability_weekday.run import validation_score as previous_score
from src import runner, submission
from src.availability import add_availability
from src.baselines import predict_available_means
from src.metrics import layer_metrics, validation_score
from src.runner import _predict, _prepare


class SharedMetricsTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"date": pd.to_datetime(["2022-01-01", "2022-01-02", "2023-01-01", "2023-01-02"]),
                                   "store_id": ["S"] * 4, "item_id": ["M"] * 4,
                                   "sales_amount": [1., 3., 4., 3.]})
        self.valid = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                                   "store_id": ["S"] * 2, "item_id": ["M"] * 2,
                                   "sales_amount": [6., 4.]})
        self.pred = np.array([3., 3.])

    def test_matches_previous_implementation_and_weighted_rmse(self):
        new = validation_score(self.train, self.valid, self.pred)
        old = previous_score(self.train, self.valid, self.pred)
        self.assertEqual(new["layers"], old["layers"])
        self.assertEqual(new["composite_normalized_rmse"], old["composite_normalized_rmse"])
        total = new["aggregated_rows"]
        squared = sum(m["rows"] * (m["weight"] * m["rmse"]) ** 2 for m in new["layers"].values())
        self.assertAlmostEqual(np.sqrt(squared / total), new["composite_normalized_rmse"])

    def test_refuses_reference_overlap_and_invalid_predictions(self):
        with self.assertRaisesRegex(ValueError, "重複"):
            validation_score(self.train, self.valid.assign(date=pd.to_datetime(["2023-02-01", "2023-02-02"])), self.pred)
        with self.assertRaisesRegex(ValueError, "長さ"):
            validation_score(self.train, self.valid, np.array([1.]))
        with self.assertRaisesRegex(ValueError, "欠損"):
            validation_score(self.train, self.valid, np.array([np.nan, 1.]))

    def test_rejects_duplicate_keys_before_aggregation(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01"]), "store_id": ["S", "S"],
                              "item_id": ["M", "M"], "actual": [1., 2.], "prediction": [1., 2.]})
        with self.assertRaisesRegex(ValueError, "重複"):
            layer_metrics(frame)

    def test_missing_reference_target_is_not_silently_averaged(self):
        train = self.train.copy()
        train.loc[0, "sales_amount"] = np.nan
        with self.assertRaisesRegex(ValueError, "欠損"):
            validation_score(train, self.valid, self.pred)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.future = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                                    "store_id": ["S", "S"], "item_id": ["M", "M"]})

    def test_prepare_rejects_reordering_and_target_leak(self):
        with self.assertRaisesRegex(ValueError, "行順"):
            _prepare(self.future, lambda x: x.iloc[::-1], with_target=False)
        with self.assertRaisesRegex(ValueError, "目的変数"):
            _prepare(self.future, lambda x: x.assign(sales_amount=1), with_target=False)

    def test_predict_rejects_mutating_order_and_protects_caller(self):
        original = self.future.copy()
        def reorder(history, future):
            future.sort_values("date", ascending=False, inplace=True)
            return np.array([1., 2.]), {}
        with self.assertRaisesRegex(ValueError, "行順"):
            _predict(reorder, pd.DataFrame(), self.future)
        pd.testing.assert_frame_equal(self.future, original)

    def test_predict_rejects_nonfinite_and_negative(self):
        for vals in ([np.nan, 1], [np.inf, 1], [-1, 1]):
            with self.subTest(vals=vals), self.assertRaisesRegex(ValueError, "有限かつ非負"):
                _predict(lambda h, f: (np.array(vals), {}), pd.DataFrame(), self.future)

    def test_provenance_includes_nested_python_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "nested").mkdir(parents=True)
            (root / "input").mkdir()
            (root / "src" / "nested" / "feature.py").write_text("VALUE = 1\n")
            (root / "input" / "train.csv").write_text("x\n")
            with patch.object(runner, "ROOT_DIR", root), patch.object(runner, "INPUT_DIR", root / "input"):
                provenance = runner._provenance()
            self.assertIn(str(Path("nested") / "feature.py"), provenance["source_sha256"])
            self.assertIn("train.csv", provenance["input_sha256"])


class SharedAvailabilityTests(unittest.TestCase):
    def test_collaboration_store_and_row_order(self):
        menus = pd.DataFrame({"item_id": ["core", "collab"], "menu_type": ["core", "collaboration"]})
        catalog = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                                "item_id": ["core", "collab"], "campaign_id": [None, "C"],
                                "catalog_active": [True, True]})
        campaigns = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "store_id": ["S1"],
                                  "campaign_id": ["C"], "campaign_active": [True]})
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                              "store_id": ["S2", "S1", "S1"], "item_id": ["collab", "collab", "core"]})
        result = add_availability(frame, (menus, catalog, campaigns))
        self.assertEqual(result.eligible.tolist(), [False, True, False])
        pd.testing.assert_frame_equal(result[frame.columns], frame)

    def test_baseline_fallback_and_zero_mask(self):
        history = pd.DataFrame({"date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-02", "2023-01-02"]),
                                "store_id": ["A", "A", "B", "B"], "item_id": ["x", "x", "x", "y"],
                                "weekday": [0, 1, 0, 0], "sales_amount": [10., 30., 50., 100.], "eligible": [True] * 4})
        future = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 5), "store_id": ["A", "A", "C", "C", "A"],
                               "item_id": ["x", "x", "x", "z", "x"], "weekday": [0, 2, 0, 0, 0],
                               "eligible": [True, True, True, True, False]})
        pred, tiers = predict_available_means(history, future)
        np.testing.assert_allclose(pred, [10., 20., 30., 47.5, 0.])
        self.assertEqual(tiers, {"store_item_weekday": 1, "store_item": 1, "item": 1, "overall": 1, "ineligible_zero": 1})


class SubmissionTests(unittest.TestCase):
    def test_wrapper_runs_script_from_zip_for_convert_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            code = '''import argparse\nfrom pathlib import Path\np=argparse.ArgumentParser()\np.add_argument("--predictions",required=True)\np.add_argument("--output")\np.add_argument("--validate",action="store_true")\na=p.parse_args()\nassert Path(a.predictions).is_file()\nif not a.validate: Path(a.output).write_text("official output")\n'''
            with zipfile.ZipFile(path / "submission_scripts.zip", "w") as z:
                z.writestr("submission_scripts/convert_submission.py", code)
            source = path / "prediction.csv"
            source.write_text("prediction")
            output = path / "out" / "submission.csv"
            with patch.object(submission, "INPUT_DIR", path), patch.object(sys, "argv", ["submission", "--predictions", str(source), "--output", str(output)]):
                submission.main()
            self.assertEqual(output.read_text(), "official output")
            with patch.object(submission, "INPUT_DIR", path), patch.object(sys, "argv", ["submission", "--predictions", str(output), "--validate"]):
                submission.main()

    def test_wrapper_refuses_overwriting_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "prediction.csv"
            source.write_text("original")
            with patch.object(sys, "argv", ["submission", "--predictions", str(source), "--output", str(source)]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    submission.main()
            self.assertEqual(source.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
