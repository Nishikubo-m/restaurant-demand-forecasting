"""販売可能性・補完・評価粒度の重要な境界条件。"""
import unittest

import numpy as np
import pandas as pd

from exp.exp002_availability_weekday.run import add_availability, layer_metrics, predict


class AvailabilityTests(unittest.TestCase):
    def test_catalog_and_collaboration_keep_original_order(self):
        menus = pd.DataFrame({"item_id": ["core", "collab"], "menu_type": ["core", "collaboration"]})
        catalog = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                                "item_id": ["core", "collab"], "campaign_id": [None, "C"],
                                "catalog_active": [True, True]})
        campaigns = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "store_id": ["S1"],
                                  "campaign_id": ["C"], "campaign_active": [True]})
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02"]),
                              "store_id": ["S2", "S1", "S1", "S1"], "item_id": ["collab", "core", "collab", "core"]})
        result = add_availability(frame, (menus, catalog, campaigns))
        self.assertEqual(result.eligible.tolist(), [False, True, True, False])
        pd.testing.assert_frame_equal(result[frame.columns], frame)

    def test_fallback_levels_and_ineligible_zero(self):
        history = pd.DataFrame({"date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-02", "2023-01-02"]),
                                "store_id": ["A", "A", "B", "B"], "item_id": ["x", "x", "x", "y"],
                                "weekday": [0, 1, 0, 0], "sales_amount": [10., 30., 50., 100.], "eligible": [True] * 4})
        future = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 5), "store_id": ["A", "A", "C", "C", "A"],
                               "item_id": ["x", "x", "x", "z", "x"], "weekday": [0, 2, 0, 0, 0],
                               "eligible": [True, True, True, True, False]})
        pred, tiers = predict(history, future)
        np.testing.assert_allclose(pred, [10., 20., 30., 47.5, 0.])
        self.assertEqual(tiers, {"store_item_weekday": 1, "store_item": 1, "item": 1, "overall": 1, "ineligible_zero": 1})

    def test_rejects_future_leak_and_ineligible_positive_history(self):
        history = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "store_id": ["A"], "item_id": ["x"],
                                "weekday": [0], "sales_amount": [10.], "eligible": [False]})
        future = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "store_id": ["A"], "item_id": ["x"],
                               "weekday": [1], "eligible": [True]})
        with self.assertRaisesRegex(ValueError, "販売条件外"):
            predict(history, future)
        history["eligible"] = True
        history["date"] = pd.Timestamp("2024-01-02")
        with self.assertRaisesRegex(ValueError, "時系列順"):
            predict(history, future)

    def test_layer_metrics_aggregate_before_scoring(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                              "store_id": ["S", "S"], "item_id": ["A", "B"],
                              "actual": [10., 20.], "prediction": [12., 17.]})
        scores = layer_metrics(frame)
        self.assertAlmostEqual(scores["date_store_item"]["rmse"], np.sqrt(6.5))
        self.assertAlmostEqual(scores["daily_store"]["rmse"], 1.0)
        self.assertEqual(scores["daily_store"]["bias"], -1.0)
        self.assertEqual(scores["daily_store"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
