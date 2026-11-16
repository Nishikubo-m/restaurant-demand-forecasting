"""最新レイアウト水準への切り替えで成長率を二重適用しないことを検証。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp019_latest_layout_level.run import predict


class LayoutLevelTests(unittest.TestCase):
    def test_replaces_growth_with_latest_layout_mean(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24",
                                    "2023-01-02", "2023-01-09", "2023-01-16", "2023-01-23"]),
            "store_id": "S", "item_id": "I", "category": "main", "eligible": True,
            "menu_type": "core", "selling_price": 10.,
            "sales_amount": [100., 100., 100., 100., 200., 200., 200., 200.],
            "layout_version": ["A"] * 4 + ["B"] * 4,
            "coupon": False, "weather_observed": ["sunny", "rainy", "sunny", "rainy"] * 2,
        })
        future = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]), "store_id": ["S"], "item_id": ["I"],
            "category": ["main"], "weekday": [0], "selling_price": [10.],
            "coupon": [False], "wet_probability_normal": [0.5],
        })
        # 共通予測器が年間成長率 2 を適用済みの予測を返すと仮定する。
        with patch("exp.exp019_latest_layout_level.run.base_predict", return_value=(np.array([200.]), {})):
            values, details = predict(history, future)
        np.testing.assert_allclose(values, [200. * (20. / 17.5) / 2.])
        self.assertEqual(details["layout_level_unavailable"], 0)


if __name__ == "__main__":
    unittest.main()
