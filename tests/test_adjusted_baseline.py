"""共通の調整付き平均予測の回帰テスト。"""
import unittest

import numpy as np
import pandas as pd

from src.adjusted_baseline import predict


class AdjustedBaselineTests(unittest.TestCase):
    def test_price_normalized_weighted_average_and_growth(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2023-01-02"]),
            "store_id": ["S", "S"], "item_id": ["I", "I"],
            "weekday": [0, 0], "category": ["food", "food"],
            "menu_type": ["core", "core"], "eligible": [True, True],
            "is_holiday": [False, False], "selling_price": [10., 20.],
            "sales_amount": [100., 240.],
        })
        future = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "store_id": ["S", "S"], "item_id": ["I", "I"],
            "weekday": [0, 0], "category": ["food", "food"],
            "eligible": [True, False], "is_holiday": [False, False],
            "selling_price": [30., 30.],
        })
        values, _ = predict(history, future)
        # ((10 + 12 * 3) / 4) * 30 * (12 / 10)
        np.testing.assert_allclose(values, [414., 0.])

    def test_history_must_precede_forecast(self):
        history = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
        future = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
        with self.assertRaises(ValueError):
            predict(history, future)


if __name__ == "__main__":
    unittest.main()
