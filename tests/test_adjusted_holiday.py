"""祝日補正で履歴不足時に既存予測が維持されることを確認する。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp032_adjusted_holiday.run import predict


class AdjustedHolidayTests(unittest.TestCase):
    def test_constant_history_and_leap_day_fallback(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24",
                                    "2023-01-02", "2023-01-09", "2023-01-16", "2023-01-23"]),
            "store_id": "S", "item_id": "I", "category": "main", "eligible": True,
            "menu_type": "core", "selling_price": 10., "sales_amount": 100.,
            "is_holiday": False, "layout_version": "A", "coupon": False,
            "weather_observed": ["sunny", "rainy", "sunny", "rainy"] * 2,
        })
        future = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-02-29"]),
            "store_id": ["S"] * 2, "item_id": ["I"] * 2, "category": ["main"] * 2,
            "weekday": [0, 3], "selling_price": [10.] * 2, "coupon": [False] * 2,
            "wet_probability_normal": [0.3, 0.4], "is_holiday": [False] * 2,
        })
        with patch("exp.exp032_adjusted_holiday.run.base_predict", return_value=(np.array([100., 100.]), {})):
            values, details = predict(history, future)
        np.testing.assert_allclose(values, [100., 100.])
        self.assertEqual(details["rain_factor_unavailable"], 1)
        self.assertEqual(details["dayofyear_factor_unavailable"], 1)
        self.assertEqual(details["adjusted_holiday_unavailable"], 1)


if __name__ == "__main__":
    unittest.main()
