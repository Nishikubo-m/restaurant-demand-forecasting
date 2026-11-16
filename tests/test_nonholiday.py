"""非祝日補正と祝日予測の維持を確認する。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp025_nonholiday.run import predict


class NonholidayTests(unittest.TestCase):
    def test_nonholiday_ratio_and_holiday_unchanged(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24",
                                    "2023-01-02", "2023-01-09", "2023-01-16", "2023-01-23"]),
            "store_id": "S", "item_id": "I", "category": "main", "eligible": True,
            "menu_type": "core", "selling_price": 10.,
            "sales_amount": [100., 100., 200., 200.] * 2,
            "is_holiday": [False, False, True, True] * 2,
            "layout_version": "A", "coupon": False,
            "weather_observed": ["sunny", "rainy", "sunny", "rainy"] * 2,
        })
        future = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "store_id": ["S"] * 2, "item_id": ["I"] * 2, "category": ["main"] * 2,
            "weekday": [0] * 2, "selling_price": [10.] * 2, "coupon": [False] * 2,
            "wet_probability_normal": [0.5] * 2, "is_holiday": [False, True],
        })
        with patch("exp.exp025_nonholiday.run.base_predict", return_value=(np.array([100., 100.]), {})):
            values, details = predict(history, future)
        np.testing.assert_allclose(values, [100. * 100. / 150., 100.])
        self.assertEqual(details["nonholiday_factor_unavailable"], 0)


if __name__ == "__main__":
    unittest.main()
