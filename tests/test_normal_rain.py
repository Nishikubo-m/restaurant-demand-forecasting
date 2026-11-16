"""雨日・非雨日平均の平年値による混合を検証する。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp018_normal_rain.run import predict


class NormalRainTests(unittest.TestCase):
    def test_mixes_only_historical_weather_using_future_normal(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24",
                                    "2023-01-02", "2023-01-09", "2023-01-16", "2023-01-23"]),
            "store_id": "S", "item_id": "I", "category": "main", "eligible": True,
            "selling_price": 10., "sales_amount": [100., 100., 200., 200.] * 2,
            "coupon": False, "weather_observed": ["sunny", "cloudy", "rainy", "rainy"] * 2,
        })
        future = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 2), "store_id": ["S"] * 2,
            "item_id": ["I"] * 2, "category": ["main"] * 2, "weekday": [0] * 2,
            "selling_price": [10.] * 2, "coupon": [False] * 2,
            "wet_probability_normal": [0.25, 0.75],
        })
        with patch("exp.exp018_normal_rain.run.base_predict", return_value=(np.array([100., 100.]), {})):
            values, details = predict(history, future)
        np.testing.assert_allclose(values, [100. * 12.5 / 15, 100. * 17.5 / 15])
        self.assertEqual(details["rain_factor_unavailable"], 0)


if __name__ == "__main__":
    unittest.main()
