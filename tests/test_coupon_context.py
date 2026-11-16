"""クーポン予定に応じた平均倍率と欠損時の動作を検証する。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp015_coupon_context.run import predict


class CouponContextTests(unittest.TestCase):
    def test_conditional_mean_and_unobserved_fallback(self):
        dates = pd.to_datetime([
            "2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24",
            "2023-01-02", "2023-01-09", "2023-01-16", "2023-01-23",
        ])
        history = pd.DataFrame({
            "date": dates, "store_id": "S", "item_id": "I", "category": "main",
            "eligible": True, "selling_price": 10.,
            "sales_amount": [100., 100., 200., 200.] * 2,
            "coupon": [False, False, True, True] * 2,
        })
        future = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 3), "store_id": ["S"] * 3,
            "item_id": ["I"] * 3, "category": ["main", "main", "other"],
            "weekday": [0] * 3, "selling_price": [10.] * 3,
            "coupon": [False, True, True],
        })
        with patch("exp.exp015_coupon_context.run.base_predict", return_value=(np.array([100., 100., 100.]), {})):
            values, details = predict(history, future)
        np.testing.assert_allclose(values, [200. / 3., 400. / 3., 100.])
        self.assertEqual(details["coupon_factor_unavailable"], 1)


if __name__ == "__main__":
    unittest.main()
