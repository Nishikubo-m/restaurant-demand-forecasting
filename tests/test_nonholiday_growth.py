"""年次比較から祝日を除き、履歴不足時は補正しない。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp045_nonholiday_growth.run import predict


class NonholidayGrowthTests(unittest.TestCase):
    def test_holidays_excluded_and_unseen_store_falls_back(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2022-01-10", "2023-01-02", "2023-01-09", "2023-01-16"]),
            "store_id": "S", "layout_version": "A", "menu_type": "core", "eligible": True,
            "sales_amount": [100., 100., 400., 400., 10000.], "selling_price": 10.,
            "is_holiday": [False, False, False, False, True],
        })
        future = pd.DataFrame({"store_id": ["S", "unseen"]})
        with patch("exp.exp045_nonholiday_growth.run.base_predict", return_value=(np.array([100., 100.]), {})):
            values, details = predict(history, future)
        np.testing.assert_allclose(values, [200., 100.])
        self.assertEqual(details["matched_growth_unavailable"], 1)


if __name__ == "__main__":
    unittest.main()
