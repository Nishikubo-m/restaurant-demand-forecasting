"""同曜日ペア不足時に年間比較を使い、未知の店舗は補正しない。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp060_fallback_level.run import predict


class FallbackLevelTests(unittest.TestCase):
    def test_annual_matched_fallback(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2022-01-10", "2023-01-02", "2023-01-09"]),
            "store_id": "S", "layout_version": "A", "menu_type": "core", "eligible": True,
            "sales_amount": [100., 100., 400., 400.], "selling_price": 10., "is_holiday": False,
        })
        future = pd.DataFrame({"date": pd.to_datetime(["2023-07-09"] * 2), "store_id": ["S", "unseen"]})
        with patch("exp.exp060_fallback_level.run.base_predict", return_value=(np.array([100., 100.]), {})):
            values, details = predict(history, future)
        elapsed = (future.date.iloc[0] - history.date.max()).days
        np.testing.assert_allclose(values, [100. * 4. ** (elapsed / 365) * 16. / 13., 100.])
        self.assertEqual(details["paired_growth_stores"], 0)
        self.assertEqual(details["recent_level_unavailable"], 1)


if __name__ == "__main__":
    unittest.main()
