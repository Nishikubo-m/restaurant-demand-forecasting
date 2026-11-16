"""同じ曜日・レイアウトの過去ペアと未観測店舗の補完を検証する。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp048_paired_growth.run import predict


class PairedGrowthTests(unittest.TestCase):
    def test_paired_weekday_growth_and_missing_store(self):
        latest = pd.date_range("2023-10-01", periods=14)
        dates = (latest - pd.Timedelta(days=364)).append(latest)
        history = pd.DataFrame({"date": dates, "store_id": "S", "layout_version": "A",
                                "menu_type": "core", "eligible": True, "is_holiday": False,
                                "selling_price": 10., "sales_amount": [100.] * 14 + [400.] * 14})
        future = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 2), "store_id": ["S", "unseen"]})
        with patch("exp.exp048_paired_growth.run.base_predict", return_value=(np.array([100., 100.]), {})):
            values, details = predict(history, future)
        elapsed = (future.date.iloc[0] - history.date.max()).days
        np.testing.assert_allclose(values, [100. * 4. ** (elapsed / 365), 100.])
        self.assertEqual(details["paired_growth_stores"], 1)
        self.assertEqual(details["matched_growth_unavailable"], 1)


if __name__ == "__main__":
    unittest.main()
