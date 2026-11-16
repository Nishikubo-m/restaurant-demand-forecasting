"""年末以外の基準日でも直近3か月の前年比を使う。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp054_rolling_quarter.run import predict


class RollingQuarterTests(unittest.TestCase):
    def test_june_cutoff_uses_june_paired_days(self):
        latest = pd.date_range("2023-06-01", periods=14)
        dates = (latest - pd.Timedelta(days=364)).append(latest)
        history = pd.DataFrame({"date": dates, "store_id": "S", "layout_version": "A",
                                "menu_type": "core", "eligible": True, "is_holiday": False,
                                "selling_price": 10., "sales_amount": [100.] * 14 + [400.] * 14})
        future = pd.DataFrame({"date": pd.to_datetime(["2023-12-14"]), "store_id": ["S"]})
        with patch("exp.exp054_rolling_quarter.run.base_predict", return_value=(np.array([100.]), {})):
            values, details = predict(history, future)
        elapsed = (future.date.iloc[0] - history.date.max()).days
        np.testing.assert_allclose(values, [100. * 4. ** (elapsed / 365)])
        self.assertEqual(details["paired_growth_stores"], 1)


if __name__ == "__main__":
    unittest.main()
