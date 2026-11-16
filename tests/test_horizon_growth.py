"""履歴だけで求めた傾向を予測期間に沿って外挿する。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp047_horizon_growth.run import predict


class HorizonGrowthTests(unittest.TestCase):
    def test_growth_scales_with_horizon_and_store_fallback(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-10-03", "2022-10-10", "2023-10-02", "2023-10-09"]),
            "store_id": "S", "layout_version": "A", "menu_type": "core", "eligible": True,
            "sales_amount": [100., 100., 400., 400.], "selling_price": 10., "is_holiday": False,
        })
        future = pd.DataFrame({"store_id": ["S", "S", "unseen"],
                               "date": pd.to_datetime(["2024-04-09", "2024-10-08", "2024-10-08"])})
        with patch("exp.exp047_horizon_growth.run.base_predict", return_value=(np.array([100., 100., 100.]), {})):
            values, details = predict(history, future)
        elapsed = (future.date - history.date.max()).dt.days.to_numpy()
        np.testing.assert_allclose(values, [100. * 4. ** (elapsed[0] / 365), 400., 100.])
        self.assertEqual(details["matched_growth_unavailable"], 1)


if __name__ == "__main__":
    unittest.main()
