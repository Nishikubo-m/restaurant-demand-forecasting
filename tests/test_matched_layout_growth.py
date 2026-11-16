"""同一条件での前年比と欠損時の補完を検証する。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp042_matched_layout_growth.run import predict


class MatchedLayoutGrowthTests(unittest.TestCase):
    def test_matched_growth_and_unseen_store(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2022-01-10", "2023-01-02", "2023-01-09"]),
            "store_id": "S", "layout_version": "A", "menu_type": "core", "eligible": True,
            "sales_amount": [100., 100., 400., 400.], "selling_price": 10.,
        })
        future = pd.DataFrame({"store_id": ["S", "unseen"]})
        with patch("exp.exp042_matched_layout_growth.run.base_predict", return_value=(np.array([100., 100.]), {})):
            values, details = predict(history, future)
        np.testing.assert_allclose(values, [200., 100.])
        self.assertEqual(details["matched_growth_groups"], 1)
        self.assertEqual(details["matched_growth_unavailable"], 1)


if __name__ == "__main__":
    unittest.main()
