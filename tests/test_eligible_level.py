"""販売可能日数に応じた直近水準補正を検証する。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp.exp062_eligible_level.run import predict


class EligibleLevelTests(unittest.TestCase):
    def test_no_old_history_does_not_need_level_correction(self):
        history = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2022-01-10", "2023-01-02", "2023-01-09", "2023-01-02", "2023-01-09"]),
            "store_id": "S", "item_id": ["I", "I", "I", "I", "C", "C"], "weekday": 0,
            "layout_version": "A", "menu_type": ["core"] * 4 + ["seasonal"] * 2,
            "eligible": True, "is_holiday": False, "selling_price": 10.,
            "sales_amount": [100., 100., 400., 400., 100., 100.],
        })
        future = pd.DataFrame({"date": pd.to_datetime(["2023-07-10"] * 3),
                               "store_id": ["S", "S", "unseen"], "item_id": ["I", "C", "I"], "weekday": 0})
        with patch("exp.exp062_eligible_level.run.base_predict", return_value=(np.array([100., 100., 100.]), {})):
            values, details = predict(history, future)
        elapsed = (future.date.iloc[0] - history.date.max()).days
        growth = 4. ** (elapsed / 365)
        np.testing.assert_allclose(values, [100. * growth * 16. / 13., 100. * growth, 100.])
        self.assertEqual(details["eligible_level_fallback"], 1)


if __name__ == "__main__":
    unittest.main()
