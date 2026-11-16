"""実験で再利用できる単純な予測方法。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TARGET_COL


def predict_available_means(history: pd.DataFrame, future: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """販売可能な過去日から平均を算出し、未観測時は段階的に補完する。"""
    if history.empty or future.empty or history.date.max() >= future.date.min():
        raise ValueError("履歴と予測期間が時系列順に分かれていません")
    if ((~history.eligible) & history[TARGET_COL].gt(0)).any():
        raise ValueError("販売条件外の過去データに正の売上があります")
    active = history[history.eligible]
    if active.empty or active[TARGET_COL].isna().any() or active[TARGET_COL].lt(0).any():
        raise ValueError("販売可能な履歴売上が空、欠損または負です")
    x = future[["store_id", "item_id", "weekday"]].copy().reset_index(drop=True)
    levels = [
        ["store_id", "item_id", "weekday"],
        ["store_id", "item_id"],
        ["item_id"],
    ]
    names = ["store_item_weekday", "store_item", "item"]
    predictions = pd.Series(np.nan, index=x.index, dtype=float)
    tiers = pd.Series("overall", index=x.index, dtype="object")
    for cols, name in zip(levels, names):
        means = active.groupby(cols, observed=True)[TARGET_COL].mean().rename("mean").reset_index()
        vals = x[cols].merge(means, on=cols, how="left", validate="many_to_one", sort=False)["mean"]
        missing = predictions.isna() & vals.notna()
        predictions.loc[missing] = vals.loc[missing]
        tiers.loc[missing] = name
    predictions = predictions.fillna(float(active[TARGET_COL].mean()))
    eligible = future.eligible.to_numpy(dtype=bool)
    predictions.loc[~eligible] = 0.0
    tiers.loc[~eligible] = "ineligible_zero"
    if not np.isfinite(predictions).all() or predictions.lt(0).any():
        raise ValueError("予測に欠損・無限大・負の値があります")
    return predictions.to_numpy(), {str(k): int(v) for k, v in tiers.value_counts().items()}
