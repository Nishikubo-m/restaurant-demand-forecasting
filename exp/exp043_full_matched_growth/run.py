"""同一席構成・月・曜日で比較できる履歴の年次変化を使う。"""
import numpy as np
import pandas as pd

from src.forecast import make_features, predict as base_predict
from src.runner import run


def predict(history, future):
    pred, details = base_predict(history, future)
    year = history.date.max().year
    core = history[history.menu_type.eq("core") & history.eligible].copy()
    core["units_proxy"] = core.sales_amount / core.selling_price
    days = core.groupby(["date", "store_id", "layout_version"], observed=True, as_index=False).units_proxy.sum()
    days["year"] = days.date.dt.year
    days = days[days.year.isin([year - 1, year])].copy()
    days["month"] = days.date.dt.month
    days["weekday"] = days.date.dt.dayofweek
    keys = ["store_id", "layout_version", "month", "weekday"]
    means = days.groupby(keys + ["year"], observed=True).units_proxy.agg(["mean", "size"])
    mean = means["mean"].unstack("year")
    count = means["size"].unstack("year")
    if not {year - 1, year}.issubset(mean.columns):
        raise ValueError("同一条件の年次比較に2年の履歴が必要です")
    ratio = (mean[year] / mean[year - 1]).where((count[year] >= 2) & (count[year - 1] >= 2))
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    # 月・曜日ごとの前年比を幾何平均し、対数変化の全量を反映する。
    store_factor = np.exp(np.log(ratio).groupby(level="store_id").mean())
    vals = future.store_id.map(store_factor)
    details["matched_growth_unavailable"] = int(vals.isna().sum())
    details["matched_growth_groups"] = int(len(ratio))
    pred = pred * vals.fillna(1.0).to_numpy(dtype=float)
    return pred, details


def main():
    run("exp043_full_matched_growth", predict, __file__, prepare=make_features(),
        params={"change": "full year growth matched by store-layout-month-weekday"})


if __name__ == "__main__":
    main()
