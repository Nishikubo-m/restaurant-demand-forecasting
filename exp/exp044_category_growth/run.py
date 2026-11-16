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
    # 月・曜日ごとの前年比を幾何平均し、対数変化の半分を反映する。
    store_factor = np.exp(np.log(ratio).groupby(level="store_id").mean() * 0.5)
    vals = future.store_id.map(store_factor)
    details["matched_growth_unavailable"] = int(vals.isna().sum())
    details["matched_growth_groups"] = int(len(ratio))
    pred = pred * vals.fillna(1.0).to_numpy(dtype=float)
    # 店舗別とカテゴリ別の前年同条件比を等重みで混ぜる。
    days = core.groupby(["date", "store_id", "category", "layout_version"], observed=True, as_index=False).units_proxy.sum()
    days["year"] = days.date.dt.year
    days = days[days.year.isin([year - 1, year])].copy()
    days["month"] = days.date.dt.month
    days["weekday"] = days.date.dt.dayofweek
    cat_keys = ["store_id", "category", "layout_version", "month", "weekday"]
    grouped = days.groupby(cat_keys + ["year"], observed=True).units_proxy.agg(["mean", "size"])
    cat_mean, cat_count = grouped["mean"].unstack("year"), grouped["size"].unstack("year")
    ratio = (cat_mean[year] / cat_mean[year - 1]).where((cat_count[year] >= 2) & (cat_count[year - 1] >= 2))
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    factors = np.exp(np.log(ratio).groupby(level=["store_id", "category"]).mean() * 0.5).rename("factor").reset_index()
    cat_factor = future[["store_id", "category"]].reset_index(drop=True).merge(factors, on=["store_id", "category"], how="left", validate="many_to_one", sort=False).factor
    original = vals.reset_index(drop=True)
    adjustment = (cat_factor / original).replace([np.inf, -np.inf], np.nan)
    adjustment = adjustment.where(adjustment > 0)
    pred = pred * (0.5 + 0.5 * adjustment.fillna(1.0).to_numpy(dtype=float))
    details["category_growth_unavailable"] = int(adjustment.isna().sum())
    return pred, details


def main():
    run("exp044_category_growth", predict, __file__, prepare=make_features(),
        params={"change": "blend category and store matched growth"})


if __name__ == "__main__":
    main()
