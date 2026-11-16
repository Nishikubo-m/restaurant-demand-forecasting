"""同一席構成・月・曜日で比較できる履歴の年次変化を使う。"""
import numpy as np
import pandas as pd

from src.forecast import make_features, predict as base_predict
from src.runner import run


def predict(history, future):
    pred, details = base_predict(history, future)
    year = history.date.max().year
    core = history[history.menu_type.eq("core") & history.eligible & ~history.is_holiday].copy()
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
    # 直近四半期で同じ条件がある店舗は、その期間の前年比を優先する。
    cutoff_month = history.date.max().to_period("M")
    recent_months = pd.period_range(end=cutoff_month, periods=3, freq="M")
    recent = ratio[ratio.index.get_level_values("month").isin(recent_months.month)]
    recent_factor = np.exp(np.log(recent).groupby(level="store_id").mean() * 0.5)
    factor = recent_factor.combine_first(store_factor)
    # 非祝日・同一レイアウトの364日前を対応付け、直近四半期を比較する。
    pairs = days[["date", "store_id", "layout_version", "units_proxy"]].copy()
    pairs["date"] = pairs.date + pd.Timedelta(days=364)
    pairs = pairs.rename(columns={"units_proxy": "prior_units"})
    recent_days = days[days.date.dt.to_period("M").isin(recent_months)]
    joined = recent_days.merge(pairs, on=["date", "store_id", "layout_version"], how="inner", validate="one_to_one")
    paired = joined.groupby("store_id", observed=True).agg(current=("units_proxy", "sum"), prior=("prior_units", "sum"), size=("date", "size"))
    paired_factor = (paired.current / paired.prior).where((paired["size"] >= 14) & (paired.prior > 0))
    paired_factor = np.sqrt(paired_factor.replace([np.inf, -np.inf], np.nan)).dropna()
    # 同一レイアウトで不足した店舗は、過去のレイアウト別水準を除いたペアで補完する。
    adjusted = days.copy()
    level = adjusted.groupby(["store_id", "layout_version", "weekday"], observed=True).units_proxy.transform("mean")
    adjusted["relative"] = adjusted.units_proxy / level
    prior = adjusted[["date", "store_id", "relative"]].rename(columns={"relative": "prior_relative"}).copy()
    prior["date"] = prior.date + pd.Timedelta(days=364)
    current = adjusted[adjusted.date.dt.to_period("M").isin(recent_months)]
    crossed = current.merge(prior, on=["date", "store_id"], how="inner", validate="one_to_one")
    grouped = crossed.groupby("store_id", observed=True).agg(current=("relative", "sum"), prior=("prior_relative", "sum"), size=("date", "size"))
    cross_factor = np.sqrt((grouped.current / grouped.prior).where((grouped["size"] >= 14) & (grouped.prior > 0)))
    cross_factor = cross_factor.replace([np.inf, -np.inf], np.nan).dropna()
    cross_factor = cross_factor.drop(index=paired_factor.index, errors="ignore")
    factor = paired_factor.combine_first(cross_factor).combine_first(factor)
    details["cross_layout_growth_stores"] = int(len(cross_factor))
    details["paired_growth_stores"] = int(len(paired_factor))
    vals = future.store_id.map(factor)
    details["recent_growth_stores"] = int(recent_factor.size)
    details["matched_growth_unavailable"] = int(vals.isna().sum())
    details["matched_growth_groups"] = int(len(ratio))
    # 年央の半分という固定倍率に代え、予測開始後の経過時間で外挿量を増やす。
    elapsed = (future.date - history.date.max()).dt.days.to_numpy(dtype=float)
    fraction = np.clip(elapsed / 365.0, 0.0, 1.0)
    pred = pred * np.power(vals.fillna(1.0).to_numpy(dtype=float), 2.0 * fraction)
    return pred, details


def main():
    run("exp056_cross_layout_growth", predict, __file__, prepare=make_features(),
        params={"change": "fallback to layout-normalized cross-layout paired growth"})


if __name__ == "__main__":
    main()
