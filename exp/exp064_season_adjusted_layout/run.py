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
    factor = paired_factor.combine_first(factor)
    details["paired_growth_stores"] = int(len(paired_factor))
    vals = future.store_id.map(factor)
    details["recent_growth_stores"] = int(recent_factor.size)
    details["matched_growth_unavailable"] = int(vals.isna().sum())
    details["matched_growth_groups"] = int(len(ratio))
    # 年央の半分という固定倍率に代え、予測開始後の経過時間で外挿量を増やす。
    elapsed = (future.date - history.date.max()).dt.days.to_numpy(dtype=float)
    fraction = np.clip(elapsed / 365.0, 0.0, 1.0)
    pred = pred * np.power(vals.fillna(1.0).to_numpy(dtype=float), 2.0 * fraction)
    # 直近年と前年のペアから、3:1の重み付き履歴平均を直近水準へ補正する。
    # 古い各年が前年と同じ水準という近似を置き、過去3年なら3:1:1を使う。
    older_years = int(history.date.dt.year.nunique() - 1)
    reference_ratio = factor.pow(2)
    level_factor = ((older_years + 3) * reference_ratio / (older_years + 3 * reference_ratio)).replace([np.inf, -np.inf], np.nan)
    level = future.store_id.map(level_factor)
    pred = pred * level.fillna(1.0).to_numpy(dtype=float)
    details["recent_level_unavailable"] = int(level.isna().sum())
    # 基礎モデルの最新レイアウト倍率を、月の季節変動を除いた倍率に置き換える。
    core = history[history.menu_type.eq("core") & history.eligible].copy()
    core["units"] = core.sales_amount / core.selling_price
    daily = core.groupby(["date", "store_id", "layout_version"], observed=True, as_index=False).units.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    daily["month"] = daily.date.dt.month
    repeat = np.where(daily.date.dt.year.eq(year), 3, 1)
    weighted = daily.iloc[np.repeat(np.arange(len(daily)), repeat)].copy()
    layout_mean = weighted.groupby(["store_id", "layout_version", "weekday"], observed=True).units.transform("mean")
    weighted["relative"] = weighted.units / layout_mean
    seasonal = weighted.groupby(["store_id", "month"], observed=True).relative.mean().rename("season").reset_index()
    cats = core.groupby(["date", "store_id", "category", "layout_version"], observed=True, as_index=False).units.sum()
    cats["weekday"] = cats.date.dt.dayofweek
    cats["month"] = cats.date.dt.month
    cats = cats.merge(seasonal, on=["store_id", "month"], how="left", validate="many_to_one", sort=False)
    cats["adjusted"] = cats.units / cats.season
    latest = cats.sort_values("date").groupby("store_id", observed=True).layout_version.last().rename("latest")
    cats = cats.join(latest, on="store_id")
    repeat = np.where(cats.date.dt.year.eq(year), 3, 1)
    cats = cats.iloc[np.repeat(np.arange(len(cats)), repeat)]
    keys = ["store_id", "category", "weekday"]
    overall = cats.groupby(keys, observed=True)[["units", "adjusted"]].mean().rename(columns={"units": "old_all", "adjusted": "new_all"})
    current = cats[cats.layout_version.eq(cats.latest)].groupby(keys, observed=True).agg(old_current=("units", "mean"), new_current=("adjusted", "mean"), size=("units", "size"))
    rates = current.join(overall)
    rates["correction"] = ((rates.new_current / rates.new_all) / (rates.old_current / rates.old_all)).where(rates["size"] >= 12)
    correction = future[keys].reset_index(drop=True).merge(rates[["correction"]].reset_index(), on=keys, how="left", validate="many_to_one", sort=False).correction
    correction = correction.where(np.isfinite(correction) & correction.gt(0))
    pred = pred * correction.fillna(1.0).to_numpy(dtype=float)
    details["season_adjusted_layout_unavailable"] = int(correction.isna().sum())
    return pred, details


def main():
    run("exp064_season_adjusted_layout", predict, __file__, prepare=make_features(),
        params={"change": "remove estimated month seasonality from latest-layout level"})


if __name__ == "__main__":
    main()
