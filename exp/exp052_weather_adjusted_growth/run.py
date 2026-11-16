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
    days = core.groupby(["date", "store_id", "layout_version", "weather_observed"], observed=True, as_index=False).units_proxy.sum()
    days["year"] = days.date.dt.year
    days = days[days.year.isin([year - 1, year])].copy()
    days["month"] = days.date.dt.month
    days["weekday"] = days.date.dt.dayofweek
    # 履歴の天候構成差を、同じレイアウト・月・曜日で正規化して軽減する。
    baseline = days.groupby(["store_id", "layout_version", "month", "weekday"], observed=True).units_proxy.transform("mean")
    days["relative"] = days.units_proxy / baseline
    days["rain"] = days.weather_observed.eq("rainy")
    total = days.groupby("store_id", observed=True).relative.mean().rename("overall")
    state = days.groupby(["store_id", "rain"], observed=True).relative.agg(["mean", "size"]).join(total)
    state["factor"] = (state["mean"] / state.overall).where(state["size"] >= 14)
    state_factor = days[["store_id", "rain"]].merge(state[["factor"]].reset_index(), on=["store_id", "rain"], how="left", validate="many_to_one", sort=False).factor.to_numpy(dtype=float)
    state_factor = np.where(np.isfinite(state_factor) & (state_factor > 0), state_factor, 1.0)
    days["units_proxy"] = days.units_proxy.to_numpy(dtype=float) / state_factor
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
    recent = ratio[ratio.index.get_level_values("month").isin([10, 11, 12])]
    recent_factor = np.exp(np.log(recent).groupby(level="store_id").mean() * 0.5)
    factor = recent_factor.combine_first(store_factor)
    # 非祝日・同一レイアウトの364日前を対応付け、直近四半期を比較する。
    pairs = days[["date", "store_id", "layout_version", "units_proxy"]].copy()
    pairs["date"] = pairs.date + pd.Timedelta(days=364)
    pairs = pairs.rename(columns={"units_proxy": "prior_units"})
    recent_days = days[days.year.eq(year) & days.month.isin([10, 11, 12])]
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
    return pred, details


def main():
    run("exp052_weather_adjusted_growth", predict, __file__, prepare=make_features(),
        params={"change": "adjust historical growth for observed weather composition"})


if __name__ == "__main__":
    main()
