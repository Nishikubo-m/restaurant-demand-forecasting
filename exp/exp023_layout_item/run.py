"""価格上昇時に数量が減る仮説を、平均予測への単一倍率で検証する。"""
import numpy as np
import pandas as pd

from src.config import INPUT_DIR

from src.adjusted_baseline import LATEST_WEIGHT, make_prepare, predict as base_predict
from src.runner import run


def predict(history, future):
    pred, details = base_predict(history, future)
    # 直近年の比重を既存の平均と合わせ、販売可能な履歴の価格を基準にする。
    weights = np.where(history.date.dt.year.eq(history.date.max().year), LATEST_WEIGHT, 1)
    weighted = history.iloc[np.repeat(np.arange(len(history)), weights)]
    active = weighted[weighted.eligible]
    keys = ["store_id", "item_id"]
    price = active.groupby(keys, observed=True).selling_price.mean()
    by_item = active.groupby("item_id", observed=True).selling_price.mean()
    reference = future[keys].merge(price.rename("reference_price").reset_index(), on=keys,
                                   how="left", validate="many_to_one", sort=False).reference_price
    reference = reference.fillna(future.item_id.map(by_item).reset_index(drop=True))
    reference = reference.fillna(float(active.selling_price.mean()))
    pred = pred * np.sqrt(reference.to_numpy(dtype=float) / future.selling_price.to_numpy(dtype=float))
    # 条件が一致する日・店舗・カテゴリの平均を、同曜日の平均と比較する。
    daily = history.copy()
    daily["units_proxy"] = daily.sales_amount / daily.selling_price
    daily = daily.groupby(["date", "store_id", "category", "coupon"], observed=True, as_index=False).units_proxy.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    repetitions = np.where(daily.date.dt.year.eq(history.date.max().year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), repetitions)]
    keys = ["store_id", "category", "weekday"]
    base = daily.groupby(keys, observed=True).units_proxy.mean().rename("base")
    grouped = daily.groupby(keys + ["coupon"], observed=True).units_proxy.agg(["mean", "size"])
    rates = grouped.join(base)
    rates["factor"] = (rates["mean"] / rates["base"]).where(rates["size"] >= 5)
    future_keys = future[keys + ["coupon"]].reset_index(drop=True)
    factors = future_keys.merge(rates[["factor"]].reset_index(), on=keys + ["coupon"],
                                how="left", validate="many_to_one", sort=False).factor
    values = factors.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    pred *= np.where(missing, 1.0, values)
    details["coupon_factor_unavailable"] = int(missing.sum())
    # 過去の雨・非雨の条件付き平均を、将来の降水確率平年値で混合する。
    daily = history.copy()
    daily["units_proxy"] = daily.sales_amount / daily.selling_price
    daily = daily.groupby(["date", "store_id", "category", "weather_observed"], observed=True, as_index=False).units_proxy.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    daily["rain"] = daily.weather_observed.eq("rainy")
    weights = np.where(daily.date.dt.year.eq(history.date.max().year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), weights)]
    keys = ["store_id", "category", "weekday"]
    overall = daily.groupby(keys, observed=True).units_proxy.mean().rename("overall")
    rain = daily.groupby(keys + ["rain"], observed=True).units_proxy.agg(["mean", "size"])
    means = rain["mean"].unstack("rain").rename(columns={False: "dry", True: "wet"})
    counts = rain["size"].unstack("rain").rename(columns={False: "dry_count", True: "wet_count"})
    rates = means.join(counts).join(overall).reset_index()
    vals = future[keys + ["wet_probability_normal"]].reset_index(drop=True).merge(rates, on=keys, how="left", validate="many_to_one", sort=False)
    expected = (vals.wet_probability_normal * vals.wet + (1 - vals.wet_probability_normal) * vals.dry) / vals.overall
    factors = expected.where((vals.wet_count >= 5) & (vals.dry_count >= 5)).to_numpy(dtype=float)
    missing = ~np.isfinite(factors)
    pred = pred * np.where(missing, 1.0, factors)
    details["rain_factor_unavailable"] = int(missing.sum())
    # 最新レイアウトの数量相当水準を店舗・カテゴリ・曜日で求める。
    core = history[history.menu_type.eq("core")].copy()
    core["units"] = core.sales_amount / core.selling_price
    store_daily = core.groupby(["date", "store_id"], observed=True, as_index=False).units.sum()
    store_daily["year"] = store_daily.date.dt.year
    yearly = store_daily.groupby(["store_id", "year"], observed=True).units.mean().unstack("year")
    year = history.date.max().year
    growth = (yearly[year] / yearly[year - 1]).replace([np.inf, -np.inf], np.nan)
    daily = core.groupby(["date", "store_id", "category", "layout_version"], observed=True, as_index=False).units.sum()
    daily["year"] = daily.date.dt.year
    daily["weekday"] = daily.date.dt.dayofweek
    latest = daily.sort_values("date").groupby("store_id", observed=True).layout_version.last().rename("latest_layout")
    daily = daily.join(latest, on="store_id")
    repetitions = np.where(daily.year.eq(year), LATEST_WEIGHT, 1)
    weighted_daily = daily.iloc[np.repeat(np.arange(len(daily)), repetitions)]
    keys = ["store_id", "category", "weekday"]
    overall = weighted_daily.groupby(keys, observed=True).units.mean().rename("overall")
    current = weighted_daily[weighted_daily.layout_version.eq(weighted_daily.latest_layout)]
    current = current.groupby(keys, observed=True).units.agg(["mean", "size"])
    rates = current.join(overall)
    rates["factor"] = (rates["mean"] / rates.overall).where(rates["size"] >= 12)
    values = future[keys].reset_index(drop=True).merge(rates[["factor"]].reset_index(), on=keys, how="left", validate="many_to_one", sort=False).factor
    annual = future.store_id.map(growth).reset_index(drop=True)
    correction = (values / annual).replace([np.inf, -np.inf], np.nan)
    pred = pred * correction.fillna(1.0).to_numpy(dtype=float)
    details["layout_level_unavailable"] = int(correction.isna().sum())
    # 商品別の変化が観測できる場合はカテゴリ倍率を商品倍率へ置き換える。
    active = history[history.eligible].copy()
    active["units"] = active.sales_amount / active.selling_price
    active = active.join(latest, on="store_id")
    repetitions = np.where(active.date.dt.year.eq(year), LATEST_WEIGHT, 1)
    active = active.iloc[np.repeat(np.arange(len(active)), repetitions)]
    item_keys = ["store_id", "item_id", "weekday"]
    item_overall = active.groupby(item_keys, observed=True).units.mean().rename("overall")
    current_item = active[active.layout_version.eq(active.latest_layout)]
    item_current = current_item.groupby(item_keys, observed=True).units.agg(["mean", "size"])
    item_rates = item_current.join(item_overall)
    item_rates["factor"] = (item_rates["mean"] / item_rates.overall).where(item_rates["size"] >= 12)
    item_factor = future[item_keys].reset_index(drop=True).merge(item_rates[["factor"]].reset_index(), on=item_keys, how="left", validate="many_to_one", sort=False).factor
    # values はカテゴリ倍率。欠損時は既存の補正を維持する。
    item_correction = (item_factor / values).replace([np.inf, -np.inf], np.nan)
    pred = pred * item_correction.fillna(1.0).to_numpy(dtype=float)
    details["layout_item_unavailable"] = int(item_correction.isna().sum())
    return pred, details


def main():
    common_prepare = make_prepare()
    stores = pd.read_csv(INPUT_DIR / "stores.csv").store_id.tolist()
    campaign = pd.read_csv(INPUT_DIR / "campaign_calendar.csv", parse_dates=["date"])
    campaign = campaign[campaign.coupon_flag].copy()
    campaign["store_id"] = campaign.store_ids.map(lambda x: stores if x == "all" else str(x).split("|"))
    campaign = campaign.explode("store_id", ignore_index=True)
    coupon = campaign[["date", "store_id", "target_item_group"]].drop_duplicates().rename(columns={"target_item_group": "category"})
    coupon["coupon"] = True
    weather = pd.read_csv(INPUT_DIR / "calendar.csv", usecols=["date", "weather_observed", "wet_probability_normal"], parse_dates=["date"])
    layouts = pd.read_csv(INPUT_DIR / "seat_layout.csv", parse_dates=["effective_start_date", "effective_end_date"])
    layouts["date"] = [pd.date_range(a, b) for a, b in zip(layouts.effective_start_date, layouts.effective_end_date)]
    layouts = layouts.explode("date", ignore_index=True)[["date", "store_id", "layout_version"]]
    if layouts.duplicated(["date", "store_id"]).any():
        raise ValueError("店舗・日のレイアウトが重複")

    def prepare(frame):
        result = common_prepare(frame).merge(coupon, on=["date", "store_id", "category"],
                                             how="left", validate="many_to_one", sort=False)
        result["coupon"] = result.coupon.fillna(False).astype(bool)
        columns = ["date", "wet_probability_normal"]
        if "sales_amount" in frame:
            columns.append("weather_observed")
        result = result.merge(weather[columns], on="date", how="left", validate="many_to_one", sort=False)
        if "sales_amount" in frame:
            result = result.merge(layouts, on=["date", "store_id"], how="left", validate="many_to_one", sort=False)
            if result.layout_version.isna().any():
                raise ValueError("履歴のレイアウトがありません")
        if result.wet_probability_normal.isna().any() or not result.wet_probability_normal.between(0, 1).all():
            raise ValueError("降水確率平年値がありません")
        if "sales_amount" in frame and result.weather_observed.isna().any():
            raise ValueError("履歴の天候実績がありません")
        return result

    run("exp023_layout_item", predict, __file__, prepare=prepare,
        params={"latest_year_weight": LATEST_WEIGHT, "price_exponent": 0.5,
                "change": "replace category layout factor with item layout factor when available"})


if __name__ == "__main__":
    main()
