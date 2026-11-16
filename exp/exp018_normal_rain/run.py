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

    def prepare(frame):
        result = common_prepare(frame).merge(coupon, on=["date", "store_id", "category"],
                                             how="left", validate="many_to_one", sort=False)
        result["coupon"] = result.coupon.fillna(False).astype(bool)
        columns = ["date", "wet_probability_normal"]
        if "sales_amount" in frame:
            columns.append("weather_observed")
        result = result.merge(weather[columns], on="date", how="left", validate="many_to_one", sort=False)
        if result.wet_probability_normal.isna().any() or not result.wet_probability_normal.between(0, 1).all():
            raise ValueError("降水確率平年値がありません")
        if "sales_amount" in frame and result.weather_observed.isna().any():
            raise ValueError("履歴の天候実績がありません")
        return result

    run("exp018_normal_rain", predict, __file__, prepare=prepare,
        params={"latest_year_weight": LATEST_WEIGHT, "price_exponent": 0.5,
                "change": "mix historical rain and no-rain means using forecast wet probability normal"})


if __name__ == "__main__":
    main()
