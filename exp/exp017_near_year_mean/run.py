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
    # 前年の近傍5週間の同曜日を平均し、同じ年の曜日平均で割る。
    recent_year = history.date.max().year
    core = history[history.menu_type.eq("core") & history.date.dt.year.eq(recent_year)].copy()
    core["units"] = core.sales_amount / core.selling_price
    daily = core.groupby(["date", "store_id", "is_holiday"], observed=True, as_index=False).units.sum()
    daily = daily[~daily.is_holiday].copy()
    daily["weekday"] = daily.date.dt.dayofweek
    baseline = daily.groupby(["store_id", "weekday"], observed=True).units.mean().rename("baseline").reset_index()
    daily = daily.merge(baseline, on=["store_id", "weekday"], validate="many_to_one")
    daily["relative"] = daily.units / daily.baseline
    source = daily[["date", "store_id", "relative"]]
    keys = future[["date", "store_id"]].drop_duplicates().reset_index(drop=True)
    matches = []
    for lag in [350, 357, 364, 371, 378]:
        lagged = keys.copy()
        lagged["date"] = lagged.date - pd.Timedelta(days=lag)
        matched = lagged.merge(source, on=["date", "store_id"], how="left", validate="many_to_one", sort=False)
        matches.append(matched.relative.to_numpy(dtype=float))
    arr = np.stack(matches, axis=1)
    available = np.isfinite(arr).sum(axis=1)
    local_mean = np.divide(np.nansum(arr, axis=1), available, out=np.ones(len(keys)), where=available >= 2)
    keys["factor"] = 0.5 + 0.5 * local_mean
    factors = future[["date", "store_id"]].merge(keys, on=["date", "store_id"], how="left", validate="many_to_one", sort=False).factor
    pred = pred * factors.to_numpy(dtype=float)
    details["lag_factor_unavailable_days_stores"] = int((available < 2).sum())
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

    def prepare(frame):
        result = common_prepare(frame).merge(coupon, on=["date", "store_id", "category"],
                                             how="left", validate="many_to_one", sort=False)
        result["coupon"] = result.coupon.fillna(False).astype(bool)
        return result

    run("exp017_near_year_mean", predict, __file__, prepare=prepare,
        params={"latest_year_weight": LATEST_WEIGHT, "price_exponent": 0.5,
                "change": "blend smoothed year-ago weekday core totals with baseline"})


if __name__ == "__main__":
    main()
