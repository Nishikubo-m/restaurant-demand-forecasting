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
    # 年間前年比を、直近四半期の同四半期前年比に置き換える。
    core = history[history.menu_type.eq("core")].copy()
    core["units_proxy"] = core.sales_amount / core.selling_price
    daily = core.groupby(["date", "store_id"], observed=True, as_index=False).units_proxy.sum()
    daily["year"] = daily.date.dt.year
    yearly = daily.groupby(["store_id", "year"], observed=True).units_proxy.mean().unstack("year")
    quarter = daily[daily.date.dt.month.isin([10, 11, 12])]
    quarter = quarter.groupby(["store_id", "year"], observed=True).units_proxy.mean().unstack("year")
    year = history.date.max().year
    old = yearly[year] / yearly[year - 1]
    recent = quarter[year] / quarter[year - 1]
    correction = (recent / old).replace([np.inf, -np.inf], np.nan)
    values = future.store_id.map(correction)
    pred = pred * values.fillna(1.0).to_numpy(dtype=float)
    details["recent_growth_unavailable"] = int(values.isna().sum())
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

    run("exp016_recent_growth", predict, __file__, prepare=prepare,
        params={"latest_year_weight": LATEST_WEIGHT, "price_exponent": 0.5,
                "change": "replace annual store growth with final-quarter year-over-year growth"})


if __name__ == "__main__":
    main()
