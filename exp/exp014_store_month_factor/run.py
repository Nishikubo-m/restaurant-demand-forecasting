"""店舗の季節性を過去の定番商品平均から補正する。"""
import numpy as np

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
    # 定番商品の販売数量相当値を曜日平均との差で標準化し、月別平均を計算する。
    # 重複された日付を先に集約すると重みが消えるため、集約後に年の重みを適用する。
    core = history[history.eligible & history.menu_type.eq("core")].copy()
    core["units"] = core.sales_amount / core.selling_price
    daily = core.groupby(["date", "store_id"], observed=True, as_index=False).units.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    daily["month"] = daily.date.dt.month
    daily_weights = np.where(daily.date.dt.year.eq(history.date.max().year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), daily_weights)].copy()
    weekday_mean = daily.groupby(["store_id", "weekday"], observed=True).units.transform("mean")
    daily["relative"] = daily.units / weekday_mean
    monthly = daily.groupby(["store_id", "month"], observed=True).relative.mean().rename("factor").reset_index()
    target = future[["store_id"]].copy().reset_index(drop=True)
    target["month"] = future.date.dt.month.to_numpy()
    factors = target.merge(monthly, on=["store_id", "month"], how="left", validate="many_to_one", sort=False).factor
    pred = pred * factors.fillna(1.0).to_numpy(dtype=float)
    details["store_month_factor_unavailable"] = int(factors.isna().sum())
    return pred, details


def main():
    run("exp014_store_month_factor", predict, __file__, prepare=make_prepare(),
        params={"latest_year_weight": LATEST_WEIGHT, "price_exponent": 0.5,
                "change": "multiply by store-month relative mean of weekday-adjusted core units"})


if __name__ == "__main__":
    main()
