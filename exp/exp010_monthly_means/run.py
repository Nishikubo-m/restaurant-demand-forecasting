"""exp009の平均を月別に細分化する。"""
import numpy as np
import pandas as pd

from src.availability import add_availability, load_availability
from src.baselines import predict_available_means
from src.config import INPUT_DIR
from src.runner import run

LATEST_WEIGHT = 3


def predict(history, future):
    if history.empty or future.empty or history.date.max() >= future.date.min():
        raise ValueError("履歴と予測期間が時系列順に分かれていません")
    latest_year = history.date.max().year
    weights = np.where(history.date.dt.year.eq(latest_year), LATEST_WEIGHT, 1)
    weighted = history.iloc[np.repeat(np.arange(len(history)), weights)].reset_index(drop=True).copy()
    weighted["sales_amount"] = weighted.sales_amount / weighted.selling_price
    # 同じ平均モデルで、月・曜日に該当する過去日を優先する。
    active = weighted[weighted.eligible].copy()
    active["month"] = active.date.dt.month
    x = future[["store_id", "item_id", "weekday"]].copy().reset_index(drop=True)
    x["month"] = future.date.dt.month.to_numpy()
    levels = [["store_id", "item_id", "month", "weekday"], ["store_id", "item_id", "weekday"],
              ["store_id", "item_id"], ["item_id"]]
    names = ["store_item_month_weekday", "store_item_weekday", "store_item", "item"]
    values = pd.Series(np.nan, index=x.index, dtype=float)
    tiers = pd.Series("overall", index=x.index, dtype="object")
    for cols, name in zip(levels, names):
        means = active.groupby(cols, observed=True).sales_amount.mean().rename("mean").reset_index()
        result = x[cols].merge(means, on=cols, how="left", validate="many_to_one", sort=False)["mean"]
        use_level = values.isna() & result.notna()
        values.loc[use_level] = result.loc[use_level]
        tiers.loc[use_level] = name
    pred = values.fillna(float(active.sales_amount.mean())).to_numpy()
    mask = future.eligible.to_numpy(dtype=bool)
    pred = np.where(mask, pred, 0.0)
    tiers.loc[~mask] = "ineligible_zero"
    details = {str(k): int(v) for k, v in tiers.value_counts().items()}
    pred = pred * future.selling_price.to_numpy(dtype=float)
    daily = history.groupby(["date", "store_id", "category", "is_holiday"], observed=True, as_index=False).sales_amount.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    repetitions = np.where(daily.date.dt.year.eq(latest_year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), repetitions)]
    keys = ["store_id", "category", "weekday"]
    overall = daily.groupby(keys, observed=True).sales_amount.mean().rename("overall")
    holidays = daily[daily.is_holiday].groupby(keys, observed=True).sales_amount.mean().rename("holiday")
    rates = pd.concat([overall, holidays], axis=1)
    rates["factor"] = rates.holiday / rates.overall
    factors = future[keys].merge(rates[["factor"]].reset_index(), on=keys, how="left", validate="many_to_one", sort=False)["factor"]
    use = future.is_holiday.to_numpy(dtype=bool)
    missing = use & (~np.isfinite(factors.to_numpy(dtype=float)))
    pred *= np.where(use, factors.fillna(1).to_numpy(dtype=float), 1.0)
    details.update(holiday_rows=int(use.sum()), holiday_factor_unavailable=int(missing.sum()))
    # 通年販売のcore商品だけを使い、直近2年の数量相当値の店舗別日平均を比較する。
    core = history[history.menu_type.eq("core")].copy()
    core["units_proxy"] = core.sales_amount / core.selling_price
    daily_core = core.groupby(["date", "store_id"], observed=True, as_index=False).units_proxy.sum()
    daily_core["year"] = daily_core.date.dt.year
    yearly = daily_core.groupby(["store_id", "year"], observed=True).units_proxy.mean().unstack("year")
    if latest_year not in yearly.columns or latest_year - 1 not in yearly.columns:
        raise ValueError("年次比較には直近2年の履歴が必要です")
    factor = (yearly[latest_year] / yearly[latest_year - 1]).replace([np.inf, -np.inf], np.nan)
    vals = future.store_id.map(factor)
    details["store_growth_factor_unavailable"] = int(vals.isna().sum())
    pred = pred * vals.fillna(1).to_numpy(dtype=float)
    return pred, details


def main():
    tables = load_availability()
    stores = pd.read_csv(INPUT_DIR / "stores.csv").store_id.tolist()
    categories = pd.read_csv(INPUT_DIR / "menus.csv", usecols=["item_id", "category"])
    calendar = pd.read_csv(INPUT_DIR / "calendar.csv", usecols=["date", "is_holiday"], parse_dates=["date"])
    if calendar.date.duplicated().any() or calendar.is_holiday.isna().any():
        raise ValueError("祝日カレンダーの日付重複または欠損")
    price = pd.read_csv(INPUT_DIR / "price_history.csv", parse_dates=["effective_start_date", "effective_end_date"])
    price["store_id"] = price.store_ids.map(lambda x: stores if x == "all" else str(x).split("|"))
    price = price.explode("store_id", ignore_index=True)
    price["date"] = [pd.date_range(a, b) for a, b in zip(price.effective_start_date, price.effective_end_date)]
    price = price.explode("date", ignore_index=True)[["date", "store_id", "item_id", "selling_price"]]
    if price.duplicated(["date", "store_id", "item_id"]).any() or price.selling_price.isna().any() or price.selling_price.le(0).any():
        raise ValueError("価格の重複・欠損・非正値")

    def prepare(frame):
        result = (add_availability(frame, tables).merge(categories, on="item_id", how="left", validate="many_to_one", sort=False)
                  .merge(calendar, on="date", how="left", validate="many_to_one", sort=False)
                  .merge(price, on=["date", "store_id", "item_id"], how="left", validate="many_to_one", sort=False))
        if result.is_holiday.isna().any() or result.selling_price.isna().any() or result.category.isna().any():
            raise ValueError("対象日の祝日または価格情報がありません")
        return result

    run("exp010_monthly_means", predict, __file__, prepare=prepare,
        params={"latest_year_weight": LATEST_WEIGHT, "change": "prefer store-item-month-weekday mean with original fallback"})


if __name__ == "__main__":
    main()
