"""exp006に商品・月別の過去の季節倍率を追加する。"""
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
    pred, details = predict_available_means(weighted, future)
    pred = pred * future.selling_price.to_numpy(dtype=float)
    # 過去の販売可能日の数量相当値から、商品別の月平均 / 通年平均を作る。
    active = weighted[weighted.eligible].copy()
    active["month"] = active.date.dt.month
    item_mean = active.groupby("item_id", observed=True).sales_amount.mean().rename("item_mean")
    month_mean = active.groupby(["item_id", "month"], observed=True).sales_amount.mean().rename("month_mean")
    rates_month = month_mean.reset_index().merge(item_mean.reset_index(), on="item_id", validate="many_to_one")
    rates_month["month_factor"] = (rates_month.month_mean / rates_month.item_mean).replace([np.inf, -np.inf], np.nan)
    future_month = future[["item_id"]].assign(month=future.date.dt.month.to_numpy())
    month_factor = future_month.merge(rates_month[["item_id", "month", "month_factor"]], on=["item_id", "month"], how="left", validate="many_to_one", sort=False).month_factor
    details["month_factor_unavailable_eligible"] = int((future.eligible.to_numpy(dtype=bool) & month_factor.isna().to_numpy()).sum())
    pred = pred * month_factor.fillna(1).to_numpy(dtype=float)
    daily = history.groupby(["date", "store_id", "is_holiday"], observed=True, as_index=False).sales_amount.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    repetitions = np.where(daily.date.dt.year.eq(latest_year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), repetitions)]
    keys = ["store_id", "weekday"]
    overall = daily.groupby(keys, observed=True).sales_amount.mean().rename("overall")
    holidays = daily[daily.is_holiday].groupby(keys, observed=True).sales_amount.mean().rename("holiday")
    rates = pd.concat([overall, holidays], axis=1)
    rates["factor"] = rates.holiday / rates.overall
    factors = future[keys].merge(rates[["factor"]].reset_index(), on=keys, how="left", validate="many_to_one", sort=False)["factor"]
    use = future.is_holiday.to_numpy(dtype=bool)
    missing = use & (~np.isfinite(factors.to_numpy(dtype=float)))
    pred *= np.where(use, factors.fillna(1).to_numpy(dtype=float), 1.0)
    details.update(holiday_rows=int(use.sum()), holiday_factor_unavailable=int(missing.sum()))
    return pred, details


def main():
    tables = load_availability()
    stores = pd.read_csv(INPUT_DIR / "stores.csv").store_id.tolist()
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
        result = (add_availability(frame, tables).merge(calendar, on="date", how="left", validate="many_to_one", sort=False)
                  .merge(price, on=["date", "store_id", "item_id"], how="left", validate="many_to_one", sort=False))
        if result.is_holiday.isna().any() or result.selling_price.isna().any():
            raise ValueError("対象日の祝日または価格情報がありません")
        return result

    run("exp007_month_factor", predict, __file__, prepare=prepare,
        params={"latest_year_weight": LATEST_WEIGHT, "change": "product month / annual quantity-equivalent mean factor"})


if __name__ == "__main__":
    main()
