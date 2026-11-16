"""exp004に、店舗・曜日別の過去の祝日倍率を一つ追加する。"""
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
    weighted = history.iloc[np.repeat(np.arange(len(history)), weights)].reset_index(drop=True)
    pred, details = predict_available_means(weighted, future)
    # 日次店舗売上を先に集約してから日単位で重み付けする。
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
    factor = factors.fillna(1).to_numpy(dtype=float)
    pred = pred * np.where(use, factor, 1.0)
    details.update(holiday_rows=int(use.sum()), holiday_factor_unavailable=int(missing.sum()))
    return pred, details


def main():
    tables = load_availability()
    calendar = pd.read_csv(INPUT_DIR / "calendar.csv", usecols=["date", "is_holiday"], parse_dates=["date"])
    if calendar.date.duplicated().any() or calendar.is_holiday.isna().any():
        raise ValueError("祝日カレンダーの日付重複または欠損")

    def prepare(frame):
        result = add_availability(frame, tables).merge(calendar, on="date", how="left", validate="many_to_one", sort=False)
        if result.is_holiday.isna().any():
            raise ValueError("対象日の祝日情報がありません")
        return result

    run("exp005_holiday_factor", predict, __file__, prepare=prepare,
        params={"latest_year_weight": LATEST_WEIGHT, "change": "historical store-weekday holiday factor"})


if __name__ == "__main__":
    main()
