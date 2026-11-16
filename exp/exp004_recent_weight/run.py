"""直近年の観測に3倍の重みを付けた販売可能日平均。"""
import numpy as np

from src.availability import add_availability, load_availability
from src.baselines import predict_available_means
from src.runner import run

LATEST_WEIGHT = 3


def predict(history, future):
    # 行の複製は整数重みによる加重平均と等価。モデル・補完順はexp003と同じ。
    if history.empty or future.empty or history.date.max() >= future.date.min():
        raise ValueError("履歴と予測期間が時系列順に分かれていません")
    latest_year = history.date.max().year
    weights = np.where(history.date.dt.year.eq(latest_year), LATEST_WEIGHT, 1)
    weighted = history.iloc[np.repeat(np.arange(len(history)), weights)].reset_index(drop=True)
    return predict_available_means(weighted, future)


def main():
    tables = load_availability()
    run("exp004_recent_weight", predict, __file__, prepare=lambda frame: add_availability(frame, tables),
        params={"change": "latest historical calendar year receives triple weight", "latest_year_weight": LATEST_WEIGHT})


if __name__ == "__main__":
    main()
