"""店舗・商品・曜日ごとの過去平均を用いるベースライン。"""
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RUNS_DIR, SUBMISSION_DIR
from src.data import KEY_COLS, TARGET_COL, load_sales

EXP_NAME = "exp001_baseline"
VALID_START = "2024-01-01"


def predict(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """未観測の商品等は店舗平均、全体平均の順で補完する。"""
    history = history.assign(weekday=history["date"].dt.dayofweek)
    future_keys = future[KEY_COLS].assign(weekday=future["date"].dt.dayofweek)
    means = history.groupby(["store_id", "item_id", "weekday"])[TARGET_COL].mean()
    store_means = history.groupby("store_id")[TARGET_COL].mean()
    predictions = future_keys.join(means, on=["store_id", "item_id", "weekday"])[TARGET_COL]
    predictions = predictions.fillna(future_keys["store_id"].map(store_means)).fillna(history[TARGET_COL].mean())
    return predictions.to_numpy()


def main() -> None:
    train, test = load_sales("train"), load_sales("test")
    history = train[train.date < VALID_START]
    valid = train[train.date >= VALID_START]
    if history.empty or valid.empty or train.date.max() >= test.date.min():
        raise ValueError("学習・検証・テスト期間の時系列を確認してください")
    valid_pred = predict(history, valid)
    rmse = float(np.sqrt(np.mean((valid[TARGET_COL].to_numpy() - valid_pred) ** 2)))
    test_pred = predict(train, test)
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{EXP_NAME}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), run_dir / "code.py")
    (run_dir / "params.json").write_text(json.dumps({"valid_start": VALID_START}, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"validation_daily_row_rmse": rmse}, indent=2), encoding="utf-8")
    valid.assign(prediction=valid_pred).to_csv(run_dir / "validation.csv", index=False)
    test.assign(**{TARGET_COL: test_pred}).to_csv(run_dir / "predictions.csv", index=False)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Validation daily row RMSE (参考値): {rmse:.3f}")
    print(f"Run: {run_dir}")
    print("提出には predictions.csv を公式スクリプトで変換してください。")


if __name__ == "__main__":
    main()
