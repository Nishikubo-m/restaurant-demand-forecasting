"""6集計層のローカル検証。2025年の提出用重みとは分けて扱う。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import KEY_COLS, TARGET_COL

LAYERS = {
    "date_store_item": ["date", "store_id", "item_id"],
    "daily_store": ["date", "store_id"],
    "daily_item": ["date", "item_id"],
    "monthly_store_item": ["month", "store_id", "item_id"],
    "monthly_store": ["month", "store_id"],
    "monthly_item": ["month", "item_id"],
}


def _check_keys(frame: pd.DataFrame, name: str) -> None:
    missing = set(KEY_COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}にキー列がありません: {sorted(missing)}")
    if frame.empty or frame[KEY_COLS].isna().any().any() or frame.duplicated(KEY_COLS).any():
        raise ValueError(f"{name}が空、または予測キーに欠損・重複があります")
    if not pd.api.types.is_datetime64_any_dtype(frame.date):
        raise ValueError(f"{name}のdateはdatetime型が必要です")


def layer_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """日次行を各層に集計した後にRMSEとバイアスを計算する。"""
    _check_keys(frame, "評価データ")
    for col in ("actual", "prediction"):
        if col not in frame or not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f"数値列 {col} が必要です")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all() or frame[col].lt(0).any():
            raise ValueError(f"{col}に欠損・無限大・負の値があります")
    data = frame.copy()
    data["month"] = data.date.dt.to_period("M")
    metrics = {}
    for name, cols in LAYERS.items():
        sums = data.groupby(cols, observed=True, sort=False)[["actual", "prediction"]].sum()
        diff = sums.prediction - sums.actual
        metrics[name] = {"rows": len(sums), "rmse": float(np.sqrt(np.mean(diff**2))), "bias": float(diff.mean())}
    return metrics


def validation_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    predictions: np.ndarray,
    *,
    reference_year: int = 2023,
) -> dict:
    """前年平均→基準年の誤差で尺度を作り、その後の期間を評価する。

    デフォルトでは2022年平均で2023年を予測した6層の誤差を尺度とし、
    2024年以降の検証データに適用する。2025年の公式提出重みは使わない。
    predictionsはvalidと同一順の1次元配列。
    """
    _check_keys(train, "尺度計算用データ")
    _check_keys(valid, "検証データ")
    for name, frame in (("尺度計算用データ", train), ("検証データ", valid)):
        if TARGET_COL not in frame or not pd.api.types.is_numeric_dtype(frame[TARGET_COL]):
            raise ValueError(f"{name}には数値の {TARGET_COL} が必要です")
        values = frame[TARGET_COL].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name}の売上に欠損・無限大・負の値があります")
    if valid.date.min() < pd.Timestamp(reference_year + 1, 1, 1):
        raise ValueError("尺度の基準年と検証期間が重複しています")
    pred = np.asarray(predictions)
    if pred.ndim != 1 or len(pred) != len(valid):
        raise ValueError("予測は検証データと同じ長さの1次元配列が必要です")
    hist = train[train.date.dt.year.eq(reference_year - 1)]
    ref = train[train.date.dt.year.eq(reference_year)].copy()
    if hist.empty or ref.empty:
        raise ValueError("尺度計算に基準年とその前年のデータが必要です")
    averages = hist.groupby(["store_id", "item_id"], observed=True)[TARGET_COL].mean().rename("prediction").reset_index()
    ref = ref.merge(averages, on=["store_id", "item_id"], how="left", validate="many_to_one", sort=False)
    if ref.prediction.isna().any():
        raise ValueError("尺度計算用の前年の店舗・商品平均が不足しています")
    reference = layer_metrics(ref.rename(columns={TARGET_COL: "actual"}))
    scored = valid[KEY_COLS + [TARGET_COL]].rename(columns={TARGET_COL: "actual"}).copy()
    scored["prediction"] = pred
    metrics = layer_metrics(scored)
    total_rows = sum(m["rows"] for m in metrics.values())
    for name, m in metrics.items():
        sigma = reference[name]["rmse"]
        if sigma <= 0 or not np.isfinite(sigma):
            raise ValueError(f"尺度が不正です: {name}")
        m["sigma"] = sigma
        m["weight"] = float(np.sqrt(total_rows / (len(LAYERS) * m["rows"])) / sigma)
        m["normalized_rmse"] = m["rmse"] / sigma
    score = float(np.sqrt(np.mean([m["normalized_rmse"] ** 2 for m in metrics.values()])))
    return {"composite_normalized_rmse": score, "validation_daily_row_rmse": metrics["date_store_item"]["rmse"],
            "aggregated_rows": total_rows, "reference_year": reference_year, "layers": metrics}
