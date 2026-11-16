"""販売可能な店舗・商品・曜日の過去平均による需要予測。"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INPUT_DIR, RUNS_DIR
from src.data import KEY_COLS, TARGET_COL, load_sales

EXP_NAME = "exp002_availability_weekday"
VALID_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")
LAYERS = {
    "date_store_item": ["date", "store_id", "item_id"],
    "daily_store": ["date", "store_id"],
    "daily_item": ["date", "item_id"],
    "monthly_store_item": ["month", "store_id", "item_id"],
    "monthly_store": ["month", "store_id"],
    "monthly_item": ["month", "item_id"],
}


def load_availability() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """カタログ期間および企画の実施店舗を日単位で展開する。"""
    menus = pd.read_csv(INPUT_DIR / "menus.csv", usecols=["item_id", "menu_type"])
    stores = pd.read_csv(INPUT_DIR / "stores.csv", usecols=["store_id"])["store_id"].tolist()
    if menus.item_id.duplicated().any() or len(stores) != len(set(stores)):
        raise ValueError("店舗または商品マスタのIDが重複しています")
    catalog = pd.read_csv(INPUT_DIR / "menu_catalog_history.csv", parse_dates=["effective_start_date", "effective_end_date"])
    if catalog[["effective_start_date", "effective_end_date"]].isna().any().any() or (catalog.effective_start_date > catalog.effective_end_date).any():
        raise ValueError("カタログの有効期間が不正です")
    catalog["date"] = [pd.date_range(a, b) for a, b in zip(catalog.effective_start_date, catalog.effective_end_date)]
    catalog = catalog.explode("date", ignore_index=True)[["date", "item_id", "campaign_id"]].assign(catalog_active=True)
    if catalog.duplicated(["date", "item_id"]).any():
        raise ValueError("カタログの日付・商品が重複しています")
    campaigns = pd.read_csv(INPUT_DIR / "campaign_calendar.csv", parse_dates=["date"])
    campaigns["store_id"] = campaigns.store_ids.map(lambda s: stores if s == "all" else str(s).split("|"))
    campaigns = campaigns.explode("store_id", ignore_index=True)[["date", "store_id", "campaign_id"]]
    if campaigns.duplicated(["date", "store_id", "campaign_id"]).any() or not campaigns.store_id.isin(stores).all():
        raise ValueError("キャンペーンの店舗・日付・企画が不正です")
    return menus, catalog, campaigns.assign(campaign_active=True)


def add_availability(frame: pd.DataFrame, tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """販売期間とコラボ参加店舗で判定し、入力の行順を維持する。"""
    if frame.duplicated(KEY_COLS).any() or frame[KEY_COLS].isna().any().any():
        raise ValueError("予測キーに重複または欠損があります")
    menus, catalog, campaigns = tables
    x = frame.assign(_row_order=np.arange(len(frame)))
    x = x.merge(menus, on="item_id", how="left", validate="many_to_one", sort=False)
    if x.menu_type.isna().any():
        raise ValueError("商品マスタに存在しない商品があります")
    x = x.merge(catalog, on=["date", "item_id"], how="left", validate="many_to_one", sort=False)
    x = x.merge(campaigns, on=["date", "store_id", "campaign_id"], how="left", validate="many_to_one", sort=False)
    x = x.sort_values("_row_order").reset_index(drop=True)
    x["eligible"] = x.catalog_active.fillna(False).astype(bool) & (
        x.menu_type.ne("collaboration") | x.campaign_active.fillna(False).astype(bool)
    )
    x["weekday"] = x.date.dt.dayofweek
    return x.drop(columns="_row_order")


def predict(history: pd.DataFrame, future: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """販売可能な過去日から平均を算出し、未観測時は段階的に補完する。"""
    if history.empty or future.empty or history.date.max() >= future.date.min():
        raise ValueError("履歴と予測期間が時系列順に分かれていません")
    if ((~history.eligible) & history[TARGET_COL].gt(0)).any():
        raise ValueError("販売条件外の過去データに正の売上があります")
    active = history[history.eligible]
    if active.empty or active[TARGET_COL].isna().any() or active[TARGET_COL].lt(0).any():
        raise ValueError("販売可能な履歴売上が空、欠損または負です")
    x = future[["store_id", "item_id", "weekday"]].copy().reset_index(drop=True)
    levels = [
        ["store_id", "item_id", "weekday"],
        ["store_id", "item_id"],
        ["item_id"],
    ]
    names = ["store_item_weekday", "store_item", "item"]
    predictions = pd.Series(np.nan, index=x.index, dtype=float)
    tiers = pd.Series("overall", index=x.index, dtype="object")
    for cols, name in zip(levels, names):
        means = active.groupby(cols, observed=True)[TARGET_COL].mean().rename("mean").reset_index()
        vals = x[cols].merge(means, on=cols, how="left", validate="many_to_one", sort=False)["mean"]
        missing = predictions.isna() & vals.notna()
        predictions.loc[missing] = vals.loc[missing]
        tiers.loc[missing] = name
    predictions = predictions.fillna(float(active[TARGET_COL].mean()))
    eligible = future.eligible.to_numpy(dtype=bool)
    predictions.loc[~eligible] = 0.0
    tiers.loc[~eligible] = "ineligible_zero"
    if not np.isfinite(predictions).all() or predictions.lt(0).any():
        raise ValueError("予測に欠損・無限大・負の値があります")
    return predictions.to_numpy(), {str(k): int(v) for k, v in tiers.value_counts().items()}


def layer_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    data = frame.copy()
    data["month"] = data.date.dt.to_period("M")
    metrics = {}
    for name, cols in LAYERS.items():
        sums = data.groupby(cols, observed=True, sort=False)[["actual", "prediction"]].sum()
        diff = sums.prediction - sums.actual
        metrics[name] = {"rows": len(sums), "rmse": float(np.sqrt(np.mean(diff**2))), "bias": float(diff.mean())}
    return metrics


def validation_score(train: pd.DataFrame, valid: pd.DataFrame, predictions: np.ndarray) -> dict:
    """2022年平均→2023年の誤差から尺度を作り、2024年を6層で評価。"""
    hist22 = train[train.date.dt.year.eq(2022)]
    ref23 = train[train.date.dt.year.eq(2023)].copy()
    averages = hist22.groupby(["store_id", "item_id"], observed=True)[TARGET_COL].mean().rename("prediction").reset_index()
    ref23 = ref23.merge(averages, on=["store_id", "item_id"], how="left", validate="many_to_one", sort=False)
    if ref23.empty or ref23.prediction.isna().any():
        raise ValueError("尺度計算用の2022年の店舗・商品平均が不足しています")
    ref23 = ref23.rename(columns={TARGET_COL: "actual"})
    reference = layer_metrics(ref23)
    scored = valid[KEY_COLS + [TARGET_COL]].rename(columns={TARGET_COL: "actual"}).copy()
    scored["prediction"] = predictions
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
            "aggregated_rows": total_rows, "layers": metrics}


def main() -> None:
    train, test = load_sales("train"), load_sales("test")
    if train[TARGET_COL].isna().any() or train[TARGET_COL].lt(0).any():
        raise ValueError("学習売上に欠損または負の値があります")
    if train.date.min() != pd.Timestamp("2022-01-01") or train.date.max() != TEST_START - pd.Timedelta(days=1):
        raise ValueError("期待する学習期間は2022〜2024年です")
    if test.date.min() != TEST_START or test.date.max() != pd.Timestamp("2025-12-31"):
        raise ValueError("期待する予測期間は2025年です")
    tables = load_availability()
    annotated = add_availability(train, tables)
    target = add_availability(test.drop(columns=TARGET_COL), tables)
    history = annotated[annotated.date.lt(VALID_START)]
    valid = annotated[annotated.date.ge(VALID_START)]
    valid_pred, valid_tiers = predict(history, valid)
    metrics = validation_score(train, valid, valid_pred)
    metrics["validation_fallback_counts"] = valid_tiers
    metrics["validation_ineligible_positive_rows"] = int(((~valid.eligible) & valid[TARGET_COL].gt(0)).sum())
    test_pred, test_tiers = predict(annotated, target)
    metrics["test_fallback_counts"] = test_tiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RUNS_DIR / f"{timestamp}_{EXP_NAME}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), run_dir / "code.py")
    (run_dir / "params.json").write_text(json.dumps({"valid_start": str(VALID_START.date()), "test_start": str(TEST_START.date()),
        "history": "eligible days", "fallback": ["store_item_weekday", "store_item", "item", "overall"],
        "validation_scale": "2022 store-item mean predicting 2023"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    valid[KEY_COLS + [TARGET_COL, "eligible"]].assign(prediction=valid_pred).to_csv(run_dir / "validation.csv", index=False)
    target[KEY_COLS].assign(**{TARGET_COL: test_pred}).to_csv(run_dir / "predictions.csv", index=False)
    print(f"2024 local 6-layer normalized RMSE: {metrics['composite_normalized_rmse']:.6f}")
    print(f"2024 daily row RMSE: {metrics['validation_daily_row_rmse']:.3f}")
    print(f"2024 validation fallback counts: {valid_tiers}")
    print(f"2025 prediction fallback counts: {test_tiers}")
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
