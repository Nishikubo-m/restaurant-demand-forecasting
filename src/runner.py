"""時系列実験の評価と成果物保存を共通化する。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import INPUT_DIR, ROOT_DIR, RUNS_DIR
from src.data import KEY_COLS, TARGET_COL, load_sales
from src.metrics import validation_score

PrepareFn = Callable[[pd.DataFrame], pd.DataFrame]
PredictFn = Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, dict]]


def _prepare(frame: pd.DataFrame, prepare: PrepareFn | None, *, with_target: bool) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    result = prepare(frame.copy()) if prepare else frame.copy()
    if not isinstance(result, pd.DataFrame) or len(result) != len(frame):
        raise ValueError("前処理は行数を変えないDataFrameを返してください")
    if not result[KEY_COLS].reset_index(drop=True).equals(frame[KEY_COLS]):
        raise ValueError("前処理で予測キーまたは行順が変更されました")
    if with_target:
        if TARGET_COL not in result or not result[TARGET_COL].reset_index(drop=True).equals(frame[TARGET_COL]):
            raise ValueError("前処理で学習目的変数が変更されました")
    elif TARGET_COL in result:
        raise ValueError("検証・テストの前処理は目的変数を返せません")
    return result.reset_index(drop=True)


def _predict(predictor: PredictFn, history: pd.DataFrame, future: pd.DataFrame) -> tuple[np.ndarray, dict]:
    if TARGET_COL in future:
        raise ValueError("予測対象に目的変数を渡すことはできません")
    # 呼び出し側のデータを保護し、破壊的な並び替えによる予測ずれを検知する。
    future_copy = future.copy()
    prediction, details = predictor(history.copy(), future_copy)
    if not future_copy[KEY_COLS].reset_index(drop=True).equals(future[KEY_COLS].reset_index(drop=True)):
        raise ValueError("予測処理が予測対象のキーまたは行順を変更しました")
    values = np.asarray(prediction, dtype=float)
    if values.ndim != 1 or len(values) != len(future) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("予測は対象と同じ長さで、有限かつ非負の1次元配列が必要です")
    if not isinstance(details, dict):
        raise ValueError("予測の付加情報はdictで返してください")
    return values, details


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL, text=True).strip())
    except (subprocess.SubprocessError, OSError):
        commit, dirty = None, None
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"git_commit": commit, "git_dirty": dirty, "python": sys.version.split()[0], "packages": versions,
            "uv_lock_sha256": _sha256(ROOT_DIR / "uv.lock") if (ROOT_DIR / "uv.lock").is_file() else None,
            "pyproject_sha256": _sha256(ROOT_DIR / "pyproject.toml") if (ROOT_DIR / "pyproject.toml").is_file() else None,
            "source_sha256": {str(p.relative_to(ROOT_DIR / "src")): _sha256(p)
                              for p in sorted((ROOT_DIR / "src").rglob("*.py")) if p.is_file()},
            "input_sha256": {p.name: _sha256(p) for p in sorted(INPUT_DIR.iterdir()) if p.is_file()}}


def run(
    exp_name: str,
    predictor: PredictFn,
    exp_file: str | Path,
    *,
    prepare: PrepareFn | None = None,
    valid_start: str = "2024-01-01",
    test_start: str = "2025-01-01",
    reference_year: int = 2023,
    params: dict | None = None,
) -> Path:
    """履歴のみで予測し、6層評価・予測・コードと入力情報を保存する。

    prepareは履歴、検証、テストに別々に呼ばれる。検証・テストには
    目的変数を渡さない。predictorは予測配列と付加情報dictを返す。
    既知の予定を使用する場合は各検証時点で利用可能だったかを実験側で明記する。
    """
    if not exp_name or Path(exp_name).name != exp_name:
        raise ValueError("実験名は単一のディレクトリ名にしてください")
    if set(params or {}) & {"valid_start", "test_start", "reference_year"}:
        raise ValueError("paramsで共通の検証設定を上書きすることはできません")
    exp_path = Path(exp_file)
    if not exp_path.is_file():
        raise ValueError(f"実験コードがありません: {exp_path}")
    exp_hash = _sha256(exp_path)
    initial_provenance = _provenance()
    train, test = load_sales("train"), load_sales("test")
    for name, frame in (("train", train), ("test", test)):
        if frame.empty or frame[KEY_COLS].isna().any().any() or frame.duplicated(KEY_COLS).any():
            raise ValueError(f"{name}に空・欠損・重複したキーがあります")
    y = train[TARGET_COL].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("学習売上に欠損・無限大・負の値があります")
    valid_date, test_date = pd.Timestamp(valid_start), pd.Timestamp(test_start)
    if valid_date >= test_date or train.date.max() >= test_date or test.date.min() < test_date:
        raise ValueError("学習・検証・テスト期間が不正です")
    raw_history = train[train.date < valid_date].reset_index(drop=True)
    raw_valid = train[train.date >= valid_date].reset_index(drop=True)
    if raw_history.empty or raw_valid.empty or raw_history.date.max() >= raw_valid.date.min():
        raise ValueError("時系列の履歴または検証期間が不正です")
    raw_test = test.drop(columns=TARGET_COL).reset_index(drop=True)
    history = _prepare(raw_history, prepare, with_target=True)
    valid_features = _prepare(raw_valid.drop(columns=TARGET_COL), prepare, with_target=False)
    test_features = _prepare(raw_test, prepare, with_target=False)
    valid_pred, valid_details = _predict(predictor, history, valid_features)
    metrics = validation_score(train, raw_valid, valid_pred, reference_year=reference_year)
    metrics["validation_prediction_details"] = valid_details
    if "eligible" in valid_features:
        metrics["validation_ineligible_positive_rows"] = int((~valid_features.eligible.astype(bool) & raw_valid[TARGET_COL].gt(0)).sum())
    full_history = _prepare(train, prepare, with_target=True)
    test_pred, test_details = _predict(predictor, full_history, test_features)
    metrics["test_prediction_details"] = test_details
    settings = {"valid_start": valid_start, "test_start": test_start, "reference_year": reference_year, **(params or {})}
    provenance = _provenance()
    checked = ("input_sha256", "uv_lock_sha256", "pyproject_sha256", "source_sha256")
    if any(provenance[k] != initial_provenance[k] for k in checked) or _sha256(exp_path) != exp_hash:
        raise RuntimeError("実験中に入力・依存関係・コードが変更されました")
    provenance["experiment_code_sha256"] = exp_hash
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(exp_path, run_dir / "code.py")
    shutil.copytree(ROOT_DIR / "src", run_dir / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "uv.lock"):
        if (ROOT_DIR / name).is_file():
            shutil.copy2(ROOT_DIR / name, run_dir / name)
    for name, obj in (("params.json", settings), ("metrics.json", metrics), ("provenance.json", provenance)):
        (run_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = raw_valid[KEY_COLS + [TARGET_COL]].copy()
    if "eligible" in valid_features:
        validation["eligible"] = valid_features.eligible.to_numpy()
    validation["prediction"] = valid_pred
    validation.to_csv(run_dir / "validation.csv", index=False)
    raw_test[KEY_COLS].assign(**{TARGET_COL: test_pred}).to_csv(run_dir / "predictions.csv", index=False)
    print(f"{valid_date.year} local 6-layer normalized RMSE: {metrics['composite_normalized_rmse']:.6f}")
    print(f"daily row RMSE: {metrics['validation_daily_row_rmse']:.3f}")
    print(f"Run saved to: {run_dir}")
    return run_dir
