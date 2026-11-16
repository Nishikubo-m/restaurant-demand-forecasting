"""2023年後半を、2022年内の誤差を尺度として参考検証する。"""
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from exp.exp054_rolling_quarter.run import predict as current_predict
from src.config import ROOT_DIR, RUNS_DIR
from src.data import KEY_COLS, TARGET_COL, load_sales
from src.forecast import make_features, predict as prior_predict
from src.metrics import layer_metrics
from src.runner import _prepare, _predict, _provenance, _sha256


def score(actual, predicted, reference):
    frame = actual[KEY_COLS + [TARGET_COL]].rename(columns={TARGET_COL: "actual"}).copy()
    frame["prediction"] = predicted
    layers = layer_metrics(frame)
    for name, layer in layers.items():
        scale = reference[name]["rmse"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"尺度が不正: {name}")
        layer["sigma"] = scale
        layer["normalized_rmse"] = layer["rmse"] / scale
    return {"score": float(np.sqrt(np.mean([v["normalized_rmse"] ** 2 for v in layers.values()]))),
            "daily_rmse": layers["date_store_item"]["rmse"], "layers": layers}


def main():
    source = Path(__file__).resolve()
    model_source = ROOT_DIR / "exp/exp054_rolling_quarter/run.py"
    before = _provenance()
    hashes = {"diagnostic": _sha256(source), "model": _sha256(model_source)}
    train = load_sales("train")
    reference_train = train[train.date.lt("2022-07-01")]
    reference_valid = train[train.date.ge("2022-07-01") & train.date.lt("2023-01-01")].copy()
    avg = reference_train.groupby(["store_id", "item_id"], observed=True)[TARGET_COL].mean().rename("prediction").reset_index()
    reference_valid = reference_valid.merge(avg, on=["store_id", "item_id"], how="left", validate="many_to_one", sort=False)
    reference = layer_metrics(reference_valid.rename(columns={TARGET_COL: "actual"}))
    history = train[train.date.lt("2023-07-01")].reset_index(drop=True)
    valid = train[train.date.ge("2023-07-01") & train.date.lt("2024-01-01")].reset_index(drop=True)
    prepare = make_features()
    past = _prepare(history, prepare, with_target=True)
    future = _prepare(valid.drop(columns=TARGET_COL), prepare, with_target=False)
    results, predictions = {}, valid[KEY_COLS + [TARGET_COL]].copy()
    for name, predictor in (("exp040", prior_predict), ("exp054", current_predict)):
        y, details = _predict(predictor, past, future)
        results[name] = score(valid, y, reference)
        results[name]["prediction_details"] = details
        predictions[name] = y
        print(f"{name}: 2023 H2 reference score {results[name]['score']:.6f}; daily RMSE {results[name]['daily_rmse']:.3f}")
    after = _provenance()
    checked = ("input_sha256", "uv_lock_sha256", "pyproject_sha256", "source_sha256")
    if any(before[k] != after[k] for k in checked) or hashes != {"diagnostic": _sha256(source), "model": _sha256(model_source)}:
        raise RuntimeError("実行中に入力またはコードが変更されました")
    out = RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_exp055_rolling_backtest"
    out.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(out / "validation.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    after["experiment_sha256"] = hashes
    (out / "provenance.json").write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(source, out / "code.py")
    shutil.copy2(model_source, out / "exp054_model.py")
    shutil.copytree(ROOT_DIR / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for file in ("pyproject.toml", "uv.lock"):
        if (ROOT_DIR / file).is_file():
            shutil.copy2(ROOT_DIR / file, out / file)
    print(f"Run saved to: {out}")


if __name__ == "__main__":
    main()
