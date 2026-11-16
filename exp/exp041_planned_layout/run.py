"""履歴の席構成変更前後平均から、予定変更後の倍率を求める。"""
import numpy as np
import pandas as pd

from src.config import INPUT_DIR
from src.forecast import make_features, predict as base_predict
from src.runner import run


def predict(history, future):
    pred, details = base_predict(history, future)
    core = history[history.menu_type.eq("core")].copy()
    core["units"] = core.sales_amount / core.selling_price
    daily = core.groupby(["date", "store_id", "layout_version"], observed=True, as_index=False).units.sum()
    daily = daily.sort_values(["store_id", "date"]).reset_index(drop=True)
    previous = daily.groupby("store_id", observed=True).layout_version.shift()
    events = daily.loc[previous.notna() & daily.layout_version.ne(previous), ["store_id", "date"]]
    cutoff = history.date.max()
    measures = []
    for event in events.itertuples(index=False):
        if event.date + pd.Timedelta(days=27) > cutoff:
            continue
        store = daily[daily.store_id.eq(event.store_id)].set_index("date").units
        before = pd.date_range(event.date - pd.Timedelta(days=28), periods=28)
        after = pd.date_range(event.date, periods=28)
        old_before = before - pd.Timedelta(days=364)
        old_after = after - pd.Timedelta(days=364)
        if not all(idx.isin(store.index).all() for idx in (before, after, old_before, old_after)):
            continue
        a, b = store.reindex(after).mean(), store.reindex(before).mean()
        c, d = store.reindex(old_after).mean(), store.reindex(old_before).mean()
        if min(a, b, c, d) > 0:
            measures.append((a / b) / (c / d))
    effect = float(np.median(measures)) if measures else 1.0
    latest = daily.groupby("store_id", observed=True).layout_version.last()
    current = future.store_id.map(latest)
    change = current.notna() & future.layout_version.notna() & current.ne(future.layout_version)
    pred = pred * np.where(change.to_numpy(), effect, 1.0)
    details["layout_change_history_events"] = len(measures)
    details["layout_change_factor"] = effect
    details["planned_layout_changed_rows"] = int(change.sum())
    return pred, details


def main():
    common_prepare = make_features()
    layouts = pd.read_csv(INPUT_DIR / "seat_layout.csv", parse_dates=["effective_start_date", "effective_end_date"])
    layouts["date"] = [pd.date_range(a, b) for a, b in zip(layouts.effective_start_date, layouts.effective_end_date)]
    layouts = layouts.explode("date", ignore_index=True)[["date", "store_id", "layout_version"]]
    if layouts.duplicated(["date", "store_id"]).any():
        raise ValueError("店舗・日のレイアウトが重複")

    def prepare(frame):
        result = common_prepare(frame)
        if "sales_amount" not in frame:
            result = result.merge(layouts, on=["date", "store_id"], how="left", validate="many_to_one", sort=False)
            if result.layout_version.isna().any():
                raise ValueError("将来のレイアウト予定がありません")
        return result

    run("exp041_planned_layout", predict, __file__, prepare=prepare,
        params={"change": "use historical 28-day layout-change comparison for known future layout changes"})


if __name__ == "__main__":
    main()
