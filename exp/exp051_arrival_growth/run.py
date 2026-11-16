"""同一席構成・月・曜日で比較できる履歴の年次変化を使う。"""
import io
import zipfile

import pyarrow.parquet as pq
import numpy as np
import pandas as pd

from src.forecast import make_features, predict as base_predict
from src.runner import run
from src.config import INPUT_DIR


def predict(history, future):
    pred, details = base_predict(history, future)
    year = history.date.max().year
    core = history[history.menu_type.eq("core") & history.eligible & ~history.is_holiday].copy()
    core["units_proxy"] = core.sales_amount / core.selling_price
    days = core.groupby(["date", "store_id", "layout_version"], observed=True, as_index=False).units_proxy.sum()
    days["year"] = days.date.dt.year
    days = days[days.year.isin([year - 1, year])].copy()
    days["month"] = days.date.dt.month
    days["weekday"] = days.date.dt.dayofweek
    keys = ["store_id", "layout_version", "month", "weekday"]
    means = days.groupby(keys + ["year"], observed=True).units_proxy.agg(["mean", "size"])
    mean = means["mean"].unstack("year")
    count = means["size"].unstack("year")
    if not {year - 1, year}.issubset(mean.columns):
        raise ValueError("同一条件の年次比較に2年の履歴が必要です")
    ratio = (mean[year] / mean[year - 1]).where((count[year] >= 2) & (count[year - 1] >= 2))
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    # 月・曜日ごとの前年比を幾何平均し、対数変化の半分を反映する。
    store_factor = np.exp(np.log(ratio).groupby(level="store_id").mean() * 0.5)
    # 直近四半期で同じ条件がある店舗は、その期間の前年比を優先する。
    recent = ratio[ratio.index.get_level_values("month").isin([10, 11, 12])]
    recent_factor = np.exp(np.log(recent).groupby(level="store_id").mean() * 0.5)
    factor = recent_factor.combine_first(store_factor)
    # 非祝日・同一レイアウトの364日前を対応付け、直近四半期を比較する。
    pairs = days[["date", "store_id", "layout_version", "units_proxy"]].copy()
    pairs["date"] = pairs.date + pd.Timedelta(days=364)
    pairs = pairs.rename(columns={"units_proxy": "prior_units"})
    recent_days = days[days.year.eq(year) & days.month.isin([10, 11, 12])]
    joined = recent_days.merge(pairs, on=["date", "store_id", "layout_version"], how="inner", validate="one_to_one")
    paired = joined.groupby("store_id", observed=True).agg(current=("units_proxy", "sum"), prior=("prior_units", "sum"), size=("date", "size"))
    paired_factor = (paired.current / paired.prior).where((paired["size"] >= 14) & (paired.prior > 0))
    paired_factor = np.sqrt(paired_factor.replace([np.inf, -np.inf], np.nan)).dropna()
    factor = paired_factor.combine_first(factor)
    # 離脱した組を含む来店組数についても同じ364日ペアを作る。
    arrivals = history.loc[~history.is_holiday, ["date", "store_id", "layout_version", "arrival_groups"]].drop_duplicates().copy()
    if arrivals.duplicated(["date", "store_id"]).any():
        raise ValueError("来店組数の店舗・日が一意ではありません")
    previous = arrivals.rename(columns={"arrival_groups": "prior_groups"}).copy()
    previous["date"] = previous.date + pd.Timedelta(days=364)
    recent_arrivals = arrivals[arrivals.date.dt.year.eq(year) & arrivals.date.dt.month.isin([10, 11, 12])]
    arrival_pairs = recent_arrivals.merge(previous, on=["date", "store_id", "layout_version"], how="inner", validate="one_to_one")
    agg = arrival_pairs.groupby("store_id", observed=True).agg(current=("arrival_groups", "sum"), prior=("prior_groups", "sum"), size=("date", "size"))
    arrivals_factor = np.sqrt((agg.current / agg.prior).where((agg["size"] >= 14) & (agg.prior > 0)))
    arrivals_factor = arrivals_factor.replace([np.inf, -np.inf], np.nan).dropna()
    both = factor.index.intersection(arrivals_factor.index)
    factor.loc[both] = (factor.loc[both] + arrivals_factor.loc[both]) * 0.5
    details["arrival_growth_stores"] = int(len(both))
    details["paired_growth_stores"] = int(len(paired_factor))
    vals = future.store_id.map(factor)
    details["recent_growth_stores"] = int(recent_factor.size)
    details["matched_growth_unavailable"] = int(vals.isna().sum())
    details["matched_growth_groups"] = int(len(ratio))
    # 年央の半分という固定倍率に代え、予測開始後の経過時間で外挿量を増やす。
    elapsed = (future.date - history.date.max()).dt.days.to_numpy(dtype=float)
    fraction = np.clip(elapsed / 365.0, 0.0, 1.0)
    pred = pred * np.power(vals.fillna(1.0).to_numpy(dtype=float), 2.0 * fraction)
    return pred, details


def main():
    common_prepare = make_features()
    stores = pd.read_csv(INPUT_DIR / "stores.csv").store_id.tolist()
    visits = []
    with zipfile.ZipFile(INPUT_DIR / "logs.zip") as archive:
        for store in stores:
            frame = pq.read_table(io.BytesIO(archive.read(f"logs/{store}/visit_log.parquet")), columns=["date", "store_id"]).to_pandas()
            visits.append(frame.groupby(["date", "store_id"], observed=True).size().rename("arrival_groups").reset_index())
    visits = pd.concat(visits, ignore_index=True)
    if visits.duplicated(["date", "store_id"]).any():
        raise ValueError("店舗・日の来店組数が重複")

    def prepare(frame):
        result = common_prepare(frame)
        if "sales_amount" in frame:
            result = result.merge(visits, on=["date", "store_id"], how="left", validate="many_to_one", sort=False)
            if result.arrival_groups.isna().any():
                raise ValueError("履歴の来店組数がありません")
        return result

    run("exp051_arrival_growth", predict, __file__, prepare=prepare,
        params={"change": "blend past arrival and revenue growth for matched layouts"})


if __name__ == "__main__":
    main()
