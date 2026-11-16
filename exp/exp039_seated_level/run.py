"""価格上昇時に数量が減る仮説を、平均予測への単一倍率で検証する。"""
import io
import zipfile

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

from src.config import INPUT_DIR

from src.adjusted_baseline import LATEST_WEIGHT, make_prepare, predict as base_predict
from src.runner import run


def predict(history, future):
    pred, details = base_predict(history, future)
    # 直近年の比重を既存の平均と合わせ、販売可能な履歴の価格を基準にする。
    weights = np.where(history.date.dt.year.eq(history.date.max().year), LATEST_WEIGHT, 1)
    weighted = history.iloc[np.repeat(np.arange(len(history)), weights)]
    active = weighted[weighted.eligible]
    keys = ["store_id", "item_id"]
    price = active.groupby(keys, observed=True).selling_price.mean()
    by_item = active.groupby("item_id", observed=True).selling_price.mean()
    reference = future[keys].merge(price.rename("reference_price").reset_index(), on=keys,
                                   how="left", validate="many_to_one", sort=False).reference_price
    reference = reference.fillna(future.item_id.map(by_item).reset_index(drop=True))
    reference = reference.fillna(float(active.selling_price.mean()))
    pred = pred * np.sqrt(reference.to_numpy(dtype=float) / future.selling_price.to_numpy(dtype=float))
    # 条件が一致する日・店舗・カテゴリの平均を、同曜日の平均と比較する。
    daily = history.copy()
    daily["units_proxy"] = daily.sales_amount / daily.selling_price
    daily = daily.groupby(["date", "store_id", "category", "coupon"], observed=True, as_index=False).units_proxy.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    repetitions = np.where(daily.date.dt.year.eq(history.date.max().year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), repetitions)]
    keys = ["store_id", "category", "weekday"]
    base = daily.groupby(keys, observed=True).units_proxy.mean().rename("base")
    grouped = daily.groupby(keys + ["coupon"], observed=True).units_proxy.agg(["mean", "size"])
    rates = grouped.join(base)
    rates["factor"] = (rates["mean"] / rates["base"]).where(rates["size"] >= 5)
    future_keys = future[keys + ["coupon"]].reset_index(drop=True)
    factors = future_keys.merge(rates[["factor"]].reset_index(), on=keys + ["coupon"],
                                how="left", validate="many_to_one", sort=False).factor
    values = factors.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    pred *= np.where(missing, 1.0, values)
    details["coupon_factor_unavailable"] = int(missing.sum())
    # 過去の雨・非雨をレイアウト・月・曜日で正規化してから比較する。
    daily = history.copy()
    daily["units_proxy"] = daily.sales_amount / daily.selling_price
    daily = daily.groupby(["date", "store_id", "category", "layout_version", "weather_observed"], observed=True, as_index=False).units_proxy.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    daily["month"] = daily.date.dt.month
    daily["rain"] = daily.weather_observed.eq("rainy")
    weights = np.where(daily.date.dt.year.eq(history.date.max().year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), weights)].copy()
    baseline = daily.groupby(["store_id", "category", "layout_version", "month", "weekday"], observed=True).units_proxy.transform("mean")
    daily["relative"] = daily.units_proxy / baseline
    keys = ["store_id", "category", "weekday"]
    overall = daily.groupby(keys, observed=True).relative.mean().rename("overall")
    rain = daily.groupby(keys + ["rain"], observed=True).relative.agg(["mean", "size"])
    means = rain["mean"].unstack("rain").rename(columns={False: "dry", True: "wet"})
    counts = rain["size"].unstack("rain").rename(columns={False: "dry_count", True: "wet_count"})
    rates = means.join(counts).join(overall).reset_index()
    vals = future[keys + ["wet_probability_normal"]].reset_index(drop=True).merge(rates, on=keys, how="left", validate="many_to_one", sort=False)
    expected = (vals.wet_probability_normal * vals.wet + (1 - vals.wet_probability_normal) * vals.dry) / vals.overall
    factors = expected.where((vals.wet_count >= 5) & (vals.dry_count >= 5)).to_numpy(dtype=float)
    missing = ~np.isfinite(factors)
    pred = pred * np.where(missing, 1.0, factors)
    details["rain_factor_unavailable"] = int(missing.sum())
    # 最新レイアウトの数量相当水準を店舗・カテゴリ・曜日で求める。
    core = history[history.menu_type.eq("core")].copy()
    core["units"] = core.sales_amount / core.selling_price
    store_daily = core.groupby(["date", "store_id"], observed=True, as_index=False).units.sum()
    store_daily["year"] = store_daily.date.dt.year
    yearly = store_daily.groupby(["store_id", "year"], observed=True).units.mean().unstack("year")
    year = history.date.max().year
    growth = (yearly[year] / yearly[year - 1]).replace([np.inf, -np.inf], np.nan)
    daily = core.groupby(["date", "store_id", "category", "layout_version"], observed=True, as_index=False).units.sum()
    daily["year"] = daily.date.dt.year
    daily["weekday"] = daily.date.dt.dayofweek
    latest = daily.sort_values("date").groupby("store_id", observed=True).layout_version.last().rename("latest_layout")
    daily = daily.join(latest, on="store_id")
    repetitions = np.where(daily.year.eq(year), LATEST_WEIGHT, 1)
    weighted_daily = daily.iloc[np.repeat(np.arange(len(daily)), repetitions)]
    keys = ["store_id", "category", "weekday"]
    overall = weighted_daily.groupby(keys, observed=True).units.mean().rename("overall")
    current = weighted_daily[weighted_daily.layout_version.eq(weighted_daily.latest_layout)]
    current = current.groupby(keys, observed=True).units.agg(["mean", "size"])
    rates = current.join(overall)
    rates["factor"] = (rates["mean"] / rates.overall).where(rates["size"] >= 12)
    values = future[keys].reset_index(drop=True).merge(rates[["factor"]].reset_index(), on=keys, how="left", validate="many_to_one", sort=False).factor
    annual = future.store_id.map(growth).reset_index(drop=True)
    correction = (values / annual).replace([np.inf, -np.inf], np.nan)
    pred = pred * correction.fillna(1.0).to_numpy(dtype=float)
    details["layout_level_unavailable"] = int(correction.isna().sum())
    # 収益のカテゴリ倍率に代え、着席組数のレイアウト別水準を使う。
    visits = history[["date", "store_id", "weekday", "layout_version", "seated_groups"]].drop_duplicates().copy()
    if visits.duplicated(["date", "store_id"]).any():
        raise ValueError("着席組数の店舗・日単位が一意ではありません")
    visits = visits.join(latest, on="store_id")
    weights = np.where(visits.date.dt.year.eq(year), LATEST_WEIGHT, 1)
    visits = visits.iloc[np.repeat(np.arange(len(visits)), weights)]
    visit_keys = ["store_id", "weekday"]
    old_mean = visits.groupby(visit_keys, observed=True).seated_groups.mean().rename("old")
    current_visit = visits[visits.layout_version.eq(visits.latest_layout)].groupby(visit_keys, observed=True).seated_groups.agg(["mean", "size"])
    visit_rates = current_visit.join(old_mean)
    visit_rates["factor"] = (visit_rates["mean"] / visit_rates.old).where(visit_rates["size"] >= 12)
    factor = future[visit_keys].reset_index(drop=True).merge(visit_rates[["factor"]].reset_index(), on=visit_keys, how="left", validate="many_to_one", sort=False).factor
    ratio = (factor / values).replace([np.inf, -np.inf], np.nan)
    pred *= ratio.fillna(1.0).to_numpy(dtype=float)
    details["seated_level_unavailable"] = int(ratio.isna().sum())
    # レイアウト・曜日の平均で過去の日別数量を正規化してから月別に集計する。
    core = history[history.eligible & history.menu_type.eq("core")].copy()
    core["units"] = core.sales_amount / core.selling_price
    daily = core.groupby(["date", "store_id", "layout_version"], observed=True, as_index=False).units.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    daily["month"] = daily.date.dt.month
    daily_weights = np.where(daily.date.dt.year.eq(year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), daily_weights)].copy()
    layout_mean = daily.groupby(["store_id", "layout_version", "weekday"], observed=True).units.transform("mean")
    daily["relative"] = daily.units / layout_mean
    monthly = daily.groupby(["store_id", "month"], observed=True).relative.mean().rename("factor").reset_index()
    target = future[["store_id"]].copy().reset_index(drop=True)
    target["month"] = future.date.dt.month.to_numpy()
    factors = target.merge(monthly, on=["store_id", "month"], how="left", validate="many_to_one", sort=False).factor
    factors = factors.replace([np.inf, -np.inf], np.nan)
    pred = pred * factors.fillna(1.0).to_numpy(dtype=float)
    details["adjusted_month_unavailable"] = int(factors.isna().sum())
    # 祝日倍率と対になる非祝日倍率を、同じ条件付き平均から計算する。
    daily = history.groupby(["date", "store_id", "category", "is_holiday"], observed=True, as_index=False).sales_amount.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    weights = np.where(daily.date.dt.year.eq(year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), weights)]
    keys = ["store_id", "category", "weekday"]
    overall = daily.groupby(keys, observed=True).sales_amount.mean().rename("overall")
    regular = daily[~daily.is_holiday].groupby(keys, observed=True).sales_amount.mean().rename("regular")
    rates = pd.concat([overall, regular], axis=1)
    rates["factor"] = rates.regular / rates.overall
    factors = future[keys].reset_index(drop=True).merge(rates[["factor"]].reset_index(), on=keys, how="left", validate="many_to_one", sort=False).factor.to_numpy(dtype=float)
    use = ~future.is_holiday.to_numpy(dtype=bool)
    missing = use & ~np.isfinite(factors)
    pred = pred * np.where(use & np.isfinite(factors), factors, 1.0)
    details["nonholiday_factor_unavailable"] = int(missing.sum())
    # レイアウト・月・曜日・祝日状態で調整した日次値を全店舗で平滑化する。
    core = history[history.eligible & history.menu_type.eq("core")].copy()
    core["units"] = core.sales_amount / core.selling_price
    daily = core.groupby(["date", "store_id", "layout_version", "is_holiday"], observed=True, as_index=False).units.sum()
    daily["weekday"] = daily.date.dt.dayofweek
    daily["month"] = daily.date.dt.month
    daily["ordinal"] = pd.to_datetime("2000-" + daily.date.dt.strftime("%m-%d")).dt.dayofyear - 1
    repetitions = np.where(daily.date.dt.year.eq(year), LATEST_WEIGHT, 1)
    daily = daily.iloc[np.repeat(np.arange(len(daily)), repetitions)].copy()
    baseline = daily.groupby(["store_id", "layout_version", "month", "weekday", "is_holiday"], observed=True).units.transform("mean")
    daily["relative"] = daily.units / baseline
    daily = daily[np.isfinite(daily.relative)]
    # 店舗別の残差を全店舗平均に半分だけ加え、過去2年の揺れを抑える。
    by_day = daily.groupby(["store_id", "ordinal"], observed=True).relative.agg(["sum", "count"])
    stores = daily.store_id.unique()
    smoothed_stores = []
    for store in stores:
        table = by_day.loc[store].reindex(range(366), fill_value=0)
        repeated = pd.concat([table] * 3, ignore_index=True)
        smoothed = repeated.rolling(15, center=True, min_periods=1).sum().iloc[366:732].reset_index(drop=True)
        smoothed["store_id"] = store
        smoothed["ordinal"] = np.arange(366)
        smoothed_stores.append(smoothed)
    all_stores = pd.concat(smoothed_stores, ignore_index=True)
    pooled = all_stores.groupby("ordinal", observed=True)[["sum", "count"]].sum()
    pooled = (pooled["sum"] / pooled["count"]).rename("pooled")
    all_stores["local"] = all_stores["sum"] / all_stores["count"]
    all_stores = all_stores.join(pooled, on="ordinal")
    all_stores["factor"] = 0.5 * all_stores.local + 0.5 * all_stores.pooled
    target = future[["store_id"]].copy().reset_index(drop=True)
    target["ordinal"] = pd.to_datetime("2000-" + future.date.dt.strftime("%m-%d")).dt.dayofyear.to_numpy() - 1
    vals = target.merge(all_stores[["store_id", "ordinal", "factor"]], on=["store_id", "ordinal"], how="left", validate="many_to_one", sort=False).factor.to_numpy(dtype=float)
    missing = ~np.isfinite(vals)
    pred = pred * np.where(missing, 1.0, vals)
    details["dayofyear_factor_unavailable"] = int(missing.sum())
    # 既存の祝日・非祝日倍率を、レイアウト別水準を除いた倍率に置き換える。
    days = history.groupby(["date", "store_id", "category", "layout_version", "is_holiday"], observed=True, as_index=False).sales_amount.sum()
    days["weekday"] = days.date.dt.dayofweek
    repetitions = np.where(days.date.dt.year.eq(year), LATEST_WEIGHT, 1)
    days = days.iloc[np.repeat(np.arange(len(days)), repetitions)].copy()
    keys = ["store_id", "category", "weekday"]
    layout_avg = days.groupby(keys + ["layout_version"], observed=True).sales_amount.transform("mean")
    days["relative"] = days.sales_amount / layout_avg
    old_all = days.groupby(keys, observed=True).sales_amount.mean().rename("old_all")
    new_all = days.groupby(keys, observed=True).relative.mean().rename("new_all")
    state = days.groupby(keys + ["is_holiday"], observed=True).agg(old_state=("sales_amount", "mean"), new_state=("relative", "mean"))
    state = state.join(old_all).join(new_all)
    state["correction"] = (state.new_state / state.new_all) / (state.old_state / state.old_all)
    vals = future[keys + ["is_holiday"]].reset_index(drop=True).merge(state[["correction"]].reset_index(), on=keys + ["is_holiday"], how="left", validate="many_to_one", sort=False).correction.to_numpy(dtype=float)
    valid = np.isfinite(vals) & (vals > 0)
    pred = pred * np.where(valid, vals, 1.0)
    details["adjusted_holiday_unavailable"] = int((~valid).sum())
    return pred, details


def main():
    common_prepare = make_prepare()
    stores = pd.read_csv(INPUT_DIR / "stores.csv").store_id.tolist()
    campaign = pd.read_csv(INPUT_DIR / "campaign_calendar.csv", parse_dates=["date"])
    campaign = campaign[campaign.coupon_flag].copy()
    campaign["store_id"] = campaign.store_ids.map(lambda x: stores if x == "all" else str(x).split("|"))
    campaign = campaign.explode("store_id", ignore_index=True)
    coupon = campaign[["date", "store_id", "target_item_group"]].drop_duplicates().rename(columns={"target_item_group": "category"})
    coupon["coupon"] = True
    weather = pd.read_csv(INPUT_DIR / "calendar.csv", usecols=["date", "weather_observed", "wet_probability_normal"], parse_dates=["date"])
    visits = []
    with zipfile.ZipFile(INPUT_DIR / "logs.zip") as archive:
        for store in stores:
            path = f"logs/{store}/visit_log.parquet"
            frame = pq.read_table(io.BytesIO(archive.read(path)), columns=["date", "store_id", "is_seated"]).to_pandas()
            frame = frame.groupby(["date", "store_id"], observed=True, as_index=False).is_seated.sum().rename(columns={"is_seated": "seated_groups"})
            visits.append(frame)
    visits = pd.concat(visits, ignore_index=True)
    if visits.duplicated(["date", "store_id"]).any():
        raise ValueError("来店ログの店舗・日が重複")
    layouts = pd.read_csv(INPUT_DIR / "seat_layout.csv", parse_dates=["effective_start_date", "effective_end_date"])
    layouts["date"] = [pd.date_range(a, b) for a, b in zip(layouts.effective_start_date, layouts.effective_end_date)]
    layouts = layouts.explode("date", ignore_index=True)[["date", "store_id", "layout_version"]]
    if layouts.duplicated(["date", "store_id"]).any():
        raise ValueError("店舗・日のレイアウトが重複")

    def prepare(frame):
        result = common_prepare(frame).merge(coupon, on=["date", "store_id", "category"],
                                             how="left", validate="many_to_one", sort=False)
        result["coupon"] = result.coupon.fillna(False).astype(bool)
        columns = ["date", "wet_probability_normal"]
        if "sales_amount" in frame:
            columns.append("weather_observed")
        result = result.merge(weather[columns], on="date", how="left", validate="many_to_one", sort=False)
        if "sales_amount" in frame:
            result = result.merge(layouts, on=["date", "store_id"], how="left", validate="many_to_one", sort=False)
            result = result.merge(visits, on=["date", "store_id"], how="left", validate="many_to_one", sort=False)
            if result.seated_groups.isna().any():
                raise ValueError("履歴の着席組数がありません")
            if result.layout_version.isna().any():
                raise ValueError("履歴のレイアウトがありません")
        if result.wet_probability_normal.isna().any() or not result.wet_probability_normal.between(0, 1).all():
            raise ValueError("降水確率平年値がありません")
        if "sales_amount" in frame and result.weather_observed.isna().any():
            raise ValueError("履歴の天候実績がありません")
        return result

    run("exp039_seated_level", predict, __file__, prepare=prepare,
        params={"latest_year_weight": LATEST_WEIGHT, "price_exponent": 0.5,
                "change": "replace category sales level with historical seated group level"})


if __name__ == "__main__":
    main()
