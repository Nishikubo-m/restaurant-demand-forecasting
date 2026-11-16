"""販売期間と企画参加店舗に基づく販売可能性。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import INPUT_DIR
from src.data import KEY_COLS


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
