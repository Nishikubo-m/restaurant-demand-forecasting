"""コンペデータの読み込み。入力ファイルは変更しない。"""
import pandas as pd

from src.config import INPUT_DIR

KEY_COLS = ["date", "store_id", "item_id"]
TARGET_COL = "sales_amount"


def load_sales(split: str) -> pd.DataFrame:
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    return pd.read_csv(INPUT_DIR / f"{split}.csv", parse_dates=["date"])


def load_table(name: str) -> pd.DataFrame:
    names = {"calendar", "campaign_calendar", "menu_catalog_history", "menus",
             "price_history", "seat_layout", "stores"}
    if name not in names:
        raise ValueError(f"Unknown table: {name}")
    return pd.read_csv(INPUT_DIR / f"{name}.csv")
