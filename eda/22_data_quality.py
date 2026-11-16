"""売上キー・異常値候補の再現可能な点検。データは変更しない。"""
import numpy as np
import pandas as pd

from src.data import load_sales


def main():
    sales = load_sales("train")
    keys = ["date", "store_id", "item_id"]
    print("rows:", len(sales))
    print("missing key rows:", int(sales[keys].isna().any(axis=1).sum()))
    print("duplicate key rows:", int(sales.duplicated(keys).sum()))
    print("missing sales rows:", int(sales.sales_amount.isna().sum()))
    print("negative sales rows:", int(sales.sales_amount.lt(0).sum()))
    sales["year"] = sales.date.dt.year
    sales["weekday"] = sales.date.dt.dayofweek
    median = sales.groupby(["year", "store_id", "item_id", "weekday"], observed=True).sales_amount.transform("median")
    ratio = (sales.sales_amount / median).where(median.gt(0)).replace([np.inf, -np.inf], np.nan)
    print("rows with positive median:", int(median.gt(0).sum()))
    print("rows over 5x median:", int(ratio.gt(5).sum()))
    print("rows over 10x median:", int(ratio.gt(10).sum()))
    candidates = sales.loc[ratio.gt(10), keys + ["sales_amount"]].copy()
    candidates["ratio"] = ratio[ratio.gt(10)]
    print(candidates.sort_values("ratio", ascending=False).to_string(index=False))
    print("候補は入力ミスの確定ではない。祝日・商品特性や分母の小ささを別途確認する。")
    print("この全期間の診断値を2024年検証の学習特徴量やデータ削除に使わない。")


if __name__ == "__main__":
    main()
