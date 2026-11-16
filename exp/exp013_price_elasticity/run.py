"""価格上昇時に数量が減る仮説を、平均予測への単一倍率で検証する。"""
import numpy as np

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
    return pred, details


def main():
    run("exp013_price_elasticity", predict, __file__, prepare=make_prepare(),
        params={"latest_year_weight": LATEST_WEIGHT, "price_exponent": 0.5,
                "change": "assume unit sales decrease with square root of price ratio"})


if __name__ == "__main__":
    main()
