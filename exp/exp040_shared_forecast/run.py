"""exp032 の共通処理化による再現を確認する。"""
from src.adjusted_baseline import LATEST_WEIGHT
from src.forecast import make_features, predict
from src.runner import run


def main():
    run("exp040_shared_forecast", predict, __file__, prepare=make_features(),
        params={"latest_year_weight": LATEST_WEIGHT, "price_exponent": 0.5,
                "change": "refactor exp032 forecast into shared src"})


if __name__ == "__main__":
    main()
