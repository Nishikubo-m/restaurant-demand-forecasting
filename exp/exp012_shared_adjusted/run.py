"""exp009の処理を共通モジュールから実行し、予測が一致するか確認する。"""
from src.adjusted_baseline import LATEST_WEIGHT, make_prepare, predict
from src.runner import run


def main():
    run("exp012_shared_adjusted", predict, __file__, prepare=make_prepare(),
        params={"latest_year_weight": LATEST_WEIGHT, "change": "refactor exp009 into shared src module"})


if __name__ == "__main__":
    main()
