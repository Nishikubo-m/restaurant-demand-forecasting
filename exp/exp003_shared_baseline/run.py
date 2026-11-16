"""共通処理でexp002を再現する。予測方法は変更しない。"""
from src.availability import add_availability, load_availability
from src.baselines import predict_available_means
from src.runner import run


def main() -> None:
    tables = load_availability()

    def prepare(frame):
        return add_availability(frame, tables)

    run(
        "exp003_shared_baseline",
        predict_available_means,
        __file__,
        prepare=prepare,
        params={"method": "eligible store-item-weekday mean", "purpose": "reproduce exp002 using shared modules"},
    )


if __name__ == "__main__":
    main()
