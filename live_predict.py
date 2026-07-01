import argparse
import time

from tensorflow.keras.models import load_model

from crypto_predictor import (
    ARTIFACT_PATH,
    MODEL_PATH,
    fetch_and_update_data,
    load_artifacts,
    predict_next_price,
    send_telegram_message,
)


def live_predict_only(symbol: str, timeframe: str, exchange: str, sleep_seconds: int, telegram: bool) -> None:
    model = load_model(MODEL_PATH, compile=False)
    artifact = load_artifacts(ARTIFACT_PATH)

    while True:
        data = fetch_and_update_data(
            symbol=symbol,
            timeframe=timeframe,
            batch_limit=1000,
            max_batches=200,
            exchange_name=exchange,
        )
        result = predict_next_price(model, data, artifact)
        message = (
            f"{result['signal']} {symbol} | "
            f"current=${result['current_price']:.2f}, "
            f"predicted=${result['predicted_price']:.2f}, "
            f"expected={result['predicted_return_pct']:.3f}%, "
            f"confidence={result['confidence']:.2f}"
        )
        print(message)

        if telegram and result["signal"] != "HOLD":
            send_telegram_message(
                f"<b>{result['signal']}</b> {symbol}\n"
                f"Current: ${result['current_price']:.2f}\n"
                f"Predicted: ${result['predicted_price']:.2f}\n"
                f"Expected: {result['predicted_return_pct']:.3f}%\n"
                f"Confidence: {result['confidence']:.2f}"
            )

        time.sleep(sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live crypto predictions.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--telegram", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    live_predict_only(args.symbol, args.timeframe, args.exchange, args.sleep_seconds, args.telegram)
