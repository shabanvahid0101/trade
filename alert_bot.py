import argparse
import json
import time
from pathlib import Path

from tensorflow.keras.models import load_model

from crypto_predictor import (
    DATA_DIR,
    combine_multi_horizon_predictions,
    fetch_and_update_data,
    horizon_model_paths,
    load_artifacts,
    load_price_csv,
    predict_next_price,
    send_telegram_message,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = BASE_DIR / "alert_state.json"


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be a comma-separated list of positive integers.")
    return horizons


def build_alert_message(symbol: str, timeframe: str, result: dict, horizon_results: list[dict]) -> str:
    direction = "لانگ" if result["signal"] == "LONG" else "شورت"
    horizon_lines = []
    for item in horizon_results:
        horizon_lines.append(
            f"h{item['horizon_candles']}: {item['signal']} | "
            f"conf {item['confidence']:.2f} | "
            f"S {item.get('class_probabilities', {}).get('SHORT', 0):.2f} "
            f"H {item.get('class_probabilities', {}).get('HOLD', 0):.2f} "
            f"L {item.get('class_probabilities', {}).get('LONG', 0):.2f}"
        )

    return (
        f"<b>سیگنال {direction} {symbol}</b>\n"
        f"Timeframe: {timeframe}\n"
        f"Time: {result['timestamp']}\n"
        f"Price: ${result['current_price']:.2f}\n"
        f"Expected: {result['predicted_return_pct']:.3f}%\n"
        f"Confidence: {result['confidence']:.2f}\n"
        f"Votes: LONG {result['long_votes']} | SHORT {result['short_votes']} | HOLD {result['hold_votes']}\n\n"
        + "\n".join(horizon_lines)
        + "\n\nاین پیام هشدار ورود است، نه تضمین سود. مدیریت ریسک و حد ضرر لازم است."
    )


def run_once(args: argparse.Namespace) -> dict:
    data_path = Path(args.data)
    data = (
        fetch_and_update_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            file=data_path,
            max_batches=args.max_fetch_batches,
            exchange_name=args.exchange,
        )
        if args.update
        else load_price_csv(data_path)
    )

    horizon_results = []
    for horizon in parse_horizons(args.horizons):
        model_path, artifact_path = horizon_model_paths(args.symbol, args.timeframe, horizon)
        if not model_path.exists() or not artifact_path.exists():
            raise FileNotFoundError(f"Missing model files for {args.symbol} {args.timeframe} horizon {horizon}.")
        model = load_model(model_path)
        artifact = load_artifacts(artifact_path)
        artifact.setdefault("timeframe", args.timeframe)
        horizon_results.append(predict_next_price(model, data, artifact))

    final = combine_multi_horizon_predictions(
        horizon_results,
        min_agree=args.min_agree,
        min_confidence=args.min_confidence,
    )
    output = {"final": final, "horizons": horizon_results}
    print(json.dumps(output, indent=2))

    state_path = Path(args.state_file)
    state = load_state(state_path)
    now = int(time.time())
    last_signal = state.get("last_signal")
    last_timestamp = state.get("last_timestamp")
    last_sent_at = int(state.get("last_sent_at", 0))
    cooldown_passed = now - last_sent_at >= args.cooldown_seconds
    is_actionable = final["signal"] in {"LONG", "SHORT"}
    changed_signal = final["signal"] != last_signal or final["timestamp"] != last_timestamp

    should_send = args.telegram and is_actionable and (changed_signal or cooldown_passed)
    if args.send_hold and args.telegram and final["signal"] == "HOLD":
        should_send = True

    if should_send:
        if final["signal"] == "HOLD":
            message = (
                f"<b>HOLD {args.symbol}</b>\n"
                f"Timeframe: {args.timeframe}\n"
                f"Time: {final['timestamp']}\n"
                f"Price: ${final['current_price']:.2f}\n"
                f"Confidence: {final['confidence']:.2f}"
            )
        else:
            message = build_alert_message(args.symbol, args.timeframe, final, horizon_results)
        send_telegram_message(message)
        state.update({"last_signal": final["signal"], "last_timestamp": final["timestamp"], "last_sent_at": now})
        save_state(state_path, state)
    elif is_actionable:
        state.update({"last_signal": final["signal"], "last_timestamp": final["timestamp"], "last_seen_at": now})
        save_state(state_path, state)

    return output


def run_loop(args: argparse.Namespace) -> None:
    while True:
        try:
            run_once(args)
        except Exception as exc:
            print(f"Alert check failed: {exc}")
        time.sleep(args.sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telegram alert runner for crypto model signals.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--data", default=str(DATA_DIR / "1h-btc_history.csv"))
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--min-agree", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--max-fetch-batches", type=int, default=20)
    parser.add_argument("--mode", choices=["single", "loop"], default="single")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--cooldown-seconds", type=int, default=3600)
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--send-hold", action="store_true")
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.mode == "loop":
        run_loop(cli_args)
    else:
        run_once(cli_args)
