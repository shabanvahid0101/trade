import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
load_dotenv(BASE_DIR / ".env")


def read_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_error": "invalid_json"}


def read_timestamps(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp"])
    return pd.to_datetime(frame["timestamp"], format="mixed").dropna().sort_values().reset_index(drop=True)


def file_status(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "size": 0}
    return {"exists": True, "size": path.stat().st_size}


def hours_since(timestamp: pd.Timestamp) -> float:
    now = pd.Timestamp.now("UTC").tz_localize(None)
    return float((now - pd.Timestamp(timestamp).tz_localize(None)).total_seconds() / 3600)


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    unit = timeframe[-1].lower()
    amount = int(timeframe[:-1])
    if unit == "m":
        return pd.to_timedelta(amount, unit="min")
    if unit == "h":
        return pd.to_timedelta(amount, unit="h")
    if unit == "d":
        return pd.to_timedelta(amount, unit="D")
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def dataset_health(path: str | Path, timeframe: str, max_age_hours: float) -> dict:
    path = Path(path)
    status = file_status(path)
    if not status["exists"]:
        return {**status, "ok": False, "error": "missing_file"}
    timestamps = read_timestamps(path)
    if timestamps.empty:
        return {**status, "ok": False, "error": "empty_dataset"}
    expected_step = timeframe_to_timedelta(timeframe)
    gap_count = int((timestamps.diff() > expected_step * 1.5).sum())
    last_timestamp = timestamps.iloc[-1]
    age = hours_since(last_timestamp)
    return {
        **status,
        "ok": age <= max_age_hours and gap_count == 0,
        "rows": int(len(timestamps)),
        "first_timestamp": str(timestamps.iloc[0]),
        "last_timestamp": str(last_timestamp),
        "age_hours": age,
        "gap_count": gap_count,
        "max_age_hours": max_age_hours,
    }


def fundamentals_health(path: str | Path, max_age_hours: float) -> dict:
    path = Path(path)
    status = file_status(path)
    if not status["exists"]:
        return {**status, "ok": False, "error": "missing_file"}
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        return {**status, "ok": False, "error": "missing_timestamp"}
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    timestamps = frame["timestamp"]
    if timestamps.empty:
        return {**status, "ok": False, "error": "empty_dataset"}
    last_timestamp = timestamps.iloc[-1]
    age = hours_since(last_timestamp)
    source_age = None
    if "fundamental_source_age_hours" in frame.columns and not frame.empty:
        source_age = pd.to_numeric(frame["fundamental_source_age_hours"], errors="coerce").iloc[-1]
        if pd.isna(source_age):
            source_age = None
    source_ok = source_age is None or float(source_age) <= max_age_hours
    return {
        **status,
        "ok": age <= max_age_hours and source_ok,
        "rows": int(len(timestamps)),
        "last_timestamp": str(last_timestamp),
        "age_hours": age,
        "source_age_hours": None if source_age is None else float(source_age),
        "max_age_hours": max_age_hours,
    }


def model_health(symbol: str, timeframe: str, horizons: list[int]) -> dict:
    safe_symbol = "".join(char for char in symbol.upper() if char.isalnum())
    files = {}
    ok = True
    for horizon in horizons:
        prefix = f"{safe_symbol}_{timeframe}_h{horizon}"
        model_path = MODELS_DIR / f"{prefix}.keras"
        artifact_path = MODELS_DIR / f"{prefix}_artifact.pkl"
        files[f"h{horizon}_model"] = file_status(model_path)
        files[f"h{horizon}_artifact"] = file_status(artifact_path)
        ok = ok and files[f"h{horizon}_model"]["exists"] and files[f"h{horizon}_artifact"]["exists"]
        ok = ok and files[f"h{horizon}_model"]["size"] > 0 and files[f"h{horizon}_artifact"]["size"] > 0
    return {"ok": ok, "files": files}


def state_health(state: dict, market_last_timestamp: str, name: str) -> dict:
    if not state:
        return {"ok": False, "error": "missing_state"}
    if state.get("_error"):
        return {"ok": False, "error": state["_error"]}
    last_timestamp = state.get("last_timestamp")
    return {
        "ok": bool(last_timestamp),
        "last_timestamp": last_timestamp,
        "matches_market_last": str(last_timestamp) == str(market_last_timestamp),
        "name": name,
    }


def paper_equity(state: dict, mark_price: float, initial_capital: float) -> dict:
    capital = float(state.get("capital", initial_capital) or initial_capital)
    position = int(state.get("position", 0) or 0)
    entry_price = float(state.get("entry_price", 0) or 0)
    notional = float(state.get("notional", 0) or 0)
    unrealized = 0.0
    if position and entry_price > 0 and notional > 0:
        unrealized = notional * position * ((mark_price - entry_price) / entry_price)
    equity = capital + unrealized
    return {
        "capital": capital,
        "equity": float(equity),
        "unrealized_pnl": float(unrealized),
        "return_pct": float((equity / initial_capital - 1) * 100) if initial_capital else 0.0,
        "position": "LONG" if position == 1 else "SHORT" if position == -1 else "FLAT",
    }


def send_telegram_message(message: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    strict = os.getenv("TELEGRAM_STRICT", "0") == "1"
    if not token or not chat_id:
        if strict:
            raise RuntimeError("Telegram TOKEN or CHAT_ID is not configured.")
        print("Telegram TOKEN or CHAT_ID is not configured.")
        return False
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
        timeout=10,
    )
    response.raise_for_status()
    print("Telegram message sent.")
    return True


def build_message(report: dict) -> str:
    status = report["status"]
    market = report["market_data"]
    fundamentals = report["fundamentals"]
    paper = report["paper_equity"]
    source_age = fundamentals.get("source_age_hours")
    source_age_text = f"{source_age:.2f}h" if source_age is not None else "n/a"
    lines = [
        f"<b>Bot Health: {status}</b>",
        f"Market last: {market.get('last_timestamp')} ({market.get('age_hours', 0):.2f}h old)",
        f"Fundamentals last: {fundamentals.get('last_timestamp')} "
        f"({fundamentals.get('age_hours', 0):.2f}h file, {source_age_text} source)",
        f"Models: {'OK' if report['models']['ok'] else 'WARN'}",
        f"Paper state: {report['paper_state'].get('last_timestamp')} | Alert state: {report['alert_state'].get('last_timestamp')}",
        f"Position: {paper['position']} | Equity: ${paper['equity']:.2f} ({paper['return_pct']:+.2f}%)",
    ]
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Health check for the crypto trading bot.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--data", default=str(DATA_DIR / "1h-btc_history.csv"))
    parser.add_argument("--fundamental-data", default=str(DATA_DIR / "1h-btc_fundamentals.csv"))
    parser.add_argument("--paper-state", default=str(BASE_DIR / "paper_state.json"))
    parser.add_argument("--alert-state", default=str(BASE_DIR / "alert_state.json"))
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--max-data-age-hours", type=float, default=4.0)
    parser.add_argument("--max-fundamental-age-hours", type=float, default=12.0)
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict:
    horizons = [int(part.strip()) for part in args.horizons.split(",") if part.strip()]
    market = dataset_health(args.data, args.timeframe, args.max_data_age_hours)
    fundamentals = fundamentals_health(args.fundamental_data, args.max_fundamental_age_hours)
    models = model_health(args.symbol, args.timeframe, horizons)
    paper_state_raw = read_json(args.paper_state)
    alert_state_raw = read_json(args.alert_state)
    paper_state = state_health(paper_state_raw, market.get("last_timestamp"), "paper")
    alert_state = state_health(alert_state_raw, market.get("last_timestamp"), "alert")

    mark_price = float(pd.read_csv(args.data).dropna(subset=["close"]).iloc[-1]["close"]) if market.get("ok") or market.get("last_timestamp") else 0.0
    equity = paper_equity(paper_state_raw, mark_price, args.initial_capital)
    warnings = []
    if not market.get("ok"):
        warnings.append("market_data_stale_or_gapped")
    if not fundamentals.get("ok"):
        warnings.append("fundamentals_stale_or_missing")
    if not models.get("ok"):
        warnings.append("model_files_missing")
    if not paper_state.get("ok"):
        warnings.append("paper_state_missing_or_invalid")
    if not alert_state.get("ok"):
        warnings.append("alert_state_missing_or_invalid")
    if alert_state.get("last_timestamp") and market.get("last_timestamp") and alert_state["last_timestamp"] != market["last_timestamp"]:
        warnings.append("alert_state_not_on_latest_market_candle")
    if paper_state.get("last_timestamp") and market.get("last_timestamp") and paper_state["last_timestamp"] != market["last_timestamp"]:
        warnings.append("paper_state_not_on_latest_market_candle")

    report = {
        "status": "OK" if not warnings else "WARNING",
        "warnings": warnings,
        "market_data": market,
        "fundamentals": fundamentals,
        "models": models,
        "paper_state": paper_state,
        "alert_state": alert_state,
        "paper_equity": equity,
    }
    print(json.dumps(report, indent=2))
    if args.telegram:
        send_telegram_message(build_message(report))
    if warnings and args.fail_on_warning:
        raise SystemExit("Health check warnings found.")
    return report


if __name__ == "__main__":
    main(build_parser().parse_args())
