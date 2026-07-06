import argparse
import json
from pathlib import Path

import pandas as pd

from crypto_predictor import DATA_DIR, fetch_and_update_with_fallbacks
from data_quality import atomic_write_csv, price_quality_report, sanitize_price_frame
from fundamental_data import update_fundamental_file


BASE_DIR = Path(__file__).resolve().parent


def load_price_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed", errors="coerce")
    return frame.sort_values("timestamp").reset_index(drop=True)


def resample_ohlcv(frame: pd.DataFrame, timeframe: str, max_rows: int | None = None) -> pd.DataFrame:
    clean, _ = sanitize_price_frame(frame)
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], format="mixed", errors="coerce")
    resampled = (
        clean.set_index("timestamp")
        .resample(timeframe)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    resampled = resampled[resampled["volume"] >= 0].reset_index(drop=True)
    if max_rows and max_rows > 0:
        resampled = resampled.tail(max_rows).reset_index(drop=True)
    resampled["timestamp"] = resampled["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return resampled


def audit_price_file(path: str | Path, timeframe: str, write: bool) -> dict:
    path = Path(path)
    raw = pd.read_csv(path)
    before = price_quality_report(raw, timeframe)
    clean, actions = sanitize_price_frame(raw)
    after = price_quality_report(clean, timeframe)
    if write:
        atomic_write_csv(clean, path)
    return {
        "path": str(path),
        "timeframe": timeframe,
        "before": before,
        "after": after,
        "actions": actions,
    }


def update_price_dataset(
    symbol: str,
    timeframe: str,
    path: str | Path,
    exchange_fallbacks: list[str],
    max_fetch_batches: int,
    max_data_age_hours: float,
) -> pd.DataFrame:
    return fetch_and_update_with_fallbacks(
        symbol=symbol,
        timeframe=timeframe,
        file=path,
        exchange_names=exchange_fallbacks,
        max_batches=max_fetch_batches,
        max_data_age_hours=max_data_age_hours,
    )


def run(args: argparse.Namespace) -> dict:
    exchange_fallbacks = [item.strip() for item in args.exchange_fallbacks.split(",") if item.strip()]
    if not exchange_fallbacks:
        raise ValueError("At least one exchange fallback is required.")

    report: dict = {"symbol": args.symbol, "datasets": {}, "fundamentals": {}}

    one_hour = update_price_dataset(
        symbol=args.symbol,
        timeframe="1h",
        path=args.one_hour_data,
        exchange_fallbacks=exchange_fallbacks,
        max_fetch_batches=args.one_hour_fetch_batches,
        max_data_age_hours=args.one_hour_max_age_hours,
    )
    report["datasets"]["1h"] = audit_price_file(args.one_hour_data, "1h", write=True)

    if args.update_fundamentals:
        fundamentals = update_fundamental_file(
            market_data=args.one_hour_data,
            output=args.one_hour_fundamentals,
            symbol=args.symbol,
            timeframe="1h",
        )
        report["fundamentals"]["1h"] = {
            "path": str(args.one_hour_fundamentals),
            "rows": int(len(fundamentals)),
            "first_timestamp": str(fundamentals["timestamp"].min()) if not fundamentals.empty else None,
            "last_timestamp": str(fundamentals["timestamp"].max()) if not fundamentals.empty else None,
        }

    five_minute = update_price_dataset(
        symbol=args.symbol,
        timeframe="5m",
        path=args.five_minute_data,
        exchange_fallbacks=exchange_fallbacks,
        max_fetch_batches=args.five_minute_fetch_batches,
        max_data_age_hours=args.five_minute_max_age_hours,
    )
    report["datasets"]["5m"] = audit_price_file(args.five_minute_data, "5m", write=True)

    if args.derive_fifteen_minute:
        fifteen_minute = resample_ohlcv(
            five_minute if not five_minute.empty else load_price_frame(args.five_minute_data),
            "15min",
            max_rows=args.fifteen_minute_rows,
        )
        atomic_write_csv(fifteen_minute, Path(args.fifteen_minute_data))
        report["datasets"]["15m"] = audit_price_file(args.fifteen_minute_data, "15m", write=True)

    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh and audit trading datasets.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--exchange-fallbacks", default="binance,okx,kucoin,bybit")
    parser.add_argument("--one-hour-data", default=str(DATA_DIR / "1h-btc_history.csv"))
    parser.add_argument("--one-hour-fundamentals", default=str(DATA_DIR / "1h-btc_fundamentals.csv"))
    parser.add_argument("--five-minute-data", default=str(DATA_DIR / "5m_btc_history.csv"))
    parser.add_argument("--fifteen-minute-data", default=str(DATA_DIR / "15m_btc_history_5000.csv"))
    parser.add_argument("--fifteen-minute-rows", type=int, default=5000)
    parser.add_argument("--one-hour-fetch-batches", type=int, default=5)
    parser.add_argument("--five-minute-fetch-batches", type=int, default=20)
    parser.add_argument("--one-hour-max-age-hours", type=float, default=2.5)
    parser.add_argument("--five-minute-max-age-hours", type=float, default=1.0)
    parser.add_argument("--derive-fifteen-minute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update-fundamentals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", default=str(BASE_DIR / "dataset_maintenance_report.json"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))
