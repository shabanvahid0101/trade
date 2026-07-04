import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_data import FUNDAMENTAL_COLUMNS, align_fundamentals_to_market, load_fundamental_csv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"


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


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def price_quality_report(frame: pd.DataFrame, timeframe: str) -> dict:
    expected_step = timeframe_to_timedelta(timeframe)
    timestamps = pd.to_datetime(frame["timestamp"], format="mixed", errors="coerce")
    sorted_ts = timestamps.sort_values().reset_index(drop=True)
    diffs = sorted_ts.diff().dropna()
    gap_mask = diffs > expected_step * 1.5
    duplicate_count = int(timestamps.duplicated().sum())

    numeric = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    bad_ohlc = (
        (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
        | (numeric["high"] < numeric["low"])
        | (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (numeric["volume"] < 0)
    )
    returns = np.log(numeric["close"] / numeric["close"].shift(1))
    return {
        "rows": int(len(frame)),
        "first_timestamp": str(sorted_ts.min()) if not sorted_ts.empty else None,
        "last_timestamp": str(sorted_ts.max()) if not sorted_ts.empty else None,
        "bad_timestamps": int(timestamps.isna().sum()),
        "duplicate_timestamps": duplicate_count,
        "monotonic": bool(timestamps.is_monotonic_increasing),
        "gap_count": int(gap_mask.sum()),
        "expected_step": str(expected_step),
        "invalid_ohlcv_rows": int(bad_ohlc.sum()),
        "nulls": {column: int(value) for column, value in frame.isna().sum().to_dict().items()},
        "max_abs_log_return": None if returns.dropna().empty else float(returns.abs().max()),
    }


def sanitize_price_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing price columns: {missing}")

    clean = frame[required].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], format="mixed", errors="coerce")
    for column in required[1:]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    before = len(clean)
    clean = clean.dropna(subset=required)
    clean = clean.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    valid = (
        (clean[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (clean["volume"] >= 0)
        & (clean["high"] >= clean[["open", "close", "low"]].max(axis=1))
        & (clean["low"] <= clean[["open", "close", "high"]].min(axis=1))
    )
    clean = clean.loc[valid].reset_index(drop=True)
    clean["timestamp"] = clean["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    actions = {
        "dropped_rows": int(before - len(clean)),
        "kept_rows": int(len(clean)),
    }
    return clean, actions


def sanitize_fundamental_frame(frame: pd.DataFrame, market_timestamps: pd.Series) -> tuple[pd.DataFrame, dict]:
    clean = frame.copy()
    if "timestamp" not in clean.columns:
        raise ValueError("Fundamental data must contain a timestamp column.")
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], format="mixed", errors="coerce")
    for column in FUNDAMENTAL_COLUMNS:
        if column not in clean.columns:
            clean[column] = np.nan
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    before = len(clean)
    clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    clean = align_fundamentals_to_market(clean[["timestamp", *FUNDAMENTAL_COLUMNS]], market_timestamps)
    clean["timestamp"] = pd.to_datetime(clean["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    source_age = pd.to_numeric(clean.get("fundamental_source_age_hours"), errors="coerce")
    actions = {
        "input_rows": int(before),
        "aligned_rows": int(len(clean)),
        "max_source_age_hours": None if source_age.dropna().empty else float(source_age.max()),
        "last_source_age_hours": None if source_age.dropna().empty else float(source_age.iloc[-1]),
    }
    return clean, actions


def audit_and_clean(args: argparse.Namespace) -> dict:
    price_path = Path(args.price_data)
    fundamental_path = Path(args.fundamental_data) if args.fundamental_data else None
    price_raw = pd.read_csv(price_path)
    price_before = price_quality_report(price_raw, args.timeframe)
    price_clean, price_actions = sanitize_price_frame(price_raw)
    price_after = price_quality_report(price_clean, args.timeframe)

    report = {
        "price_data": {
            "path": str(price_path),
            "before": price_before,
            "after": price_after,
            "actions": price_actions,
        }
    }

    if args.write:
        atomic_write_csv(price_clean, price_path)

    if fundamental_path:
        market_timestamps = pd.to_datetime(price_clean["timestamp"], format="mixed")
        fundamentals_raw = load_fundamental_csv(fundamental_path) if fundamental_path.exists() else pd.DataFrame(columns=["timestamp", *FUNDAMENTAL_COLUMNS])
        fundamentals_clean, fundamental_actions = sanitize_fundamental_frame(fundamentals_raw, market_timestamps)
        fundamental_source_age = pd.to_numeric(fundamentals_clean["fundamental_source_age_hours"], errors="coerce")
        report["fundamental_data"] = {
            "path": str(fundamental_path),
            "rows": int(len(fundamentals_clean)),
            "first_timestamp": str(pd.to_datetime(fundamentals_clean["timestamp"]).min()) if not fundamentals_clean.empty else None,
            "last_timestamp": str(pd.to_datetime(fundamentals_clean["timestamp"]).max()) if not fundamentals_clean.empty else None,
            "nulls": {column: int(value) for column, value in fundamentals_clean.isna().sum().to_dict().items()},
            "actions": fundamental_actions,
            "source_age_hours": {
                "max": None if fundamental_source_age.dropna().empty else float(fundamental_source_age.max()),
                "last": None if fundamental_source_age.dropna().empty else float(fundamental_source_age.iloc[-1]),
            },
        }
        if args.write:
            atomic_write_csv(fundamentals_clean, fundamental_path)

    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and sanitize market/fundamental datasets.")
    parser.add_argument("--price-data", default=str(DATA_DIR / "1h-btc_history.csv"))
    parser.add_argument("--fundamental-data", default=str(DATA_DIR / "1h-btc_fundamentals.csv"))
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--report", default=str(BASE_DIR / "data_quality_report.json"))
    parser.add_argument("--write", action="store_true")
    return parser


if __name__ == "__main__":
    result = audit_and_clean(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
