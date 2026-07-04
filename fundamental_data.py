import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

FUNDAMENTAL_COLUMNS = [
    "funding_rate",
    "open_interest",
    "open_interest_value",
    "global_long_short_ratio",
    "global_long_account",
    "global_short_account",
    "taker_buy_sell_ratio",
    "taker_buy_vol",
    "taker_sell_vol",
    "fundamental_source_age_hours",
]

DERIVED_FUNDAMENTAL_FEATURE_COLUMNS = [
    "funding_rate_change",
    "funding_rate_zscore",
    "funding_rate_ema_24",
    "funding_rate_extreme",
    "open_interest_change",
    "open_interest_change_6",
    "open_interest_change_24",
    "open_interest_zscore",
    "open_interest_value_change",
    "open_interest_value_change_24",
    "global_long_short_ratio_change",
    "global_long_short_ratio_zscore",
    "long_account_change",
    "short_account_change",
    "long_short_crowding",
    "taker_buy_sell_ratio_change",
    "taker_buy_pressure",
    "taker_buy_pressure_change",
    "taker_buy_pressure_zscore",
    "taker_buy_imbalance",
    "taker_buy_imbalance_zscore",
    "futures_leverage_pressure",
    "futures_directional_pressure",
    "futures_crowding_pressure",
    "futures_squeeze_risk",
    "futures_pain_risk",
    "oi_price_divergence_24",
    "volume_oi_confirmation",
    "fundamental_source_stale",
]

FUNDAMENTAL_FEATURE_COLUMNS = [
    "funding_rate",
    *DERIVED_FUNDAMENTAL_FEATURE_COLUMNS[:10],
    "global_long_short_ratio",
    "global_long_short_ratio_change",
    "global_long_short_ratio_zscore",
    "global_long_account",
    "global_short_account",
    "long_account_change",
    "short_account_change",
    "long_short_crowding",
    "taker_buy_sell_ratio",
    *DERIVED_FUNDAMENTAL_FEATURE_COLUMNS[15:28],
    "fundamental_source_age_hours",
    "fundamental_source_stale",
]


def binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").upper()


def timeframe_to_milliseconds(timeframe: str) -> int:
    units = {
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
    }
    unit = timeframe[-1].lower()
    if unit not in units:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return int(timeframe[:-1]) * units[unit]


def request_json(url: str, params: dict, retries: int = 3, sleep_seconds: float = 0.4) -> list[dict]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "code" in payload:
                raise RuntimeError(payload)
            return payload if isinstance(payload, list) else []
        except Exception as exc:
            body = getattr(getattr(exc, "response", None), "text", "")
            last_error = f"{exc} {body}".strip()
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Request failed for {url}: {last_error}")


def fetch_paginated(
    url: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    time_key: str,
    limit: int = 500,
    period: str | None = None,
    max_window_ms: int | None = None,
) -> list[dict]:
    rows = []
    cursor = start_ms
    while cursor <= end_ms:
        window_end = min(end_ms, cursor + max_window_ms) if max_window_ms else end_ms
        params = {"symbol": symbol, "startTime": cursor, "endTime": window_end, "limit": limit}
        if period:
            params["period"] = period
        batch = request_json(url, params=params)
        if not batch:
            if max_window_ms and window_end < end_ms:
                cursor = window_end + 1
                continue
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][time_key]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < limit and (not max_window_ms or window_end >= end_ms):
            break
        if len(batch) < limit and max_window_ms:
            cursor = window_end + 1
        time.sleep(0.2)
    return rows


def frame_from_rows(rows: list[dict], time_key: str, rename: dict[str, str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", *rename.values()])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame[time_key].astype("int64"), unit="ms", utc=False)
    for source, target in rename.items():
        frame[target] = pd.to_numeric(frame.get(source), errors="coerce")
    return frame[["timestamp", *rename.values()]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")


def load_market_span(path: str | Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    frame = pd.read_csv(path, usecols=["timestamp"])
    timestamps = pd.to_datetime(frame["timestamp"])
    return timestamps.min(), timestamps.max()


def load_market_timestamps(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp"])
    return pd.to_datetime(frame["timestamp"], format="mixed").dropna().sort_values().drop_duplicates().reset_index(drop=True)


def align_fundamentals_to_market(fundamentals: pd.DataFrame, market_timestamps: pd.Series) -> pd.DataFrame:
    market = pd.DataFrame({"timestamp": pd.to_datetime(market_timestamps).astype("datetime64[ns]")})
    values = fundamentals.copy().sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    values["timestamp"] = pd.to_datetime(values["timestamp"], format="mixed").astype("datetime64[ns]")
    if market.empty:
        return values[["timestamp", *FUNDAMENTAL_COLUMNS]]
    raw_columns = [column for column in FUNDAMENTAL_COLUMNS if column != "fundamental_source_age_hours"]
    for column in raw_columns:
        if column not in values.columns:
            values[column] = np.nan
    if values.empty:
        values = pd.DataFrame(columns=["timestamp", "source_timestamp", *raw_columns])
    else:
        non_empty = values[raw_columns].notna().any(axis=1)
        values["source_timestamp"] = values["timestamp"].where(non_empty)
    aligned = pd.merge_asof(
        market,
        values[["timestamp", "source_timestamp", *raw_columns]],
        on="timestamp",
        direction="backward",
    )
    aligned[raw_columns] = aligned[raw_columns].ffill()
    aligned["source_timestamp"] = pd.to_datetime(aligned["source_timestamp"]).ffill()
    aligned["fundamental_source_age_hours"] = (
        (aligned["timestamp"] - aligned["source_timestamp"]).dt.total_seconds() / 3600
    ).clip(lower=0)
    return aligned[["timestamp", *FUNDAMENTAL_COLUMNS]]


def fetch_binance_fundamentals(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    api_symbol = binance_symbol(symbol)
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    period = timeframe if timeframe in {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"} else "1h"
    futures_data_window_ms = 29 * 24 * 60 * 60 * 1000
    futures_start_ms = max(start_ms, end_ms - futures_data_window_ms)

    def safe_fetch(name: str, *args, **kwargs) -> list[dict]:
        try:
            return fetch_paginated(*args, **kwargs)
        except Exception as exc:
            print(f"Warning: skipped {name} fundamentals: {exc}")
            return []

    funding_rows = safe_fetch(
        "funding_rate",
        "https://fapi.binance.com/fapi/v1/fundingRate",
        api_symbol,
        start_ms,
        end_ms,
        time_key="fundingTime",
        limit=1000,
    )
    open_interest_rows = safe_fetch(
        "open_interest",
        "https://fapi.binance.com/futures/data/openInterestHist",
        api_symbol,
        futures_start_ms,
        end_ms,
        time_key="timestamp",
        limit=500,
        period=period,
        max_window_ms=futures_data_window_ms,
    )
    long_short_rows = safe_fetch(
        "global_long_short_ratio",
        "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
        api_symbol,
        futures_start_ms,
        end_ms,
        time_key="timestamp",
        limit=500,
        period=period,
        max_window_ms=futures_data_window_ms,
    )
    taker_rows = safe_fetch(
        "taker_buy_sell_ratio",
        "https://fapi.binance.com/futures/data/takerlongshortRatio",
        api_symbol,
        futures_start_ms,
        end_ms,
        time_key="timestamp",
        limit=500,
        period=period,
        max_window_ms=futures_data_window_ms,
    )

    frames = [
        frame_from_rows(funding_rows, "fundingTime", {"fundingRate": "funding_rate"}),
        frame_from_rows(
            open_interest_rows,
            "timestamp",
            {"sumOpenInterest": "open_interest", "sumOpenInterestValue": "open_interest_value"},
        ),
        frame_from_rows(
            long_short_rows,
            "timestamp",
            {
                "longShortRatio": "global_long_short_ratio",
                "longAccount": "global_long_account",
                "shortAccount": "global_short_account",
            },
        ),
        frame_from_rows(
            taker_rows,
            "timestamp",
            {
                "buySellRatio": "taker_buy_sell_ratio",
                "buyVol": "taker_buy_vol",
                "sellVol": "taker_sell_vol",
            },
        ),
    ]

    merged = None
    for frame in frames:
        merged = frame if merged is None else pd.merge(merged, frame, on="timestamp", how="outer")
    if merged is None:
        merged = pd.DataFrame(columns=["timestamp", *FUNDAMENTAL_COLUMNS])
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    for column in FUNDAMENTAL_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[["timestamp", *FUNDAMENTAL_COLUMNS]]


def load_fundamental_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} must contain a timestamp column.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed")
    for column in FUNDAMENTAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["timestamp", *FUNDAMENTAL_COLUMNS]].sort_values("timestamp").drop_duplicates(subset=["timestamp"])


def merge_fundamentals_asof(price_data: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    prices = price_data.copy().sort_values("timestamp")
    values = fundamentals.copy().sort_values("timestamp")
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    values["timestamp"] = pd.to_datetime(values["timestamp"])
    merged = pd.merge_asof(prices, values, on="timestamp", direction="backward")
    merged[FUNDAMENTAL_COLUMNS] = merged[FUNDAMENTAL_COLUMNS].ffill()
    return merged


def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if not set(FUNDAMENTAL_COLUMNS).issubset(frame.columns):
        return frame

    for column in FUNDAMENTAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["fundamental_source_age_hours"] = frame["fundamental_source_age_hours"].fillna(0)
    frame["fundamental_source_stale"] = (frame["fundamental_source_age_hours"] / 24).clip(lower=0, upper=7)

    funding = frame["funding_rate"]
    funding_std = funding.rolling(72).std().replace(0, np.nan)
    frame["funding_rate_change"] = funding.diff()
    frame["funding_rate_zscore"] = (funding - funding.rolling(72).mean()) / funding_std
    frame["funding_rate_ema_24"] = funding.ewm(span=24, adjust=False).mean()
    frame["funding_rate_extreme"] = frame["funding_rate_zscore"].abs()

    open_interest = frame["open_interest"].replace(0, np.nan)
    oi_std = open_interest.rolling(72).std().replace(0, np.nan)
    frame["open_interest_change"] = np.log(open_interest / open_interest.shift(1))
    frame["open_interest_change_6"] = np.log(open_interest / open_interest.shift(6))
    frame["open_interest_change_24"] = np.log(open_interest / open_interest.shift(24))
    frame["open_interest_zscore"] = (open_interest - open_interest.rolling(72).mean()) / oi_std
    open_interest_value = frame["open_interest_value"].replace(0, np.nan)
    frame["open_interest_value_change"] = np.log(open_interest_value / open_interest_value.shift(1))
    frame["open_interest_value_change_24"] = np.log(open_interest_value / open_interest_value.shift(24))

    long_short_ratio = frame["global_long_short_ratio"].replace(0, np.nan)
    ratio_std = long_short_ratio.rolling(72).std().replace(0, np.nan)
    frame["global_long_short_ratio_change"] = np.log(long_short_ratio / long_short_ratio.shift(1))
    frame["global_long_short_ratio_zscore"] = (long_short_ratio - long_short_ratio.rolling(72).mean()) / ratio_std
    frame["long_account_change"] = frame["global_long_account"].diff()
    frame["short_account_change"] = frame["global_short_account"].diff()
    frame["long_short_crowding"] = frame["global_long_account"] - frame["global_short_account"]

    taker_ratio = frame["taker_buy_sell_ratio"].replace(0, np.nan)
    frame["taker_buy_sell_ratio_change"] = np.log(taker_ratio / taker_ratio.shift(1))
    total_taker = frame["taker_buy_vol"] + frame["taker_sell_vol"]
    frame["taker_buy_pressure"] = frame["taker_buy_vol"] / total_taker.replace(0, np.nan)
    pressure_std = frame["taker_buy_pressure"].rolling(72).std().replace(0, np.nan)
    frame["taker_buy_pressure_change"] = frame["taker_buy_pressure"].diff()
    frame["taker_buy_pressure_zscore"] = (
        frame["taker_buy_pressure"] - frame["taker_buy_pressure"].rolling(72).mean()
    ) / pressure_std
    frame["taker_buy_imbalance"] = (frame["taker_buy_vol"] - frame["taker_sell_vol"]) / total_taker.replace(0, np.nan)
    imbalance_std = frame["taker_buy_imbalance"].rolling(72).std().replace(0, np.nan)
    frame["taker_buy_imbalance_zscore"] = (
        frame["taker_buy_imbalance"] - frame["taker_buy_imbalance"].rolling(72).mean()
    ) / imbalance_std

    if "close" in frame.columns:
        close = frame["close"].replace(0, np.nan)
        price_return_1 = np.log(close / close.shift(1))
        price_return_24 = np.log(close / close.shift(24))
    else:
        price_return_1 = pd.Series(np.nan, index=frame.index)
        price_return_24 = pd.Series(np.nan, index=frame.index)
    if "volume" in frame.columns:
        volume = frame["volume"].replace(0, np.nan)
        volume_change_24 = np.log(volume / volume.shift(24))
    else:
        volume_change_24 = pd.Series(np.nan, index=frame.index)

    frame["futures_leverage_pressure"] = frame["open_interest_change_24"] * frame["funding_rate_zscore"]
    frame["futures_directional_pressure"] = frame["taker_buy_imbalance_zscore"] * frame["open_interest_change_6"]
    frame["futures_crowding_pressure"] = frame["funding_rate_zscore"] * frame["long_short_crowding"]
    frame["futures_squeeze_risk"] = frame["open_interest_zscore"] * (-frame["long_short_crowding"]) * price_return_1.clip(lower=0)
    frame["futures_pain_risk"] = frame["open_interest_zscore"] * frame["long_short_crowding"] * (-price_return_1.clip(upper=0))
    frame["oi_price_divergence_24"] = frame["open_interest_change_24"] - price_return_24
    frame["volume_oi_confirmation"] = volume_change_24 * frame["open_interest_change_24"]
    for column in DERIVED_FUNDAMENTAL_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def update_fundamental_file(
    market_data: str | Path,
    output: str | Path,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    start, end = load_market_span(market_data)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_fundamental_csv(output) if output.exists() else pd.DataFrame(columns=["timestamp", *FUNDAMENTAL_COLUMNS])
    fetch_start = start
    if not existing.empty and existing["timestamp"].notna().any():
        refresh_start = min(existing["timestamp"].max() - pd.to_timedelta(2, unit="D"), end - pd.to_timedelta(29, unit="D"))
        fetch_start = max(start, refresh_start)
    fetched = fetch_binance_fundamentals(symbol=symbol, timeframe=timeframe, start=fetch_start, end=end)
    combined = (
        pd.concat([existing, fetched], ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined = align_fundamentals_to_market(combined, load_market_timestamps(market_data))
    temp_output = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    combined.to_csv(temp_output, index=False)
    temp_output.replace(output)
    max_source_age = combined["fundamental_source_age_hours"].max() if "fundamental_source_age_hours" in combined else None
    print(
        f"Fundamental data updated: {output} rows={len(combined)} "
        f"last={combined['timestamp'].max() if not combined.empty else None} "
        f"max_source_age_hours={max_source_age}"
    )
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Binance Futures market/fundamental datasets.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--market-data", default=str(DATA_DIR / "1h-btc_history.csv"))
    parser.add_argument("--output", default=str(DATA_DIR / "1h-btc_fundamentals.csv"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    update_fundamental_file(
        market_data=args.market_data,
        output=args.output,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
