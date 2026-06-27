import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler

try:
    import ccxt
except ImportError:  # Allows offline evaluation on an existing CSV.
    ccxt = None

try:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam
except ImportError:
    EarlyStopping = None
    ReduceLROnPlateau = None
    LSTM = None
    Dense = None
    Dropout = None
    Input = None
    Huber = None
    Sequential = None
    load_model = None
    Adam = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "btc_lstm_model.keras"
ARTIFACT_PATH = BASE_DIR / "model_artifacts.pkl"
LEGACY_SCALER_PATH = BASE_DIR / "scaler.pkl"
LOG_PATH = BASE_DIR / "logging.log"
MODELS_DIR = BASE_DIR / "models"

CORE_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_3",
    "log_return_6",
    "log_return_12",
    "volume_change",
    "volume_zscore",
    "hl_pct",
    "oc_pct",
    "sma_20_dist",
    "sma_50_dist",
    "ema_20_dist",
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "bb_width",
    "bb_position",
    "atr_14_pct",
    "volatility_20",
    "fib_reaction",
]

ADVANCED_FEATURE_COLUMNS = [
    *CORE_FEATURE_COLUMNS,
    "log_return_24",
    "log_return_48",
    "roc_12",
    "roc_24",
    "momentum_10",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "cci_20",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "mfi_14",
    "obv_zscore",
    "vwap_48_dist",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_position",
    "trend_20_50",
    "trend_50_100",
    "volatility_ratio",
    "return_skew_20",
    "return_kurt_20",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

FEATURE_COLUMNS = ADVANCED_FEATURE_COLUMNS
FEATURE_SETS = {
    "core": CORE_FEATURE_COLUMNS,
    "advanced": ADVANCED_FEATURE_COLUMNS,
}


def timeframe_to_milliseconds(timeframe: str) -> int:
    units = {
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "w": 7 * 24 * 60 * 60 * 1000,
    }
    unit = timeframe[-1].lower()
    if unit not in units:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return int(timeframe[:-1]) * units[unit]


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be a comma-separated list of positive integers.")
    return horizons


def safe_symbol_name(symbol: str) -> str:
    return "".join(char for char in symbol.upper() if char.isalnum())


def horizon_model_paths(symbol: str, timeframe: str, horizon: int) -> tuple[Path, Path]:
    prefix = f"{safe_symbol_name(symbol)}_{timeframe}_h{horizon}"
    return MODELS_DIR / f"{prefix}.keras", MODELS_DIR / f"{prefix}_artifact.pkl"


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame
    feature_scaler: RobustScaler
    target_scaler: RobustScaler | None
    feature_columns: list[str]
    sequence_length: int
    horizon: int
    timeframe: str
    target_mode: str
    target_threshold: float
    feature_selection_report: dict


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        return
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


setup_logging()
load_dotenv(BASE_DIR / ".env")


def send_telegram_message(message: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    strict = os.getenv("TELEGRAM_STRICT", "0") == "1"
    if not token or not chat_id:
        error = "Telegram TOKEN or CHAT_ID is not configured."
        logging.warning(error)
        print(error)
        if strict:
            raise RuntimeError(error)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        print("Telegram message sent.")
        return True
    except Exception as exc:
        error = f"Telegram send failed: {exc}"
        logging.error(error)
        print(error)
        if strict:
            raise RuntimeError(error) from exc
        return False


def create_exchange(exchange_name: str):
    if ccxt is None:
        raise RuntimeError("ccxt is not installed. Install requirements or use --data with an existing CSV.")
    if not hasattr(ccxt, exchange_name):
        raise ValueError(f"Unsupported exchange: {exchange_name}")

    exchange_config = {"enableRateLimit": True}
    if exchange_name == "coinex" and os.getenv("Access_ID") and os.getenv("Secret_Key"):
        exchange_config.update({"apiKey": os.getenv("Access_ID"), "secret": os.getenv("Secret_Key")})
    return getattr(ccxt, exchange_name)(exchange_config)


def fetch_and_update_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    batch_limit: int = 1000,
    file: str | Path = DATA_DIR / "5m_btc_history.csv",
    retries: int = 3,
    max_batches: int = 200,
    repair_gaps: bool = True,
    exchange_name: str = "binance",
) -> pd.DataFrame:
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    exchange = create_exchange(exchange_name)

    old_df = pd.DataFrame()
    since = None
    if file.exists():
        old_df = load_price_csv(file)
        if repair_gaps and not old_df.empty:
            original_len = len(old_df)
            old_df = repair_history_gaps(
                exchange=exchange,
                df=old_df,
                symbol=symbol,
                timeframe=timeframe,
                batch_limit=batch_limit,
                retries=retries,
                max_batches=max_batches,
            )
            if len(old_df) != original_len:
                old_df.to_csv(file, index=False)
                logging.info("Repaired dataset gaps: %s -> %s rows saved to %s", original_len, len(old_df), file)
        if not old_df.empty:
            since = int(old_df["timestamp"].max().timestamp() * 1000) + 1

    new_data = []
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    for batch_number in range(max_batches):
        candles = []
        for attempt in range(1, retries + 1):
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
                break
            except Exception as exc:
                logging.error("Fetch retry %s/%s failed: %s", attempt, retries, exc)
                time.sleep(5)

        if not candles:
            break

        if since is not None:
            candles = [candle for candle in candles if candle[0] >= since]
        if not candles:
            break

        new_data.extend(candles)
        since = int(candles[-1][0]) + timeframe_ms
        logging.info("Fetched batch %s: %s candles for %s %s", batch_number + 1, len(candles), symbol, timeframe)

        if len(candles) < batch_limit:
            break
        time.sleep(exchange.rateLimit / 1000 if getattr(exchange, "rateLimit", None) else 0.2)

    if not new_data:
        return old_df

    new_df = pd.DataFrame(new_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=False)
    combined = (
        pd.concat([old_df, new_df], ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined.to_csv(file, index=False)
    logging.info("Dataset updated: %s rows saved to %s", len(combined), file)
    return combined


def fetch_ohlcv_batches(
    exchange,
    symbol: str,
    timeframe: str,
    since: int | None,
    batch_limit: int,
    retries: int,
    max_batches: int,
    until: int | None = None,
) -> list[list[float]]:
    new_data = []
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    cursor = since

    for batch_number in range(max_batches):
        candles = []
        for attempt in range(1, retries + 1):
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=batch_limit)
                break
            except Exception as exc:
                logging.error("Fetch retry %s/%s failed: %s", attempt, retries, exc)
                time.sleep(5)

        if not candles:
            break

        if cursor is not None:
            candles = [candle for candle in candles if candle[0] >= cursor]
        if until is not None:
            candles = [candle for candle in candles if candle[0] < until]
        if not candles:
            break

        new_data.extend(candles)
        cursor = int(candles[-1][0]) + timeframe_ms
        logging.info("Fetched batch %s: %s candles for %s %s", batch_number + 1, len(candles), symbol, timeframe)

        if len(candles) < batch_limit:
            break
        time.sleep(exchange.rateLimit / 1000 if getattr(exchange, "rateLimit", None) else 0.2)

    return new_data


def repair_history_gaps(
    exchange,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    batch_limit: int,
    retries: int,
    max_batches: int,
) -> pd.DataFrame:
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    expected_step = pd.to_timedelta(timeframe_ms, unit="ms")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    gaps = df["timestamp"].diff() > expected_step * 1.5
    if not gaps.any():
        return df

    repaired_batches = []
    for gap_idx in np.flatnonzero(gaps.to_numpy()):
        gap_start = df["timestamp"].iloc[gap_idx - 1]
        gap_end = df["timestamp"].iloc[gap_idx]
        since = int(gap_start.timestamp() * 1000) + timeframe_ms
        until = int(gap_end.timestamp() * 1000)
        logging.info("Repairing history gap from %s to %s", gap_start, gap_end)
        repaired_batches.extend(
            fetch_ohlcv_batches(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                batch_limit=batch_limit,
                retries=retries,
                max_batches=max_batches,
                until=until,
            )
        )

    if not repaired_batches:
        return df

    repair_df = pd.DataFrame(repaired_batches, columns=["timestamp", "open", "high", "low", "close", "volume"])
    repair_df["timestamp"] = pd.to_datetime(repair_df["timestamp"], unit="ms", utc=False)
    repaired = (
        pd.concat([df, repair_df], ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return repaired


def load_price_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} must contain a timestamp column.")

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[["timestamp", *required]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def latest_continuous_block(df: pd.DataFrame, timeframe: str = "5m") -> pd.DataFrame:
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    expected_step = pd.to_timedelta(timeframe_to_milliseconds(timeframe), unit="ms")
    gap_mask = df["timestamp"].diff() > expected_step * 1.5
    if not gap_mask.any():
        return df
    last_gap_idx = int(np.flatnonzero(gap_mask.to_numpy())[-1])
    return df.iloc[last_gap_idx:].reset_index(drop=True)


def add_features(df: pd.DataFrame, horizon: int = 1, require_target: bool = True) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    open_ = df["open"].replace(0, np.nan)

    for lag in (1, 3, 6, 12, 24, 48):
        df[f"log_return_{lag}"] = np.log(close / close.shift(lag))

    volume_mean = df["volume"].rolling(50).mean()
    volume_std = df["volume"].rolling(50).std().replace(0, np.nan)
    df["volume_change"] = np.log1p(df["volume"]) - np.log1p(df["volume"].shift(1))
    df["volume_zscore"] = (df["volume"] - volume_mean) / volume_std
    df["hl_pct"] = (high - low) / close
    df["oc_pct"] = (df["close"] - open_) / open_

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    df["sma_20_dist"] = close / sma_20 - 1
    df["sma_50_dist"] = close / sma_50 - 1
    df["ema_20_dist"] = close / ema_20 - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50) / 100

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_pct"] = macd / close
    df["macd_signal_pct"] = macd_signal / close

    bb_mid = sma_20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / close
    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14_pct"] = true_range.rolling(14).mean() / close
    df["volatility_20"] = df["log_return_1"].rolling(20).std()
    df["roc_12"] = close.pct_change(12)
    df["roc_24"] = close.pct_change(24)
    df["momentum_10"] = close / close.shift(10) - 1

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_range = (high_14 - low_14).replace(0, np.nan)
    df["stoch_k"] = (close - low_14) / stoch_range
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["williams_r"] = (high_14 - close) / stoch_range

    typical_price = (high + low + close) / 3
    cci_mean = typical_price.rolling(20).mean()
    cci_mad = typical_price.rolling(20).apply(lambda values: np.mean(np.abs(values - np.mean(values))), raw=True)
    df["cci_20"] = (typical_price - cci_mean) / (0.015 * cci_mad.replace(0, np.nan))

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_14 = true_range.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr_14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr_14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.rolling(14).mean() / 100
    df["plus_di_14"] = plus_di / 100
    df["minus_di_14"] = minus_di / 100

    money_flow = typical_price * df["volume"]
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0.0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0.0).rolling(14).sum()
    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    df["mfi_14"] = (100 - (100 / (1 + money_ratio))) / 100

    obv_direction = np.sign(close.diff()).fillna(0)
    obv = (obv_direction * df["volume"]).cumsum()
    obv_mean = obv.rolling(100).mean()
    obv_std = obv.rolling(100).std().replace(0, np.nan)
    df["obv_zscore"] = (obv - obv_mean) / obv_std

    rolling_vwap = (typical_price * df["volume"]).rolling(48).sum() / df["volume"].rolling(48).sum().replace(0, np.nan)
    df["vwap_48_dist"] = close / rolling_vwap - 1

    candle_range = (high - low).replace(0, np.nan)
    body = df["close"] - open_
    df["body_pct"] = body / open_
    df["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range
    df["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / candle_range
    df["close_position"] = (close - low) / candle_range

    sma_100 = close.rolling(100).mean()
    df["trend_20_50"] = sma_20 / sma_50 - 1
    df["trend_50_100"] = sma_50 / sma_100 - 1
    df["volatility_ratio"] = df["log_return_1"].rolling(12).std() / df["log_return_1"].rolling(48).std().replace(0, np.nan)
    df["return_skew_20"] = df["log_return_1"].rolling(20).skew()
    df["return_kurt_20"] = df["log_return_1"].rolling(20).kurt()

    timestamp = pd.to_datetime(df["timestamp"])
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    df["weekday_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)

    swing_high = high.rolling(48).max().shift(1)
    swing_low = low.rolling(48).min().shift(1)
    price_range = (swing_high - swing_low).replace(0, np.nan)
    fib_ratios = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
    levels = np.vstack([swing_low + price_range * ratio for ratio in fib_ratios]).T
    fib_distance = np.abs(levels - close.to_numpy()[:, None])
    valid_fib_rows = ~np.isnan(fib_distance).all(axis=1)
    nearest_fib_distance = np.full(len(df), np.nan)
    nearest_fib_distance[valid_fib_rows] = np.nanmin(fib_distance[valid_fib_rows], axis=1)
    df["fib_reaction"] = nearest_fib_distance / close

    df["future_close"] = close.shift(-horizon)
    df["target_return"] = df["future_close"] / close - 1
    df = df.replace([np.inf, -np.inf], np.nan)
    required_columns = FEATURE_COLUMNS + (["target_return", "future_close"] if require_target else [])
    return df.dropna(subset=required_columns).reset_index(drop=True)


def select_features_by_correlation(
    featured: pd.DataFrame,
    candidate_columns: list[str],
    sequence_length: int,
    train_end: int,
    max_features: int = 18,
    min_abs_corr: float = 0.005,
    max_pair_corr: float = 0.92,
    min_features: int = 8,
    method: str = "spearman",
) -> tuple[list[str], dict]:
    if method not in {"spearman", "pearson"}:
        raise ValueError("feature correlation method must be 'spearman' or 'pearson'.")
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    if min_features <= 0:
        raise ValueError("min_features must be positive.")

    target_start = sequence_length - 1
    target_stop = target_start + train_end
    train_frame = featured.iloc[target_start:target_stop].copy()
    if train_frame.empty:
        return candidate_columns, {"enabled": False, "reason": "empty_train_window"}

    target = train_frame["target_return"]
    scores = []
    for column in candidate_columns:
        series = train_frame[column]
        corr = series.corr(target, method=method)
        if pd.isna(corr) or np.isinf(corr):
            corr = 0.0
        scores.append({"feature": column, "correlation": float(corr), "abs_correlation": float(abs(corr))})

    ranked = sorted(scores, key=lambda item: item["abs_correlation"], reverse=True)
    strong = [item for item in ranked if item["abs_correlation"] >= min_abs_corr]
    if len(strong) < min_features:
        strong = ranked[:min_features]

    selected: list[str] = []
    skipped_redundant: list[dict] = []
    corr_frame = train_frame[candidate_columns].corr(method=method).abs()
    for item in strong:
        feature = item["feature"]
        if len(selected) >= max_features:
            break
        if selected:
            max_existing_corr = float(corr_frame.loc[feature, selected].max())
            if max_existing_corr > max_pair_corr and len(selected) >= min_features:
                skipped_redundant.append(
                    {
                        "feature": feature,
                        "abs_correlation": item["abs_correlation"],
                        "max_pair_correlation": max_existing_corr,
                    }
                )
                continue
        selected.append(feature)

    if len(selected) < min_features:
        for item in ranked:
            feature = item["feature"]
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= min(min_features, len(candidate_columns)):
                break

    selected_set = set(selected)
    report = {
        "enabled": True,
        "method": method,
        "candidate_count": len(candidate_columns),
        "selected_count": len(selected),
        "max_features": max_features,
        "min_abs_corr": min_abs_corr,
        "max_pair_corr": max_pair_corr,
        "top_ranked": ranked[:12],
        "selected": [item for item in ranked if item["feature"] in selected_set],
        "skipped_redundant": skipped_redundant[:12],
    }
    return selected, report


def prepare_datasets(
    df: pd.DataFrame,
    sequence_length: int = 96,
    horizon: int = 1,
    timeframe: str = "5m",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_rows: int | None = None,
    feature_columns: list[str] | None = None,
    target_mode: str = "regression",
    target_threshold: float = 0.0015,
    feature_selection: str = "none",
    max_selected_features: int = 18,
    min_feature_correlation: float = 0.005,
    max_feature_pair_correlation: float = 0.92,
    min_selected_features: int = 8,
    feature_correlation_method: str = "spearman",
) -> SplitData:
    if target_mode not in {"regression", "classification"}:
        raise ValueError("target_mode must be 'regression' or 'classification'.")
    if feature_selection not in {"none", "correlation"}:
        raise ValueError("feature_selection must be 'none' or 'correlation'.")
    feature_columns = feature_columns or FEATURE_COLUMNS
    df = latest_continuous_block(df, timeframe=timeframe)
    if max_rows is not None and max_rows > 0 and len(df) > max_rows + 250:
        df = df.tail(max_rows + 250).reset_index(drop=True)
    featured = add_features(df, horizon=horizon, require_target=True)
    if max_rows is not None and max_rows > 0 and len(featured) > max_rows:
        featured = featured.tail(max_rows).reset_index(drop=True)
    featured_gap_count = int((featured["timestamp"].diff() > pd.to_timedelta(timeframe_to_milliseconds(timeframe), unit="ms") * 1.5).sum())
    if featured_gap_count:
        raise ValueError(f"Training data still contains {featured_gap_count} timestamp gaps after repair.")
    if len(featured) < sequence_length + 100:
        raise ValueError("Not enough rows after feature engineering. Fetch more history before training.")

    if target_mode == "classification":
        target_return = featured["target_return"].to_numpy(dtype=np.float32)
        targets = np.where(target_return > target_threshold, 2, np.where(target_return < -target_threshold, 0, 1)).astype(np.int32)
    else:
        targets = featured["target_return"].to_numpy(dtype=np.float32)
    meta = featured[["timestamp", "close", "future_close", "target_return"]].copy()

    n_samples = len(featured) - sequence_length + 1
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n_samples:
        raise ValueError("Invalid split sizes. Adjust train_ratio/val_ratio.")

    if feature_selection == "correlation":
        feature_columns, feature_selection_report = select_features_by_correlation(
            featured=featured,
            candidate_columns=feature_columns,
            sequence_length=sequence_length,
            train_end=train_end,
            max_features=max_selected_features,
            min_abs_corr=min_feature_correlation,
            max_pair_corr=max_feature_pair_correlation,
            min_features=min_selected_features,
            method=feature_correlation_method,
        )
    else:
        feature_selection_report = {
            "enabled": False,
            "method": "none",
            "candidate_count": len(feature_columns),
            "selected_count": len(feature_columns),
            "selected": feature_columns,
        }

    features = featured[feature_columns].to_numpy(dtype=np.float32)

    X, y, rows = [], [], []
    for end_idx in range(sequence_length - 1, len(featured)):
        X.append(features[end_idx - sequence_length + 1 : end_idx + 1])
        y.append(targets[end_idx])
        rows.append(meta.iloc[end_idx])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    meta_all = pd.DataFrame(rows).reset_index(drop=True)

    X_train_raw, X_val_raw, X_test_raw = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train_raw, y_val_raw, y_test_raw = y[:train_end], y[train_end:val_end], y[val_end:]

    feature_scaler = RobustScaler()
    feature_scaler.fit(X_train_raw.reshape(-1, X.shape[-1]))

    def scale_x(values: np.ndarray) -> np.ndarray:
        scaled = feature_scaler.transform(values.reshape(-1, values.shape[-1]))
        return scaled.reshape(values.shape).astype(np.float32)

    X_train = scale_x(X_train_raw)
    X_val = scale_x(X_val_raw)
    X_test = scale_x(X_test_raw)
    if target_mode == "classification":
        target_scaler = None
        y_train = y_train_raw.astype(np.int32)
        y_val = y_val_raw.astype(np.int32)
        y_test = y_test_raw.astype(np.int32)
    else:
        target_scaler = RobustScaler()
        target_scaler.fit(y_train_raw.reshape(-1, 1))
        y_train = target_scaler.transform(y_train_raw.reshape(-1, 1)).ravel().astype(np.float32)
        y_val = target_scaler.transform(y_val_raw.reshape(-1, 1)).ravel().astype(np.float32)
        y_test = target_scaler.transform(y_test_raw.reshape(-1, 1)).ravel().astype(np.float32)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        meta_train=meta_all.iloc[:train_end].reset_index(drop=True),
        meta_val=meta_all.iloc[train_end:val_end].reset_index(drop=True),
        meta_test=meta_all.iloc[val_end:].reset_index(drop=True),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
        timeframe=timeframe,
        target_mode=target_mode,
        target_threshold=target_threshold,
        feature_selection_report=feature_selection_report,
    )


def require_tensorflow() -> None:
    if Sequential is None:
        raise RuntimeError("TensorFlow is not installed. Install project requirements before training or predicting.")


def build_model(sequence_length: int, n_features: int, target_mode: str = "regression") -> Sequential:
    require_tensorflow()
    output_layer = Dense(3, activation="softmax") if target_mode == "classification" else Dense(1)
    model = Sequential(
        [
            Input(shape=(sequence_length, n_features)),
            LSTM(96, return_sequences=True),
            Dropout(0.25),
            LSTM(48),
            Dropout(0.20),
            Dense(32, activation="relu"),
            output_layer,
        ]
    )
    if target_mode == "classification":
        model.compile(optimizer=Adam(learning_rate=0.0005), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber(delta=0.01), metrics=["mae"])
    return model


def train_model(split: SplitData, epochs: int = 80, batch_size: int = 32, verbose: int = 1):
    model = build_model(split.sequence_length, split.X_train.shape[-1], split.target_mode)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=verbose),
    ]
    fit_kwargs = {}
    if split.target_mode == "classification":
        classes = np.array([0, 1, 2])
        present_classes = np.unique(split.y_train)
        weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=split.y_train)
        class_weight = {int(label): float(weight) for label, weight in zip(present_classes, weights)}
        for label in classes:
            class_weight.setdefault(int(label), 1.0)
        fit_kwargs["class_weight"] = class_weight
    history = model.fit(
        split.X_train,
        split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        **fit_kwargs,
    )
    return model, history


def inverse_target(values: np.ndarray, target_scaler: RobustScaler) -> np.ndarray:
    return target_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def evaluate_predictions(actual_return: np.ndarray, predicted_return: np.ndarray, current_close: np.ndarray) -> dict:
    actual_price = current_close * (1 + actual_return)
    predicted_price = current_close * (1 + predicted_return)
    naive_price = current_close
    mae = mean_absolute_error(actual_price, predicted_price)
    rmse = float(np.sqrt(mean_squared_error(actual_price, predicted_price)))
    naive_mae = mean_absolute_error(actual_price, naive_price)
    direction_accuracy = float(np.mean(np.sign(actual_return) == np.sign(predicted_return)))
    mape = float(np.mean(np.abs((actual_price - predicted_price) / actual_price)) * 100)

    return {
        "mae_usd": float(mae),
        "rmse_usd": rmse,
        "mape_pct": mape,
        "naive_mae_usd": float(naive_mae),
        "mae_vs_naive_pct": float((1 - mae / naive_mae) * 100) if naive_mae else 0.0,
        "direction_accuracy_pct": direction_accuracy * 100,
    }


def evaluate_model(model, split: SplitData) -> dict:
    if split.target_mode == "classification":
        probabilities = model.predict(split.X_test, verbose=0)
        predicted_class = probabilities.argmax(axis=1)
        actual_class = split.y_test.astype(np.int32)
        accuracy = float(np.mean(predicted_class == actual_class) * 100)
        actionable = predicted_class != 1
        actual_actionable = actual_class != 1
        directional_when_actionable = (
            float(np.mean(predicted_class[actionable] == actual_class[actionable]) * 100) if actionable.any() else 0.0
        )
        long_mask = predicted_class == 2
        short_mask = predicted_class == 0
        metrics = {
            "target_mode": "classification",
            "target_threshold_pct": split.target_threshold * 100,
            "class_accuracy_pct": accuracy,
            "actionable_accuracy_pct": directional_when_actionable,
            "action_rate_pct": float(np.mean(actionable) * 100),
            "actual_action_rate_pct": float(np.mean(actual_actionable) * 100),
            "long_signal_count": int(long_mask.sum()),
            "short_signal_count": int(short_mask.sum()),
            "hold_signal_count": int((predicted_class == 1).sum()),
            "long_precision_pct": float(np.mean(actual_class[long_mask] == 2) * 100) if long_mask.any() else 0.0,
            "short_precision_pct": float(np.mean(actual_class[short_mask] == 0) * 100) if short_mask.any() else 0.0,
        }
        logging.info("Classification evaluation metrics: %s", metrics)
        return metrics

    predicted_scaled = model.predict(split.X_test, verbose=0).ravel()
    predicted_return = inverse_target(predicted_scaled, split.target_scaler)
    actual_return = split.meta_test["target_return"].to_numpy()
    current_close = split.meta_test["close"].to_numpy()
    metrics = evaluate_predictions(actual_return, predicted_return, current_close)
    logging.info("Evaluation metrics: %s", metrics)
    return metrics


def class_predictions_to_signal_returns(predicted_class: np.ndarray, threshold: float) -> np.ndarray:
    signal_returns = np.zeros(len(predicted_class), dtype=np.float32)
    signal_returns[predicted_class == 2] = threshold * 1.01
    signal_returns[predicted_class == 0] = -threshold * 1.01
    return signal_returns


def backtest_predictions(
    meta: pd.DataFrame,
    predicted_return: np.ndarray,
    threshold: float = 0.0015,
    fee_rate: float = 0.001,
    initial_capital: float = 10_000.0,
    market_mode: str = "futures",
    leverage: float = 1.0,
) -> dict:
    if market_mode not in {"spot", "futures"}:
        raise ValueError("market_mode must be 'spot' or 'futures'.")
    if leverage <= 0:
        raise ValueError("leverage must be positive.")
    if market_mode == "spot":
        return backtest_spot_long_only(meta, predicted_return, threshold, fee_rate, initial_capital)

    return backtest_futures_long_short(meta, predicted_return, threshold, fee_rate, initial_capital, leverage)


def backtest_spot_long_only(
    meta: pd.DataFrame,
    predicted_return: np.ndarray,
    threshold: float,
    fee_rate: float,
    initial_capital: float,
) -> dict:
    capital = initial_capital
    units = 0.0
    equity_curve = []
    trades = []

    for row, pred_ret in zip(meta.itertuples(index=False), predicted_return):
        price = float(row.close)
        future_price = float(row.future_close)
        signal = 1 if pred_ret > threshold else -1 if pred_ret < -threshold else 0

        if signal == 1 and units == 0:
            units = (capital * (1 - fee_rate)) / price
            trades.append({"timestamp": str(row.timestamp), "side": "BUY", "price": price})
            capital = 0.0
        elif signal == -1 and units > 0:
            capital = units * price * (1 - fee_rate)
            trades.append({"timestamp": str(row.timestamp), "side": "SELL", "price": price})
            units = 0.0

        equity = capital + units * future_price
        equity_curve.append(equity)

    if units > 0:
        final_price = float(meta["future_close"].iloc[-1])
        capital = units * final_price * (1 - fee_rate)
        trades.append({"timestamp": str(meta["timestamp"].iloc[-1]), "side": "FINAL_SELL", "price": final_price})
    else:
        capital = capital if capital else equity_curve[-1]

    returns = pd.Series(equity_curve).pct_change().dropna()
    max_drawdown = 0.0
    if equity_curve:
        equity_series = pd.Series(equity_curve)
        drawdown = equity_series / equity_series.cummax() - 1
        max_drawdown = float(drawdown.min() * 100)

    return {
        "initial_capital": initial_capital,
        "final_capital": float(capital),
        "total_return_pct": float((capital / initial_capital - 1) * 100),
        "max_drawdown_pct": max_drawdown,
        "trade_count": len(trades),
        "sharpe_like": float((returns.mean() / returns.std()) * np.sqrt(365 * 24 * 12)) if len(returns) > 2 and returns.std() else 0.0,
        "last_trades": trades[-10:],
    }


def backtest_futures_long_short(
    meta: pd.DataFrame,
    predicted_return: np.ndarray,
    threshold: float,
    fee_rate: float,
    initial_capital: float,
    leverage: float,
) -> dict:
    capital = initial_capital
    position = 0
    entry_price = 0.0
    notional = 0.0
    equity_curve = []
    trades = []

    def unrealized_pnl(mark_price: float) -> float:
        if position == 0:
            return 0.0
        return notional * position * ((mark_price - entry_price) / entry_price)

    def close_position(timestamp, price: float, reason: str) -> None:
        nonlocal capital, position, entry_price, notional
        if position == 0:
            return
        pnl = unrealized_pnl(price)
        close_fee = notional * fee_rate
        capital += pnl - close_fee
        trades.append(
            {
                "timestamp": str(timestamp),
                "side": "CLOSE_LONG" if position == 1 else "CLOSE_SHORT",
                "price": price,
                "pnl": float(pnl - close_fee),
                "reason": reason,
            }
        )
        position = 0
        entry_price = 0.0
        notional = 0.0

    def open_position(timestamp, side: int, price: float) -> None:
        nonlocal capital, position, entry_price, notional
        notional = capital * leverage
        open_fee = notional * fee_rate
        capital -= open_fee
        position = side
        entry_price = price
        trades.append(
            {
                "timestamp": str(timestamp),
                "side": "OPEN_LONG" if side == 1 else "OPEN_SHORT",
                "price": price,
                "notional": float(notional),
                "fee": float(open_fee),
            }
        )

    for row, pred_ret in zip(meta.itertuples(index=False), predicted_return):
        price = float(row.close)
        future_price = float(row.future_close)
        signal = 1 if pred_ret > threshold else -1 if pred_ret < -threshold else 0

        if signal == 0 and position != 0:
            close_position(row.timestamp, price, "signal_flat")
        elif signal != 0 and position != signal:
            close_position(row.timestamp, price, "signal_flip")
            if capital > 0:
                open_position(row.timestamp, signal, price)
        elif signal != 0 and position == 0 and capital > 0:
            open_position(row.timestamp, signal, price)

        equity_curve.append(capital + unrealized_pnl(future_price))

    if position != 0:
        close_position(meta["timestamp"].iloc[-1], float(meta["future_close"].iloc[-1]), "final_close")

    returns = pd.Series(equity_curve).pct_change().dropna()
    max_drawdown = 0.0
    if equity_curve:
        equity_series = pd.Series(equity_curve)
        drawdown = equity_series / equity_series.cummax() - 1
        max_drawdown = float(drawdown.min() * 100)

    return {
        "market_mode": "futures",
        "leverage": leverage,
        "initial_capital": initial_capital,
        "final_capital": float(capital),
        "total_return_pct": float((capital / initial_capital - 1) * 100),
        "max_drawdown_pct": max_drawdown,
        "trade_count": len(trades),
        "sharpe_like": float((returns.mean() / returns.std()) * np.sqrt(365 * 24 * 12)) if len(returns) > 2 and returns.std() else 0.0,
        "last_trades": trades[-10:],
    }


def save_artifacts(split: SplitData, metrics: dict, path: str | Path = ARTIFACT_PATH) -> None:
    if split.feature_columns == ADVANCED_FEATURE_COLUMNS:
        feature_set = "advanced"
    elif split.feature_columns == CORE_FEATURE_COLUMNS:
        feature_set = "core"
    else:
        feature_set = "selected"
    artifact = {
        "feature_scaler": split.feature_scaler,
        "target_scaler": split.target_scaler,
        "feature_columns": split.feature_columns,
        "feature_set": feature_set,
        "feature_selection": split.feature_selection_report,
        "sequence_length": split.sequence_length,
        "horizon": split.horizon,
        "timeframe": split.timeframe,
        "target_mode": split.target_mode,
        "target_threshold": split.target_threshold,
        "metrics": metrics,
        "created_at": pd.Timestamp.now("UTC").isoformat(),
    }
    joblib.dump(artifact, path)
    joblib.dump(split.feature_scaler, LEGACY_SCALER_PATH)


def load_artifacts(path: str | Path = ARTIFACT_PATH) -> dict:
    return joblib.load(path)


def predict_next_price(model, data: pd.DataFrame, artifact: dict) -> dict:
    sequence_length = int(artifact["sequence_length"])
    horizon = int(artifact["horizon"])
    timeframe = artifact.get("timeframe", "5m")
    data = latest_continuous_block(data, timeframe=timeframe)
    featured = add_features(data, horizon=horizon, require_target=False)
    if len(featured) < sequence_length:
        raise ValueError("Not enough processed rows for prediction.")

    feature_columns = artifact["feature_columns"]
    raw_sequence = featured[feature_columns].tail(sequence_length).to_numpy(dtype=np.float32)
    scaled_sequence = artifact["feature_scaler"].transform(raw_sequence).reshape(1, sequence_length, len(feature_columns))
    current_price = float(featured["close"].iloc[-1])

    if artifact.get("target_mode") == "classification":
        probabilities = model.predict(scaled_sequence, verbose=0).ravel()
        predicted_class = int(probabilities.argmax())
        class_names = {0: "SHORT", 1: "HOLD", 2: "LONG"}
        confidence = float(probabilities[predicted_class])
        min_confidence = float(artifact.get("metrics", {}).get("min_confidence", 0.50))
        signal = class_names[predicted_class] if confidence >= min_confidence else "HOLD"
        expected_return = 0.0
        if signal == "LONG":
            expected_return = float(artifact.get("target_threshold", 0.0015))
        elif signal == "SHORT":
            expected_return = -float(artifact.get("target_threshold", 0.0015))
        return {
            "timestamp": str(featured["timestamp"].iloc[-1]),
            "current_price": current_price,
            "predicted_price": float(current_price * (1 + expected_return)),
            "predicted_return_pct": float(expected_return * 100),
            "confidence": confidence,
            "signal": signal,
            "class_probabilities": {
                "SHORT": float(probabilities[0]),
                "HOLD": float(probabilities[1]),
                "LONG": float(probabilities[2]),
            },
            "horizon_candles": horizon,
        }

    pred_scaled = model.predict(scaled_sequence, verbose=0).ravel()[0]
    pred_return = inverse_target(np.array([pred_scaled]), artifact["target_scaler"])[0]
    predicted_price = current_price * (1 + pred_return)

    confidence_floor = max(artifact.get("metrics", {}).get("mae_usd", 0.0) / current_price, 0.0005)
    confidence = min(abs(pred_return) / confidence_floor, 3.0) / 3.0
    signal = "LONG" if pred_return > confidence_floor else "SHORT" if pred_return < -confidence_floor else "HOLD"

    return {
        "timestamp": str(featured["timestamp"].iloc[-1]),
        "current_price": current_price,
        "predicted_price": float(predicted_price),
        "predicted_return_pct": float(pred_return * 100),
        "confidence": float(confidence),
        "signal": signal,
        "horizon_candles": horizon,
    }


def run_training_pipeline(
    args: argparse.Namespace,
    data: pd.DataFrame,
    horizon: int,
    model_path: str | Path,
    artifact_path: str | Path,
) -> dict:
    split = prepare_datasets(
        data,
        sequence_length=args.sequence_length,
        horizon=horizon,
        timeframe=args.timeframe,
        max_rows=args.max_train_rows,
        feature_columns=FEATURE_SETS[args.feature_set],
        target_mode=args.target_mode,
        target_threshold=args.threshold,
        feature_selection=args.feature_selection,
        max_selected_features=args.max_selected_features,
        min_feature_correlation=args.min_feature_correlation,
        max_feature_pair_correlation=args.max_feature_pair_correlation,
        min_selected_features=args.min_selected_features,
        feature_correlation_method=args.feature_correlation_method,
    )
    model, _ = train_model(split, epochs=args.epochs, batch_size=args.batch_size, verbose=args.training_verbose)
    metrics = evaluate_model(model, split)
    metrics["min_confidence"] = args.min_confidence
    metrics["selected_feature_count"] = len(split.feature_columns)
    metrics["selected_features"] = split.feature_columns
    metrics["feature_selection"] = split.feature_selection_report

    if split.target_mode == "classification":
        probabilities = model.predict(split.X_test, verbose=0)
        predicted_class = probabilities.argmax(axis=1)
        predicted_class[probabilities.max(axis=1) < args.min_confidence] = 1
        predicted_return = class_predictions_to_signal_returns(predicted_class, args.threshold)
    else:
        pred_scaled = model.predict(split.X_test, verbose=0).ravel()
        predicted_return = inverse_target(pred_scaled, split.target_scaler)
    backtest = backtest_predictions(
        split.meta_test,
        predicted_return,
        threshold=args.threshold,
        fee_rate=args.fee_rate,
        market_mode=args.market_mode,
        leverage=args.leverage,
    )

    model_path = Path(model_path)
    artifact_path = Path(artifact_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    save_artifacts(split, {**metrics, "backtest": backtest}, path=artifact_path)

    return {
        "horizon": horizon,
        "model_path": str(model_path),
        "artifact_path": str(artifact_path),
        "metrics": metrics,
        "backtest": backtest,
    }


def load_training_data(args: argparse.Namespace) -> pd.DataFrame:
    return (
        fetch_and_update_data(
            args.symbol,
            args.timeframe,
            file=args.data,
            max_batches=args.max_fetch_batches,
            exchange_name=args.exchange,
        )
        if args.update
        else load_price_csv(args.data)
    )


def train_command(args: argparse.Namespace) -> None:
    data = load_training_data(args)
    result = run_training_pipeline(args, data, args.horizon, MODEL_PATH, ARTIFACT_PATH)

    print(json.dumps({"metrics": result["metrics"], "backtest": result["backtest"]}, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved artifacts: {ARTIFACT_PATH}")


def train_multi_command(args: argparse.Namespace) -> None:
    data = load_training_data(args)
    results = []
    for horizon in parse_horizons(args.horizons):
        model_path, artifact_path = horizon_model_paths(args.symbol, args.timeframe, horizon)
        print(f"Training horizon {horizon} -> {model_path}")
        results.append(run_training_pipeline(args, data, horizon, model_path, artifact_path))
    print(json.dumps({"results": results}, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    require_tensorflow()
    data = (
        fetch_and_update_data(
            args.symbol,
            args.timeframe,
            file=args.data,
            max_batches=args.max_fetch_batches,
            exchange_name=args.exchange,
        )
        if args.update
        else load_price_csv(args.data)
    )
    model = load_model(MODEL_PATH)
    artifact = load_artifacts()
    artifact.setdefault("timeframe", args.timeframe)
    result = predict_next_price(model, data, artifact)
    print(json.dumps(result, indent=2))
    if args.telegram:
        send_telegram_message(
            f"<b>{result['signal']}</b> {args.symbol}\n"
            f"Current: ${result['current_price']:.2f}\n"
            f"Predicted: ${result['predicted_price']:.2f}\n"
            f"Expected: {result['predicted_return_pct']:.3f}%\n"
            f"Confidence: {result['confidence']:.2f}"
        )


def combine_multi_horizon_predictions(results: list[dict], min_agree: int, min_confidence: float) -> dict:
    if not results:
        raise ValueError("No horizon predictions were provided.")
    actionable = [result for result in results if result["signal"] in {"LONG", "SHORT"} and result["confidence"] >= min_confidence]
    long_votes = [result for result in actionable if result["signal"] == "LONG"]
    short_votes = [result for result in actionable if result["signal"] == "SHORT"]

    if len(long_votes) >= min_agree and len(short_votes) == 0:
        signal = "LONG"
        voters = long_votes
    elif len(short_votes) >= min_agree and len(long_votes) == 0:
        signal = "SHORT"
        voters = short_votes
    else:
        signal = "HOLD"
        voters = actionable

    current_price = float(results[0]["current_price"])
    expected_return = float(np.mean([result["predicted_return_pct"] for result in voters]) / 100) if voters and signal != "HOLD" else 0.0
    confidence = float(np.mean([result["confidence"] for result in voters])) if voters else float(np.mean([result["confidence"] for result in results]))
    return {
        "timestamp": results[0]["timestamp"],
        "current_price": current_price,
        "signal": signal,
        "predicted_price": float(current_price * (1 + expected_return)),
        "predicted_return_pct": float(expected_return * 100),
        "confidence": confidence,
        "min_agree": min_agree,
        "min_confidence": min_confidence,
        "long_votes": len(long_votes),
        "short_votes": len(short_votes),
        "hold_votes": int(sum(1 for result in results if result["signal"] == "HOLD")),
    }


def predict_multi_command(args: argparse.Namespace) -> None:
    require_tensorflow()
    data = (
        fetch_and_update_data(
            args.symbol,
            args.timeframe,
            file=args.data,
            max_batches=args.max_fetch_batches,
            exchange_name=args.exchange,
        )
        if args.update
        else load_price_csv(args.data)
    )
    results = []
    for horizon in parse_horizons(args.horizons):
        model_path, artifact_path = horizon_model_paths(args.symbol, args.timeframe, horizon)
        if not model_path.exists() or not artifact_path.exists():
            raise FileNotFoundError(f"Missing model/artifact for horizon {horizon}. Run train-multi first.")
        model = load_model(model_path)
        artifact = load_artifacts(artifact_path)
        artifact.setdefault("timeframe", args.timeframe)
        results.append(predict_next_price(model, data, artifact))

    final_signal = combine_multi_horizon_predictions(results, args.min_agree, args.min_confidence)
    output = {"final": final_signal, "horizons": results}
    print(json.dumps(output, indent=2))
    if args.telegram:
        send_telegram_message(
            f"<b>{final_signal['signal']}</b> {args.symbol} multi-horizon\n"
            f"Current: ${final_signal['current_price']:.2f}\n"
            f"Expected: {final_signal['predicted_return_pct']:.3f}%\n"
            f"Confidence: {final_signal['confidence']:.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crypto price forecasting pipeline.")
    parser.add_argument("--data", default=str(DATA_DIR / "5m_btc_history.csv"))
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--exchange", default="binance")

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_train_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--data", default=str(DATA_DIR / "5m_btc_history.csv"))
        command_parser.add_argument("--symbol", default="BTC/USDT")
        command_parser.add_argument("--timeframe", default="5m")
        command_parser.add_argument("--exchange", default="binance")
        command_parser.add_argument("--update", action="store_true", help="Fetch fresh candles before training.")
        command_parser.add_argument("--sequence-length", type=int, default=96)
        command_parser.add_argument("--epochs", type=int, default=80)
        command_parser.add_argument("--batch-size", type=int, default=32)
        command_parser.add_argument("--training-verbose", type=int, choices=[0, 1, 2], default=2)
        command_parser.add_argument("--threshold", type=float, default=0.0015)
        command_parser.add_argument("--fee-rate", type=float, default=0.001)
        command_parser.add_argument("--market-mode", choices=["spot", "futures"], default="futures")
        command_parser.add_argument("--leverage", type=float, default=1.0)
        command_parser.add_argument("--max-fetch-batches", type=int, default=200)
        command_parser.add_argument("--max-train-rows", type=int, default=5000)
        command_parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="core")
        command_parser.add_argument("--feature-selection", choices=["none", "correlation"], default="correlation")
        command_parser.add_argument("--feature-correlation-method", choices=["spearman", "pearson"], default="spearman")
        command_parser.add_argument("--max-selected-features", type=int, default=18)
        command_parser.add_argument("--min-selected-features", type=int, default=8)
        command_parser.add_argument("--min-feature-correlation", type=float, default=0.005)
        command_parser.add_argument("--max-feature-pair-correlation", type=float, default=0.92)
        command_parser.add_argument("--target-mode", choices=["regression", "classification"], default="classification")
        command_parser.add_argument("--min-confidence", type=float, default=0.50)

    train = subparsers.add_parser("train")
    add_train_options(train)
    train.add_argument("--horizon", type=int, default=1)
    train.set_defaults(func=train_command)

    train_multi = subparsers.add_parser("train-multi")
    add_train_options(train_multi)
    train_multi.add_argument("--horizons", default="1,3,6,12")
    train_multi.set_defaults(func=train_multi_command)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--data", default=str(DATA_DIR / "5m_btc_history.csv"))
    predict.add_argument("--symbol", default="BTC/USDT")
    predict.add_argument("--timeframe", default="5m")
    predict.add_argument("--exchange", default="binance")
    predict.add_argument("--update", action="store_true", help="Fetch fresh candles before prediction.")
    predict.add_argument("--max-fetch-batches", type=int, default=200)
    predict.add_argument("--telegram", action="store_true")
    predict.set_defaults(func=predict_command)

    predict_multi = subparsers.add_parser("predict-multi")
    predict_multi.add_argument("--data", default=str(DATA_DIR / "5m_btc_history.csv"))
    predict_multi.add_argument("--symbol", default="BTC/USDT")
    predict_multi.add_argument("--timeframe", default="5m")
    predict_multi.add_argument("--exchange", default="binance")
    predict_multi.add_argument("--update", action="store_true", help="Fetch fresh candles before prediction.")
    predict_multi.add_argument("--max-fetch-batches", type=int, default=200)
    predict_multi.add_argument("--horizons", default="1,3,6,12")
    predict_multi.add_argument("--min-agree", type=int, default=2)
    predict_multi.add_argument("--min-confidence", type=float, default=0.50)
    predict_multi.add_argument("--telegram", action="store_true")
    predict_multi.set_defaults(func=predict_multi_command)
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)
