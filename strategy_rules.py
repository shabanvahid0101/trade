import numpy as np
import pandas as pd


def add_range_context(data: pd.DataFrame, lookback: int = 48) -> pd.DataFrame:
    frame = data[["timestamp", "open", "high", "low", "close"]].copy().sort_values("timestamp")
    close = frame["close"].replace(0, np.nan)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    range_low = frame["low"].rolling(lookback).min().shift(1)
    range_high = frame["high"].rolling(lookback).max().shift(1)
    range_span = (range_high - range_low).replace(0, np.nan)
    frame["atr_14_pct"] = true_range.rolling(14).mean() / close
    frame["trend_20_50"] = sma_20 / sma_50 - 1
    frame["trend_strength_pct"] = frame["trend_20_50"].abs()
    frame["range_low"] = range_low
    frame["range_high"] = range_high
    frame["range_position"] = (close - range_low) / range_span
    frame["range_width_pct"] = range_span / close
    return frame


def is_range_row(
    row,
    range_atr_max: float,
    range_trend_max: float,
    range_width_min: float,
    range_width_max: float,
) -> bool:
    values = [
        float(row.atr_14_pct),
        float(row.trend_strength_pct),
        float(row.range_width_pct),
        float(row.range_position),
    ]
    if any(np.isnan(value) for value in values):
        return False
    return (
        values[0] <= range_atr_max
        and values[1] <= range_trend_max
        and range_width_min <= values[2] <= range_width_max
        and 0 <= values[3] <= 1
    )


def range_side(row, range_lower: float, range_upper: float) -> int:
    position = float(row.range_position)
    if position <= range_lower:
        return 1
    if position >= range_upper:
        return -1
    return 0


def side_to_signal(side: int) -> str:
    if side == 1:
        return "LONG"
    if side == -1:
        return "SHORT"
    return "HOLD"


def signal_to_return(signal: str, threshold: float) -> float:
    if signal == "LONG":
        return threshold
    if signal == "SHORT":
        return -threshold
    return 0.0


def apply_hybrid_to_latest(
    final: dict,
    data: pd.DataFrame,
    strategy: str,
    threshold: float,
    range_lower: float,
    range_upper: float,
    range_atr_max: float,
    range_trend_max: float,
    range_width_min: float,
    range_width_max: float,
    range_lookback: int = 48,
) -> dict:
    if strategy == "model":
        return {**final, "strategy": "model", "market_regime": "model"}

    context = add_range_context(data, lookback=range_lookback).dropna(subset=["range_position", "range_width_pct"])
    if context.empty:
        return {**final, "strategy": strategy, "market_regime": "unknown"}

    row = context.iloc[-1]
    in_range = is_range_row(row, range_atr_max, range_trend_max, range_width_min, range_width_max)
    if strategy == "range" or (strategy == "hybrid" and in_range):
        side = range_side(row, range_lower, range_upper) if in_range else 0
        signal = side_to_signal(side)
        expected_return = signal_to_return(signal, threshold)
        return {
            **final,
            "signal": signal,
            "predicted_price": float(float(final["current_price"]) * (1 + expected_return)),
            "predicted_return_pct": float(expected_return * 100),
            "strategy": strategy,
            "market_regime": "range" if in_range else "not_range",
            "range_position": float(row.range_position),
            "range_low": float(row.range_low),
            "range_high": float(row.range_high),
        }

    return {
        **final,
        "strategy": strategy,
        "market_regime": "trend_or_unclear",
        "range_position": float(row.range_position),
        "range_low": float(row.range_low),
        "range_high": float(row.range_high),
    }


def apply_hybrid_to_returns(
    meta: pd.DataFrame,
    predicted_return: np.ndarray,
    data: pd.DataFrame,
    strategy: str,
    threshold: float,
    range_lower: float,
    range_upper: float,
    range_atr_max: float,
    range_trend_max: float,
    range_width_min: float,
    range_width_max: float,
    range_lookback: int = 48,
) -> np.ndarray:
    if strategy == "model":
        return predicted_return

    context = add_range_context(data, lookback=range_lookback)
    merged = meta[["timestamp"]].merge(context, on="timestamp", how="left")
    adjusted = predicted_return.copy()
    for idx, row in enumerate(merged.itertuples(index=False)):
        in_range = is_range_row(row, range_atr_max, range_trend_max, range_width_min, range_width_max)
        if strategy == "range" or (strategy == "hybrid" and in_range):
            side = range_side(row, range_lower, range_upper) if in_range else 0
            adjusted[idx] = signal_to_return(side_to_signal(side), threshold)
    return adjusted
