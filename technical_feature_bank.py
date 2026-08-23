from __future__ import annotations

import numpy as np
import pandas as pd


TA_WIDE_FEATURE_COLUMNS = [
    "ta_ema_8_dist",
    "ta_ema_21_dist",
    "ta_ema_100_dist",
    "ta_sma_200_dist",
    "ta_ema_8_21_spread",
    "ta_ema_21_100_spread",
    "ta_ema_8_slope_3",
    "ta_ema_21_slope_6",
    "ta_sma_200_slope_12",
    "ta_hma_20_dist",
    "ta_hma_20_slope_3",
    "ta_aroon_up_25",
    "ta_aroon_down_25",
    "ta_aroon_osc_25",
    "ta_ichimoku_conversion_dist",
    "ta_ichimoku_base_dist",
    "ta_ichimoku_cloud_width",
    "ta_ichimoku_cloud_position",
    "ta_ichimoku_bull_flag",
    "ta_donchian_20_width",
    "ta_donchian_20_position",
    "ta_donchian_55_width",
    "ta_donchian_55_position",
    "ta_keltner_20_width",
    "ta_keltner_20_position",
    "ta_bollinger_keltner_squeeze",
    "ta_supertrend_10_3_dir",
    "ta_supertrend_10_3_dist",
    "ta_chandelier_long_dist",
    "ta_chandelier_short_dist",
    "ta_atr_14_zscore_100",
    "ta_parkinson_vol_20",
    "ta_garman_klass_vol_20",
    "ta_ppo",
    "ta_ppo_signal",
    "ta_ppo_hist",
    "ta_stoch_rsi_k",
    "ta_stoch_rsi_d",
    "ta_tsi",
    "ta_tsi_signal",
    "ta_ultimate_osc",
    "ta_fisher_10",
    "ta_fisher_signal_10",
    "ta_rsi_7",
    "ta_rsi_21",
    "ta_rsi_7_21_spread",
    "ta_cmf_20",
    "ta_chaikin_osc",
    "ta_eom_14",
    "ta_force_index_13_zscore",
    "ta_vpt_zscore_100",
    "ta_volume_price_corr_20",
    "ta_volume_regime_20_100",
    "ta_candle_body_share",
    "ta_candle_upper_lower_wick_ratio",
    "ta_candle_doji",
    "ta_candle_hammer",
    "ta_candle_shooting_star",
    "ta_candle_bullish_engulfing",
    "ta_candle_bearish_engulfing",
    "ta_candle_morning_star",
    "ta_candle_evening_star",
    "ta_candle_three_bar_reversal_up",
    "ta_candle_three_bar_reversal_down",
    "ta_gap_pct",
    "ta_gap_zscore_50",
    "ta_liquidity_sweep_high_20",
    "ta_liquidity_sweep_low_20",
]


def _safe_div(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(numerator).div(pd.Series(denominator).replace(0, np.nan))


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50) / 100


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda values: float(np.dot(values, weights) / weights.sum()), raw=True)


def _hma(series: pd.Series, period: int) -> pd.Series:
    half = max(int(period / 2), 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    return _wma(2 * _wma(series, half) - _wma(series, period), sqrt_period)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)


def _aroon(high: pd.Series, low: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    def up_position(values: np.ndarray) -> float:
        return 1 - ((len(values) - 1 - int(np.argmax(values))) / (len(values) - 1))

    def down_position(values: np.ndarray) -> float:
        return 1 - ((len(values) - 1 - int(np.argmin(values))) / (len(values) - 1))

    return high.rolling(period).apply(up_position, raw=True), low.rolling(period).apply(down_position, raw=True)


def _supertrend(close: pd.Series, high: pd.Series, low: pd.Series, atr: pd.Series, multiplier: float) -> tuple[pd.Series, pd.Series]:
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    direction = np.full(len(close), np.nan)
    trend_line = np.full(len(close), np.nan)

    for idx in range(len(close)):
        if np.isnan(atr.iloc[idx]):
            continue
        if idx == 0 or np.isnan(direction[idx - 1]):
            direction[idx] = 1.0
            trend_line[idx] = lower.iloc[idx]
            continue

        prev_line = trend_line[idx - 1]
        prev_direction = direction[idx - 1]
        if close.iloc[idx] > prev_line:
            direction[idx] = 1.0
        elif close.iloc[idx] < prev_line:
            direction[idx] = -1.0
        else:
            direction[idx] = prev_direction

        if direction[idx] > 0:
            trend_line[idx] = max(lower.iloc[idx], prev_line if prev_direction > 0 else lower.iloc[idx])
        else:
            trend_line[idx] = min(upper.iloc[idx], prev_line if prev_direction < 0 else upper.iloc[idx])

    return pd.Series(direction, index=close.index), pd.Series(trend_line, index=close.index)


def add_ta_wide_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a broad, leakage-safe technical-analysis feature bank.

    Features only use current and prior completed candle data. The model's
    existing scaler/feature-selection layer decides which of these candidates
    are useful for a specific symbol/timeframe/horizon.
    """
    df = df.copy()
    close = df["close"].replace(0, np.nan)
    open_ = df["open"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_100 = close.ewm(span=100, adjust=False).mean()
    sma_200 = close.rolling(200).mean()
    hma_20 = _hma(close, 20)
    df["ta_ema_8_dist"] = close / ema_8 - 1
    df["ta_ema_21_dist"] = close / ema_21 - 1
    df["ta_ema_100_dist"] = close / ema_100 - 1
    df["ta_sma_200_dist"] = close / sma_200 - 1
    df["ta_ema_8_21_spread"] = ema_8 / ema_21 - 1
    df["ta_ema_21_100_spread"] = ema_21 / ema_100 - 1
    df["ta_ema_8_slope_3"] = ema_8 / ema_8.shift(3) - 1
    df["ta_ema_21_slope_6"] = ema_21 / ema_21.shift(6) - 1
    df["ta_sma_200_slope_12"] = sma_200 / sma_200.shift(12) - 1
    df["ta_hma_20_dist"] = close / hma_20 - 1
    df["ta_hma_20_slope_3"] = hma_20 / hma_20.shift(3) - 1

    aroon_up, aroon_down = _aroon(high, low, 25)
    df["ta_aroon_up_25"] = aroon_up
    df["ta_aroon_down_25"] = aroon_down
    df["ta_aroon_osc_25"] = aroon_up - aroon_down

    conv_high = high.rolling(9).max()
    conv_low = low.rolling(9).min()
    base_high = high.rolling(26).max()
    base_low = low.rolling(26).min()
    span_high = high.rolling(52).max()
    span_low = low.rolling(52).min()
    conversion = (conv_high + conv_low) / 2
    base = (base_high + base_low) / 2
    span_a = (conversion + base) / 2
    span_b = (span_high + span_low) / 2
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    cloud_range = (cloud_top - cloud_bottom).replace(0, np.nan)
    df["ta_ichimoku_conversion_dist"] = close / conversion - 1
    df["ta_ichimoku_base_dist"] = close / base - 1
    df["ta_ichimoku_cloud_width"] = cloud_range / close
    df["ta_ichimoku_cloud_position"] = ((close - cloud_bottom) / cloud_range).clip(0, 1)
    df["ta_ichimoku_bull_flag"] = ((close > cloud_top) & (conversion > base)).astype(float)

    true_range = _true_range(high, low, close)
    atr_10 = true_range.rolling(10).mean()
    atr_14 = true_range.rolling(14).mean()
    atr_20 = true_range.rolling(20).mean()
    for period in (20, 55):
        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        width = (upper - lower).replace(0, np.nan)
        df[f"ta_donchian_{period}_width"] = width / close
        df[f"ta_donchian_{period}_position"] = ((close - lower) / width).clip(0, 1)

    keltner_mid = close.ewm(span=20, adjust=False).mean()
    keltner_upper = keltner_mid + 2 * atr_20
    keltner_lower = keltner_mid - 2 * atr_20
    keltner_width = (keltner_upper - keltner_lower).replace(0, np.nan)
    bollinger_mid = close.rolling(20).mean()
    bollinger_std = close.rolling(20).std()
    bollinger_width = (4 * bollinger_std).replace(0, np.nan)
    df["ta_keltner_20_width"] = keltner_width / close
    df["ta_keltner_20_position"] = ((close - keltner_lower) / keltner_width).clip(0, 1)
    df["ta_bollinger_keltner_squeeze"] = (bollinger_width / keltner_width).clip(0, 10)

    supertrend_dir, supertrend_line = _supertrend(close, high, low, atr_10, multiplier=3)
    df["ta_supertrend_10_3_dir"] = supertrend_dir
    df["ta_supertrend_10_3_dist"] = close / supertrend_line - 1
    chandelier_high = high.rolling(22).max()
    chandelier_low = low.rolling(22).min()
    df["ta_chandelier_long_dist"] = close / (chandelier_high - 3 * atr_14) - 1
    df["ta_chandelier_short_dist"] = (chandelier_low + 3 * atr_14) / close - 1
    df["ta_atr_14_zscore_100"] = _zscore(atr_14 / close, 100)
    df["ta_parkinson_vol_20"] = np.sqrt(((np.log(high / low) ** 2).rolling(20).mean()) / (4 * np.log(2)))
    df["ta_garman_klass_vol_20"] = np.sqrt(
        (0.5 * (np.log(high / low) ** 2) - (2 * np.log(2) - 1) * (np.log(close / open_) ** 2)).rolling(20).mean().clip(lower=0)
    )

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    ppo = (ema_12 - ema_26) / ema_26.replace(0, np.nan)
    ppo_signal = ppo.ewm(span=9, adjust=False).mean()
    df["ta_ppo"] = ppo
    df["ta_ppo_signal"] = ppo_signal
    df["ta_ppo_hist"] = ppo - ppo_signal

    rsi_14 = _rsi(close, 14)
    rsi_min = rsi_14.rolling(14).min()
    rsi_max = rsi_14.rolling(14).max()
    stoch_rsi = (rsi_14 - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df["ta_stoch_rsi_k"] = stoch_rsi.rolling(3).mean()
    df["ta_stoch_rsi_d"] = df["ta_stoch_rsi_k"].rolling(3).mean()

    momentum = close.diff()
    abs_momentum = momentum.abs()
    tsi = 100 * momentum.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean() / abs_momentum.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean().replace(0, np.nan)
    df["ta_tsi"] = tsi / 100
    df["ta_tsi_signal"] = tsi.ewm(span=7, adjust=False).mean() / 100

    buying_pressure = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    true_high = pd.concat([high, close.shift(1)], axis=1).max(axis=1)
    true_low = pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    true_range_uo = (true_high - true_low).replace(0, np.nan)
    avg_7 = buying_pressure.rolling(7).sum() / true_range_uo.rolling(7).sum()
    avg_14 = buying_pressure.rolling(14).sum() / true_range_uo.rolling(14).sum()
    avg_28 = buying_pressure.rolling(28).sum() / true_range_uo.rolling(28).sum()
    df["ta_ultimate_osc"] = (4 * avg_7 + 2 * avg_14 + avg_28) / 7

    range_10_high = high.rolling(10).max()
    range_10_low = low.rolling(10).min()
    price_position = (2 * ((close - range_10_low) / (range_10_high - range_10_low).replace(0, np.nan)) - 1).clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + price_position) / (1 - price_position))
    df["ta_fisher_10"] = fisher
    df["ta_fisher_signal_10"] = fisher.shift(1)
    df["ta_rsi_7"] = _rsi(close, 7)
    df["ta_rsi_21"] = _rsi(close, 21)
    df["ta_rsi_7_21_spread"] = df["ta_rsi_7"] - df["ta_rsi_21"]

    money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_volume = money_flow_multiplier * volume
    df["ta_cmf_20"] = money_flow_volume.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    ad_line = money_flow_volume.fillna(0).cumsum()
    df["ta_chaikin_osc"] = (ad_line.ewm(span=3, adjust=False).mean() - ad_line.ewm(span=10, adjust=False).mean()) / volume.rolling(20).sum().replace(0, np.nan)
    distance_moved = ((high + low) / 2).diff()
    box_ratio = volume / (high - low).replace(0, np.nan)
    df["ta_eom_14"] = (distance_moved / box_ratio).rolling(14).mean()
    force_index = close.diff() * volume
    df["ta_force_index_13_zscore"] = _zscore(force_index.ewm(span=13, adjust=False).mean(), 100)
    vpt = (volume * close.pct_change()).fillna(0).cumsum()
    df["ta_vpt_zscore_100"] = _zscore(vpt, 100)
    df["ta_volume_price_corr_20"] = close.pct_change().rolling(20).corr(volume.pct_change())
    df["ta_volume_regime_20_100"] = volume.rolling(20).mean() / volume.rolling(100).mean().replace(0, np.nan) - 1

    candle_range = (high - low).replace(0, np.nan)
    body = close - open_
    body_abs = body.abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    df["ta_candle_body_share"] = body_abs / candle_range
    df["ta_candle_upper_lower_wick_ratio"] = upper_wick / lower_wick.replace(0, np.nan)
    df["ta_candle_doji"] = (body_abs <= candle_range * 0.1).astype(float)
    df["ta_candle_hammer"] = ((lower_wick >= body_abs * 2) & (upper_wick <= body_abs) & (body > 0)).astype(float)
    df["ta_candle_shooting_star"] = ((upper_wick >= body_abs * 2) & (lower_wick <= body_abs) & (body < 0)).astype(float)
    previous_body = close.shift(1) - open_.shift(1)
    df["ta_candle_bullish_engulfing"] = ((previous_body < 0) & (body > 0) & (open_ <= close.shift(1)) & (close >= open_.shift(1))).astype(float)
    df["ta_candle_bearish_engulfing"] = ((previous_body > 0) & (body < 0) & (open_ >= close.shift(1)) & (close <= open_.shift(1))).astype(float)
    df["ta_candle_morning_star"] = ((close.shift(2) < open_.shift(2)) & (body_abs.shift(1) < body_abs.shift(2) * 0.5) & (body > 0) & (close > (open_.shift(2) + close.shift(2)) / 2)).astype(float)
    df["ta_candle_evening_star"] = ((close.shift(2) > open_.shift(2)) & (body_abs.shift(1) < body_abs.shift(2) * 0.5) & (body < 0) & (close < (open_.shift(2) + close.shift(2)) / 2)).astype(float)
    df["ta_candle_three_bar_reversal_up"] = ((low < low.shift(1)) & (low.shift(1) < low.shift(2)) & (close > high.shift(1))).astype(float)
    df["ta_candle_three_bar_reversal_down"] = ((high > high.shift(1)) & (high.shift(1) > high.shift(2)) & (close < low.shift(1))).astype(float)

    df["ta_gap_pct"] = open_ / close.shift(1) - 1
    df["ta_gap_zscore_50"] = _zscore(df["ta_gap_pct"], 50)
    prior_high_20 = high.rolling(20).max().shift(1)
    prior_low_20 = low.rolling(20).min().shift(1)
    df["ta_liquidity_sweep_high_20"] = ((high > prior_high_20) & (close < prior_high_20)).astype(float)
    df["ta_liquidity_sweep_low_20"] = ((low < prior_low_20) & (close > prior_low_20)).astype(float)

    neutral_values = {
        "ta_ichimoku_cloud_position": 0.5,
        "ta_donchian_20_position": 0.5,
        "ta_donchian_55_position": 0.5,
        "ta_keltner_20_position": 0.5,
        "ta_stoch_rsi_k": 0.5,
        "ta_stoch_rsi_d": 0.5,
        "ta_rsi_7": 0.5,
        "ta_rsi_21": 0.5,
        "ta_ultimate_osc": 0.5,
        "ta_bollinger_keltner_squeeze": 1.0,
        "ta_supertrend_10_3_dir": 0.0,
    }
    for column in TA_WIDE_FEATURE_COLUMNS:
        neutral = neutral_values.get(column, 0.0)
        df[column] = df[column].replace([np.inf, -np.inf], np.nan).ffill().fillna(neutral)

    return df
