import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from crypto_predictor import (
    DATA_DIR,
    attach_fundamentals,
    columns_need_fundamentals,
    execution_price,
    horizon_model_paths,
    load_artifacts,
    load_price_csv,
)


def parse_float_grid(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Grid must contain at least one value.")
    return values


def parse_int_grid(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Grid must contain at least one value.")
    return values


def parse_horizon_sets(value: str) -> list[list[int]]:
    sets = []
    for group in value.split(";"):
        horizons = sorted({int(part.strip()) for part in group.split(",") if part.strip()})
        if horizons:
            sets.append(horizons)
    if not sets:
        raise ValueError("At least one horizon set is required.")
    return sets


def parse_string_grid(value: str, allowed: set[str]) -> list[str]:
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"Invalid grid values: {invalid}. Allowed values: {sorted(allowed)}")
    if not values:
        raise ValueError("Grid must contain at least one value.")
    return values


def load_horizon_predictions(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    horizons: list[int],
    max_rows: int,
) -> pd.DataFrame:
    prediction_frames = []
    base = None

    for horizon in sorted(set(horizons)):
        model_path, artifact_path = horizon_model_paths(symbol, timeframe, horizon)
        if not model_path.exists() or not artifact_path.exists():
            raise FileNotFoundError(f"Missing model or artifact for horizon {horizon}: {model_path}")

        artifact = load_artifacts(artifact_path)
        feature_columns = artifact["feature_columns"]
        sequence_length = int(artifact["sequence_length"])
        featured = prepare_historical_features(
            data=data,
            horizon=int(artifact["horizon"]),
            timeframe=timeframe,
            max_rows=max_rows,
            feature_columns=feature_columns,
        )
        features = featured[feature_columns].to_numpy(dtype=np.float32)
        scaled_features = artifact["feature_scaler"].transform(features)
        X = np.asarray(
            [scaled_features[end_idx - sequence_length + 1 : end_idx + 1] for end_idx in range(sequence_length - 1, len(featured))],
            dtype=np.float32,
        )
        model = load_model(model_path, compile=False)
        probabilities = model.predict(X, verbose=0)
        frame = featured[["timestamp", "close"]].iloc[sequence_length - 1 :].reset_index(drop=True).copy()
        frame[f"class_h{horizon}"] = probabilities.argmax(axis=1).astype(int)
        frame[f"confidence_h{horizon}"] = probabilities.max(axis=1)
        frame[f"short_prob_h{horizon}"] = probabilities[:, 0]
        frame[f"hold_prob_h{horizon}"] = probabilities[:, 1]
        frame[f"long_prob_h{horizon}"] = probabilities[:, 2]
        prediction_frames.append(frame.drop(columns=["close"]))
        base = frame[["timestamp", "close"]] if base is None else base

    merged = base
    for frame in prediction_frames:
        merged = merged.merge(frame, on="timestamp", how="inner")

    ohlc = data[["timestamp", "open", "high", "low", "close"]].copy().sort_values("timestamp")
    previous_close = ohlc["close"].shift(1)
    true_range = pd.concat(
        [
            ohlc["high"] - ohlc["low"],
            (ohlc["high"] - previous_close).abs(),
            (ohlc["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    sma_20 = ohlc["close"].rolling(20).mean()
    sma_50 = ohlc["close"].rolling(50).mean()
    range_low = ohlc["low"].rolling(48).min().shift(1)
    range_high = ohlc["high"].rolling(48).max().shift(1)
    range_span = (range_high - range_low).replace(0, np.nan)
    ohlc["atr_14_pct"] = true_range.rolling(14).mean() / ohlc["close"]
    ohlc["trend_20_50"] = sma_20 / sma_50 - 1
    ohlc["trend_strength_pct"] = ohlc["trend_20_50"].abs()
    ohlc["volatility_20"] = np.log(ohlc["close"] / previous_close).rolling(20).std()
    ohlc["range_low"] = range_low
    ohlc["range_high"] = range_high
    ohlc["range_position"] = (ohlc["close"] - range_low) / range_span
    ohlc["range_width_pct"] = range_span / ohlc["close"]
    ohlc["next_open"] = ohlc["open"].shift(-1)
    ohlc["next_high"] = ohlc["high"].shift(-1)
    ohlc["next_low"] = ohlc["low"].shift(-1)
    ohlc["next_close"] = ohlc["close"].shift(-1)
    merged = merged.drop(columns=["close"]).merge(ohlc, on="timestamp", how="inner")
    return merged.dropna(subset=["next_high", "next_low", "next_close"]).reset_index(drop=True)


def prepare_historical_features(
    data: pd.DataFrame,
    horizon: int,
    timeframe: str,
    max_rows: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    from crypto_predictor import add_features, latest_continuous_block, timeframe_to_milliseconds

    frame = latest_continuous_block(data, timeframe=timeframe)
    if max_rows is not None and max_rows > 0 and len(frame) > max_rows + 250:
        frame = frame.tail(max_rows + 250).reset_index(drop=True)
    featured = add_features(frame, horizon=horizon, require_target=True, feature_columns=feature_columns)
    if max_rows is not None and max_rows > 0 and len(featured) > max_rows:
        featured = featured.tail(max_rows).reset_index(drop=True)
    gap_count = int((featured["timestamp"].diff() > pd.to_timedelta(timeframe_to_milliseconds(timeframe), unit="ms") * 1.5).sum())
    if gap_count:
        raise ValueError(f"Historical prediction data contains {gap_count} timestamp gaps after repair.")
    return featured


def build_ensemble_signal(
    row,
    horizons: list[int],
    min_confidence: float,
    min_agree: int,
    atr_min: float,
    trend_min: float,
    volatility_min: float,
    trend_filter: str,
) -> int:
    if float(row.atr_14_pct) < atr_min:
        return 0
    if trend_filter == "strength" and float(row.trend_strength_pct) < trend_min:
        return 0
    if float(row.volatility_20) < volatility_min:
        return 0

    votes = []
    for horizon in horizons:
        predicted_class = int(getattr(row, f"class_h{horizon}"))
        confidence = float(getattr(row, f"confidence_h{horizon}"))
        votes.append(predicted_class if confidence >= min_confidence else 1)

    long_votes = sum(vote == 2 for vote in votes)
    short_votes = sum(vote == 0 for vote in votes)
    if long_votes >= min_agree and short_votes == 0:
        signal = 1
    elif short_votes >= min_agree and long_votes == 0:
        signal = -1
    else:
        return 0

    if trend_filter == "follow":
        trend = float(row.trend_20_50)
        if signal == 1 and trend < trend_min:
            return 0
        if signal == -1 and trend > -trend_min:
            return 0
    return signal


def is_range_regime(
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


def build_range_signal(
    row,
    range_lower: float,
    range_upper: float,
    range_atr_max: float,
    range_trend_max: float,
    range_width_min: float,
    range_width_max: float,
) -> int:
    if not is_range_regime(row, range_atr_max, range_trend_max, range_width_min, range_width_max):
        return 0
    position = float(row.range_position)
    if position <= range_lower:
        return 1
    if position >= range_upper:
        return -1
    return 0


def build_strategy_signal(
    row,
    horizons: list[int],
    min_confidence: float,
    min_agree: int,
    atr_min: float,
    trend_min: float,
    volatility_min: float,
    trend_filter: str,
    strategy: str,
    range_lower: float,
    range_upper: float,
    range_atr_max: float,
    range_trend_max: float,
    range_width_min: float,
    range_width_max: float,
) -> int:
    range_signal = build_range_signal(
        row,
        range_lower,
        range_upper,
        range_atr_max,
        range_trend_max,
        range_width_min,
        range_width_max,
    )
    if strategy == "range":
        return range_signal
    if strategy == "hybrid" and is_range_regime(row, range_atr_max, range_trend_max, range_width_min, range_width_max):
        return range_signal
    return build_ensemble_signal(
        row,
        horizons,
        min_confidence,
        min_agree,
        atr_min,
        trend_min,
        volatility_min,
        trend_filter,
    )


def simulate_futures(
    frame: pd.DataFrame,
    horizons: list[int],
    min_confidence: float,
    min_agree: int,
    initial_capital: float,
    fee_rate: float,
    leverage: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    atr_min: float,
    trend_min: float,
    volatility_min: float,
    trend_filter: str,
    strategy: str,
    range_lower: float,
    range_upper: float,
    range_atr_max: float,
    range_trend_max: float,
    range_width_min: float,
    range_width_max: float,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict:
    capital = initial_capital
    position = 0
    entry_price = 0.0
    notional = 0.0
    trades = []
    equity_curve = []

    def pnl_at(price: float) -> float:
        if position == 0:
            return 0.0
        return notional * position * ((price - entry_price) / entry_price)

    def close_position(timestamp, price: float, reason: str) -> None:
        nonlocal capital, position, entry_price, notional
        if position == 0:
            return
        fill_price = execution_price(price, position, "close", spread_bps, slippage_bps)
        net_pnl = pnl_at(fill_price) - notional * fee_rate
        capital += net_pnl
        trades.append(
            {
                "timestamp": str(timestamp),
                "side": "CLOSE_LONG" if position == 1 else "CLOSE_SHORT",
                "price": float(fill_price),
                "mid_price": float(price),
                "pnl": float(net_pnl),
                "reason": reason,
                "capital": float(capital),
            }
        )
        position = 0
        entry_price = 0.0
        notional = 0.0

    def open_position(timestamp, side: int, price: float) -> None:
        nonlocal capital, position, entry_price, notional
        if side == 0 or capital <= 0:
            return
        notional = capital * leverage
        fee = notional * fee_rate
        capital -= fee
        position = side
        fill_price = execution_price(price, side, "open", spread_bps, slippage_bps)
        entry_price = fill_price
        trades.append(
            {
                "timestamp": str(timestamp),
                "side": "OPEN_LONG" if side == 1 else "OPEN_SHORT",
                "price": float(fill_price),
                "mid_price": float(price),
                "notional": float(notional),
                "fee": float(fee),
                "capital": float(capital),
            }
        )

    for row in frame.itertuples(index=False):
        timestamp = row.timestamp
        price = float(row.close)
        signal = build_strategy_signal(
            row,
            horizons,
            min_confidence,
            min_agree,
            atr_min,
            trend_min,
            volatility_min,
            trend_filter,
            strategy,
            range_lower,
            range_upper,
            range_atr_max,
            range_trend_max,
            range_width_min,
            range_width_max,
        )

        if signal == 0 and position != 0:
            close_position(timestamp, price, "signal_hold")
        elif signal != 0 and signal != position:
            close_position(timestamp, price, "signal_flip")
            open_position(timestamp, signal, price)
        elif signal != 0 and position == 0:
            open_position(timestamp, signal, price)

        exit_price = None
        exit_reason = None
        if position == 1:
            stop_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct > 0 else None
            take_price = entry_price * (1 + take_profit_pct) if take_profit_pct > 0 else None
            if stop_price and float(row.next_low) <= stop_price:
                exit_price, exit_reason = stop_price, "stop_loss"
            elif take_price and float(row.next_high) >= take_price:
                exit_price, exit_reason = take_price, "take_profit"
        elif position == -1:
            stop_price = entry_price * (1 + stop_loss_pct) if stop_loss_pct > 0 else None
            take_price = entry_price * (1 - take_profit_pct) if take_profit_pct > 0 else None
            if stop_price and float(row.next_high) >= stop_price:
                exit_price, exit_reason = stop_price, "stop_loss"
            elif take_price and float(row.next_low) <= take_price:
                exit_price, exit_reason = take_price, "take_profit"

        if exit_price is not None:
            close_position(timestamp, exit_price, exit_reason)

        mark_price = float(row.next_close)
        equity_curve.append(capital + pnl_at(mark_price))

    if position != 0:
        close_position(frame["timestamp"].iloc[-1], float(frame["next_close"].iloc[-1]), "final_close")

    equity = pd.Series(equity_curve, dtype=float)
    returns = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1 if not equity.empty else pd.Series(dtype=float)
    closes = [trade for trade in trades if trade["side"].startswith("CLOSE")]
    wins = [trade for trade in closes if trade.get("pnl", 0) > 0]
    return {
        "initial_capital": initial_capital,
        "final_capital": float(capital),
        "total_return_pct": float((capital / initial_capital - 1) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100) if not drawdown.empty else 0.0,
        "trade_count": len(trades),
        "closed_trade_count": len(closes),
        "win_rate_pct": float(len(wins) / len(closes) * 100) if closes else 0.0,
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
        "sharpe_like": float((returns.mean() / returns.std()) * np.sqrt(365 * 24)) if len(returns) > 2 and returns.std() else 0.0,
        "last_trades": trades[-8:],
    }


def iter_grid(args: argparse.Namespace):
    for (
        horizons,
        min_confidence,
        min_agree,
        stop_loss,
        take_profit,
        atr_min,
        trend_min,
        volatility_min,
        trend_filter,
        strategy,
        range_lower,
        range_upper,
        range_atr_max,
        range_trend_max,
        range_width_min,
        range_width_max,
    ) in itertools.product(
        parse_horizon_sets(args.horizon_sets),
        parse_float_grid(args.confidence_grid),
        parse_int_grid(args.min_agree_grid),
        parse_float_grid(args.stop_loss_grid),
        parse_float_grid(args.take_profit_grid),
        parse_float_grid(args.atr_min_grid),
        parse_float_grid(args.trend_min_grid),
        parse_float_grid(args.volatility_min_grid),
        parse_string_grid(args.trend_filter_grid, {"off", "strength", "follow"}),
        parse_string_grid(args.strategy_grid, {"model", "range", "hybrid"}),
        parse_float_grid(args.range_lower_grid),
        parse_float_grid(args.range_upper_grid),
        parse_float_grid(args.range_atr_max_grid),
        parse_float_grid(args.range_trend_max_grid),
        parse_float_grid(args.range_width_min_grid),
        parse_float_grid(args.range_width_max_grid),
    ):
        if min_agree <= len(horizons) and range_lower < range_upper and range_width_min <= range_width_max:
            yield (
                horizons,
                min_confidence,
                min_agree,
                stop_loss,
                take_profit,
                atr_min,
                trend_min,
                volatility_min,
                trend_filter,
                strategy,
                range_lower,
                range_upper,
                range_atr_max,
                range_trend_max,
                range_width_min,
                range_width_max,
            )


def evaluate_grid(frame: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    results = []
    for (
        horizons,
        min_confidence,
        min_agree,
        stop_loss,
        take_profit,
        atr_min,
        trend_min,
        volatility_min,
        trend_filter,
        strategy,
        range_lower,
        range_upper,
        range_atr_max,
        range_trend_max,
        range_width_min,
        range_width_max,
    ) in iter_grid(args):
        metrics = simulate_futures(
            frame=frame,
            horizons=horizons,
            min_confidence=min_confidence,
            min_agree=min_agree,
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
            leverage=args.leverage,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            atr_min=atr_min,
            trend_min=trend_min,
            volatility_min=volatility_min,
            trend_filter=trend_filter,
            strategy=strategy,
            range_lower=range_lower,
            range_upper=range_upper,
            range_atr_max=range_atr_max,
            range_trend_max=range_trend_max,
            range_width_min=range_width_min,
            range_width_max=range_width_max,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
        )
        results.append(
            {
                "horizons": horizons,
                "min_confidence": min_confidence,
                "min_agree": min_agree,
                "stop_loss_pct": stop_loss * 100,
                "take_profit_pct": take_profit * 100,
                "atr_min_pct": atr_min * 100,
                "trend_min_pct": trend_min * 100,
                "volatility_min_pct": volatility_min * 100,
                "trend_filter": trend_filter,
                "strategy": strategy,
                "range_lower": range_lower,
                "range_upper": range_upper,
                "range_atr_max_pct": range_atr_max * 100,
                "range_trend_max_pct": range_trend_max * 100,
                "range_width_min_pct": range_width_min * 100,
                "range_width_max_pct": range_width_max * 100,
                **metrics,
            }
        )
    return results


def rank_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda item: (
            item["final_capital"],
            item["max_drawdown_pct"],
            item["closed_trade_count"],
            item["win_rate_pct"],
        ),
        reverse=True,
    )


def prepare_prediction_frame(args: argparse.Namespace) -> pd.DataFrame:
    data = load_price_csv(args.data)
    horizon_sets = parse_horizon_sets(args.horizon_sets)
    all_horizons = sorted(set(itertools.chain.from_iterable(horizon_sets)))
    artifacts = []
    for horizon in all_horizons:
        _, artifact_path = horizon_model_paths(args.symbol, args.timeframe, horizon)
        if artifact_path.exists():
            artifacts.append(load_artifacts(artifact_path))
    data = attach_fundamentals(
        data=data,
        data_path=args.data,
        symbol=args.symbol,
        timeframe=args.timeframe,
        fundamental_data=args.fundamental_data,
        update_fundamentals=args.update_fundamentals,
        required=any(columns_need_fundamentals(artifact["feature_columns"]) for artifact in artifacts),
    )
    return load_horizon_predictions(
        data=data,
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizons=all_horizons,
        max_rows=args.max_train_rows,
    )


def bars_per_day(timeframe: str) -> int:
    return 24 if timeframe.endswith("h") else 24 * 12


def run_optimizer(args: argparse.Namespace) -> dict:
    prediction_frame = prepare_prediction_frame(args)
    rows = args.days * bars_per_day(args.timeframe)
    test_frame = prediction_frame.tail(rows).reset_index(drop=True)

    results = evaluate_grid(test_frame, args)
    ranked = rank_results(results)
    output = {
        "start": str(test_frame["timestamp"].iloc[0]),
        "end": str(test_frame["timestamp"].iloc[-1]),
        "days": args.days,
        "tested_configs": len(results),
        "top": ranked[: args.top],
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def config_key(config: dict) -> str:
    return (
        f"h={','.join(map(str, config['horizons']))}|"
        f"conf={config['min_confidence']}|agree={config['min_agree']}|"
        f"sl={config['stop_loss_pct']}|tp={config['take_profit_pct']}|"
        f"atr={config.get('atr_min_pct', 0)}|trend={config.get('trend_min_pct', 0)}|"
        f"vol={config.get('volatility_min_pct', 0)}|trend_filter={config.get('trend_filter', 'off')}|"
        f"strategy={config.get('strategy', 'model')}|"
        f"range={config.get('range_lower', 0)}-{config.get('range_upper', 1)}"
    )


def run_walk_forward(args: argparse.Namespace) -> dict:
    prediction_frame = prepare_prediction_frame(args)
    train_rows = args.train_days * bars_per_day(args.timeframe)
    test_rows = args.test_days * bars_per_day(args.timeframe)
    step_rows = args.step_days * bars_per_day(args.timeframe)
    start_index = max(0, len(prediction_frame) - args.walkforward_days * bars_per_day(args.timeframe))

    folds = []
    cursor = start_index
    while cursor + train_rows + test_rows <= len(prediction_frame):
        train_frame = prediction_frame.iloc[cursor : cursor + train_rows].reset_index(drop=True)
        test_frame = prediction_frame.iloc[cursor + train_rows : cursor + train_rows + test_rows].reset_index(drop=True)
        train_ranked = rank_results(evaluate_grid(train_frame, args))
        if not train_ranked:
            break
        best_config = train_ranked[0]
        test_metrics = simulate_futures(
            frame=test_frame,
            horizons=best_config["horizons"],
            min_confidence=best_config["min_confidence"],
            min_agree=best_config["min_agree"],
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
            leverage=args.leverage,
            stop_loss_pct=best_config["stop_loss_pct"] / 100,
            take_profit_pct=best_config["take_profit_pct"] / 100,
            atr_min=best_config.get("atr_min_pct", 0) / 100,
            trend_min=best_config.get("trend_min_pct", 0) / 100,
            volatility_min=best_config.get("volatility_min_pct", 0) / 100,
            trend_filter=best_config.get("trend_filter", "off"),
            strategy=best_config.get("strategy", "model"),
            range_lower=best_config.get("range_lower", 0.2),
            range_upper=best_config.get("range_upper", 0.8),
            range_atr_max=best_config.get("range_atr_max_pct", 0.8) / 100,
            range_trend_max=best_config.get("range_trend_max_pct", 0.3) / 100,
            range_width_min=best_config.get("range_width_min_pct", 0.8) / 100,
            range_width_max=best_config.get("range_width_max_pct", 6.0) / 100,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
        )
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_start": str(train_frame["timestamp"].iloc[0]),
                "train_end": str(train_frame["timestamp"].iloc[-1]),
                "test_start": str(test_frame["timestamp"].iloc[0]),
                "test_end": str(test_frame["timestamp"].iloc[-1]),
                "selected_config": {
                    "horizons": best_config["horizons"],
                    "min_confidence": best_config["min_confidence"],
                    "min_agree": best_config["min_agree"],
                    "stop_loss_pct": best_config["stop_loss_pct"],
                    "take_profit_pct": best_config["take_profit_pct"],
                    "atr_min_pct": best_config.get("atr_min_pct", 0),
                    "trend_min_pct": best_config.get("trend_min_pct", 0),
                    "volatility_min_pct": best_config.get("volatility_min_pct", 0),
                    "trend_filter": best_config.get("trend_filter", "off"),
                    "strategy": best_config.get("strategy", "model"),
                    "range_lower": best_config.get("range_lower", 0.2),
                    "range_upper": best_config.get("range_upper", 0.8),
                    "range_atr_max_pct": best_config.get("range_atr_max_pct", 0.8),
                    "range_trend_max_pct": best_config.get("range_trend_max_pct", 0.3),
                    "range_width_min_pct": best_config.get("range_width_min_pct", 0.8),
                    "range_width_max_pct": best_config.get("range_width_max_pct", 6.0),
                    "train_final_capital": best_config["final_capital"],
                    "train_return_pct": best_config["total_return_pct"],
                    "train_drawdown_pct": best_config["max_drawdown_pct"],
                },
                "test": test_metrics,
            }
        )
        cursor += step_rows

    if not folds:
        raise ValueError("Not enough rows for walk-forward validation. Reduce train/test days.")

    returns = [fold["test"]["total_return_pct"] for fold in folds]
    final_capitals = [fold["test"]["final_capital"] for fold in folds]
    drawdowns = [fold["test"]["max_drawdown_pct"] for fold in folds]
    config_counts = {}
    for fold in folds:
        key = config_key(fold["selected_config"])
        config_counts[key] = config_counts.get(key, 0) + 1

    output = {
        "mode": "walk-forward",
        "fold_count": len(folds),
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "walkforward_days": args.walkforward_days,
        "average_test_return_pct": float(np.mean(returns)),
        "median_test_return_pct": float(np.median(returns)),
        "profitable_fold_pct": float(np.mean([value > 0 for value in returns]) * 100),
        "average_final_capital": float(np.mean(final_capitals)),
        "worst_drawdown_pct": float(min(drawdowns)),
        "selected_config_counts": config_counts,
        "folds": folds,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize crypto paper-trading strategy parameters.")
    parser.add_argument("--mode", choices=["optimize", "walk-forward"], default="optimize")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--data", default=str(DATA_DIR / "1h-btc_history.csv"))
    parser.add_argument("--fundamental-data", default=None)
    parser.add_argument("--update-fundamentals", action="store_true")
    parser.add_argument("--horizon-sets", default="1,3,6;1,3;3,6;1,6;6")
    parser.add_argument("--confidence-grid", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--min-agree-grid", default="1,2,3")
    parser.add_argument("--stop-loss-grid", default="0,0.005,0.01")
    parser.add_argument("--take-profit-grid", default="0,0.01,0.02")
    parser.add_argument("--atr-min-grid", default="0,0.003,0.006")
    parser.add_argument("--trend-min-grid", default="0,0.003,0.006")
    parser.add_argument("--volatility-min-grid", default="0")
    parser.add_argument("--trend-filter-grid", default="off,follow")
    parser.add_argument("--strategy-grid", default="model,hybrid")
    parser.add_argument("--range-lower-grid", default="0.20")
    parser.add_argument("--range-upper-grid", default="0.80")
    parser.add_argument("--range-atr-max-grid", default="0.008")
    parser.add_argument("--range-trend-max-grid", default="0.003")
    parser.add_argument("--range-width-min-grid", default="0.008")
    parser.add_argument("--range-width-max-grid", default="0.06")
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--max-train-rows", type=int, default=5000)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--test-days", type=int, default=3)
    parser.add_argument("--step-days", type=int, default=3)
    parser.add_argument("--walkforward-days", type=int, default=45)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", default="strategy_optimization.json")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    if parsed_args.mode == "walk-forward":
        run_walk_forward(parsed_args)
    else:
        run_optimizer(parsed_args)
