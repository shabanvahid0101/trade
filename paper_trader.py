import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from crypto_predictor import (
    DATA_DIR,
    MODELS_DIR,
    attach_fundamentals,
    backtest_predictions,
    class_predictions_to_signal_returns,
    columns_need_fundamentals,
    combine_multi_horizon_predictions,
    execution_price,
    fetch_and_update_with_fallbacks,
    horizon_model_paths,
    latest_continuous_block,
    load_artifacts,
    load_price_csv,
    parse_exchange_names,
    predict_next_price,
    send_telegram_message,
    timeframe_to_milliseconds,
)
from risk_manager import apply_dynamic_position_sizing, decision_to_dict, evaluate_risk
from signal_explainer import build_signal_explanation, fa_code, fa_regime, fa_signal, fa_strategy, format_explanation_for_telegram
from strategy_rules import apply_hybrid_to_latest, apply_hybrid_to_returns


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = BASE_DIR / "paper_state.json"


def bars_per_day(timeframe: str) -> int:
    day_ms = 24 * 60 * 60 * 1000
    return max(1, int(round(day_ms / timeframe_to_milliseconds(timeframe))))


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be a comma-separated list of positive integers.")
    return horizons


def load_state(path: Path, initial_capital: float) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "capital": initial_capital,
        "position": 0,
        "entry_price": 0.0,
        "notional": 0.0,
        "trades": [],
        "last_timestamp": None,
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def signal_to_side(signal: str) -> int:
    if signal == "LONG":
        return 1
    if signal == "SHORT":
        return -1
    return 0


def unrealized_pnl(state: dict, mark_price: float) -> float:
    position = int(state.get("position", 0))
    if position == 0:
        return 0.0
    entry_price = float(state["entry_price"])
    notional = float(state["notional"])
    return notional * position * ((mark_price - entry_price) / entry_price)


def close_position(
    state: dict,
    timestamp: str,
    price: float,
    fee_rate: float,
    reason: str,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_explanation: dict | None = None,
) -> None:
    position = int(state.get("position", 0))
    if position == 0:
        return
    notional = float(state["notional"])
    fill_price = execution_price(price, position, "close", spread_bps, slippage_bps)
    pnl = unrealized_pnl(state, fill_price)
    close_fee = notional * fee_rate
    net_pnl = pnl - close_fee
    state["capital"] = float(state["capital"] + net_pnl)
    state["trades"].append(
        {
            "timestamp": timestamp,
            "side": "CLOSE_LONG" if position == 1 else "CLOSE_SHORT",
            "price": fill_price,
            "mid_price": price,
            "pnl": float(net_pnl),
            "reason": reason,
            "signal_reason": signal_explanation.get("summary") if signal_explanation else None,
            "capital": float(state["capital"]),
        }
    )
    state["position"] = 0
    state["entry_price"] = 0.0
    state["notional"] = 0.0


def open_position(
    state: dict,
    timestamp: str,
    side: int,
    price: float,
    fee_rate: float,
    leverage: float,
    position_size_pct: float,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_explanation: dict | None = None,
) -> None:
    if side == 0 or float(state["capital"]) <= 0:
        return
    position_size_pct = max(0.0, min(position_size_pct, 1.0))
    notional = float(state["capital"]) * leverage * position_size_pct
    if notional <= 0:
        return
    open_fee = notional * fee_rate
    fill_price = execution_price(price, side, "open", spread_bps, slippage_bps)
    state["capital"] = float(state["capital"] - open_fee)
    state["position"] = side
    state["entry_price"] = fill_price
    state["notional"] = notional
    state["trades"].append(
        {
            "timestamp": timestamp,
            "side": "OPEN_LONG" if side == 1 else "OPEN_SHORT",
            "price": fill_price,
            "mid_price": price,
            "notional": notional,
            "fee": float(open_fee),
            "position_size_pct": float(position_size_pct),
            "signal_reason": signal_explanation.get("summary") if signal_explanation else None,
            "capital": float(state["capital"]),
        }
    )


def apply_signal(
    state: dict,
    signal: str,
    timestamp: str,
    price: float,
    fee_rate: float,
    leverage: float,
    risk: dict,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_explanation: dict | None = None,
) -> dict:
    desired_side = signal_to_side(signal)
    current_side = int(state.get("position", 0))
    if state.get("last_timestamp") == timestamp:
        equity = float(state["capital"]) + unrealized_pnl(state, price)
        return {"changed": False, "equity": equity, "reason": "already_processed"}

    if desired_side == 0 and current_side != 0:
        close_position(state, timestamp, price, fee_rate, "signal_hold", spread_bps, slippage_bps, signal_explanation)
    elif desired_side != 0 and desired_side != current_side:
        close_position(state, timestamp, price, fee_rate, "signal_flip", spread_bps, slippage_bps, signal_explanation)
        if risk["allow_new_position"]:
            open_position(
                state, timestamp, desired_side, price, fee_rate, leverage, risk["position_size_pct"],
                spread_bps, slippage_bps, signal_explanation
            )
    elif desired_side != 0 and current_side == 0:
        if risk["allow_new_position"]:
            open_position(
                state, timestamp, desired_side, price, fee_rate, leverage, risk["position_size_pct"],
                spread_bps, slippage_bps, signal_explanation
            )

    state["last_timestamp"] = timestamp
    if signal_explanation:
        state["last_signal_reason"] = signal_explanation.get("summary")
        state["last_signal_reasons"] = signal_explanation.get("reasons", [])
    equity = float(state["capital"]) + unrealized_pnl(state, price)
    reason = "processed" if risk["allow_new_position"] or desired_side == 0 or current_side != 0 else f"risk_blocked:{risk['reason']}"
    return {"changed": True, "equity": equity, "reason": reason}


def protective_exit_reason(state: dict, mark_price: float, stop_loss_pct: float, take_profit_pct: float) -> str | None:
    position = int(state.get("position", 0) or 0)
    if position == 0:
        return None
    entry_price = float(state.get("entry_price", 0) or 0)
    if entry_price <= 0:
        return None

    if position == 1:
        if stop_loss_pct > 0 and mark_price <= entry_price * (1 - stop_loss_pct):
            return "stop_loss"
        if take_profit_pct > 0 and mark_price >= entry_price * (1 + take_profit_pct):
            return "take_profit"
    if position == -1:
        if stop_loss_pct > 0 and mark_price >= entry_price * (1 + stop_loss_pct):
            return "stop_loss"
        if take_profit_pct > 0 and mark_price <= entry_price * (1 - take_profit_pct):
            return "take_profit"
    return None


def holding_candles(state: dict, timestamp: str, timeframe: str) -> float | None:
    position = int(state.get("position", 0) or 0)
    if position == 0:
        return None
    open_prefix = "OPEN_LONG" if position == 1 else "OPEN_SHORT"
    open_trade = next(
        (trade for trade in reversed(state.get("trades", [])) if trade.get("side") == open_prefix),
        None,
    )
    if not open_trade or not open_trade.get("timestamp"):
        return None
    try:
        elapsed_ms = pd.Timestamp(timestamp).value // 1_000_000 - pd.Timestamp(open_trade["timestamp"]).value // 1_000_000
        step_ms = timeframe_to_milliseconds(timeframe)
    except Exception:
        return None
    if step_ms <= 0 or elapsed_ms < 0:
        return None
    return elapsed_ms / step_ms


def apply_protective_exit(
    state: dict,
    timestamp: str,
    price: float,
    fee_rate: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_holding_candles: int,
    timeframe: str,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_explanation: dict | None = None,
) -> dict | None:
    if state.get("last_timestamp") == timestamp:
        return None
    reason = protective_exit_reason(state, price, stop_loss_pct, take_profit_pct)
    if not reason and max_holding_candles > 0:
        candles = holding_candles(state, timestamp, timeframe)
        if candles is not None and candles >= max_holding_candles:
            reason = "max_holding_candles"
    if not reason:
        return None
    close_position(state, timestamp, price, fee_rate, reason, spread_bps, slippage_bps, signal_explanation)
    state["last_timestamp"] = timestamp
    state["last_signal_reason"] = f"protective_exit:{reason}"
    state["last_signal_reasons"] = [f"Protective exit triggered: {reason}"]
    equity = float(state["capital"]) + unrealized_pnl(state, price)
    return {"changed": True, "equity": equity, "reason": f"protective_exit:{reason}"}


def build_paper_telegram_message(
    args: argparse.Namespace,
    final: dict,
    risk: dict,
    state: dict,
    trade_result: dict,
    position_name: str,
    quality_text: str,
    signal_explanation: dict,
) -> str:
    return (
        f"<b>{args.telegram_label} - {args.symbol} {args.timeframe}</b>\n"
        f"Signal: {fa_signal(final['signal'])} ({final['signal']})\n"
        f"Strategy: {fa_strategy(final.get('strategy', 'model'))} | Regime: {fa_regime(final.get('market_regime', 'model'))}\n"
        f"Risk: {fa_code(risk['risk_level'])} | Size: {risk['position_size_pct']:.2f} | Reason: {fa_code(risk['reason'])}\n"
        f"Size reason: {fa_code(risk.get('position_size_reason'))} | Signal quality: {quality_text}\n"
        f"Costs: fee {args.fee_rate:.4f} | spread {args.spread_bps:.2f} bps | slippage {args.slippage_bps:.2f} bps\n"
        f"Protective exits: SL {args.stop_loss_pct:.4f} | TP {args.take_profit_pct:.4f} | Max hold {args.max_holding_candles} candles\n"
        f"Position: {fa_signal(position_name)} ({position_name})\n"
        f"Price: ${final['current_price']:.2f}\n"
        f"Equity: ${trade_result['equity']:.2f}\n"
        f"Free capital: ${float(state['capital']):.2f}\n"
        f"Model confidence: {final['confidence']:.2f}\n\n"
        f"{format_explanation_for_telegram(signal_explanation)}"
    )


def build_historical_ensemble(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    horizons: list[int],
    max_rows: int,
    min_confidence: float,
    min_agree: int,
    model_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    prediction_frames = []
    base_meta = None
    target_threshold = None
    for horizon in horizons:
        model_path, artifact_path = horizon_model_paths(symbol, timeframe, horizon, model_dir)
        artifact = load_artifacts(artifact_path)
        target_threshold = float(artifact["target_threshold"])
        featured = build_historical_feature_frame(
            data=data,
            horizon=int(artifact["horizon"]),
            timeframe=timeframe,
            max_rows=max_rows,
            feature_columns=artifact["feature_columns"],
        )
        sequence_length = int(artifact["sequence_length"])
        features = featured[artifact["feature_columns"]].to_numpy(dtype=np.float32)
        scaled_features = artifact["feature_scaler"].transform(features)
        X = np.asarray(
            [scaled_features[end_idx - sequence_length + 1 : end_idx + 1] for end_idx in range(sequence_length - 1, len(featured))],
            dtype=np.float32,
        )
        model = load_model(model_path, compile=False)
        probabilities = model.predict(X, verbose=0)
        predicted_class = probabilities.argmax(axis=1)
        predicted_class[probabilities.max(axis=1) < min_confidence] = 1
        frame = featured[["timestamp", "close", "future_close", "target_return"]].iloc[sequence_length - 1 :].reset_index(drop=True).copy()
        frame[f"class_h{horizon}"] = predicted_class
        frame[f"confidence_h{horizon}"] = probabilities.max(axis=1)
        prediction_frames.append(frame[["timestamp", f"class_h{horizon}", f"confidence_h{horizon}"]])
        base_meta = frame[["timestamp", "close", "future_close", "target_return"]]

    if base_meta is None or target_threshold is None:
        raise ValueError("No models were loaded.")

    merged = base_meta
    for frame in prediction_frames:
        merged = merged.merge(frame, on="timestamp", how="inner")

    final_class = []
    for row in merged.itertuples(index=False):
        votes = [getattr(row, f"class_h{horizon}") for horizon in horizons]
        long_votes = sum(vote == 2 for vote in votes)
        short_votes = sum(vote == 0 for vote in votes)
        if long_votes >= min_agree and short_votes == 0:
            final_class.append(2)
        elif short_votes >= min_agree and long_votes == 0:
            final_class.append(0)
        else:
            final_class.append(1)

    predicted_return = class_predictions_to_signal_returns(np.array(final_class), target_threshold)
    return merged, predicted_return


def build_historical_feature_frame(
    data: pd.DataFrame,
    horizon: int,
    timeframe: str,
    max_rows: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    from crypto_predictor import add_features, timeframe_to_milliseconds

    frame = latest_continuous_block(data, timeframe=timeframe)
    if max_rows is not None and max_rows > 0 and len(frame) > max_rows + 250:
        frame = frame.tail(max_rows + 250).reset_index(drop=True)
    featured = add_features(frame, horizon=horizon, require_target=True, feature_columns=feature_columns)
    if max_rows is not None and max_rows > 0 and len(featured) > max_rows:
        featured = featured.tail(max_rows).reset_index(drop=True)
    gap_count = int((featured["timestamp"].diff() > pd.to_timedelta(timeframe_to_milliseconds(timeframe), unit="ms") * 1.5).sum())
    if gap_count:
        raise ValueError(f"Historical paper data contains {gap_count} timestamp gaps after repair.")
    return featured


def run_backtest(args: argparse.Namespace) -> dict:
    data = load_price_csv(args.data)
    horizons = parse_horizons(args.horizons)
    needs_fundamentals = False
    for horizon in horizons:
        _, artifact_path = horizon_model_paths(args.symbol, args.timeframe, horizon, args.model_dir)
        artifact = load_artifacts(artifact_path)
        needs_fundamentals = needs_fundamentals or columns_need_fundamentals(artifact["feature_columns"])
    data = attach_fundamentals(
        data=data,
        data_path=args.data,
        symbol=args.symbol,
        timeframe=args.timeframe,
        fundamental_data=args.fundamental_data,
        update_fundamentals=args.update_fundamentals,
        required=needs_fundamentals,
    )
    meta, predicted_return = build_historical_ensemble(
        data=data,
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizons=horizons,
        max_rows=args.max_train_rows,
        min_confidence=args.min_confidence,
        min_agree=args.min_agree,
        model_dir=args.model_dir,
    )
    rows = args.days * bars_per_day(args.timeframe)
    meta_window = meta.tail(rows).reset_index(drop=True)
    predicted_window = predicted_return[-len(meta_window) :]
    predicted_window = apply_hybrid_to_returns(
        meta=meta_window,
        predicted_return=predicted_window,
        data=data,
        strategy=args.strategy,
        threshold=args.threshold,
        range_lower=args.range_lower,
        range_upper=args.range_upper,
        range_atr_max=args.range_atr_max,
        range_trend_max=args.range_trend_max,
        range_width_min=args.range_width_min,
        range_width_max=args.range_width_max,
        range_lookback=args.range_lookback,
    )
    result = backtest_predictions(
        meta_window,
        predicted_window,
        threshold=args.threshold,
        fee_rate=args.fee_rate,
        initial_capital=args.initial_capital,
        market_mode="futures",
        leverage=args.leverage,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
    )
    output = {
        "start": str(meta_window["timestamp"].iloc[0]),
        "end": str(meta_window["timestamp"].iloc[-1]),
        "days": args.days,
        "horizons": horizons,
        "strategy": args.strategy,
        "backtest": result,
    }
    print(json.dumps(output, indent=2))
    return output


def run_single(args: argparse.Namespace) -> dict:
    horizons = parse_horizons(args.horizons)
    artifacts = []
    for horizon in horizons:
        model_path, artifact_path = horizon_model_paths(args.symbol, args.timeframe, horizon, args.model_dir)
        artifact = load_artifacts(artifact_path)
        artifact.setdefault("timeframe", args.timeframe)
        artifacts.append((horizon, model_path, artifact))

    data = (
        fetch_and_update_with_fallbacks(
            symbol=args.symbol,
            timeframe=args.timeframe,
            file=args.data,
            exchange_names=parse_exchange_names(args.exchange_fallbacks),
            max_batches=args.max_fetch_batches,
            max_data_age_hours=args.max_data_age_hours,
        )
        if args.update
        else load_price_csv(args.data)
    )
    data = attach_fundamentals(
        data=data,
        data_path=args.data,
        symbol=args.symbol,
        timeframe=args.timeframe,
        fundamental_data=args.fundamental_data,
        update_fundamentals=args.update_fundamentals,
        required=any(columns_need_fundamentals(artifact["feature_columns"]) for _, _, artifact in artifacts),
    )
    dataset_last_timestamp = str(data["timestamp"].max()) if not data.empty else None
    print(f"Paper data last timestamp: {dataset_last_timestamp}")
    horizon_results = []
    for horizon, model_path, artifact in artifacts:
        model = load_model(model_path, compile=False)
        horizon_results.append(predict_next_price(model, data, artifact, min_confidence=args.min_confidence))

    final = combine_multi_horizon_predictions(horizon_results, args.min_agree, args.min_confidence)
    final = apply_hybrid_to_latest(
        final=final,
        data=data,
        strategy=args.strategy,
        threshold=args.threshold,
        range_lower=args.range_lower,
        range_upper=args.range_upper,
        range_atr_max=args.range_atr_max,
        range_trend_max=args.range_trend_max,
        range_width_min=args.range_width_min,
        range_width_max=args.range_width_max,
        range_lookback=args.range_lookback,
    )
    state_path = Path(args.state_file)
    state = load_state(state_path, args.initial_capital)
    risk_decision = evaluate_risk(
        state=state,
        mark_price=float(final["current_price"]),
        initial_capital=args.initial_capital,
        max_drawdown_pct=args.risk_max_drawdown_pct,
        reduce_drawdown_pct=args.risk_reduce_drawdown_pct,
        max_loss_streak=args.risk_max_loss_streak,
        base_position_size_pct=args.risk_position_size_pct,
        reduced_position_size_pct=args.risk_reduced_position_size_pct,
        min_equity=args.risk_min_equity,
    )
    risk = decision_to_dict(risk_decision) if args.risk_enabled else {
        "allow_new_position": True,
        "risk_level": "disabled",
        "position_size_pct": args.risk_position_size_pct,
        "reason": "risk_disabled",
        "drawdown_pct": 0.0,
        "loss_streak": 0,
    }
    if args.dynamic_position_sizing:
        risk = apply_dynamic_position_sizing(
            risk=risk,
            final=final,
            horizon_results=horizon_results,
            min_position_size_pct=args.dynamic_min_position_size_pct,
            max_position_size_pct=args.dynamic_max_position_size_pct,
            recovery_enabled=args.risk_recovery_enabled,
            recovery_position_size_pct=args.risk_recovery_position_size_pct,
            recovery_min_signal_quality=args.risk_recovery_min_signal_quality,
            recovery_max_loss_streak=args.risk_recovery_max_loss_streak,
        )
    else:
        risk["dynamic_position_sizing"] = False
        risk["base_position_size_pct"] = risk["position_size_pct"]
        risk["signal_quality_score"] = None
        risk["position_size_reason"] = "dynamic_position_sizing_disabled"
    signal_explanation = build_signal_explanation(final, horizon_results, risk=risk)
    trade_result = apply_protective_exit(
        state=state,
        timestamp=final["timestamp"],
        price=float(final["current_price"]),
        fee_rate=args.fee_rate,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_holding_candles=args.max_holding_candles,
        timeframe=args.timeframe,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        signal_explanation=signal_explanation,
    )
    if trade_result is None:
        trade_result = apply_signal(
            state,
            signal=final["signal"],
            timestamp=final["timestamp"],
            price=float(final["current_price"]),
            fee_rate=args.fee_rate,
            leverage=args.leverage,
            risk=risk,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            signal_explanation=signal_explanation,
        )
    save_state(state_path, state)
    output = {
        "data_last_timestamp": dataset_last_timestamp,
        "final": final,
        "risk": risk,
        "signal_explanation": signal_explanation,
        "execution_costs": {"fee_rate": args.fee_rate, "spread_bps": args.spread_bps, "slippage_bps": args.slippage_bps},
        "paper": {"state": state, **trade_result},
    }
    print(json.dumps(output, indent=2))

    if args.telegram:
        position = int(state.get("position", 0))
        position_name = "LONG" if position == 1 else "SHORT" if position == -1 else "FLAT"
        quality = risk.get("signal_quality_score")
        quality_text = f"{float(quality):.2f}" if quality is not None else "n/a"
        send_telegram_message(build_paper_telegram_message(args, final, risk, state, trade_result, position_name, quality_text, signal_explanation))
        return output
        send_telegram_message(
            f"<b>{args.telegram_label} - {args.symbol} {args.timeframe}</b>\n"
            f"سیگنال: {fa_signal(final['signal'])} ({final['signal']})\n"
            f"استراتژی: {fa_strategy(final.get('strategy', 'model'))} | وضعیت بازار: {fa_regime(final.get('market_regime', 'model'))}\n"
            f"ریسک: {fa_code(risk['risk_level'])} | حجم معامله: {risk['position_size_pct']:.2f} | دلیل: {fa_code(risk['reason'])}\n"
            f"دلیل حجم: {fa_code(risk.get('position_size_reason'))} | کیفیت سیگنال: {quality_text}\n"
            f"هزینه‌ها: کارمزد {args.fee_rate:.4f} | اسپرد {args.spread_bps:.2f} bps | اسلیپیج {args.slippage_bps:.2f} bps\n"
            f"پوزیشن فعلی: {fa_signal(position_name)} ({position_name})\n"
            f"قیمت: ${final['current_price']:.2f}\n"
            f"ارزش حساب: ${trade_result['equity']:.2f}\n"
            f"سرمایه آزاد: ${float(state['capital']):.2f}\n"
            f"اطمینان مدل: {final['confidence']:.2f}\n\n"
            f"{format_explanation_for_telegram(signal_explanation)}"
        )
        return output
        send_telegram_message(
            f"<b>{args.telegram_label} - {args.symbol} {args.timeframe}</b>\n"
            f"سیگنال: {fa_signal(final['signal'])} ({final['signal']})\n"
            f"استراتژی: {fa_strategy(final.get('strategy', 'model'))} | وضعیت بازار: {fa_regime(final.get('market_regime', 'model'))}\n"
            f"ریسک: {fa_code(risk['risk_level'])} | حجم معامله: {risk['position_size_pct']:.2f} | دلیل: {fa_code(risk['reason'])}\n"
            f"دلیل حجم: {fa_code(risk.get('position_size_reason'))} | کیفیت سیگنال: {quality_text}\n"
            f"هزینه‌ها: کارمزد {args.fee_rate:.4f} | اسپرد {args.spread_bps:.2f} bps | اسلیپیج {args.slippage_bps:.2f} bps\n"
            f"پوزیشن فعلی: {fa_signal(position_name)} ({position_name})\n"
            f"قیمت: ${final['current_price']:.2f}\n"
            f"ارزش حساب: ${trade_result['equity']:.2f}\n"
            f"سرمایه آزاد: ${float(state['capital']):.2f}\n"
            f"اطمینان مدل: {final['confidence']:.2f}\n\n"
            f"{format_explanation_for_telegram(signal_explanation)}"
        )
        return output
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper trading runner and backtester.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--exchange-fallbacks", default="binance,okx,kucoin,bybit")
    parser.add_argument("--data", default=str(DATA_DIR / "1h-btc_history.csv"))
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    parser.add_argument("--fundamental-data", default=None)
    parser.add_argument("--update-fundamentals", action="store_true")
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--min-agree", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--strategy", choices=["model", "range", "hybrid"], default="hybrid")
    parser.add_argument("--range-lower", type=float, default=0.15)
    parser.add_argument("--range-upper", type=float, default=0.75)
    parser.add_argument("--range-atr-max", type=float, default=0.006)
    parser.add_argument("--range-trend-max", type=float, default=0.002)
    parser.add_argument("--range-width-min", type=float, default=0.005)
    parser.add_argument("--range-width-max", type=float, default=0.04)
    parser.add_argument("--range-lookback", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.0015)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--stop-loss-pct", type=float, default=0.0)
    parser.add_argument("--take-profit-pct", type=float, default=0.0)
    parser.add_argument("--max-holding-candles", type=int, default=0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--risk-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--risk-position-size-pct", type=float, default=1.0)
    parser.add_argument("--risk-reduced-position-size-pct", type=float, default=0.5)
    parser.add_argument("--risk-reduce-drawdown-pct", type=float, default=3.0)
    parser.add_argument("--risk-max-drawdown-pct", type=float, default=5.0)
    parser.add_argument("--risk-max-loss-streak", type=int, default=3)
    parser.add_argument("--risk-min-equity", type=float, default=50.0)
    parser.add_argument("--risk-recovery-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--risk-recovery-position-size-pct", type=float, default=0.25)
    parser.add_argument("--risk-recovery-min-signal-quality", type=float, default=0.55)
    parser.add_argument("--risk-recovery-max-loss-streak", type=int, default=3)
    parser.add_argument("--dynamic-position-sizing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-min-position-size-pct", type=float, default=0.25)
    parser.add_argument("--dynamic-max-position-size-pct", type=float, default=1.0)
    parser.add_argument("--max-train-rows", type=int, default=5000)
    parser.add_argument("--max-fetch-batches", type=int, default=5)
    parser.add_argument("--max-data-age-hours", type=float, default=4.0)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--mode", choices=["backtest", "single"], default="backtest")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--telegram-label", default="Paper Trading")
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.mode == "single":
        run_single(cli_args)
    else:
        run_backtest(cli_args)
