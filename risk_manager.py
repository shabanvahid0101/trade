from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    allow_new_position: bool
    risk_level: str
    position_size_pct: float
    reason: str
    drawdown_pct: float
    loss_streak: int


def unrealized_pnl(state: dict, mark_price: float) -> float:
    position = int(state.get("position", 0) or 0)
    if position == 0:
        return 0.0
    entry_price = float(state.get("entry_price", 0) or 0)
    notional = float(state.get("notional", 0) or 0)
    if entry_price <= 0 or notional <= 0:
        return 0.0
    return notional * position * ((mark_price - entry_price) / entry_price)


def current_equity(state: dict, mark_price: float, initial_capital: float) -> float:
    return float(state.get("capital", initial_capital) or initial_capital) + unrealized_pnl(state, mark_price)


def consecutive_losses(state: dict) -> int:
    losses = 0
    closed = [trade for trade in state.get("trades", []) if str(trade.get("side", "")).startswith("CLOSE")]
    for trade in reversed(closed):
        pnl = float(trade.get("pnl", 0) or 0)
        if pnl < 0:
            losses += 1
        else:
            break
    return losses


def peak_capital(state: dict, initial_capital: float, current_equity_value: float) -> float:
    values = [initial_capital, current_equity_value]
    for trade in state.get("trades", []):
        if "capital" in trade:
            values.append(float(trade["capital"]))
    return max(values)


def evaluate_risk(
    state: dict,
    mark_price: float,
    initial_capital: float,
    max_drawdown_pct: float,
    reduce_drawdown_pct: float,
    max_loss_streak: int,
    base_position_size_pct: float,
    reduced_position_size_pct: float,
    min_equity: float,
) -> RiskDecision:
    equity = current_equity(state, mark_price, initial_capital)
    peak = peak_capital(state, initial_capital, equity)
    drawdown_pct = (equity / peak - 1) * 100 if peak > 0 else 0.0
    loss_streak = consecutive_losses(state)

    if equity <= min_equity:
        return RiskDecision(False, "blocked", 0.0, "equity_below_minimum", drawdown_pct, loss_streak)
    if drawdown_pct <= -abs(max_drawdown_pct):
        return RiskDecision(False, "blocked", 0.0, "max_drawdown_exceeded", drawdown_pct, loss_streak)
    if loss_streak >= max_loss_streak:
        return RiskDecision(True, "reduced", reduced_position_size_pct, "loss_streak_reduced_size", drawdown_pct, loss_streak)
    if drawdown_pct <= -abs(reduce_drawdown_pct):
        return RiskDecision(True, "reduced", reduced_position_size_pct, "drawdown_reduced_size", drawdown_pct, loss_streak)

    return RiskDecision(True, "normal", base_position_size_pct, "ok", drawdown_pct, loss_streak)


def decision_to_dict(decision: RiskDecision) -> dict:
    return {
        "allow_new_position": decision.allow_new_position,
        "risk_level": decision.risk_level,
        "position_size_pct": decision.position_size_pct,
        "reason": decision.reason,
        "drawdown_pct": decision.drawdown_pct,
        "loss_streak": decision.loss_streak,
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def signal_quality_score(final: dict, horizon_results: list[dict]) -> tuple[float, list[str]]:
    signal = final.get("signal", "HOLD")
    if signal not in {"LONG", "SHORT"}:
        return 0.0, ["no_trade_signal"]

    confidence = _clamp(float(final.get("confidence", 0.0)), 0.0, 1.0)
    horizon_count = max(1, len(horizon_results))
    vote_key = "long_votes" if signal == "LONG" else "short_votes"
    agree_votes = int(final.get(vote_key, 0) or 0)
    agreement = _clamp(agree_votes / horizon_count, 0.0, 1.0)

    strongest = max((float(item.get("confidence", 0.0) or 0.0) for item in horizon_results), default=confidence)
    strongest = _clamp(strongest, 0.0, 1.0)

    regime = final.get("market_regime", "model")
    if regime == "range":
        regime_factor = 0.85
        regime_reason = "range_regime_size_discount"
    elif regime == "trend_or_unclear":
        regime_factor = 1.0
        regime_reason = "trend_or_unclear"
    else:
        regime_factor = 0.95
        regime_reason = f"regime_{regime}"

    score = (0.50 * confidence + 0.35 * agreement + 0.15 * strongest) * regime_factor
    reasons = [
        f"confidence={confidence:.2f}",
        f"agreement={agree_votes}/{horizon_count}",
        f"strongest_confidence={strongest:.2f}",
        regime_reason,
    ]
    return _clamp(score, 0.0, 1.0), reasons


def apply_dynamic_position_sizing(
    risk: dict,
    final: dict,
    horizon_results: list[dict],
    min_position_size_pct: float,
    max_position_size_pct: float,
) -> dict:
    updated = dict(risk)
    original_cap = _clamp(float(risk.get("position_size_pct", 0.0) or 0.0), 0.0, 1.0)
    min_size = _clamp(min_position_size_pct, 0.0, 1.0)
    max_size = _clamp(max_position_size_pct, min_size, 1.0)

    if not risk.get("allow_new_position", False):
        updated.update(
            {
                "dynamic_position_sizing": True,
                "base_position_size_pct": original_cap,
                "signal_quality_score": 0.0,
                "position_size_pct": 0.0,
                "position_size_reason": "risk_blocked",
            }
        )
        return updated

    score, quality_reasons = signal_quality_score(final, horizon_results)
    if final.get("signal") not in {"LONG", "SHORT"}:
        dynamic_size = 0.0
        reason = "no_trade_signal"
    else:
        dynamic_size = min_size + (max_size - min_size) * score
        dynamic_size = min(dynamic_size, original_cap)
        reason = "dynamic_signal_quality"

    updated.update(
        {
            "dynamic_position_sizing": True,
            "base_position_size_pct": original_cap,
            "signal_quality_score": float(score),
            "position_size_pct": float(_clamp(dynamic_size, 0.0, original_cap)),
            "position_size_reason": reason,
            "position_size_inputs": quality_reasons,
        }
    )
    return updated
