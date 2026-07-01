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
        return RiskDecision(False, "blocked", 0.0, "loss_streak_exceeded", drawdown_pct, loss_streak)
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
