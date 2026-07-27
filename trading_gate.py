import argparse
import json
import os
from pathlib import Path

from performance_report import build_report, current_mark_price, load_state, send_telegram_message


def profit_factor(state: dict) -> float | None:
    closed = [trade for trade in state.get("trades", []) if str(trade.get("side", "")).startswith("CLOSE")]
    gross_profit = sum(float(trade.get("pnl", 0) or 0) for trade in closed if float(trade.get("pnl", 0) or 0) > 0)
    gross_loss = abs(sum(float(trade.get("pnl", 0) or 0) for trade in closed if float(trade.get("pnl", 0) or 0) < 0))
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return gross_profit / gross_loss


def set_output(name: str, value: str) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def pause_state(path: Path, state: dict, reason: str, report: dict, factor: float | None) -> None:
    state["trading_paused"] = True
    state["trading_pause_reason"] = reason
    state["trading_pause_report"] = {
        "closed_trade_count": report["closed_trade_count"],
        "total_return_pct": report["total_return_pct"],
        "max_drawdown_pct": report["max_drawdown_pct"],
        "profit_factor": factor,
        "last_timestamp": report.get("last_timestamp"),
    }
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_message(label: str, reason: str, report: dict, factor: float | None) -> str:
    factor_text = "n/a" if factor is None else f"{factor:.2f}" if factor != float("inf") else "inf"
    return (
        f"<b>{label} Gate</b>\n"
        f"Status: PAUSED\n"
        f"Reason: {reason}\n"
        f"Closed trades: {report['closed_trade_count']}\n"
        f"Return: {report['total_return_pct']:+.2f}%\n"
        f"Drawdown: {report['max_drawdown_pct']:.2f}%\n"
        f"Profit factor: {factor_text}\n"
        f"Last processed: {report.get('last_timestamp')}"
    )


def evaluate(args: argparse.Namespace) -> dict:
    path = Path(args.state_file)
    state = load_state(path)
    mark_price, mark_timestamp = current_mark_price(args.data)
    report = build_report(state, args.initial_capital, mark_price, mark_timestamp)
    factor = profit_factor(state)
    reasons = []

    if state.get("trading_paused"):
        reason = str(state.get("trading_pause_reason") or "already_paused")
        set_output("allowed", "false")
        set_output("reason", reason)
        output = {"allowed": False, "reason": reason, "report": report, "profit_factor": factor}
        print(json.dumps(output, indent=2))
        return output

    if report["max_drawdown_pct"] <= -abs(args.max_drawdown_pct):
        reasons.append(f"drawdown {report['max_drawdown_pct']:.2f}% <= -{abs(args.max_drawdown_pct):.2f}%")

    enough_trades = report["closed_trade_count"] >= args.min_closed_trades
    if enough_trades:
        if report["total_return_pct"] < args.min_return_pct:
            reasons.append(f"return {report['total_return_pct']:.2f}% < {args.min_return_pct:.2f}% after {args.min_closed_trades} trades")
        if factor is not None and factor < args.min_profit_factor:
            reasons.append(f"profit_factor {factor:.2f} < {args.min_profit_factor:.2f} after {args.min_closed_trades} trades")

    allowed = not reasons
    reason = "; ".join(reasons)
    if allowed:
        state.pop("trading_pause_pending", None)
    else:
        pause_state(path, state, reason, report, factor)
        if args.telegram:
            send_telegram_message(build_message(args.label, reason, report, factor))

    set_output("allowed", "true" if allowed else "false")
    set_output("reason", reason)
    output = {"allowed": allowed, "reason": reason, "report": report, "profit_factor": factor}
    print(json.dumps(output, indent=2))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate paper trading based on live paper performance.")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--label", default="Paper Trading")
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--min-closed-trades", type=int, default=50)
    parser.add_argument("--min-return-pct", type=float, default=0.0)
    parser.add_argument("--min-profit-factor", type=float, default=1.1)
    parser.add_argument("--max-drawdown-pct", type=float, default=2.0)
    parser.add_argument("--telegram", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
