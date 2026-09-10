import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from performance_report import build_report, current_mark_price, load_state, send_telegram_message
from trading_gate import profit_factor


def load_empty_state(initial_capital: float) -> dict:
    return {
        "capital": initial_capital,
        "position": 0,
        "entry_price": 0.0,
        "notional": 0.0,
        "trades": [],
        "last_timestamp": None,
    }


def closed_trades(state: dict) -> list[dict]:
    return [trade for trade in state.get("trades", []) if str(trade.get("side", "")).startswith("CLOSE")]


def first_trade_timestamp(state: dict) -> str | None:
    timestamps = [trade.get("timestamp") for trade in state.get("trades", []) if trade.get("timestamp")]
    return min(timestamps) if timestamps else state.get("last_timestamp")


def benchmark_buy_hold_pct(data_path: str | Path, start_timestamp: str | None, end_timestamp: str | None) -> dict:
    data = pd.read_csv(data_path)
    if data.empty:
        return {"ok": False, "reason": "empty_data"}
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed")
    data = data.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    if data.empty:
        return {"ok": False, "reason": "empty_data_after_cleaning"}

    start_ts = pd.to_datetime(start_timestamp, format="mixed") if start_timestamp else data.iloc[0]["timestamp"]
    end_ts = pd.to_datetime(end_timestamp, format="mixed") if end_timestamp else data.iloc[-1]["timestamp"]
    start_rows = data[data["timestamp"] >= start_ts]
    end_rows = data[data["timestamp"] <= end_ts]
    if start_rows.empty or end_rows.empty:
        return {"ok": False, "reason": "timestamp_out_of_data_range"}
    start = start_rows.iloc[0]
    end = end_rows.iloc[-1]
    start_price = float(start["close"])
    end_price = float(end["close"])
    return {
        "ok": True,
        "start_timestamp": str(start["timestamp"]),
        "end_timestamp": str(end["timestamp"]),
        "start_price": start_price,
        "end_price": end_price,
        "return_pct": float((end_price / start_price - 1) * 100) if start_price else 0.0,
    }


def pnl_groups(trades: list[dict], key_fn) -> dict:
    groups: dict[str, dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for trade in trades:
        key = str(key_fn(trade))
        pnl = float(trade.get("pnl", 0) or 0)
        groups[key]["count"] += 1
        groups[key]["pnl"] += pnl
    return {key: {"count": item["count"], "pnl": float(item["pnl"])} for key, item in sorted(groups.items())}


def expectancy(closes: list[dict]) -> dict:
    wins = [float(trade.get("pnl", 0) or 0) for trade in closes if float(trade.get("pnl", 0) or 0) > 0]
    losses = [float(trade.get("pnl", 0) or 0) for trade in closes if float(trade.get("pnl", 0) or 0) < 0]
    count = len(closes)
    win_rate = len(wins) / count if count else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy_per_trade": float(win_rate * avg_win + (1 - win_rate) * avg_loss) if count else 0.0,
        "payoff_ratio": float(avg_win / abs(avg_loss)) if avg_loss else None,
    }


def verdict(report: dict, factor: float | None, alpha_pct: float | None, args: argparse.Namespace) -> dict:
    failures = []
    warnings = []
    if report["closed_trade_count"] < args.min_closed_trades:
        warnings.append(f"closed trades {report['closed_trade_count']} < sample target {args.min_closed_trades}")
    if report["total_return_pct"] < args.min_return_pct:
        failures.append(f"return {report['total_return_pct']:.2f}% < {args.min_return_pct:.2f}%")
    if alpha_pct is not None and alpha_pct < args.min_alpha_pct:
        failures.append(f"alpha_vs_buy_hold {alpha_pct:.2f}% < {args.min_alpha_pct:.2f}%")
    if factor is not None and factor < args.min_profit_factor:
        failures.append(f"profit factor {factor:.2f} < {args.min_profit_factor:.2f}")
    if report["max_drawdown_pct"] <= -abs(args.max_drawdown_pct):
        failures.append(f"drawdown {report['max_drawdown_pct']:.2f}% <= -{abs(args.max_drawdown_pct):.2f}%")
    status = "REJECT" if failures else "CANDIDATE" if warnings else "PASS"
    return {"status": status, "failures": failures, "warnings": warnings}


def analyze_system(system: dict, args: argparse.Namespace) -> dict:
    state_path = Path(system["state_path"])
    state = load_state(state_path) if state_path.exists() else load_empty_state(args.initial_capital)
    mark_price, mark_timestamp = current_mark_price(system["data_path"])
    report = build_report(state, args.initial_capital, mark_price, mark_timestamp)
    closes = closed_trades(state)
    factor = profit_factor(state)
    bench = benchmark_buy_hold_pct(system["data_path"], first_trade_timestamp(state), report.get("last_timestamp"))
    alpha = None
    if bench.get("ok"):
        alpha = float(report["total_return_pct"]) - float(bench["return_pct"])
    return {
        **system,
        "report": report,
        "buy_hold": bench,
        "alpha_vs_buy_hold_pct": alpha,
        "profit_factor": factor,
        "expectancy": expectancy(closes),
        "by_side": pnl_groups(closes, lambda trade: "LONG" if trade.get("side") == "CLOSE_LONG" else "SHORT"),
        "by_reason": pnl_groups(closes, lambda trade: trade.get("reason") or "unknown"),
        "verdict": verdict(report, factor, alpha, args),
        "trading_paused": bool(state.get("trading_paused")),
        "trading_pause_reason": state.get("trading_pause_reason"),
    }


def parse_systems(value: str) -> list[dict]:
    systems = []
    for raw in value.split(";"):
        item = raw.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) != 4:
            raise ValueError("Each system must be formatted as label|symbol|data_path|state_path")
        label, symbol, data_path, state_path = parts
        systems.append({"label": label, "symbol": symbol, "data_path": data_path, "state_path": state_path})
    if not systems:
        raise ValueError("At least one system is required.")
    return systems


def format_factor(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def build_message(results: list[dict]) -> str:
    lines = ["<b>Trading Edge Report</b>"]
    for item in results:
        report = item["report"]
        buy_hold = item["buy_hold"]
        buy_hold_text = f"{buy_hold['return_pct']:+.2f}%" if buy_hold.get("ok") else "n/a"
        alpha = item.get("alpha_vs_buy_hold_pct")
        alpha_text = "n/a" if alpha is None else f"{alpha:+.2f}%"
        verdict_item = item["verdict"]
        pause = " | PAUSED" if item.get("trading_paused") else ""
        lines.extend(
            [
                "",
                f"<b>{item['label']} - {verdict_item['status']}{pause}</b>",
                f"Return: {report['total_return_pct']:+.2f}% | Buy&Hold: {buy_hold_text} | Alpha: {alpha_text}",
                f"Closed trades: {report['closed_trade_count']} | Win rate: {report['win_rate_pct']:.1f}% | Profit factor: {format_factor(item['profit_factor'])}",
                f"Drawdown: {report['max_drawdown_pct']:.2f}% | Expectancy/trade: {item['expectancy']['expectancy_per_trade']:+.4f}",
            ]
        )
        if item.get("trading_pause_reason"):
            lines.append(f"Pause reason: {item['trading_pause_reason']}")
        if verdict_item["failures"]:
            lines.append("Failures: " + "; ".join(verdict_item["failures"]))
        if verdict_item["warnings"]:
            lines.append("Warnings: " + "; ".join(verdict_item["warnings"]))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark paper systems against buy-and-hold and reject weak edges.")
    parser.add_argument(
        "--systems",
        default=(
            "1h Production|BTC/USDT|dataset/1h-btc_history.csv|paper_state.json;"
            "15m Staging|BTC/USDT|dataset/15m_btc_history_5000.csv|paper_state_15m_staging.json;"
            "5m Staging|BTC/USDT|dataset/5m_btc_history.csv|paper_state_5m_staging.json"
        ),
    )
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--min-closed-trades", type=int, default=50)
    parser.add_argument("--min-return-pct", type=float, default=0.0)
    parser.add_argument("--min-alpha-pct", type=float, default=0.0)
    parser.add_argument("--min-profit-factor", type=float, default=1.10)
    parser.add_argument("--max-drawdown-pct", type=float, default=2.0)
    parser.add_argument("--telegram", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict:
    results = [analyze_system(system, args) for system in parse_systems(args.systems)]
    output = {"systems": results}
    print(json.dumps(output, indent=2, default=str))
    if args.telegram:
        send_telegram_message(build_message(results))
    return output


if __name__ == "__main__":
    main(build_parser().parse_args())
