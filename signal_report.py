import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from telegram_utils import print_telegram_disabled, telegram_disabled


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = BASE_DIR / "paper_state.json"
load_dotenv(BASE_DIR / ".env")


def load_state(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Paper state file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def send_telegram_message(message: str) -> bool:
    if telegram_disabled():
        print_telegram_disabled()
        return False

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    strict = os.getenv("TELEGRAM_STRICT", "0") == "1"
    if not token or not chat_id:
        error = "Telegram TOKEN or CHAT_ID is not configured."
        print(error)
        if strict:
            raise RuntimeError(error)
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        response.raise_for_status()
        print("Telegram message sent.")
        return True
    except Exception as exc:
        print(f"Telegram send failed: {exc}")
        if strict:
            raise
        return False


def _trade_signal(side: str) -> str:
    if "LONG" in side:
        return "LONG"
    if "SHORT" in side:
        return "SHORT"
    return "UNKNOWN"


def _parse_summary(summary: str | None) -> dict:
    if not summary:
        return {"signal": "UNKNOWN", "strategy": "unknown", "regime": "unknown", "confidence": None}
    parts = summary.split(";")
    first = parts[0].strip()
    signal = first.split(" ", 1)[0] if first else "UNKNOWN"
    strategy = "unknown"
    regime = "unknown"
    if " via " in first:
        context = first.split(" via ", 1)[1]
        if "/" in context:
            strategy, regime = context.split("/", 1)
        else:
            strategy = context
    confidence = None
    for part in parts:
        part = part.strip()
        if part.startswith("confidence="):
            try:
                confidence = float(part.split("=", 1)[1].replace("%", ""))
            except ValueError:
                confidence = None
    return {"signal": signal, "strategy": strategy, "regime": regime, "confidence": confidence}


def _group_closed_trades(trades: list[dict]) -> list[dict]:
    open_by_direction: dict[str, dict] = {}
    closed = []
    for trade in trades:
        side = str(trade.get("side", ""))
        direction = _trade_signal(side)
        if side.startswith("OPEN"):
            open_by_direction[direction] = trade
        elif side.startswith("CLOSE"):
            opened = open_by_direction.pop(direction, {})
            signal_reason = trade.get("signal_reason") or opened.get("signal_reason")
            parsed = _parse_summary(signal_reason)
            closed.append(
                {
                    "open_timestamp": opened.get("timestamp"),
                    "close_timestamp": trade.get("timestamp"),
                    "direction": direction,
                    "pnl": float(trade.get("pnl", 0) or 0),
                    "reason": trade.get("reason", "unknown"),
                    "signal_reason": signal_reason,
                    "signal": parsed["signal"] if parsed["signal"] != "UNKNOWN" else direction,
                    "strategy": parsed["strategy"],
                    "regime": parsed["regime"],
                    "confidence": parsed["confidence"],
                }
            )
    return closed


def _summarize_group(items: list[dict]) -> dict:
    pnl_values = [float(item.get("pnl", 0) or 0) for item in items]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    confidences = [item["confidence"] for item in items if item.get("confidence") is not None]
    return {
        "count": len(items),
        "pnl": float(sum(pnl_values)),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": float(len(wins) / len(items) * 100) if items else 0.0,
        "avg_pnl": float(sum(pnl_values) / len(items)) if items else 0.0,
        "avg_confidence": float(sum(confidences) / len(confidences)) if confidences else None,
    }


def _group_by(items: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[str(item.get(key) or "unknown")].append(item)
    return {name: _summarize_group(group) for name, group in sorted(groups.items())}


def build_report(state: dict) -> dict:
    trades = state.get("trades", [])
    closed = _group_closed_trades(trades)
    open_trades = [trade for trade in trades if str(trade.get("side", "")).startswith("OPEN")]
    close_reasons = Counter(str(trade.get("reason", "unknown")) for trade in closed)
    best = max(closed, key=lambda trade: trade["pnl"], default=None)
    worst = min(closed, key=lambda trade: trade["pnl"], default=None)
    missing_reason_count = sum(1 for trade in trades if not trade.get("signal_reason"))

    return {
        "closed_trade_count": len(closed),
        "open_trade_count": len(open_trades),
        "missing_signal_reason_count": missing_reason_count,
        "last_signal_reason": state.get("last_signal_reason"),
        "last_signal_reasons": state.get("last_signal_reasons", []),
        "overall": _summarize_group(closed),
        "by_signal": _group_by(closed, "signal"),
        "by_direction": _group_by(closed, "direction"),
        "by_regime": _group_by(closed, "regime"),
        "by_strategy": _group_by(closed, "strategy"),
        "close_reasons": dict(close_reasons),
        "best_trade": best,
        "worst_trade": worst,
    }


def _money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}${value:.2f}"


def _format_group_lines(groups: dict[str, dict], title: str, limit: int = 6) -> list[str]:
    if not groups:
        return [f"{title}: n/a"]
    lines = [f"<b>{title}</b>"]
    ranked = sorted(groups.items(), key=lambda item: item[1]["pnl"], reverse=True)
    for name, stats in ranked[:limit]:
        confidence = stats.get("avg_confidence")
        confidence_text = f"، اطمینان {confidence:.2f}" if confidence is not None else ""
        lines.append(
            f"- {name}: {stats['count']} معامله، سود/ضرر {_money(stats['pnl'])}، "
            f"نرخ برد {stats['win_rate_pct']:.1f}%{confidence_text}"
        )
    return lines


def build_message(report: dict, symbol: str) -> str:
    overall = report["overall"]
    lines = [
        f"<b>گزارش کیفیت سیگنال‌ها - {symbol}</b>",
        f"معاملات بسته‌شده: {report['closed_trade_count']} | رویدادهای باز: {report['open_trade_count']}",
        f"سود/ضرر کل سیگنال‌ها: {_money(overall['pnl'])}",
        f"نرخ برد: {overall['win_rate_pct']:.1f}% | میانگین هر معامله: {_money(overall['avg_pnl'])}",
    ]
    if report["missing_signal_reason_count"]:
        lines.append(f"رویدادهای قدیمی بدون دلیل ثبت‌شده: {report['missing_signal_reason_count']}")

    lines.append("")
    lines.extend(_format_group_lines(report["by_signal"], "بر اساس سیگنال"))
    lines.append("")
    lines.extend(_format_group_lines(report["by_regime"], "بر اساس وضعیت بازار"))

    best = report.get("best_trade")
    worst = report.get("worst_trade")
    lines.append("")
    if best:
        lines.append(f"بهترین: {_money(best['pnl'])} | {best.get('signal_reason') or best.get('reason')}")
    else:
        lines.append("بهترین: n/a")
    if worst:
        lines.append(f"بدترین: {_money(worst['pnl'])} | {worst.get('signal_reason') or worst.get('reason')}")
    else:
        lines.append("بدترین: n/a")

    last_reason = report.get("last_signal_reason")
    if last_reason:
        lines.append("")
        lines.append(f"آخرین سیگنال: {last_reason}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze paper-trading performance by signal explanation.")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--telegram", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict:
    state = load_state(args.state_file)
    report = build_report(state)
    output = {"report": report}
    print(json.dumps(output, indent=2))
    if args.telegram:
        send_telegram_message(build_message(report, args.symbol))
    return output


if __name__ == "__main__":
    main(build_parser().parse_args())
