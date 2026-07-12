import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
DEFAULT_STATE_PATH = BASE_DIR / "paper_state.json"
load_dotenv(BASE_DIR / ".env")


def load_state(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Paper state file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def current_mark_price(data_path: str | Path) -> tuple[float, str]:
    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError(f"Market data is empty: {data_path}")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    row = data.iloc[-1]
    return float(row["close"]), str(row["timestamp"])


def send_telegram_message(message: str) -> bool:
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


def position_name(position: int) -> str:
    if position == 1:
        return "LONG"
    if position == -1:
        return "SHORT"
    return "FLAT"


def unrealized_pnl(state: dict, mark_price: float) -> float:
    position = int(state.get("position", 0))
    if position == 0:
        return 0.0
    entry_price = float(state.get("entry_price", 0) or 0)
    notional = float(state.get("notional", 0) or 0)
    if entry_price <= 0 or notional <= 0:
        return 0.0
    return notional * position * ((mark_price - entry_price) / entry_price)


def max_drawdown_pct(values: list[float]) -> float:
    if not values:
        return 0.0
    series = pd.Series(values, dtype=float)
    drawdown = series / series.cummax() - 1
    return float(drawdown.min() * 100)


def daily_realized_pnl(closed_trades: list[dict]) -> list[dict]:
    if not closed_trades:
        return []
    frame = pd.DataFrame(closed_trades)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["day"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    grouped = frame.groupby("day")["pnl"].sum().reset_index()
    return [{"day": row["day"], "pnl": float(row["pnl"])} for _, row in grouped.iterrows()]


def build_report(state: dict, initial_capital: float, mark_price: float, mark_timestamp: str) -> dict:
    trades = state.get("trades", [])
    closed_trades = [trade for trade in trades if str(trade.get("side", "")).startswith("CLOSE")]
    open_trades = [trade for trade in trades if str(trade.get("side", "")).startswith("OPEN")]
    wins = [trade for trade in closed_trades if float(trade.get("pnl", 0) or 0) > 0]
    losses = [trade for trade in closed_trades if float(trade.get("pnl", 0) or 0) < 0]
    realized_pnl = float(sum(float(trade.get("pnl", 0) or 0) for trade in closed_trades))
    fees = float(sum(float(trade.get("fee", 0) or 0) for trade in trades))
    unrealized = unrealized_pnl(state, mark_price)
    capital = float(state.get("capital", initial_capital))
    equity = capital + unrealized
    closed_capital_curve = [initial_capital]
    for trade in closed_trades:
        if "capital" in trade:
            closed_capital_curve.append(float(trade["capital"]))
    closed_capital_curve.append(equity)

    best_trade = max(closed_trades, key=lambda trade: float(trade.get("pnl", 0) or 0), default=None)
    worst_trade = min(closed_trades, key=lambda trade: float(trade.get("pnl", 0) or 0), default=None)
    position = int(state.get("position", 0) or 0)
    total_return_pct = (equity / initial_capital - 1) * 100 if initial_capital else 0.0

    return {
        "mark_timestamp": mark_timestamp,
        "mark_price": mark_price,
        "initial_capital": initial_capital,
        "capital": capital,
        "equity": float(equity),
        "realized_pnl": realized_pnl,
        "unrealized_pnl": float(unrealized),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": max_drawdown_pct(closed_capital_curve),
        "trade_count": len(trades),
        "open_trade_count": len(open_trades),
        "closed_trade_count": len(closed_trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": float(len(wins) / len(closed_trades) * 100) if closed_trades else 0.0,
        "fees": fees,
        "position": position_name(position),
        "entry_price": float(state.get("entry_price", 0) or 0),
        "notional": float(state.get("notional", 0) or 0),
        "last_timestamp": state.get("last_timestamp"),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "daily_realized_pnl": daily_realized_pnl(closed_trades),
    }


def format_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}${value:.2f}"


def build_message(report: dict, symbol: str, drawdown_alert_pct: float) -> str:
    alert_line = ""
    if report["max_drawdown_pct"] <= -abs(drawdown_alert_pct):
        alert_line = f"\nهشدار ریسک: افت سرمایه {report['max_drawdown_pct']:.2f}%"

    best = report.get("best_trade") or {}
    worst = report.get("worst_trade") or {}
    best_line = f"بهترین معامله: {format_money(float(best.get('pnl', 0) or 0))}" if best else "بهترین معامله: n/a"
    worst_line = f"بدترین معامله: {format_money(float(worst.get('pnl', 0) or 0))}" if worst else "بدترین معامله: n/a"

    return (
        f"<b>گزارش عملکرد Paper Trading - {symbol}</b>\n"
        f"زمان: {report['mark_timestamp']} UTC\n"
        f"قیمت: ${report['mark_price']:.2f}\n"
        f"ارزش حساب: ${report['equity']:.2f} ({report['total_return_pct']:+.2f}%)\n"
        f"سرمایه آزاد: ${report['capital']:.2f}\n"
        f"سود/ضرر قطعی‌شده: {format_money(report['realized_pnl'])}\n"
        f"سود/ضرر باز: {format_money(report['unrealized_pnl'])}\n"
        f"پوزیشن: {report['position']}\n"
        f"معاملات: {report['closed_trade_count']} بسته‌شده / {report['trade_count']} رویداد\n"
        f"نرخ برد: {report['win_rate_pct']:.1f}%\n"
        f"بیشترین افت سرمایه: {report['max_drawdown_pct']:.2f}%\n"
        f"{best_line} | {worst_line}"
        f"{alert_line}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and send paper-trading performance reports.")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--data", default=str(DATA_DIR / "1h-btc_history.csv"))
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--drawdown-alert-pct", type=float, default=5.0)
    parser.add_argument("--telegram-label", default="Paper Trading")
    parser.add_argument("--telegram", action="store_true")
    return parser


def build_message_fa(report: dict, symbol: str, drawdown_alert_pct: float, label: str) -> str:
    alert_line = ""
    if report["max_drawdown_pct"] <= -abs(drawdown_alert_pct):
        alert_line = f"\nهشدار ریسک: افت سرمایه {report['max_drawdown_pct']:.2f}%"

    best = report.get("best_trade") or {}
    worst = report.get("worst_trade") or {}
    best_line = f"بهترین معامله: {format_money(float(best.get('pnl', 0) or 0))}" if best else "بهترین معامله: n/a"
    worst_line = f"بدترین معامله: {format_money(float(worst.get('pnl', 0) or 0))}" if worst else "بدترین معامله: n/a"

    return (
        f"<b>گزارش عملکرد {label} - {symbol}</b>\n"
        f"زمان: {report['mark_timestamp']} UTC\n"
        f"قیمت: ${report['mark_price']:.2f}\n"
        f"ارزش حساب: ${report['equity']:.2f} ({report['total_return_pct']:+.2f}%)\n"
        f"سرمایه آزاد: ${report['capital']:.2f}\n"
        f"سود/ضرر قطعی‌شده: {format_money(report['realized_pnl'])}\n"
        f"سود/ضرر باز: {format_money(report['unrealized_pnl'])}\n"
        f"پوزیشن: {report['position']}\n"
        f"معاملات: {report['closed_trade_count']} بسته‌شده / {report['trade_count']} رویداد\n"
        f"نرخ برد: {report['win_rate_pct']:.1f}%\n"
        f"بیشترین افت سرمایه: {report['max_drawdown_pct']:.2f}%\n"
        f"{best_line} | {worst_line}"
        f"{alert_line}"
    )


def main(args: argparse.Namespace) -> dict:
    state = load_state(args.state_file)
    mark_price, mark_timestamp = current_mark_price(args.data)
    report = build_report(state, args.initial_capital, mark_price, mark_timestamp)
    output = {"report": report}
    print(json.dumps(output, indent=2))
    if args.telegram:
        send_telegram_message(build_message_clean(report, args.symbol, args.drawdown_alert_pct, args.telegram_label))
    return output


def build_message_clean(report: dict, symbol: str, drawdown_alert_pct: float, label: str) -> str:
    alert_line = ""
    if report["max_drawdown_pct"] <= -abs(drawdown_alert_pct):
        alert_line = f"\nRisk alert: drawdown {report['max_drawdown_pct']:.2f}%"

    best = report.get("best_trade") or {}
    worst = report.get("worst_trade") or {}
    best_line = f"Best trade: {format_money(float(best.get('pnl', 0) or 0))}" if best else "Best trade: n/a"
    worst_line = f"Worst trade: {format_money(float(worst.get('pnl', 0) or 0))}" if worst else "Worst trade: n/a"

    return (
        f"<b>{label} Performance - {symbol}</b>\n"
        f"Time: {report['mark_timestamp']} UTC\n"
        f"Price: ${report['mark_price']:.2f}\n"
        f"Equity: ${report['equity']:.2f} ({report['total_return_pct']:+.2f}%)\n"
        f"Free capital: ${report['capital']:.2f}\n"
        f"Realized PnL: {format_money(report['realized_pnl'])}\n"
        f"Open PnL: {format_money(report['unrealized_pnl'])}\n"
        f"Position: {report['position']}\n"
        f"Trades: {report['closed_trade_count']} closed / {report['trade_count']} events\n"
        f"Win rate: {report['win_rate_pct']:.1f}%\n"
        f"Max drawdown: {report['max_drawdown_pct']:.2f}%\n"
        f"{best_line} | {worst_line}"
        f"{alert_line}"
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
