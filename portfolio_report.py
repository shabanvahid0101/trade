import argparse
import json
from pathlib import Path

from performance_report import build_report, current_mark_price, format_money, load_state, send_telegram_message


def parse_systems(value: str) -> list[dict]:
    systems = []
    for raw_item in value.split(";"):
        item = raw_item.strip()
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


def system_report(system: dict, initial_capital: float) -> dict:
    state = load_state(system["state_path"])
    mark_price, mark_timestamp = current_mark_price(system["data_path"])
    report = build_report(state, initial_capital, mark_price, mark_timestamp)
    return {**system, "report": report}


def build_message(reports: list[dict]) -> str:
    lines = ["<b>Trading Tool Portfolio Report</b>"]
    total_equity = 0.0
    total_initial = 0.0
    for item in reports:
        report = item["report"]
        total_equity += float(report["equity"])
        total_initial += float(report["initial_capital"])
        lines.extend(
            [
                "",
                f"<b>{item['label']} - {item['symbol']}</b>",
                f"Data time: {report['mark_timestamp']} UTC",
                f"Price: ${report['mark_price']:.2f}",
                f"Equity: ${report['equity']:.2f} ({report['total_return_pct']:+.2f}%)",
                f"Realized PnL: {format_money(report['realized_pnl'])} | Open PnL: {format_money(report['unrealized_pnl'])}",
                f"Position: {report['position']} | Closed trades: {report['closed_trade_count']} | Win rate: {report['win_rate_pct']:.1f}%",
                f"Drawdown: {report['max_drawdown_pct']:.2f}% | Last processed: {report.get('last_timestamp')}",
            ]
        )
    if total_initial:
        total_return = (total_equity / total_initial - 1) * 100
        lines.extend(["", f"<b>Combined paper equity:</b> ${total_equity:.2f} ({total_return:+.2f}%)"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combined paper-trading report for production and staging systems.")
    parser.add_argument(
        "--systems",
        default=(
            "1h Production|BTC/USDT|dataset/1h-btc_history.csv|paper_state.json;"
            "15m Staging|BTC/USDT|dataset/15m_btc_history_5000.csv|paper_state_15m_staging.json"
        ),
    )
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--telegram", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict:
    reports = [system_report(system, args.initial_capital) for system in parse_systems(args.systems)]
    output = {"systems": reports}
    print(json.dumps(output, indent=2, default=str))
    if args.telegram:
        send_telegram_message(build_message(reports))
    return output


if __name__ == "__main__":
    main(build_parser().parse_args())
