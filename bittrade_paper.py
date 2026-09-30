"""Forward-only BitTrade paper portfolio and durable Telegram outbox.

Only public exchange GET endpoints are used. No historical entry/exit fills.
Persist portfolio/outbox to Git BEFORE sending, then persist delivery receipts.
Telegram has no idempotency key: an ambiguous timeout/crash can duplicate a
message on retry, but cannot open another paper trade for the same candle.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bittrade_data import PublicClient, PERIODS, utc, write_json, atomic_text

DEFAULTS = {"mode": "experimental", "initial_jpy": 100000.0, "position_fraction": .1,
            "max_positions": 3, "fee_rate": .001, "slippage_bps": 5,
            "stop_loss_pct": .01, "take_profit_pct": .02, "exit_probability": .40,
            "max_drawdown_pct": 5., "max_loss_streak": 5,
            "max_spread_bps": 40., "min_24h_jpy": 10000000.}


def epoch(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return stamp.timestamp()


def create_state(now, mode="experimental"):
    config = {**DEFAULTS, "mode": mode}
    if mode not in ("experimental", "strict"):
        raise ValueError("Unknown paper mode")
    return {"version": 1, "created_at": utc(now), "config": config,
            "cash_jpy": config["initial_jpy"], "positions": {}, "closed_trades": [],
            "seen_signals": {}, "outbox": [], "peak_equity_jpy": config["initial_jpy"],
            "max_drawdown_pct": 0., "paused": False, "live_trading_enabled": False}


def event(state, event_id, kind, text, now):
    if any(e["id"] == event_id for e in state["outbox"]):
        return
    state["outbox"].append({"id": event_id, "kind": kind, "created_at": utc(now),
                             "text": text, "sent_at": None, "attempts": 0})


def book_levels(book, side, now):
    age = now - float(book["ts"])/1000
    if not -5 <= age <= 60:
        raise ValueError("Stale order book")
    levels = book["tick"][side]
    result = []
    for price, size in levels:
        price, size = float(price), float(size)
        if not all(math.isfinite(v) and v > 0 for v in (price, size)):
            raise ValueError("Invalid order book level")
        result.append((price, size))
    if not result:
        raise ValueError("Empty order book")
    return sorted(result, reverse=side == "bids")


def fill_cost(levels, quantity, slippage, buy):
    if not math.isfinite(quantity) or quantity <= 0:
        raise ValueError("Positive finite fill quantity required")
    remaining = quantity
    value = 0.
    for price, size in levels:
        take = min(remaining, size)
        value += take * price * (1+slippage if buy else 1-slippage)
        remaining -= take
        if remaining <= max(1e-12, quantity*1e-10):
            return value
    raise ValueError("Insufficient visible depth for full paper fill")


def buy_fill(levels, budget, market, config):
    remaining = budget
    quantity = 0.
    fee, slip = config["fee_rate"], config["slippage_bps"]/10000
    for price, size in levels:
        unit_cost = price*(1+slip)*(1+fee)
        take = min(size, remaining/unit_cost)
        quantity += take
        remaining -= take*unit_cost
        if remaining <= 1e-8:
            break
    precision = int(market["amount-precision"])
    quantity = float(Decimal(str(quantity)).quantize(Decimal(1).scaleb(-precision), rounding=ROUND_DOWN))
    quantity = min(quantity, float(market.get("sell-market-max-order-amt", quantity)))
    if quantity <= 0 or quantity < float(market.get("sell-market-min-order-amt", 0)):
        raise ValueError("Paper order below minimum quantity")
    value = fill_cost(levels, quantity, slip, True)
    if value < float(market.get("min-order-value", 0)) or value*(1+fee) > budget+1e-7:
        raise ValueError("Paper order below minimum value or above budget")
    return quantity, value, value*fee


def mark(state, tickers):
    config = state["config"]
    equity = state["cash_jpy"]
    unrealized = 0.
    stale = []
    for key, p in state["positions"].items():
        bid = float(tickers.get(p["symbol"], {}).get("bid", 0))
        if not math.isfinite(bid) or bid <= 0:
            bid = p.get("last_bid", p["entry_price"])
            stale.append(key)
        else:
            p["last_bid"] = bid
        liquidation = p["quantity"]*bid*(1-config["slippage_bps"]/10000)*(1-config["fee_rate"])
        equity += liquidation
        unrealized += liquidation-p["cost_jpy"]
    trades = state["closed_trades"]
    profit = sum(t["net_pnl_jpy"] for t in trades)
    wins = sum(max(t["net_pnl_jpy"], 0) for t in trades)
    losses = -sum(min(t["net_pnl_jpy"], 0) for t in trades)
    state["peak_equity_jpy"] = max(state["peak_equity_jpy"], equity)
    drawdown = (1-equity/state["peak_equity_jpy"])*100
    state["max_drawdown_pct"] = max(state["max_drawdown_pct"], drawdown)
    return {"equity_jpy": equity, "cash_jpy": state["cash_jpy"], "realized_pnl_jpy": profit,
            "unrealized_pnl_jpy": unrealized, "return_pct": (equity/config["initial_jpy"]-1)*100,
            "closed_trades": len(trades), "open_positions": len(state["positions"]),
            "profit_factor": wins/losses if losses else None,
            "win_rate_pct": 100*sum(t["net_pnl_jpy"] > 0 for t in trades)/len(trades) if trades else None,
            "drawdown_pct": drawdown, "max_drawdown_pct": state["max_drawdown_pct"],
            "stale_marks": stale}


def validate_snapshot(report, ticker_data, now):
    if not -5 <= now-float(ticker_data["ts"])/1000 <= 300:
        raise ValueError("Fresh market snapshot required")
    report_fresh = -5 <= now-epoch(report["created_at"]) <= 900
    return report_fresh


def validate_state(state, now):
    if state.get("version") != 1 or state.get("live_trading_enabled") is not False:
        raise ValueError("Unsupported or non-paper state")
    config = state["config"]
    if config.get("mode") not in ("experimental", "strict"):
        raise ValueError("Invalid paper mode")
    for name, value in DEFAULTS.items():
        if name != "mode" and config.get(name) != value:
            raise ValueError("Paper configuration changed; create an explicitly separate experiment")
    # datetime ISO serialization rounds fractional timestamps to microseconds.
    if now + 0.000001 < epoch(state.get("updated_at", state["created_at"])):
        raise ValueError("Cannot run paper ledger backwards in time")
    cash = state["cash_jpy"]
    if not math.isfinite(cash) or cash < -1e-7:
        raise ValueError("Invalid paper cash")
    costs = 0.
    symbols = set()
    ids = set()
    for key, p in state["positions"].items():
        if p["symbol"] in symbols or p["id"] in ids or key != f"{p['symbol']}_{p['timeframe']}":
            raise ValueError("Duplicate or inconsistent paper position")
        symbols.add(p["symbol"])
        ids.add(p["id"])
        for field in ("quantity", "cost_jpy", "entry_price", "entry_fee_jpy"):
            if not math.isfinite(p[field]) or p[field] <= 0:
                raise ValueError("Invalid paper position amount")
        costs += p["cost_jpy"]
    realized = 0.
    for trade in state["closed_trades"]:
        if trade["id"] in ids or not math.isfinite(trade["net_pnl_jpy"]):
            raise ValueError("Invalid or duplicate closed trade")
        ids.add(trade["id"])
        if not math.isclose(trade["net_pnl_jpy"], trade["net_proceeds_jpy"]-trade["cost_jpy"], abs_tol=1e-6):
            raise ValueError("Trade PnL does not reconcile")
        realized += trade["net_pnl_jpy"]
    if not math.isclose(cash, config["initial_jpy"]+realized-costs, abs_tol=1e-5):
        raise ValueError("Paper cash does not reconcile with trade ledger")


def advance(state, report, ticker_data, markets, get_book, now, exits_only=False):
    """Return a new state. Callers must durably persist before notifications."""
    state = copy.deepcopy(state)
    validate_state(state, now)
    report_fresh = validate_snapshot(report, ticker_data, now)
    ticker_map = {t["symbol"]: t for t in ticker_data["data"]}
    market_map = {m["symbol"]: m for m in markets}
    results = report.get("results", {}) if report_fresh and not exits_only else {}
    config = state["config"]
    event(state, "bittrade-paper-start-v1", "activation",
          "✅ پیپر تریدینگ BitTrade و اعلان ورود/خروج فعال شد.\n"
          f"سرمایه مجازی: {config['initial_jpy']:,.0f} ین | حالت: {config['mode']}\n"
          "فقط معامله کاغذی؛ ورود اجباری انجام نمی‌شود. بررسی هر ۳۰ دقیقه است.\n"
          "سوددهی مدل هنوز تأیید نشده است. پیام ورود و خروج شامل هزینه‌ها خواهد بود.", now)
    decisions = {}
    closed_symbols = set()
    # Exit management runs even if the new research fails or rejects the model.
    for key, position in list(state["positions"].items()):
        symbol = position["symbol"]
        ticker = ticker_map.get(symbol, {})
        bid = float(ticker.get("bid", 0))
        if not math.isfinite(bid) or bid <= 0 or market_map.get(symbol, {}).get("state") != "online":
            decisions[key] = "exit_deferred_market_unavailable"
            event(state, position["id"]+":unavailable", "warning",
                  f"⚠️ پیپر BitTrade: قیمت معتبر برای مدیریت {key} موجود نیست؛ خروج جعلی ثبت نشد.", now)
            continue
        reason = None
        latest = results.get(key, {}).get("latest", {})
        fresh_signal = (latest.get("candle") and epoch(latest["candle"]) > epoch(position["signal_candle"])
                        and 0 <= now-epoch(latest["candle"])-PERIODS[position["timeframe"]][1]
                        <= PERIODS[position["timeframe"]][1]*2)
        if bid <= position["entry_price"]*(1-config["stop_loss_pct"]):
            reason = "stop_loss_observed"
        elif bid >= position["entry_price"]*(1+config["take_profit_pct"]):
            reason = "take_profit_observed"
        elif now >= epoch(position["exit_due_at"]):
            reason = "holding_horizon_elapsed"
        elif fresh_signal and latest.get("probability_positive_net_return", 1) < config["exit_probability"]:
            reason = "model_exit"
        if not reason:
            decisions[key] = "position_held"
            continue
        try:
            levels = book_levels(get_book(symbol), "bids", now)
            proceeds = fill_cost(levels, position["quantity"], config["slippage_bps"]/10000, False)
        except (ValueError, OSError, KeyError):
            decisions[key] = "exit_deferred_book_unavailable"
            event(state, position["id"]+":depth", "warning",
                  f"⚠️ خروج پیپر {key} منتظر دفتر سفارش معتبر و عمق کافی است؛ قیمت خروج فرضی ثبت نشد.", now)
            continue
        exit_fee = proceeds*config["fee_rate"]
        net = proceeds-exit_fee
        pnl = net-position["cost_jpy"]
        trade = {**position, "closed_at": utc(now), "exit_price": proceeds/position["quantity"],
                 "exit_fee_jpy": exit_fee, "net_proceeds_jpy": net, "net_pnl_jpy": pnl,
                 "net_return_pct": pnl/position["cost_jpy"]*100, "exit_reason": reason}
        state["cash_jpy"] += net
        state["closed_trades"].append(trade)
        del state["positions"][key]
        closed_symbols.add(symbol)
        stats = mark(state, ticker_map)
        reason_fa = {"model_exit": "افت احتمال مدل", "holding_horizon_elapsed": "پایان زمان نگهداری",
                     "stop_loss_observed": "مشاهده حد ضرر", "take_profit_observed": "مشاهده حد سود"}[reason]
        event(state, position["id"]+":exit", "exit",
              f"🔴 خروج کاغذی BitTrade | {key}\nعلت: {reason_fa}\n"
              f"قیمت خروج: {trade['exit_price']:,.4f} ین\n"
              f"سود/زیان خالص: {pnl:+,.2f} ین ({trade['net_return_pct']:+.2f}٪)\n"
              f"سود/زیان بسته‌شده حساب: {stats['realized_pnl_jpy']:+,.2f} ین\n"
              f"ارزش حساب: {stats['equity_jpy']:,.2f} ین | بازده: {stats['return_pct']:+.2f}٪\n"
              f"تعداد معاملات بسته‌شده: {stats['closed_trades']}\nشناسه: {position['id']}", now)
        decisions[key] = "closed:"+reason
    stats = mark(state, ticker_map)
    streak = 0
    for trade in reversed(state["closed_trades"]):
        if trade["net_pnl_jpy"] >= 0:
            break
        streak += 1
    if not state["paused"] and (stats["max_drawdown_pct"] >= config["max_drawdown_pct"]
                                or streak >= config["max_loss_streak"]):
        state["paused"] = True
        event(state, "bittrade-paper-risk-pause-v1", "warning",
              "⏸ ورود جدید پیپر BitTrade به‌دلیل افت سرمایه یا زیان‌های متوالی متوقف شد. مدیریت خروج ادامه دارد.", now)
    ordered = sorted(results.items(), key=lambda pair: (-pair[1]["latest"]["probability_positive_net_return"], pair[0]))
    for key, result in ordered:
        if key in state["positions"]:
            continue
        source = result["source"]
        symbol, timeframe = source["symbol"], source["timeframe"]
        if source.get("provider") != "bittrade" or source.get("market_type") != "spot" or timeframe not in PERIODS:
            raise ValueError("Invalid signal source")
        latest = result["latest"]
        stamp = latest["candle"]
        previous = state["seen_signals"].get(key)
        if previous and epoch(previous) >= epoch(stamp):
            continue
        # Mark evaluated candles even on HOLD; do not retroactively enter them.
        state["seen_signals"][key] = stamp
        age = now-epoch(stamp)-PERIODS[timeframe][1]
        prob = float(latest["probability_positive_net_return"])
        threshold = float(result["selected_confidence"])
        if not (math.isfinite(prob) and math.isfinite(threshold) and 0 <= prob <= 1 and 0 < threshold <= 1):
            decisions[key] = "invalid_probability"
            continue
        ticker, market = ticker_map.get(symbol, {}), market_map.get(symbol, {})
        bid, ask = float(ticker.get("bid", 0)), float(ticker.get("ask", 0))
        spread = (ask-bid)/((ask+bid)/2)*10000 if math.isfinite(ask) and math.isfinite(bid) and ask >= bid > 0 else float("inf")
        why = None
        if not 0 <= age <= 2*PERIODS[timeframe][1]:
            why = "stale_signal"
        elif state["paused"]:
            why = "risk_paused"
        elif config["mode"] == "strict" and not result.get("eligible_for_further_paper_evaluation", False):
            why = "research_gate_rejected"
        elif latest["probability_positive_net_return"] < result["selected_confidence"]:
            why = "probability_below_threshold"
        elif (market.get("state") != "online" or market.get("api-trading") != "enabled"
              or spread > config["max_spread_bps"] or ticker.get("vol", 0) < config["min_24h_jpy"]
              or "current_market_liquidity_or_status_gate_failed" in result.get("gate_reasons", [])):
            why = "liquidity_or_market_gate_rejected"
        elif (len(state["positions"]) >= config["max_positions"] or symbol in closed_symbols
              or any(p["symbol"] == symbol for p in state["positions"].values())):
            why = "portfolio_capacity_or_symbol_already_used"
        if why:
            decisions[key] = why
            continue
        budget = min(state["cash_jpy"], mark(state, ticker_map)["equity_jpy"]*config["position_fraction"])
        budget = min(budget, float(market.get("buy-market-max-order-value", budget)))
        try:
            book = get_book(symbol)
            asks, bids = book_levels(book, "asks", now), book_levels(book, "bids", now)
            book_spread = (asks[0][0]-bids[0][0])/((asks[0][0]+bids[0][0])/2)*10000
            if not 0 <= book_spread <= config["max_spread_bps"]:
                raise ValueError("Invalid or wide order book spread")
            quantity, value, fee = buy_fill(asks, budget, market, config)
        except (ValueError, OSError, KeyError):
            decisions[key] = "entry_deferred_book_or_limits"
            continue
        trade_id = hashlib.sha256(f"{state['created_at']}:{key}:{stamp}".encode()).hexdigest()[:16]
        position = {"id": trade_id, "symbol": symbol, "timeframe": timeframe, "opened_at": utc(now),
                    "signal_candle": stamp, "quantity": quantity, "entry_price": value/quantity,
                    "entry_fee_jpy": fee, "cost_jpy": value+fee, "last_bid": bid,
                    "exit_due_at": utc(now+result["horizon_candles"]*PERIODS[timeframe][1]),
                    "probability": latest["probability_positive_net_return"],
                    "confidence_threshold": result["selected_confidence"], "mode": config["mode"],
                    "research_created_at": report["created_at"], "training_data_sha256": result["sha256"],
                    "research_gate_reasons": result.get("gate_reasons", [])}
        state["cash_jpy"] -= value+fee
        state["positions"][key] = position
        event(state, trade_id+":entry", "entry",
              f"🟢 ورود کاغذی BitTrade | {key}\nحالت: {config['mode']} — پول مجازی\n"
              f"قیمت ورود: {position['entry_price']:,.4f} ین\nمقدار: {quantity:.10g}\n"
              f"هزینه با کارمزد ورود: {value+fee:,.2f} ین\n"
              f"احتمال مدل: {position['probability']:.1%} | آستانه: {position['confidence_threshold']:.0%}\n"
              f"حد ضرر پایشی: {config['stop_loss_pct']:.0%} | حد سود پایشی: {config['take_profit_pct']:.0%}\n"
              f"حداکثر نگهداری: {result['horizon_candles']} کندل\n"
              "هزینه خروج نیز محاسبه می‌شود؛ اجرای سفارش واقعی نیست.\n"
              f"شناسه: {trade_id}", now)
        decisions[key] = "paper_opened"
    state["updated_at"] = utc(now)
    state["decisions"] = decisions
    state["research_fresh"] = report_fresh
    state["exits_only"] = exits_only
    state["research_blocked"] = report.get("blocked", {})
    state["research_errors"] = report.get("errors", {})
    state["metrics"] = mark(state, ticker_map)
    validate_state(state, now)
    return state


def summary(state):
    m = state["metrics"]
    lines = ["# BitTrade forward paper trading", "", state["updated_at"], "",
             f"Mode: **{state['config']['mode']}**. Virtual JPY only. Live trading: disabled.", "",
             f"Initial: {state['config']['initial_jpy']:,.2f} JPY | Equity: {m['equity_jpy']:,.2f} JPY | Return: {m['return_pct']:+.2f}%",
             f"Realized: {m['realized_pnl_jpy']:+,.2f} JPY | Unrealized (estimated liquidation): {m['unrealized_pnl_jpy']:+,.2f} JPY",
             f"Closed trades: {m['closed_trades']} | Open positions: {m['open_positions']} | Max observed drawdown: {m['max_drawdown_pct']:.2f}%",
             f"Paused: {state['paused']} | Fresh research: {state['research_fresh']}", "",
             "Book snapshots approximate full fills, including visible spread, 5 bps extra slippage and 0.1% fees per side.",
             "Stops and exits execute only on polling at the then-current book; missed historical prices are never assumed.",
             "Backtest results and this forward portfolio are separate; repeated retraining is part of this paper policy.", "",
             "## Open positions", "", "| Strategy | Entry JPY | Quantity | Cost JPY | Opened UTC |",
             "|---|---:|---:|---:|---|"]
    for key, p in state["positions"].items():
        lines.append(f"| {key} | {p['entry_price']:.4f} | {p['quantity']:.10g} | {p['cost_jpy']:.2f} | {p['opened_at']} |")
    lines += ["", "## Recent exits", "", "| Strategy | Net PnL JPY | Net return | Reason | Closed UTC |",
              "|---|---:|---:|---|---|"]
    for p in state["closed_trades"][-20:]:
        lines.append(f"| {p['symbol']}_{p['timeframe']} | {p['net_pnl_jpy']:+.2f} | {p['net_return_pct']:+.2f}% | {p['exit_reason']} | {p['closed_at']} |")
    lines += ["", "## Last decisions", ""]
    lines.extend(f"- {key}: {value}" for key, value in sorted(state["decisions"].items()))
    return "\n".join(lines)+"\n"


def send_telegram(text):
    # Keep credentials out of files, logs, exceptions and public research state.
    if os.getenv("BITTRADE_TELEGRAM_ENABLED") != "1":
        raise RuntimeError("BitTrade Telegram is not enabled")
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise RuntimeError("Missing Telegram repository secrets")
    payload = json.dumps({"chat_id": chat, "text": text,
                          "link_preview_options": {"is_disabled": True}}).encode()
    req = Request(f"https://api.telegram.org/bot{token}/sendMessage", data=payload,
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=20) as response:
            body = json.load(response)
        if body.get("ok") is not True:
            raise RuntimeError("Telegram rejected notification")
        return int(body["result"]["message_id"])
    except (HTTPError, URLError, OSError, ValueError, KeyError):
        raise RuntimeError("Telegram delivery failed; check bot access and repository secrets") from None


def notify(path, sender=send_telegram):
    state = json.loads(Path(path).read_text())
    failures = 0
    for item in state["outbox"]:
        if item["sent_at"]:
            continue
        item["attempts"] += 1
        try:
            sender(item["text"])
            item["sent_at"] = utc(time.time())
            item.pop("delivery_error", None)
            print("Telegram delivered:", item["id"], item["kind"], flush=True)
        except RuntimeError:
            item["delivery_error"] = "delivery_failed_or_ambiguous"
            failures += 1
            print("Telegram delivery pending:", item["id"], flush=True)
        write_json(path, state)
        if failures:
            break  # Preserve order and avoid flooding on an outage.
        time.sleep(.1)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["step", "notify"])
    parser.add_argument("--state", default="reports/bittrade/paper_state.json")
    parser.add_argument("--report", default="reports/bittrade/research_report.json")
    parser.add_argument("--data", default="dataset/bittrade")
    parser.add_argument("--mode", choices=["experimental", "strict"], default="experimental")
    parser.add_argument("--exits-only", action="store_true", help="Manage existing positions if data/research failed")
    args = parser.parse_args()
    path = Path(args.state)
    if args.command == "notify":
        raise SystemExit(1 if notify(path) else 0)
    now = time.time()
    state = json.loads(path.read_text()) if path.exists() else create_state(now, args.mode)
    if state["config"]["mode"] != args.mode:
        raise ValueError("Existing paper mode differs; do not silently change experiment")
    try:
        report = json.loads(Path(args.report).read_text())
    except (OSError, ValueError):
        report = {"created_at": utc(0), "results": {}, "errors": {"report": "unavailable"}}
        args.exits_only = True
    client = PublicClient()
    ticker_data = client.get("/market/tickers")
    markets = client.get("/v1/common/symbols")["data"]
    def get_book(symbol):
        return client.get("/market/depth", symbol=symbol, type="step0")
    state = advance(state, report, ticker_data, markets, get_book, time.time(), args.exits_only)
    write_json(path, state)
    atomic_text(path.with_name("PAPER_SUMMARY.md"), summary(state))
    print(json.dumps(state["metrics"], allow_nan=False))


if __name__ == "__main__":
    main()
