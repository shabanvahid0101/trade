"""Chronological, cost-aware BitTrade spot research. Never sends orders.

Every daily result is exploratory (the expanding test window is reused).
Passing gates means eligible for further paper evaluation, never live approval.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from bittrade_data import PERIODS, validate_rows, write_json

FEATURES = ["return_1", "return_3", "return_6", "return_12", "ema_gap", "rsi", "volatility",
            "range", "body", "volume_ratio", "turnover_ratio"]
SPREAD_BPS = {"btcjpy": 5.0, "ethjpy": 30.0, "xrpjpy": 30.0}


class MarketNotEvaluable(ValueError):
    """Expected research exclusion, not a collector or programming failure."""


def features(frame):
    f = frame.copy()
    for lag in (1, 3, 6, 12):
        f[f"return_{lag}"] = f.close.pct_change(lag)
    f["ema_gap"] = f.close.ewm(span=12, adjust=False).mean() / f.close.ewm(span=48, adjust=False).mean() - 1
    delta = f.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = (gain / (gain + loss)).fillna(0.5)
    f["volatility"] = f.return_1.rolling(24).std()
    f["range"] = (f.high - f.low) / f.close
    f["body"] = (f.close - f.open) / f.open
    f["volume_ratio"] = f.volume / f.volume.rolling(48).mean().replace(0, np.nan)
    f["turnover_ratio"] = f.quote_volume / f.quote_volume.rolling(48).mean().replace(0, np.nan)
    return f.replace([np.inf, -np.inf], np.nan)


def net_return(entry, exit_price, fee, friction):
    return exit_price * (1-friction) * (1-fee) / (entry * (1+friction) * (1+fee)) - 1


def simulate(frame, probability, horizon, confidence, fee, friction):
    """One long position, next-open entry, horizon-th close exit; otherwise cash.

    Signals require a trade in the signal candle. Volume-zero entry/exit bars
    are not treated as guaranteed fills: encountering one invalidates the run.
    """
    equity = 100000.0
    curve = [equity]
    pnls = []
    i = 0
    while i + horizon < len(frame):
        if probability[i] < confidence or frame.iloc[i].volume <= 0:
            i += 1
            continue
        entry_bar = frame.iloc[i+1]
        exit_bar = frame.iloc[i+horizon]
        if entry_bar.volume <= 0 or exit_bar.volume <= 0:
            raise MarketNotEvaluable("Cannot assume fills on zero-volume candles")
        entry = float(entry_bar.open) * (1+friction)
        quantity = equity / (entry * (1+fee))
        for j in range(i+1, i+horizon+1):
            # Conservative intrabar drawdown mark and close liquidation mark.
            curve.append(quantity * float(frame.iloc[j].low) * (1-friction) * (1-fee))
            curve.append(quantity * float(frame.iloc[j].close) * (1-friction) * (1-fee))
        after = quantity * float(exit_bar.close) * (1-friction) * (1-fee)
        pnls.append(after-equity)
        equity = after
        curve.append(equity)
        i += horizon
    a = np.array(curve)
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    benchmark = 100 * net_return(float(frame.iloc[1].open), float(frame.iloc[-1].close), fee, friction)
    return {"return_pct": (equity/100000-1)*100, "closed_trades": len(pnls),
            "profit_factor": wins/losses if losses else None,
            "win_rate_pct": float(np.mean(np.array(pnls) > 0)*100) if pnls else None,
            "max_drawdown_pct": float(np.min(a / np.maximum.accumulate(a)-1)*100),
            "buy_hold_net_return_pct": benchmark, "cash_return_pct": 0.0,
            "alpha_vs_buy_hold_pct": (equity/100000-1)*100-benchmark}


def split_indices(n, horizon):
    train_end = int(n * .6)
    validation_end = int(n * .8)
    # Labels at i access i+horizon: remove these before the next partition.
    return {"train": (0, train_end-horizon), "validation": (train_end, validation_end-horizon),
            "test": (validation_end, n-horizon)}


def probability(model, values):
    if 1 not in model.classes_:
        return np.zeros(len(values))
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def run_one(path, output, fee=.001, slippage_bps=5, horizon=6):
    manifest = json.loads(path.with_suffix(".source.json").read_text())
    identity = manifest["source"]
    if identity["provider"] != "bittrade" or identity["market_type"] != "spot":
        raise ValueError("Expected BitTrade spot data")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("CSV hash differs from source manifest")
    symbol, timeframe = identity["symbol"], identity["timeframe"]
    seconds = PERIODS[timeframe][1]
    raw = pd.read_csv(path)
    validate_rows(raw.to_dict("records"), seconds)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    if not raw.timestamp.is_monotonic_increasing:
        raise ValueError("Unsorted dataset")
    age = (pd.Timestamp.now(tz="UTC") - raw.timestamp.iloc[-1]).total_seconds() - seconds
    if age < -60 or age > max(3*seconds, 7200):
        raise ValueError("Dataset is stale or in the future")
    # A REST outage longer than its window may leave gaps. Never bridge them.
    gaps = np.flatnonzero(raw.timestamp.diff().dt.total_seconds().fillna(seconds).to_numpy() != seconds)
    discarded = int(gaps[-1]) if len(gaps) else 0
    raw = raw.iloc[discarded:].reset_index(drop=True)
    f = features(raw).iloc[48:].reset_index(drop=True)
    if len(f) < 1200:
        raise MarketNotEvaluable("Need at least 1200 contiguous candles after feature warmup")
    if f[FEATURES].isna().any().any():
        raise MarketNotEvaluable("Missing features; insufficient trading activity")
    spread = SPREAD_BPS.get(symbol, 50.0)
    friction = (spread/2 + slippage_bps)/10000
    future = net_return(f.open.shift(-1), f.close.shift(-horizon), fee, friction)
    y = (future > 0).astype(int).to_numpy()
    x = f[FEATURES].to_numpy()
    splits = split_indices(len(f), horizon)
    ta, tb = splits["train"]
    va, vb = splits["validation"]
    sa, sb = splits["test"]
    if len(np.unique(y[ta:tb])) < 2:
        raise MarketNotEvaluable("Training target has only one class")
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=30,
                                          l2_regularization=10, early_stopping=False, random_state=42)
    model.fit(x[ta:tb], y[ta:tb])
    vp = probability(model, x[va:vb])
    options = []
    for confidence in (.50, .55, .60, .65, .70):
        # Include the final horizon's prices to settle labels near the boundary.
        m = simulate(f.iloc[va:vb+horizon].reset_index(drop=True), vp,
                     horizon, confidence, fee, friction)
        options.append({"confidence": confidence, **m})
    eligible = [m for m in options if m["closed_trades"] >= 20 and m["return_pct"] > 0
                and m["alpha_vs_buy_hold_pct"] > 0 and m["profit_factor"] is not None
                and m["profit_factor"] >= 1.1 and m["max_drawdown_pct"] >= -2]
    selected = max(eligible, key=lambda m: m["return_pct"]) if eligible else None
    confidence = selected["confidence"] if selected else .60
    tp = probability(model, x[sa:sb])
    test_frame = f.iloc[sa:sb+horizon].reset_index(drop=True)
    candidate = simulate(test_frame, tp, horizon, confidence, fee, friction)
    # If validation rejected every threshold, the actual policy remains cash.
    policy = candidate if selected else simulate(test_frame, np.zeros(len(tp)), horizon, 1., fee, friction)
    days = (raw.timestamp.iloc[-1]-raw.timestamp.iloc[0]).total_seconds()/86400
    reasons = []
    if days < 30:
        reasons.append("less_than_30_days_of_history")
    if not selected:
        reasons.append("no_validation_candidate_passed")
    if candidate["closed_trades"] < 50:
        reasons.append("fewer_than_50_test_trades")
    if candidate["return_pct"] <= 0 or candidate["alpha_vs_buy_hold_pct"] <= 0:
        reasons.append("no_positive_net_edge_on_test")
    if candidate["profit_factor"] is None or candidate["profit_factor"] < 1.1:
        reasons.append("test_profit_factor_below_1_1_or_undefined")
    if candidate["max_drawdown_pct"] < -2:
        reasons.append("test_drawdown_exceeds_2_percent")
    markets = json.loads((path.parent / "markets.json").read_text())["data"]
    market = next(m for m in markets if m["symbol"] == symbol)
    ticker_data = json.loads((path.parent / "tickers.json").read_text())
    ticker_age = pd.Timestamp.now(tz="UTC").timestamp()-ticker_data["ts"]/1000
    ticker = next(t for t in ticker_data["data"] if t["symbol"] == symbol)
    bid, ask = ticker.get("bid", 0), ticker.get("ask", 0)
    observed_spread = (ask-bid)/((ask+bid)/2)*10000 if ask and bid and ask >= bid else None
    liquidity_ok = (0 <= ticker_age <= 7200 and market["state"] == "online"
                    and market.get("api-trading") == "enabled" and observed_spread is not None
                    and observed_spread <= 40 and ticker.get("vol", 0) >= 10000000
                    and float(raw.volume.iloc[-1]) > 0)
    if not liquidity_ok:
        reasons.append("current_market_liquidity_or_status_gate_failed")
    latest_probability = float(probability(model, x[-1:])[0])
    latest_signal = "BUY_PAPER_CANDIDATE" if not reasons and latest_probability >= confidence else "HOLD"
    report = {"source": identity, "sha256": manifest["sha256"], "history_days": days,
        "contiguous_rows": len(raw), "discarded_rows_before_last_gap": discarded,
        "model": "HistGradientBoostingClassifier", "features": FEATURES, "horizon_candles": horizon,
        "cost_assumptions": {"fee_per_side": fee, "full_spread_bps": spread,
                             "slippage_per_side_bps": slippage_bps, "historical_orderbook_available": False},
        "windows": {k: {"start": str(f.timestamp.iloc[a]), "end": str(f.timestamp.iloc[b-1]), "rows": b-a}
                    for k, (a, b) in splits.items()},
        "validation_options": options, "selected_confidence": confidence,
        "validation_passed": selected is not None,
        "test_accuracy_pct": float(accuracy_score(y[sa:sb], tp >= .5)*100),
        "test_balanced_accuracy_pct": float(balanced_accuracy_score(y[sa:sb], tp >= .5)*100),
        "train_majority_test_accuracy_pct": float(np.mean(y[sa:sb] == int(np.mean(y[ta:tb]) >= .5))*100),
        "test_candidate": candidate, "test_gated_policy": policy,
        "gate_reasons": reasons, "eligible_for_further_paper_evaluation": not reasons,
        "latest": {"candle": str(raw.timestamp.iloc[-1]), "probability_positive_net_return": latest_probability,
                   "signal": latest_signal, "observed_spread_bps": observed_spread,
                   "volume_24h_jpy": ticker.get("vol")}, "live_trading_enabled": False}
    key = path.stem
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "source": identity, "report": report}, output/f"{key}.joblib")
    return report


def run(data, output):
    data, output = Path(data), Path(output)
    report = {"created_at": pd.Timestamp.now(tz="UTC").isoformat(),
              "protocol": "60/20/20 chronological split; horizon-purged train/validation; next-open long-only fills; confidence selected on validation only",
              "limitation": "Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.",
              "live_trading_enabled": False, "results": {}, "blocked": {}, "errors": {}}
    paths = sorted(data.glob("*.csv"))
    if not paths:
        raise ValueError("No BitTrade datasets found")
    for path in paths:
        try:
            result = run_one(path, output)
            report["results"][path.stem] = result
            print(path.stem, "test_return=", round(result["test_candidate"]["return_pct"], 3),
                  "signal=", result["latest"]["signal"], flush=True)
        except MarketNotEvaluable as exc:
            report["blocked"][path.stem] = {"reason": str(exc), "signal": "HOLD"}
            (output/f"{path.stem}.joblib").unlink(missing_ok=True)
            print("BLOCKED", path.stem, str(exc), flush=True)
        except (ValueError, OSError, KeyError, StopIteration) as exc:
            report["errors"][path.stem] = str(exc)
            (output/f"{path.stem}.joblib").unlink(missing_ok=True)
            print("ERROR", path.stem, str(exc), flush=True)
    write_json(output/"research_report.json", report)
    lines = ["# BitTrade spot research", "", report["created_at"], "", report["limitation"], "",
             "Candidate return is diagnostic. The gated policy stays in cash if validation fails.", "",
             "| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for key, r in report["results"].items():
        t = r["test_candidate"]
        lines.append(f"| {key} | {r['history_days']:.1f} | {t['closed_trades']} | {t['return_pct']:.2f}% | {t['buy_hold_net_return_pct']:.2f}% | {r['test_gated_policy']['return_pct']:.2f}% | {r['latest']['signal']} |")
    for key, error in report["errors"].items():
        lines.append(f"\nError {key}: {error}")
    for key, blocked in report["blocked"].items():
        lines.append(f"\nBlocked {key}: {blocked['reason']}. Signal: HOLD.")
    from bittrade_data import atomic_text
    atomic_text(output/"SUMMARY.md", "\n".join(lines)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/bittrade")
    parser.add_argument("--output", default="reports/bittrade")
    args = parser.parse_args()
    result = run(args.data, args.output)
    raise SystemExit(1 if result["errors"] else 0)
