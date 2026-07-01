from __future__ import annotations


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _horizon_label(item: dict) -> str:
    horizon = item.get("horizon_candles", "?")
    signal = item.get("signal", "UNKNOWN")
    confidence = float(item.get("confidence", 0.0))
    probabilities = item.get("class_probabilities", {}) or {}
    short_prob = float(probabilities.get("SHORT", 0.0))
    hold_prob = float(probabilities.get("HOLD", 0.0))
    long_prob = float(probabilities.get("LONG", 0.0))
    return (
        f"h{horizon}: {signal} conf={confidence:.2f} "
        f"(S={short_prob:.2f}, H={hold_prob:.2f}, L={long_prob:.2f})"
    )


def build_signal_explanation(final: dict, horizon_results: list[dict], risk: dict | None = None) -> dict:
    signal = final.get("signal", "UNKNOWN")
    regime = final.get("market_regime", "model")
    strategy = final.get("strategy", "model")
    confidence = float(final.get("confidence", 0.0))
    expected_pct = float(final.get("predicted_return_pct", 0.0))
    long_votes = int(final.get("long_votes", 0))
    short_votes = int(final.get("short_votes", 0))
    hold_votes = int(final.get("hold_votes", 0))

    horizon_lines = [_horizon_label(item) for item in horizon_results]
    supporting = [item for item in horizon_results if item.get("signal") == signal]
    strongest = max(horizon_results, key=lambda item: float(item.get("confidence", 0.0)), default=None)

    reasons: list[str] = []
    if signal == "LONG":
        reasons.append(f"LONG selected because bullish votes reached {long_votes}.")
    elif signal == "SHORT":
        reasons.append(f"SHORT selected because bearish votes reached {short_votes}.")
    else:
        reasons.append("HOLD selected because the models did not produce a strong trade signal.")

    if strongest:
        reasons.append(
            f"Strongest horizon is h{strongest.get('horizon_candles')} "
            f"with {strongest.get('signal')} confidence {float(strongest.get('confidence', 0.0)):.2f}."
        )
    if supporting and signal in {"LONG", "SHORT"}:
        horizons = ", ".join(f"h{item.get('horizon_candles')}" for item in supporting)
        reasons.append(f"Supporting horizons: {horizons}.")

    range_position = final.get("range_position")
    range_low = final.get("range_low")
    range_high = final.get("range_high")
    if range_position is not None:
        reasons.append(
            "Range context: "
            f"position={float(range_position) * 100:.1f}% between "
            f"{_fmt_float(range_low)} and {_fmt_float(range_high)}."
        )
    if regime == "range":
        reasons.append("Range regime is active, so support/resistance mean reversion can override the model.")
    elif regime == "trend_or_unclear":
        reasons.append("Range filter is not active, so the model vote remains the main signal.")

    risk_summary = None
    if risk is not None:
        quality = risk.get("signal_quality_score")
        quality_text = f" | quality={float(quality):.2f}" if quality is not None else ""
        size_reason = risk.get("position_size_reason")
        reason_text = f" | size_reason={size_reason}" if size_reason else ""
        risk_summary = (
            f"{risk.get('risk_level', 'unknown')} | "
            f"size={float(risk.get('position_size_pct', 0.0)):.2f} | "
            f"{risk.get('reason', 'unknown')}"
            f"{quality_text}"
            f"{reason_text}"
        )
        reasons.append(f"Risk decision: {risk_summary}.")

    summary = (
        f"{signal} via {strategy}/{regime}; "
        f"votes L/S/H={long_votes}/{short_votes}/{hold_votes}; "
        f"confidence={confidence:.2f}; expected={expected_pct:.3f}%."
    )

    return {
        "summary": summary,
        "reasons": reasons,
        "horizon_lines": horizon_lines,
        "risk_summary": risk_summary,
    }


def format_explanation_for_telegram(explanation: dict, max_reasons: int = 5) -> str:
    lines = ["<b>Why</b>", explanation.get("summary", "No explanation available.")]
    for reason in explanation.get("reasons", [])[:max_reasons]:
        lines.append(f"- {reason}")
    horizon_lines = explanation.get("horizon_lines", [])
    if horizon_lines:
        lines.append("")
        lines.append("<b>Horizons</b>")
        lines.extend(f"- {line}" for line in horizon_lines)
    return "\n".join(lines)
