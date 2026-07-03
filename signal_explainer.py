from __future__ import annotations


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def fa_signal(signal: str) -> str:
    return {"LONG": "لانگ", "SHORT": "شورت", "HOLD": "هولد", "FLAT": "بدون پوزیشن"}.get(str(signal), str(signal))


def fa_regime(regime: str) -> str:
    return {
        "range": "بازار رنج",
        "trend_or_unclear": "رونددار یا نامشخص",
        "model": "مدل",
    }.get(str(regime), str(regime))


def fa_strategy(strategy: str) -> str:
    return {"hybrid": "هیبرید", "range": "رنج", "model": "مدل"}.get(str(strategy), str(strategy))


def fa_code(value: str | None) -> str:
    if value is None:
        return "نامشخص"
    return {
        "ok": "عادی",
        "normal": "عادی",
        "reduced": "کاهش حجم",
        "blocked": "مسدود",
        "equity_below_minimum": "سرمایه کمتر از حداقل مجاز است",
        "max_drawdown_exceeded": "حداکثر افت سرمایه رد شده",
        "loss_streak_exceeded": "تعداد ضررهای پشت‌سرهم زیاد شده",
        "drawdown_reduced_size": "به خاطر افت سرمایه، حجم کمتر شد",
        "risk_blocked": "ورود به خاطر ریسک مسدود شد",
        "no_trade_signal": "سیگنال قابل معامله وجود ندارد",
        "dynamic_signal_quality": "حجم بر اساس کیفیت سیگنال تنظیم شد",
        "range_regime_size_discount": "به خاطر بازار رنج، حجم کمتر شد",
        "trend_or_unclear": "بازار رونددار یا نامشخص",
    }.get(str(value), str(value))


def _horizon_label(item: dict) -> str:
    horizon = item.get("horizon_candles", "?")
    signal = item.get("signal", "UNKNOWN")
    confidence = float(item.get("confidence", 0.0))
    probabilities = item.get("class_probabilities", {}) or {}
    short_prob = float(probabilities.get("SHORT", 0.0))
    hold_prob = float(probabilities.get("HOLD", 0.0))
    long_prob = float(probabilities.get("LONG", 0.0))
    return (
        f"h{horizon}: {fa_signal(signal)} | اطمینان {confidence:.2f} "
        f"(شورت {short_prob:.2f}، هولد {hold_prob:.2f}، لانگ {long_prob:.2f})"
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
        reasons.append(f"سیگنال لانگ شد چون تعداد رأی‌های صعودی به {long_votes} رسید.")
    elif signal == "SHORT":
        reasons.append(f"سیگنال شورت شد چون تعداد رأی‌های نزولی به {short_votes} رسید.")
    else:
        reasons.append("هولد شد چون مدل‌ها سیگنال قوی و قابل معامله ندادند.")

    if strongest:
        reasons.append(
            f"قوی‌ترین افق h{strongest.get('horizon_candles')} است؛ "
            f"سیگنال آن {fa_signal(strongest.get('signal'))} با اطمینان {float(strongest.get('confidence', 0.0)):.2f} بود."
        )
    if supporting and signal in {"LONG", "SHORT"}:
        horizons = ", ".join(f"h{item.get('horizon_candles')}" for item in supporting)
        reasons.append(f"افق‌های تأییدکننده: {horizons}.")

    range_position = final.get("range_position")
    range_low = final.get("range_low")
    range_high = final.get("range_high")
    if range_position is not None:
        reasons.append(
            "وضعیت رنج: "
            f"قیمت در {float(range_position) * 100:.1f}% محدوده بین "
            f"{_fmt_float(range_low)} و {_fmt_float(range_high)} قرار دارد."
        )
    if regime == "range":
        reasons.append("فیلتر بازار رنج فعال است؛ برگشت از حمایت/مقاومت می‌تواند رأی مدل را تغییر دهد.")
    elif regime == "trend_or_unclear":
        reasons.append("فیلتر رنج فعال نیست؛ رأی مدل سیگنال اصلی باقی می‌ماند.")

    risk_summary = None
    if risk is not None:
        quality = risk.get("signal_quality_score")
        quality_text = f" | کیفیت={float(quality):.2f}" if quality is not None else ""
        size_reason = risk.get("position_size_reason")
        reason_text = f" | دلیل حجم={fa_code(size_reason)}" if size_reason else ""
        risk_summary = (
            f"{fa_code(risk.get('risk_level', 'unknown'))} | "
            f"حجم={float(risk.get('position_size_pct', 0.0)):.2f} | "
            f"{fa_code(risk.get('reason', 'unknown'))}"
            f"{quality_text}"
            f"{reason_text}"
        )
        reasons.append(f"تصمیم مدیریت ریسک: {risk_summary}.")

    summary = (
        f"{fa_signal(signal)} با استراتژی {fa_strategy(strategy)} / وضعیت {fa_regime(regime)}؛ "
        f"رأی‌ها لانگ/شورت/هولد = {long_votes}/{short_votes}/{hold_votes}؛ "
        f"اطمینان {confidence:.2f}؛ حرکت مورد انتظار {expected_pct:.3f}%."
    )

    return {
        "summary": summary,
        "reasons": reasons,
        "horizon_lines": horizon_lines,
        "risk_summary": risk_summary,
    }


def format_explanation_for_telegram(explanation: dict, max_reasons: int = 5) -> str:
    lines = ["<b>دلیل تصمیم</b>", explanation.get("summary", "توضیحی موجود نیست.")]
    for reason in explanation.get("reasons", [])[:max_reasons]:
        lines.append(f"- {reason}")
    horizon_lines = explanation.get("horizon_lines", [])
    if horizon_lines:
        lines.append("")
        lines.append("<b>افق‌های مدل</b>")
        lines.extend(f"- {line}" for line in horizon_lines)
    return "\n".join(lines)
