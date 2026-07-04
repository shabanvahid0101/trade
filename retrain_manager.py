import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib

from crypto_predictor import MODELS_DIR, horizon_model_paths, send_telegram_message
from validate_retrain import load_metrics, validate_metrics


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = BASE_DIR / "model_registry.json"
CALENDAR_FEATURES = {
    "hour_return_mean",
    "hour_abs_return_mean",
    "hour_big_move_rate",
    "weekday_return_mean",
    "weekday_abs_return_mean",
    "weekday_big_move_rate",
    "is_us_session",
    "is_weekend",
}


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons:
        raise ValueError("At least one horizon is required.")
    return horizons


def git_short_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def promotion_files(symbol: str, timeframe: str, horizons: list[int], staging_dir: Path, production_dir: Path) -> list[tuple[Path, Path]]:
    files = []
    for horizon in horizons:
        staging_model, staging_artifact = horizon_model_paths(symbol, timeframe, horizon, staging_dir)
        production_model, production_artifact = horizon_model_paths(symbol, timeframe, horizon, production_dir)
        files.extend([(staging_model, production_model), (staging_artifact, production_artifact)])
    return files


def promote_models(symbol: str, timeframe: str, horizons: list[int], staging_dir: Path, production_dir: Path) -> list[str]:
    promoted = []
    for source, target in promotion_files(symbol, timeframe, horizons, staging_dir, production_dir):
        if not source.exists():
            raise FileNotFoundError(f"Missing staging artifact: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target_path = target.resolve()
        try:
            promoted.append(str(target_path.relative_to(BASE_DIR.resolve())))
        except ValueError:
            promoted.append(str(target_path))
    return promoted


def load_json_file(path: str | Path | None) -> dict:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def collect_feature_summary(symbol: str, timeframe: str, horizons: list[int], artifact_dir: Path) -> dict:
    by_horizon = {}
    calendar_hits = set()
    for horizon in horizons:
        _, artifact_path = horizon_model_paths(symbol, timeframe, horizon, artifact_dir)
        if not artifact_path.exists():
            continue
        artifact = joblib.load(artifact_path)
        features = list(artifact.get("feature_columns", []))
        calendar_features = [feature for feature in features if feature in CALENDAR_FEATURES]
        calendar_hits.update(calendar_features)
        by_horizon[str(horizon)] = {
            "feature_count": len(features),
            "calendar_features": calendar_features,
            "top_features": features[:8],
        }
    return {
        "by_horizon": by_horizon,
        "calendar_features_used": sorted(calendar_hits),
    }


def data_quality_summary(report: dict) -> dict:
    price_after = report.get("price_data", {}).get("after", {})
    fundamentals = report.get("fundamental_data", {})
    source_age = fundamentals.get("source_age_hours", {})
    return {
        "price_rows": price_after.get("rows"),
        "last_price_timestamp": price_after.get("last_timestamp"),
        "price_gap_count": price_after.get("gap_count"),
        "invalid_ohlcv_rows": price_after.get("invalid_ohlcv_rows"),
        "fundamental_rows": fundamentals.get("rows"),
        "last_fundamental_timestamp": fundamentals.get("last_timestamp"),
        "last_fundamental_source_age_hours": source_age.get("last"),
    }


def build_registry(
    status: str,
    symbol: str,
    timeframe: str,
    horizons: list[int],
    metrics: dict,
    failures: list[str],
    promoted_files: list[str],
    staging_dir: Path,
    production_dir: Path,
    feature_summary: dict,
    data_quality: dict,
) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "updated_at": now,
        "active_status": "production_updated" if status == "passed" else "production_unchanged",
        "last_retrain_status": status,
        "symbol": symbol,
        "timeframe": timeframe,
        "horizons": horizons,
        "git_sha": git_short_sha(),
        "staging_dir": str(staging_dir),
        "production_dir": str(production_dir),
        "promoted_files": promoted_files,
        "feature_summary": feature_summary,
        "data_quality": data_quality,
        "validation_metrics": {
            "mode": metrics.get("mode"),
            "selected_config_counts": metrics.get("selected_config_counts", {}),
        },
        "quality_gate": {
            "passed": status == "passed",
            "failures": failures,
            "fold_count": metrics.get("fold_count"),
            "average_test_return_pct": metrics.get("average_test_return_pct"),
            "profitable_fold_pct": metrics.get("profitable_fold_pct"),
            "worst_drawdown_pct": metrics.get("worst_drawdown_pct"),
        },
    }


def send_retrain_report(message: str) -> bool:
    try:
        return send_telegram_message(message)
    except Exception as exc:
        print(f"Retrain report telegram send failed: {exc}")
        return False


def format_value(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def format_signed_value(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def format_feature_summary(feature_summary: dict) -> str:
    by_horizon = feature_summary.get("by_horizon") or {}
    if not by_horizon:
        return "در دسترس نیست"
    lines = []
    for horizon, info in sorted(by_horizon.items(), key=lambda item: int(item[0])):
        calendar_features = info.get("calendar_features") or []
        calendar_text = "، ".join(calendar_features) if calendar_features else "ندارد"
        lines.append(f"h{horizon}: {info.get('feature_count', 'n/a')} فیچر | زمانی: {calendar_text}")
    return "\n".join(lines)


def format_top_configs(metrics: dict, limit: int = 2) -> str:
    counts = metrics.get("selected_config_counts") or {}
    if not counts:
        return "در دسترس نیست"
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return "\n".join(f"- {count} fold: {config}" for config, count in ranked)


def build_message(registry: dict) -> str:
    gate = registry["quality_gate"]
    status_text = "قبول شد - مدل اصلی آپدیت شد" if gate["passed"] else "رد شد - مدل قبلی فعال ماند"
    failures = gate.get("failures") or []
    failure_text = "\n".join(f"- {failure}" for failure in failures) if failures else "ندارد"
    data_quality = registry.get("data_quality") or {}
    metrics = registry.get("validation_metrics") or {}
    promoted_count = len(registry.get("promoted_files") or [])
    return (
        f"<b>گزارش بازآموزی مدل بیت‌کوین</b>\n"
        f"وضعیت: {status_text}\n"
        f"نماد: {registry['symbol']} | تایم‌فریم: {registry['timeframe']}\n"
        f"افق‌ها: {','.join(str(h) for h in registry['horizons'])}\n"
        f"فایل‌های مدل آپدیت‌شده: {promoted_count}\n\n"
        f"<b>نتیجه تست Walk-forward</b>\n"
        f"میانگین بازده تست: {format_signed_value(gate.get('average_test_return_pct'), '%')}\n"
        f"foldهای سودده: {format_value(gate.get('profitable_fold_pct'), '%')}\n"
        f"بدترین افت سرمایه: {format_signed_value(gate.get('worst_drawdown_pct'), '%')}\n"
        f"تعداد fold: {gate.get('fold_count', 'n/a')}\n\n"
        f"<b>کیفیت دیتای آموزشی</b>\n"
        f"تعداد کندل قیمت: {data_quality.get('price_rows', 'n/a')}\n"
        f"آخرین کندل قیمت: {data_quality.get('last_price_timestamp', 'n/a')}\n"
        f"گپ کندل‌ها: {data_quality.get('price_gap_count', 'n/a')}\n"
        f"کندل خراب: {data_quality.get('invalid_ohlcv_rows', 'n/a')}\n"
        f"آخرین دیتای فاندامنتال: {data_quality.get('last_fundamental_timestamp', 'n/a')}\n"
        f"سن دیتای فاندامنتال: {format_value(data_quality.get('last_fundamental_source_age_hours'), ' ساعت')}\n\n"
        f"<b>فیچرهای مدل جدید</b>\n"
        f"{format_feature_summary(registry.get('feature_summary') or {})}\n\n"
        f"<b>تنظیم برنده در validation</b>\n"
        f"{format_top_configs(metrics)}\n\n"
        f"Git: {registry.get('git_sha') or 'n/a'}\n"
        f"دلایل رد شدن: {failure_text}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote validated staging models and send retrain report.")
    parser.add_argument("--metrics", default="strategy_walkforward_retrain.json")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--staging-dir", default=str(MODELS_DIR / "staging"))
    parser.add_argument("--production-dir", default=str(MODELS_DIR))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--data-quality-report", default="data_quality_report.json")
    parser.add_argument("--min-average-return-pct", type=float, default=0.0)
    parser.add_argument("--min-profitable-fold-pct", type=float, default=50.0)
    parser.add_argument("--max-worst-drawdown-pct", type=float, default=-5.0)
    parser.add_argument("--min-fold-count", type=int, default=3)
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--fail-on-reject", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(args: argparse.Namespace) -> dict:
    metrics = load_metrics(args.metrics)
    passed, failures = validate_metrics(
        metrics,
        min_average_return_pct=args.min_average_return_pct,
        min_profitable_fold_pct=args.min_profitable_fold_pct,
        max_worst_drawdown_pct=args.max_worst_drawdown_pct,
        min_fold_count=args.min_fold_count,
    )
    horizons = parse_horizons(args.horizons)
    staging_dir = Path(args.staging_dir)
    production_dir = Path(args.production_dir)
    promoted_files = promote_models(args.symbol, args.timeframe, horizons, staging_dir, production_dir) if passed else []
    feature_artifact_dir = production_dir if passed else staging_dir
    feature_summary = collect_feature_summary(args.symbol, args.timeframe, horizons, feature_artifact_dir)
    data_quality = data_quality_summary(load_json_file(args.data_quality_report))
    registry = build_registry(
        status="passed" if passed else "failed",
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizons=horizons,
        metrics=metrics,
        failures=failures,
        promoted_files=promoted_files,
        staging_dir=staging_dir,
        production_dir=production_dir,
        feature_summary=feature_summary,
        data_quality=data_quality,
    )
    Path(args.registry).write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(registry, indent=2, ensure_ascii=False))
    if args.telegram:
        send_retrain_report(build_message(registry))
    if not passed and args.fail_on_reject:
        raise SystemExit("Retrain quality gate failed; production models were not changed.")
    return registry


if __name__ == "__main__":
    main(build_parser().parse_args())
