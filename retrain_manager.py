import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from crypto_predictor import MODELS_DIR, horizon_model_paths, send_telegram_message
from validate_retrain import load_metrics, validate_metrics


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = BASE_DIR / "model_registry.json"


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
        "quality_gate": {
            "passed": status == "passed",
            "failures": failures,
            "fold_count": metrics.get("fold_count"),
            "average_test_return_pct": metrics.get("average_test_return_pct"),
            "profitable_fold_pct": metrics.get("profitable_fold_pct"),
            "worst_drawdown_pct": metrics.get("worst_drawdown_pct"),
        },
    }


def format_value(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def build_message(registry: dict) -> str:
    gate = registry["quality_gate"]
    status_text = "PASSED - production updated" if gate["passed"] else "FAILED - production unchanged"
    failures = gate.get("failures") or []
    failure_text = "\n".join(f"- {failure}" for failure in failures) if failures else "none"
    return (
        f"<b>Nightly Model Retrain</b>\n"
        f"Status: {status_text}\n"
        f"Symbol: {registry['symbol']} {registry['timeframe']}\n"
        f"Horizons: {','.join(str(h) for h in registry['horizons'])}\n"
        f"Avg return: {format_value(gate.get('average_test_return_pct'), '%')}\n"
        f"Profitable folds: {format_value(gate.get('profitable_fold_pct'), '%')}\n"
        f"Worst drawdown: {format_value(gate.get('worst_drawdown_pct'), '%')}\n"
        f"Fold count: {gate.get('fold_count', 'n/a')}\n"
        f"Git: {registry.get('git_sha') or 'n/a'}\n"
        f"Failures: {failure_text}"
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
    )
    Path(args.registry).write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))
    if args.telegram:
        send_telegram_message(build_message(registry))
    if not passed and args.fail_on_reject:
        raise SystemExit("Retrain quality gate failed; production models were not changed.")
    return registry


if __name__ == "__main__":
    main(build_parser().parse_args())
