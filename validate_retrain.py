import argparse
import json
from pathlib import Path


def load_metrics(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Validation output not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_metrics(
    metrics: dict,
    min_average_return_pct: float,
    min_profitable_fold_pct: float,
    max_worst_drawdown_pct: float,
    min_fold_count: int,
    min_average_closed_trades: float,
) -> tuple[bool, list[str]]:
    failures = []
    fold_count = int(metrics.get("fold_count", 0))
    average_return = float(metrics.get("average_test_return_pct", 0.0))
    profitable_fold_pct = float(metrics.get("profitable_fold_pct", 0.0))
    worst_drawdown = float(metrics.get("worst_drawdown_pct", 0.0))
    average_closed_trades = float(metrics.get("average_test_closed_trade_count", 0.0))

    if fold_count < min_fold_count:
        failures.append(f"fold_count {fold_count} < required {min_fold_count}")
    if average_return < min_average_return_pct:
        failures.append(f"average_test_return_pct {average_return:.4f} < required {min_average_return_pct:.4f}")
    if profitable_fold_pct < min_profitable_fold_pct:
        failures.append(f"profitable_fold_pct {profitable_fold_pct:.2f} < required {min_profitable_fold_pct:.2f}")
    allowed_worst_drawdown = -abs(max_worst_drawdown_pct)
    if worst_drawdown < allowed_worst_drawdown:
        failures.append(f"worst_drawdown_pct {worst_drawdown:.2f} < allowed {allowed_worst_drawdown:.2f}")
    if average_closed_trades < min_average_closed_trades:
        failures.append(
            f"average_test_closed_trade_count {average_closed_trades:.2f} < required {min_average_closed_trades:.2f}"
        )

    return not failures, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quality gate for nightly model retraining.")
    parser.add_argument("--input", default="strategy_walkforward_retrain.json")
    parser.add_argument("--min-average-return-pct", type=float, default=0.0)
    parser.add_argument("--min-profitable-fold-pct", type=float, default=50.0)
    parser.add_argument("--max-worst-drawdown-pct", type=float, default=-5.0)
    parser.add_argument("--min-fold-count", type=int, default=3)
    parser.add_argument("--min-average-closed-trades", type=float, default=2.0)
    return parser


def main(args: argparse.Namespace) -> None:
    metrics = load_metrics(args.input)
    passed, failures = validate_metrics(
        metrics,
        min_average_return_pct=args.min_average_return_pct,
        min_profitable_fold_pct=args.min_profitable_fold_pct,
        max_worst_drawdown_pct=args.max_worst_drawdown_pct,
        min_fold_count=args.min_fold_count,
        min_average_closed_trades=args.min_average_closed_trades,
    )
    summary = {
        "passed": passed,
        "fold_count": metrics.get("fold_count"),
        "average_test_return_pct": metrics.get("average_test_return_pct"),
        "profitable_fold_pct": metrics.get("profitable_fold_pct"),
        "worst_drawdown_pct": metrics.get("worst_drawdown_pct"),
        "average_test_closed_trade_count": metrics.get("average_test_closed_trade_count"),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit("Retrain validation failed; refreshed model weights will not be committed.")


if __name__ == "__main__":
    main(build_parser().parse_args())
