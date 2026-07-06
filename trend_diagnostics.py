import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from crypto_predictor import (
    add_features,
    attach_fundamentals,
    columns_need_fundamentals,
    horizon_model_paths,
    latest_continuous_block,
    load_price_csv,
)


def parse_horizons(value: str) -> list[int]:
    horizons = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not horizons:
        raise ValueError("At least one horizon is required.")
    return horizons


def class_name(value: int) -> str:
    return {0: "SHORT", 1: "HOLD", 2: "LONG"}.get(int(value), "UNKNOWN")


def direction(value: float, dead_zone: float) -> str:
    if value > dead_zone:
        return "up"
    if value < -dead_zone:
        return "down"
    return "flat"


def bucket(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def summarize_group(frame: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for name, group in frame.groupby(column, dropna=False):
        actionable = group[group["predicted_class"] != 1]
        rows.append(
            {
                column: str(name),
                "rows": int(len(group)),
                "accuracy_pct": float(group["correct"].mean() * 100) if len(group) else 0.0,
                "actionable_rows": int(len(actionable)),
                "actionable_accuracy_pct": float(actionable["correct"].mean() * 100) if len(actionable) else 0.0,
                "wrong_long": int(((group["predicted_class"] == 2) & (~group["correct"])).sum()),
                "wrong_short": int(((group["predicted_class"] == 0) & (~group["correct"])).sum()),
                "actual_long_pct": float((group["actual_class"] == 2).mean() * 100) if len(group) else 0.0,
                "actual_short_pct": float((group["actual_class"] == 0).mean() * 100) if len(group) else 0.0,
            }
        )
    return sorted(rows, key=lambda item: item["rows"], reverse=True)


def confusion_matrix(frame: pd.DataFrame) -> dict:
    matrix: dict[str, dict[str, int]] = {}
    for actual in [0, 1, 2]:
        actual_name = class_name(actual)
        matrix[actual_name] = {}
        rows = frame[frame["actual_class"] == actual]
        for predicted in [0, 1, 2]:
            matrix[actual_name][class_name(predicted)] = int((rows["predicted_class"] == predicted).sum())
    return matrix


def build_prediction_frame(
    data: pd.DataFrame,
    data_path: str,
    fundamental_data: str | None,
    symbol: str,
    timeframe: str,
    horizon: int,
    model_dir: str,
    min_confidence: float,
) -> tuple[pd.DataFrame, dict]:
    model_path, artifact_path = horizon_model_paths(symbol, timeframe, horizon, model_dir)
    artifact = joblib.load(artifact_path)
    feature_columns = artifact["feature_columns"]
    if columns_need_fundamentals(feature_columns):
        data = attach_fundamentals(
            data=data,
            data_path=data_path,
            symbol=symbol,
            timeframe=timeframe,
            fundamental_data=fundamental_data,
            update_fundamentals=False,
            required=True,
        )
    sequence_length = int(artifact["sequence_length"])
    threshold = float(artifact["target_threshold"])
    model = load_model(model_path, compile=False)

    frame = latest_continuous_block(data, timeframe=timeframe)
    featured = add_features(frame, horizon=horizon, require_target=True, feature_columns=feature_columns)
    features = featured[feature_columns].to_numpy(dtype=np.float32)
    scaled_features = artifact["feature_scaler"].transform(features)
    X = np.asarray(
        [scaled_features[end_idx - sequence_length + 1 : end_idx + 1] for end_idx in range(sequence_length - 1, len(featured))],
        dtype=np.float32,
    )
    meta = featured.iloc[sequence_length - 1 :].reset_index(drop=True).copy()
    probabilities = model.predict(X, verbose=0)
    predicted_class = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    predicted_class = np.where(confidence >= min_confidence, predicted_class, 1)
    actual_class = np.where(meta["target_return"] > threshold, 2, np.where(meta["target_return"] < -threshold, 0, 1))

    meta["predicted_class"] = predicted_class.astype(int)
    meta["actual_class"] = actual_class.astype(int)
    meta["confidence"] = confidence.astype(float)
    meta["correct"] = meta["predicted_class"] == meta["actual_class"]
    meta["target_abs_return"] = meta["target_return"].abs()
    meta["trend_strength"] = meta.get("trend_20_50", pd.Series(0, index=meta.index)).abs()
    meta["trend_direction"] = meta.get("trend_20_50", pd.Series(0, index=meta.index)).apply(lambda value: direction(value, threshold / 2))
    meta["future_direction"] = meta["target_return"].apply(lambda value: direction(value, threshold))
    meta["prediction_direction"] = meta["predicted_class"].map({0: "down", 1: "flat", 2: "up"})
    meta["trend_alignment"] = np.where(
        meta["prediction_direction"].eq("flat") | meta["trend_direction"].eq("flat"),
        "neutral",
        np.where(meta["prediction_direction"].eq(meta["trend_direction"]), "with_trend", "against_trend"),
    )
    meta["confidence_bucket"] = meta["confidence"].apply(lambda value: bucket(value, [0.40, 0.45, 0.50, 0.55, 0.60], ["<=0.40", "0.40-0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", ">0.60"]))
    meta["trend_bucket"] = meta["trend_strength"].apply(lambda value: bucket(value, [0.001, 0.002, 0.004, 0.008], ["very_low", "low", "medium", "high", "very_high"]))
    meta["adx_bucket"] = meta.get("adx_14", pd.Series(0, index=meta.index)).apply(lambda value: bucket(value, [0.15, 0.25, 0.35], ["weak", "normal", "strong", "very_strong"]))
    meta["atr_bucket"] = meta.get("atr_14_pct", pd.Series(0, index=meta.index)).apply(lambda value: bucket(value, [0.002, 0.004, 0.008], ["quiet", "normal", "active", "very_active"]))

    metadata = {
        "model_path": str(model_path),
        "artifact_path": str(artifact_path),
        "target_threshold_pct": threshold * 100,
        "sequence_length": sequence_length,
        "feature_count": len(feature_columns),
    }
    return meta, metadata


def analyze(args: argparse.Namespace) -> dict:
    data = load_price_csv(args.data)
    report = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "model_dir": args.model_dir,
        "min_confidence": args.min_confidence,
        "horizons": {},
    }

    for horizon in parse_horizons(args.horizons):
        frame, metadata = build_prediction_frame(
            data=data,
            data_path=args.data,
            fundamental_data=args.fundamental_data,
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizon=horizon,
            model_dir=args.model_dir,
            min_confidence=args.min_confidence,
        )
        actionable = frame[frame["predicted_class"] != 1]
        wrong = frame[~frame["correct"]]
        high_conf_wrong = wrong[wrong["confidence"] >= max(args.min_confidence, args.high_confidence)]
        report["horizons"][str(horizon)] = {
            **metadata,
            "rows": int(len(frame)),
            "start": str(frame["timestamp"].min()) if not frame.empty else None,
            "end": str(frame["timestamp"].max()) if not frame.empty else None,
            "accuracy_pct": float(frame["correct"].mean() * 100) if len(frame) else 0.0,
            "actionable_rows": int(len(actionable)),
            "actionable_rate_pct": float(len(actionable) / len(frame) * 100) if len(frame) else 0.0,
            "actionable_accuracy_pct": float(actionable["correct"].mean() * 100) if len(actionable) else 0.0,
            "confusion": confusion_matrix(frame),
            "by_confidence": summarize_group(frame, "confidence_bucket"),
            "by_trend_strength": summarize_group(frame, "trend_bucket"),
            "by_trend_direction": summarize_group(frame, "trend_direction"),
            "by_trend_alignment": summarize_group(frame, "trend_alignment"),
            "by_adx": summarize_group(frame, "adx_bucket"),
            "by_atr": summarize_group(frame, "atr_bucket"),
            "high_confidence_wrong_examples": high_conf_wrong[
                [
                    "timestamp",
                    "close",
                    "target_return",
                    "confidence",
                    "predicted_class",
                    "actual_class",
                    "trend_20_50",
                    "adx_14",
                    "atr_14_pct",
                ]
            ]
            .tail(args.examples)
            .to_dict(orient="records"),
        }

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze where trend predictions fail.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--data", default="dataset/15m_btc_history_5000.csv")
    parser.add_argument("--fundamental-data", default=None)
    parser.add_argument("--model-dir", default="models/15m_staging")
    parser.add_argument("--horizons", default="3")
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--high-confidence", type=float, default=0.55)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--output", default="trend_diagnostics_report.json")
    return parser


if __name__ == "__main__":
    print(json.dumps(analyze(build_parser().parse_args()), indent=2, ensure_ascii=False, default=str))
