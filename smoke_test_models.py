import argparse
from pathlib import Path

import numpy as np
import joblib
from tensorflow.keras.models import load_model

from crypto_predictor import MODELS_DIR, horizon_model_paths
from sanitize_keras_models import sanitize_model


def parse_horizons(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def smoke_test(symbol: str, timeframe: str, horizons: list[int], sanitize: bool, model_dir: str | Path = MODELS_DIR) -> None:
    for horizon in horizons:
        model_path, artifact_path = horizon_model_paths(symbol, timeframe, horizon, model_dir)
        if sanitize:
            sanitize_model(model_path)
        if not model_path.exists() or not artifact_path.exists():
            raise FileNotFoundError(f"Missing model/artifact for horizon {horizon}: {model_path}, {artifact_path}")
        artifact = joblib.load(artifact_path)
        sequence_length = int(artifact["sequence_length"])
        feature_count = len(artifact["feature_columns"])
        model = load_model(model_path, compile=False)
        sample = np.zeros((1, sequence_length, feature_count), dtype=np.float32)
        prediction = model.predict(sample, verbose=0)
        if prediction.shape[0] != 1:
            raise RuntimeError(f"Unexpected prediction shape for h{horizon}: {prediction.shape}")
        print(f"OK h{horizon}: model={model_path.name}, input={(sequence_length, feature_count)}, output={prediction.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test saved Keras models and artifacts.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    parser.add_argument("--sanitize", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    smoke_test(args.symbol, args.timeframe, parse_horizons(args.horizons), args.sanitize, args.model_dir)
