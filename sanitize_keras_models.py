import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


UNSUPPORTED_CONFIG_KEYS = {"input_axes", "output_axes"}


def clean_config(value):
    if isinstance(value, dict):
        return {key: clean_config(item) for key, item in value.items() if key not in UNSUPPORTED_CONFIG_KEYS}
    if isinstance(value, list):
        return [clean_config(item) for item in value]
    return value


def sanitize_model(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with zipfile.ZipFile(path, "r") as source:
        if "config.json" not in source.namelist():
            return False
        config_text = source.read("config.json").decode("utf-8")
        if not any(f'"{key}"' in config_text for key in UNSUPPORTED_CONFIG_KEYS):
            return False

        import json

        cleaned_config = clean_config(json.loads(config_text))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_file:
            temp_path = Path(temp_file.name)

        try:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as target:
                for item in source.infolist():
                    if item.filename == "config.json":
                        target.writestr(item, json.dumps(cleaned_config, separators=(",", ":")))
                    else:
                        target.writestr(item, source.read(item.filename))
            shutil.move(str(temp_path), path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove unsupported Keras config keys from .keras model archives.")
    parser.add_argument("--paths", nargs="+", required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    for item in args.paths:
        for path in sorted(Path().glob(item)):
            changed = sanitize_model(path)
            print(f"{path}: {'sanitized' if changed else 'unchanged'}")
