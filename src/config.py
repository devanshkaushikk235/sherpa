from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    return _load("default.yaml")


def load_games() -> dict:
    return _load("games.yaml")["games"]
