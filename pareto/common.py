"""Paths, config and small helpers shared by all modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
PKG = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / ".cache" / "llm"
STANFORD_DIR = ROOT / "third_party" / "stanford_meta_harness"  # vendored data + scorers
TEMPLATE_DIR = PKG / "workspace_template"
PROMPTS_DIR = PKG / "prompts"
BASELINES_DIR = PKG / "baselines"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no extra dependency). Does not override existing vars."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
