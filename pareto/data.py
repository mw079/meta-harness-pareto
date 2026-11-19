"""Datasets and scoring, reused read-only from the vendored Stanford Meta-Harness subset."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from .common import STANFORD_DIR

if str(STANFORD_DIR) not in sys.path:
    sys.path.insert(0, str(STANFORD_DIR))

from text_classification.data import load_dataset_splits_3way  # noqa: E402


def load_splits(name: str, cfg: dict) -> tuple[list[dict], list[dict], list[dict], Callable]:
    sizes = cfg["datasets"]["sizes"][name]
    return load_dataset_splits_3way(
        name,
        num_train=sizes["train"],
        num_val=sizes["val"],
        num_test=sizes["test"],
        shuffle_seed=cfg["datasets"].get("shuffle_seed", 42),
    )


def eval_kwargs(ex: dict[str, Any]) -> dict[str, Any]:
    """Same as Stanford inner_loop._get_eval_kwargs."""
    kwargs = {k: v for k, v in ex.items() if k not in ("input", "target")}
    if "raw_input" in ex:
        kwargs["input_nums"] = ex["raw_input"]
    return kwargs


def unpack_eval(raw) -> tuple[bool, dict]:
    """Same as Stanford inner_loop._unpack_eval_result."""
    if isinstance(raw, dict):
        return bool(raw["was_correct"]), raw.get("metrics", {})
    return bool(raw), {}
