"""Load a single-file harness. Deliberately imports NO dataset code, so the workers'
self-check (pareto.checks) runs inside the sandbox, where third_party/ (Stanford data) is hidden."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

from .common import TEMPLATE_DIR

if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

import harness_base  # noqa: E402


def load_harness_class(path: Path):
    """Import a single-file harness and return its MemorySystem subclass."""
    path = Path(path).resolve()
    mod_name = f"harness_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    classes = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, harness_base.MemorySystem)
        and obj is not harness_base.MemorySystem
        and obj.__module__ == mod_name
        and not inspect.isabstract(obj)
    ]
    if len(classes) != 1:
        raise ValueError(f"expected exactly 1 concrete MemorySystem subclass in {path.name}, found {len(classes)}")
    return classes[0]
