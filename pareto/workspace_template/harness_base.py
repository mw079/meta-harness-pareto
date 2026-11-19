"""Base class every harness extends. DO NOT EDIT (it is replaced on every run).

A harness is ONE Python file with ONE subclass of `MemorySystem`.

    from harness_base import MemorySystem, extract_json_field

The evaluator does, per dataset:
    h = YourHarness(llm)                      # fresh instance
    for batch in train:                       # learning phase (labels visible)
        h.learn_from_batch(batch)
    h.set_state(h.get_state())                # state must round-trip
    for x in questions (parallel threads):    # prediction phase (dev, val or test)
        answer, metadata = h.predict(x["input"])

Cost of a harness = $ spent by `predict` per question (input + output tokens, where
output includes hidden reasoning tokens). Learning spend is reported separately.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Protocol


class LLMCallable(Protocol):
    def __call__(self, prompt: str) -> str: ...

    def batch(self, prompts: list[str]) -> list[str]: ...


def extract_json_field(text: str, field: str, default: str = "") -> str:
    """Extract `field` from a JSON object in an LLM response (robust to code fences,
    surrounding text and trailing commas). Copied from Stanford Meta-Harness."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get(field, default))
    except (json.JSONDecodeError, TypeError):
        pass

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text or ""):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return str(data.get(field, default))
        except json.JSONDecodeError:
            pass

    text = text or ""
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth, pos, in_str = 1, start + 1, False
        while pos < len(text) and depth > 0:
            c = text[pos]
            if c == '"' and text[pos - 1] != "\\":
                in_str = not in_str
            elif not in_str:
                depth += 1 if c == "{" else (-1 if c == "}" else 0)
            pos += 1
        if depth == 0:
            candidate = re.sub(r",\s*([\]}])", r"\1", text[start:pos])
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return str(data.get(field, default))
            except json.JSONDecodeError:
                pass

    match = re.findall(rf'"{field}"\s*:\s*"([^"]*)"', text)
    return match[-1] if match else default


class MemorySystem(ABC):
    """Harness interface (same as Stanford Meta-Harness text classification)."""

    def __init__(self, llm: LLMCallable):
        self._llm = llm

    # -- the only ways to reach the model ------------------------------------
    def call_llm(self, prompt: str) -> str:
        """One model call. Model and reasoning effort are fixed by the evaluator."""
        return self._llm(prompt)

    def call_llm_batch(self, prompts: list[str]) -> list[str]:
        """Several independent calls in parallel (same cost as calling them one by one)."""
        return self._llm.batch(prompts)

    # -- interface to implement ---------------------------------------------
    @abstractmethod
    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        """Answer one question. Must be thread-safe (called from parallel threads)
        and must work with no prior learning. Returns (answer, metadata)."""

    @abstractmethod
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        """Learn from labeled examples. Each dict has: input, prediction, ground_truth,
        was_correct, and optionally raw_question / context."""

    @abstractmethod
    def get_state(self) -> str:
        """Serialize all learned state to a string."""

    @abstractmethod
    def set_state(self, state: str) -> None:
        """Restore state produced by get_state()."""
