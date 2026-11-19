"""The harness model client: gpt-6-luna via the OpenAI API, with exact cost tracking.

Design points
- Model and reasoning effort are locked by config; harnesses only pass a prompt.
- Every call is metered: input / cached input / output (incl. reasoning) tokens and $.
- Disk cache keyed by (prompt, occurrence index). The occurrence index means a harness
  that asks the same prompt 5 times (e.g. voting) gets 5 different samples, and a rerun
  gets the same 5. Cache hits are still COUNTED in cost (the frontier must not depend on
  cache luck); only `real_usd` (what we actually paid) skips them.
- Spend limits per phase: if a harness exceeds them, calls raise BudgetExceeded.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from .common import CACHE_DIR


class BudgetExceeded(RuntimeError):
    pass


def _empty_usage() -> dict[str, float]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "usd": 0.0,
        "real_usd": 0.0,
        "cache_hits": 0,
    }


class UsageTracker:
    """Thread-safe meter. Usage is split by phase ('learn' / 'predict') and, during
    prediction, by example (the evaluator calls `set_example` in each prediction thread)."""

    def __init__(self, phase_limits_usd: dict[str, float] | None = None):
        self._lock = threading.Lock()
        self._local = threading.local()
        self.phase = "learn"
        self.phase_limits_usd = phase_limits_usd or {}
        self.by_phase: dict[str, dict] = defaultdict(_empty_usage)
        self.by_example: dict[Any, dict] = defaultdict(_empty_usage)
        self.calls_by_example: dict[Any, list] = defaultdict(list)
        self._occurrences: Counter = Counter()

    # -- scoping -------------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def set_example(self, key: Any) -> None:
        self._local.example = key

    def current_example(self) -> Any:
        return getattr(self._local, "example", None)

    def next_occurrence(self, prompt_hash: str) -> int:
        with self._lock:
            n = self._occurrences[prompt_hash]
            self._occurrences[prompt_hash] += 1
            return n

    # -- metering ------------------------------------------------------------
    def check_budget(self) -> None:
        limit = self.phase_limits_usd.get(self.phase)
        if limit is not None and self.by_phase[self.phase]["usd"] > limit:
            raise BudgetExceeded(
                f"{self.phase} phase spent ${self.by_phase[self.phase]['usd']:.4f} > limit ${limit}"
            )

    def record(self, usage: dict, cached: bool, prompt: str, response: str) -> None:
        with self._lock:
            targets = [self.by_phase[self.phase]]
            example = self.current_example()
            if self.phase == "predict" and example is not None:
                targets.append(self.by_example[example])
                self.calls_by_example[example].append(
                    {
                        "prompt": _elide(prompt, 4000, 1500),
                        "response": _elide(response, 3000, 1000),
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "reasoning_tokens": usage["reasoning_tokens"],
                        "usd": round(usage["usd"], 6),
                    }
                )
            for t in targets:
                t["calls"] += 1
                for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "usd"):
                    t[k] += usage[k]
                if cached:
                    t["cache_hits"] += 1
                else:
                    t["real_usd"] += usage["usd"]

    def phase_summary(self, phase: str) -> dict:
        with self._lock:
            u = dict(self.by_phase[phase])
        u["usd"] = round(u["usd"], 6)
        u["real_usd"] = round(u["real_usd"], 6)
        return u


def _elide(text: str, head: int, tail: int) -> str:
    if text is None:
        return ""
    if len(text) <= head + tail + 50:
        return text
    return f"{text[:head]}\n\n...[{len(text) - head - tail} chars elided]...\n\n{text[-tail:]}"


def price_usage(prompt_tokens: int, cached: int, completion: int, prices: dict) -> float:
    uncached = max(0, prompt_tokens - cached)
    return (
        uncached * prices["input"]
        + cached * prices.get("cached_input", prices["input"])
        + completion * prices["output"]
    ) / 1_000_000


class HarnessLLM:
    """Callable handed to harnesses: `llm(prompt) -> str` and `llm.batch(prompts)`."""

    def __init__(self, model_cfg: dict, tracker: UsageTracker, use_cache: bool = True):
        self.cfg = model_cfg
        self.tracker = tracker
        self.use_cache = use_cache
        self._sem = threading.Semaphore(model_cfg.get("max_concurrency", 16))
        self._client = None
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                timeout=self.cfg.get("request_timeout_s", 300),
                max_retries=0,  # tenacity handles retries
            )
        return self._client

    # -- public API used by harnesses ---------------------------------------
    def __call__(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a str")
        self.tracker.check_budget()
        limit = self.cfg.get("max_prompt_chars", 200_000)
        if len(prompt) > limit:
            half = limit // 2
            prompt = prompt[:half] + "\n\n...[TRUNCATED]...\n\n" + prompt[-half:]

        phash = hashlib.sha256(prompt.encode()).hexdigest()
        occurrence = self.tracker.next_occurrence(phash)
        cache_path = self._cache_path(phash, occurrence)

        cached = self._load_cache(cache_path) if self.use_cache else None
        if cached is not None:
            content, usage = cached["content"], dict(cached["usage"])
            # Re-price from the cached token counts so a price change in config.yaml applies to
            # cached and fresh calls alike.
            usage["usd"] = price_usage(usage["input_tokens"], usage["cached_input_tokens"],
                                       usage["output_tokens"], self.cfg["price_per_million"])
            self.tracker.record(usage, cached=True, prompt=prompt, response=content)
            return content

        with self._sem:
            content, usage = self._complete(prompt)
        if self.use_cache:
            self._save_cache(cache_path, {"content": content, "usage": usage})
        self.tracker.record(usage, cached=False, prompt=prompt, response=content)
        return content

    def batch(self, prompts: list[str]) -> list[str]:
        """Parallel calls; per-example attribution is preserved."""
        example = self.tracker.current_example()

        def one(p: str) -> str:
            self.tracker.set_example(example)
            return self(p)

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(prompts)))) as ex:
            return list(ex.map(one, prompts))

    # -- internals -------------------------------------------------------------
    def _cache_path(self, phash: str, occurrence: int) -> Path:
        key = json.dumps(
            [self.cfg["model"], self.cfg["reasoning_effort"], self.cfg.get("max_completion_tokens"), phash, occurrence]
        )
        return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    @staticmethod
    def _load_cache(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _save_cache(path: Path, payload: dict) -> None:
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")  # unique per process+thread
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except OSError:
            pass

    def _complete(self, prompt: str) -> tuple[str, dict]:
        import openai

        @retry(
            retry=retry_if_exception_type(
                (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
            ),
            wait=wait_exponential(multiplier=2, min=2, max=60) + wait_random(0, 3),
            stop=stop_after_attempt(6),
            reraise=True,
        )
        def call():
            return self.client.chat.completions.create(
                model=self.cfg["model"],
                reasoning_effort=self.cfg["reasoning_effort"],
                max_completion_tokens=self.cfg.get("max_completion_tokens", 16000),
                messages=[{"role": "user", "content": prompt}],
            )

        resp = call()
        content = resp.choices[0].message.content or ""
        u = resp.usage
        prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
        completion = getattr(u, "completion_tokens", 0) or 0
        ptd = getattr(u, "prompt_tokens_details", None)
        ctd = getattr(u, "completion_tokens_details", None)
        cached = (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
        reasoning = (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
        usage = {
            "input_tokens": prompt_tokens,
            "cached_input_tokens": cached,
            "output_tokens": completion,
            "reasoning_tokens": reasoning,
            "usd": price_usage(prompt_tokens, cached, completion, self.cfg["price_per_million"]),
        }
        return content, usage


class FakeLLM:
    """Offline stand-in with the same interface. Used by interface checks and tests."""

    def __init__(self, tracker: UsageTracker | None = None, answer: str = "fake_label"):
        self.tracker = tracker or UsageTracker()
        self.answer = answer

    def __call__(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a str")
        content = json.dumps({"reasoning": "fake", "final_answer": self.answer})
        usage = {
            "input_tokens": len(prompt) // 4,
            "cached_input_tokens": 0,
            "output_tokens": 20,
            "reasoning_tokens": 0,
            "usd": 0.0,
        }
        self.tracker.record(usage, cached=False, prompt=prompt, response=content)
        return content

    def batch(self, prompts: list[str]) -> list[str]:
        return [self(p) for p in prompts]
