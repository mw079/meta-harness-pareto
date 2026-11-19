"""Baseline: few-shot with 10 sampled training examples (cheap point; variant of Stanford
fewshot_memory.py / fewshot_all.py)."""

import hashlib
import json
import random
from typing import Any

from harness_base import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MAX_CHARS = 30000
MAX_EXAMPLES = 10


def _seed_for_input(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class FewShot10(MemorySystem):
    def __init__(self, llm):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []

    def _examples_section(self, seed: int) -> str:
        if not self.examples:
            return ""
        rng = random.Random(seed)
        if len(self.examples) > MAX_EXAMPLES:
            to_use = rng.sample(self.examples, MAX_EXAMPLES)
        else:
            to_use = list(self.examples[-MAX_EXAMPLES:])
            rng.shuffle(to_use)
        parts, total = [], 0
        for ex in to_use:
            part = f"Q: {ex.get('raw_question', ex['input'])}\nA: {ex['target']}"
            if total + len(part) > MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        prompt = PROMPT_TEMPLATE.format(
            examples_section=self._examples_section(_seed_for_input(input)), input=input
        )
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer"), {"full_response": response}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples})

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
