# The fixed setup (you cannot change any of this)

- **Model:** `{model}`, reasoning effort `{effort}`, called ONLY via `self.call_llm(prompt)`
  or `self.call_llm_batch(prompts)`. No temperature or other parameters.
- **Price:** ${price_in} per 1M input tokens, ${price_out} per 1M output tokens. Hidden reasoning
  tokens are billed as **output**. Rough rule: 4 characters ≈ 1 token.
  So cost is driven by (a) prompt length × number of calls, and (b) how much the model
  writes and *thinks* per call. Asking for long written reasoning costs output tokens
  on top of the hidden reasoning.
- **Task family:** text classification / short-answer tasks. Each search dataset gives
  the harness labeled training examples (`learn_from_batch`), then asks it to answer
  new questions (`predict`) in parallel threads.
- **Search datasets:** 3 (see `data/`). The final evaluation ALSO uses a held-out test set
  and **unseen datasets from different domains** that nobody has seen. Harnesses that
  only work because of quirks of these 3 datasets will fail there.

# How the data is used

Each dataset's training split is divided into **learn** (75%) and **dev** (25%):
- The **dev run** teaches the harness with `learn` and has it predict `dev`. Its full traces
  are in `archive/<id>/dev/traces/` — this is what you study.
- The **validation run** teaches the harness with the full training split and has it
  predict a hidden validation set. It decides which harnesses are kept. You only see its
  aggregate numbers (`archive/<id>/val_summary.json`). You cannot read validation examples
  and must not try: fix what the dev traces show, in a way that will generalize.

Accuracy = exact match (Stanford Meta-Harness scorers), averaged over the 3 datasets.
Cost = USD spent by `predict` per question. Learning-phase spend is capped and reported.

Traces are large; search them with `grep`, `jq` or small Python scripts — don't read them whole.
Use these scripts only on files in your directory; never on anything outside it.

# Hard rules (violations are rejected automatically)

- A harness is ONE file, at most {max_lines} lines, exactly one class extending `MemorySystem`.
- Imports: only `harness_base` and these standard-library modules: {allowed_imports}.
  No numpy, no network, no files, no `os`/`sys`/`subprocess`, no `open`/`eval`/`exec`, and no
  reaching into module internals (`module._name`, `getattr(module, ...)`).
- Reach the model only through `self.call_llm` / `self.call_llm_batch`.
- **No dataset-specific knowledge**: no dataset names, no hard-coded label lists, no rules
  that only make sense for one of these datasets. Everything must be *learned* from the
  training examples in `learn_from_batch`. General strategies are fine
  ("collect the label set from training data", "retrieve similar examples").
- `predict` must be thread-safe and work with no prior learning (cold start).
- `get_state` / `set_state` must round-trip all learned state as a string.
- **Diagnosis is required:** before writing code, study the dev traces and write
  `diagnosis.md` with concrete numbers (error types and counts per dataset, 2-3 real
  examples of the most common error, where the cost goes: input/output/reasoning tokens,
  calls per question) and your hypothesis. Candidates without `diagnosis.md`, or whose
  author never opened any dev traces, are rejected before evaluation.
- **Self-check** every harness until it passes (free, fake model, no API calls). Run it
  exactly as written from your directory — never `cd` elsewhere (it reads its rules from
  `self_check_rules.json` in this directory; do not edit that file):
  `{self_check_cmd} <path/to/harness.py>`
- Do not run the real model. Do not look for test, validation or other data. Stay inside
  this directory; files outside it are blocked. Do not stop early; produce real candidates.

# Taste

- **Simplicity criterion:** equal results with less code is a win. A tiny gain that adds
  a lot of fragile code is not worth it. Removing something with no loss is great.
- **Change a mechanism, not just a number.** Pure parameter tweaks (k=10 → k=12) rarely
  help. Removing or shrinking something is a legitimate mechanism if the traces show it is
  not pulling its weight.
- **Spend compute where it matters.** Extra calls on every question are expensive; extra
  calls only on hard/uncertain questions can be cheap.
- **Be honest in predictions.** Calibrated predictions help the next researchers.

# notes.json schema (one per candidate)

```json
{
  "name": "short_snake_case_name",
  "base": "<id of the harness you started from>",
  "mechanism": "one sentence: what the harness does differently",
  "hypothesis": "falsifiable claim, linked to evidence from the dev traces",
  "evidence": ["concrete numbers from traces that motivated this"],
  "expected_accuracy_delta": 0.0,
  "expected_cost_delta_pct": 0.0,
  "risks": "what could go wrong / how it could fail on unseen domains",
  "components": ["tags", "for", "the", "archive"]
}
```
`expected_accuracy_delta` is in accuracy points vs. the base (e.g. +0.03 = +3 points);
`expected_cost_delta_pct` is relative to the base (e.g. -40 = 40% cheaper).
