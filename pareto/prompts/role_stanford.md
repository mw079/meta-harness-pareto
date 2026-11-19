# Role

You run ONE iteration of harness evolution (adapted from Stanford Meta-Harness).
A **harness** (memory system) is the code around ONE fixed model: what it remembers from
training examples, what it puts in the prompt, how many calls it makes, and how it turns
responses into an answer. The model never changes. Only the harness does.

**Goal: the highest average validation accuracy.**

# Your workspace (current directory)

| Path | What it is |
|---|---|
| `TASK.md` | This iteration: the leaderboard of all harnesses so far |
| `archive/<id>/` | **Every harness tried so far**: `harness.py`, `meta.json`, `dev/eval.json`, `dev/traces/<dataset>.jsonl`, `val_summary.json` (failed ones have code + meta only) |
| `history.jsonl` | One line per harness: what was tried and what happened |
| `harness_base.py` | Base class + `extract_json_field`. Read it; do not edit |
| `data/<dataset>/learn.jsonl`, `dev.jsonl` | The training data, split into learn and dev |
| `c1/`, `c2/`, `c3/` | **Your output**: one candidate per folder: `candidate.py` + `notes.json` |
| `diagnosis.md` | **Your output** (required, see rules) |

# Critical constraints

- You MUST implement **3 new harnesses**, one in each of `c1/`, `c2/`, `c3/`.
- Design exactly 3 candidates: a mix of exploitation and exploration.
- Do NOT write "the best is optimal" or stop early. Complete all steps.

**Anti-parameter-tuning.** The most common failure is a candidate that is just a parameter
variant of an existing one. Check `history.jsonl` for what has been tried — parameter sweeps
(pool sizes, retrieval counts, context budgets, similarity metrics) almost always regress or
tie. Good candidates change a fundamental mechanism: a new retrieval algorithm, a new prompt
architecture, a new learning strategy, a new memory structure. Combining parts of two
harnesses is valid. Exploitation axes: A=prompt template, B=memory content, C=selection
algorithm, D=memory sizing, E=learning trigger, F=LLM usage in learning. If the last 3
iterations explored the same axis, pick different ones.

**Anti-overfitting.** No dataset-specific hints, never mention dataset names; general patterns
("prioritize recent errors", "balance label coverage") are fine.

# Workflow

1. **Analyze.** Read `history.jsonl`, the leaderboard in `TASK.md`, and the code and dev
   traces of the strongest harnesses. Write `diagnosis.md` (required, see rules) and
   formulate 3 falsifiable hypotheses, each targeting a different mechanism.
2. **Prototype.** Before writing a final harness, test the core retrieval/learning logic in
   isolation with a small script on real examples from `data/` (in this directory, no model
   calls). Try 2-3 variants. Delete the scripts when done.
3. **Implement.** For each candidate: copy a top-performing harness into `c<k>/candidate.py`,
   then implement the new mechanism. Set `base` in `c<k>/notes.json` to the id you copied.
4. **Self-critique.** Re-read each file: a genuinely NEW mechanism, or a parameter variant? If
   `predict()` and `learn_from_batch()` are identical to the base except for numbers, rewrite.
5. **Self-check** every candidate until it passes. Write the three `notes.json` files and stop.
