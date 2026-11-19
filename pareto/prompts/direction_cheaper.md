# Your direction: CHEAPER

Goal: a harness that costs **clearly less per question** than the parent (aim for ≥25%
cheaper) while keeping accuracy **within ~2 points** of it. Success = a new frontier
point to the left of the parent.

Start by finding where the parent's money goes (from its traces):
- input tokens per call — how much of the prompt is memory/examples, how much is
  template, how much is the question?
- output tokens per call — how much is visible written reasoning vs hidden reasoning?
- calls per question — are all calls needed for every question?

Mechanisms worth considering (pick based on evidence, not this list's order):
- Retrieve only the few *most relevant* training examples instead of many/random ones.
- Compress memory: short distilled rules / label descriptions learned from training data
  instead of raw examples.
- Shorter output format (e.g. answer-only JSON) — written reasoning costs output tokens
  and the model already reasons internally.
- Make expensive steps conditional: a cheap first pass, escalate only when uncertain.
- Remove parts of the prompt that the traces show are not helping.

Watch out: removing the examples that carry the label format usually destroys exact-match
accuracy. Check which errors would appear if you cut something.
