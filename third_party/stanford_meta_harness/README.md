# Stanford Meta-Harness — vendored subset

Source: <https://github.com/stanford-iris-lab/meta-harness>, commit
`f805ad3aa4ada290458c1e0deadbe132fbbdea4e`, directory `reference_examples/text_classification/`.
License: MIT (see `LICENSE`) for the code. The bundled datasets (USPTO, Symptom2Disease, LawBench,
FiNER, AEGIS 2.0) are redistributed as published in the Stanford repository and keep their original
licenses; check them before reusing the data.

Only what this project uses is kept, unmodified:

- `text_classification/__init__.py`
- `text_classification/data/`: dataset loaders, the bundled datasets, and the exact-match scorers.

Not vendored: Stanford's own search loop, agents, LLM client and tests. Our baselines
(`pareto/baselines/`) are single-file ports of their `no_memory` and `fewshot` agents, and the
Stanford strategy's prompt (`pareto/prompts/role_stanford.md`) is adapted from their
`.claude/skills/meta-harness/SKILL.md`.

Coding agents can never read this folder: it contains the test splits and the unseen datasets.
