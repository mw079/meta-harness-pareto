# Your direction: MORE ACCURATE

Goal: a harness that is **clearly more accurate** than the parent (aim for ≥ +3 points
averaged over the datasets) at a cost increase you can justify. Success = a new frontier
point above the parent. Cost still matters: +1 point for 5× cost is a bad trade; prefer
mechanisms that spend extra compute only where it helps.

Start from the parent's errors (from its traces), per dataset:
- Which errors are *format* problems (right idea, wrong string — casing, extra words,
  synonyms, wrong separator, multi-label order)? These are often the cheapest wins.
- Which are *knowledge/judgment* errors? Would better examples, label definitions, or
  a contrastive example have prevented them?
- Are errors concentrated on certain labels or input types?

Mechanisms worth considering (pick based on evidence, not this list's order):
- Learn the exact answer format / label vocabulary from training data and normalize or
  constrain outputs to it.
- Retrieval of similar training examples; contrastive pairs for commonly confused labels.
- Error-driven lessons: during learning, have the model predict training examples and
  distill short rules from its mistakes (mind the learning-phase budget).
- Targeted extra compute: verify or re-ask only when the first answer is outside the
  learned label set or looks uncertain; vote only on hard cases.

Watch out: anything that silently fits these 3 datasets will be caught on unseen domains.
