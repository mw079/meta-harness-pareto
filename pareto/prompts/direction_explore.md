# Your direction: EXPLORE

Goal: try a **mechanism that none of the frontier harnesses uses**. You may land anywhere on
the curve; the most valuable result is a point that fills a **gap** in the frontier (see the
table in TASK.md) or beats several frontier points at once.

Two other workers run at the same time as you, from the same parent: CHEAPER and
MORE ACCURATE. The accurate worker will usually take the most obvious next step
(e.g. retrieving similar training examples, constraining answers to a learned label set).
Do NOT spend your slot on that obvious step — pick a different mechanism.

Before choosing:
- List the mechanisms the frontier harnesses use (`components` in their `meta.json` and
  their code in `archive/`). Pick something genuinely different, not a variant.
- Look at the frontier for gaps: big cost jumps between neighbors, or a missing very
  cheap point.

Idea space (examples, not a menu): memory structure (label prototypes, hierarchical
rules, confusion clusters), prompt architecture (decompose → classify, label-description
first), learning strategy (model-written lessons, active selection of informative
examples), inference strategy (cascades, early exit, consistency checks on hard cases),
combinations of the best parts of two different frontier harnesses.

You may start from the parent or rewrite from scratch; say which in notes.json (`base`).
