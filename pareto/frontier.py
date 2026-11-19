"""Noise-aware Pareto frontier over (accuracy up, cost down) and the parent queue.

Frontier, recomputed from ALL valid candidates every time (global comparison):
  1. exact Pareto set: drop b if some a is at least as accurate and at least as cheap
     (strictly better in one; exact duplicates: the older one wins);
  2. noise pruning, one sweep from cheapest to most expensive: a point is dropped if the last
     KEPT point is at least cost_eps cheaper (relative to the point's cost) and the point is not
     more than acc_eps more accurate (the gain is within noise, the extra cost is real).
Comparing only against KEPT points avoids the chain effect of pairwise ε-dominance (A drops
B, B drops C, so C is lost although nothing kept is within ε of it).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    id: str
    acc: float
    cost: float
    order: int = 0  # creation order, for tie-breaks


def dominates(a: Point, b: Point) -> bool:
    """Exact Pareto dominance (duplicates: older wins)."""
    if a.id == b.id:
        return False
    if a.acc == b.acc and a.cost == b.cost:
        return a.order < b.order
    return a.acc >= b.acc and a.cost <= b.cost


def compute_frontier(points: list[Point], acc_eps: float, cost_eps: float) -> list[Point]:
    exact = sorted((p for p in points if not any(dominates(q, p) for q in points)),
                   key=lambda p: (p.cost, -p.acc))
    kept: list[Point] = []
    for p in exact:  # cheapest first; accuracy increases along the exact frontier
        if kept:
            last = kept[-1]
            meaningfully_pricier = last.cost < p.cost * (1.0 - cost_eps)
            if meaningfully_pricier and p.acc <= last.acc + acc_eps:
                continue
        kept.append(p)
    return kept


def hypervolume(points: list[Point], ref_acc: float = 0.0, ref_cost: float | None = None) -> float:
    """Area dominated by the frontier in (cost, acc) space up to a reference point
    (ref_cost defaults to the max cost). Bigger is better. Single-number summary."""
    if not points:
        return 0.0
    ref_cost = ref_cost if ref_cost is not None else max(p.cost for p in points) * 1.0001
    pts = sorted(points, key=lambda p: p.cost)
    area, best_acc = 0.0, ref_acc
    for i, p in enumerate(pts):
        best_acc = max(best_acc, p.acc)
        nxt = pts[i + 1].cost if i + 1 < len(pts) else ref_cost
        area += max(0.0, nxt - p.cost) * max(0.0, best_acc - ref_acc)
    return area


class ParentQueue:
    """Round-robin parent queue over frontier members.

    - pop(): next member still on the frontier; refills from the frontier (cheapest
      first) when empty, so every part of the curve keeps getting expanded.
    - sync(): enqueue new frontier members at the back.
    """

    def __init__(self, items: list[str] | None = None, seen: list[str] | None = None):
        self.q: deque[str] = deque(items or [])
        self.seen: set[str] = set(seen or [])

    def sync(self, frontier_ids: list[str]) -> None:
        for fid in frontier_ids:
            if fid not in self.seen:
                self.q.append(fid)
                self.seen.add(fid)

    def pop(self, frontier_ids: list[str]) -> str | None:
        front = set(frontier_ids)
        while self.q:
            cand = self.q.popleft()
            if cand in front:
                return cand
        if not frontier_ids:
            return None
        self.q.extend(frontier_ids)  # new cycle, ordered cheapest -> most accurate
        return self.q.popleft()

    def to_dict(self) -> dict:
        return {"queue": list(self.q), "seen": sorted(self.seen)}

    @classmethod
    def from_dict(cls, d: dict | None) -> ParentQueue:
        d = d or {}
        return cls(d.get("queue"), d.get("seen"))
