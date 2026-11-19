"""Evaluate one harness file on one split, with exact cost metering.

    dev       : learn on the `learn` part of train, predict the `dev` part (traces shown to agents)
    val, test : learn on the full train split, predict val / test

Run as a subprocess by the search loop (crash + timeout isolation):

    python -m pareto.evaluate --harness path.py --split dev --datasets USPTO,LawBench \
        --out runs/<run>/archive/h0001/dev [--limit-train 6 --limit-eval 3] [--fake-llm]

Writes <out>/eval.json and <out>/traces/<dataset>.jsonl (one line per example).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .common import load_config, load_dotenv, write_json
from .data import eval_kwargs, load_splits, unpack_eval
from .llm import BudgetExceeded, FakeLLM, HarnessLLM, UsageTracker
from .loader import load_harness_class


def _learn(harness, train: list[dict], batch_size: int) -> None:
    for start in range(0, len(train), batch_size):
        batch = []
        for ex in train[start : start + batch_size]:
            r = {
                "input": ex["input"],
                "prediction": ex["target"],  # offline mode: ground truth visible (as Stanford)
                "ground_truth": ex["target"],
                "was_correct": True,
            }
            for k, v in ex.items():
                if k not in ("input", "target") and k not in r:
                    r[k] = v
            batch.append(r)
        harness.learn_from_batch(batch)


def split_learn_dev(train: list[dict], dev_fraction: float) -> tuple[list[dict], list[dict]]:
    """Deterministic learn/dev split of the (already seed-shuffled) train split.
    dev = last round(dev_fraction * n) examples, at least 1."""
    n_dev = max(1, round(dev_fraction * len(train)))
    return train[: len(train) - n_dev], train[len(train) - n_dev :]


def resolve_split(name: str, split: str, cfg: dict) -> tuple[list[dict], list[dict], object]:
    """Returns (examples the harness learns from, examples it predicts, evaluator).
    dev: learn on 'learn' part of train, predict 'dev' part (traces shown to agents).
    val / test: learn on the FULL train split, predict val / test."""
    train, val, test, evaluator = load_splits(name, cfg)
    if split == "dev":
        learn, dev = split_learn_dev(train, cfg["evaluation"].get("dev_fraction", 0.25))
        return learn, dev, evaluator
    return train, {"val": val, "test": test}[split], evaluator


def evaluate_dataset(harness_cls, name: str, split: str, cfg: dict, fake: bool,
                     limit_train: int | None, limit_eval: int | None) -> tuple[dict, list[dict]]:
    ev_cfg = cfg["evaluation"]
    train, examples, evaluator = resolve_split(name, split, cfg)
    if limit_train is not None:
        train = train[:limit_train]
    if limit_eval is not None:
        examples = examples[:limit_eval]

    tracker = UsageTracker(
        {"learn": ev_cfg["max_learn_usd_per_dataset"], "predict": ev_cfg["max_predict_usd_per_dataset"]}
    )
    llm = FakeLLM(tracker) if fake else HarnessLLM(cfg["harness_model"], tracker)
    result: dict[str, Any] = {"dataset": name, "split": split, "n": len(examples), "status": "ok"}
    t0 = time.time()

    # ---- learning phase ----
    try:
        tracker.set_phase("learn")
        harness = harness_cls(llm)
        _learn(harness, train, ev_cfg.get("learn_batch_size", 1))
        state = harness.get_state()
        if not isinstance(state, str):
            raise TypeError("get_state() must return a str")
        harness.set_state(state)
        result["state_chars"] = len(state)
    except BudgetExceeded as e:
        result.update(status="over_budget", error=str(e))
    except Exception as e:  # noqa: BLE001 - any harness bug fails the dataset
        result.update(status="crashed", error=f"learn: {type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    result["learn_usage"] = tracker.phase_summary("learn")
    result["learn_seconds"] = round(time.time() - t0, 1)
    if result["status"] != "ok":
        return result, []

    # ---- prediction phase ----
    tracker.set_phase("predict")
    t1 = time.time()

    def predict_one(idx: int, ex: dict) -> dict:
        tracker.set_example(idx)
        rec: dict[str, Any] = {"idx": idx, "input": ex["input"], "target": ex["target"]}
        try:
            pred, meta = harness.predict(ex["input"])
            ok, metrics = unpack_eval(evaluator(pred, ex["target"], **eval_kwargs(ex)))
            rec.update(prediction=pred, correct=ok, metrics=metrics,
                       metadata={k: v for k, v in (meta or {}).items() if k != "full_response"})
        except BudgetExceeded as e:
            rec.update(prediction=None, correct=False, error=f"BudgetExceeded: {e}")
        except Exception as e:  # noqa: BLE001
            rec.update(prediction=None, correct=False, error=f"{type(e).__name__}: {e}")
        return rec

    with ThreadPoolExecutor(max_workers=max(1, min(ev_cfg.get("eval_threads", 8), len(examples)))) as pool:
        records = list(pool.map(lambda a: predict_one(*a), enumerate(examples)))

    for rec in records:
        u = tracker.by_example.get(rec["idx"])
        rec["usd"] = round(u["usd"], 6) if u else 0.0
        rec["tokens"] = (
            {k: u[k] for k in ("calls", "input_tokens", "output_tokens", "reasoning_tokens")} if u else {}
        )
        rec["calls"] = tracker.calls_by_example.get(rec["idx"], [])

    n = len(records)
    errors = sum(1 for r in records if r.get("error"))
    predict_usage = tracker.phase_summary("predict")
    result.update(
        correct=sum(1 for r in records if r["correct"]),
        accuracy=(sum(1 for r in records if r["correct"]) / n) if n else 0.0,
        errors=errors,
        predict_usage=predict_usage,
        usd_per_question=(predict_usage["usd"] / n) if n else 0.0,
        tokens_per_question=((predict_usage["input_tokens"] + predict_usage["output_tokens"]) / n) if n else 0.0,
        calls_per_question=(predict_usage["calls"] / n) if n else 0.0,
        predict_seconds=round(time.time() - t1, 1),
    )
    if any("BudgetExceeded" in (r.get("error") or "") for r in records):
        result["status"] = "over_budget"
    elif n and errors / n > ev_cfg.get("max_error_rate", 0.2):
        result["status"] = "crashed"
        result["error"] = next(r["error"] for r in records if r.get("error"))
    return result, records


def evaluate_harness(harness_path: Path, datasets: list[str], split: str, out_dir: Path, cfg: dict,
                     fake: bool = False, limit_train: int | None = None, limit_eval: int | None = None) -> dict:
    out_dir = Path(out_dir)
    t0 = time.time()
    summary: dict[str, Any] = {"harness": str(harness_path), "split": split, "datasets": {}, "status": "ok"}
    try:
        harness_cls = load_harness_class(harness_path)
    except Exception as e:  # noqa: BLE001
        summary.update(status="crashed", error=f"import: {type(e).__name__}: {e}")
        write_json(out_dir / "eval.json", summary)
        return summary

    for name in datasets:
        res, records = evaluate_dataset(harness_cls, name, split, cfg, fake, limit_train, limit_eval)
        summary["datasets"][name] = res
        if records:
            trace_path = out_dir / "traces" / f"{name}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, default=str) + "\n")
        if res["status"] != "ok":
            summary["status"] = res["status"]
            summary["error"] = f"{name}: {res.get('error', '')}"
            break

    ds = [summary["datasets"][d] for d in datasets if d in summary["datasets"]]
    if summary["status"] == "ok" and ds:
        summary["accuracy"] = sum(d["accuracy"] for d in ds) / len(ds)
        summary["usd_per_question"] = sum(d["usd_per_question"] for d in ds) / len(ds)
        summary["tokens_per_question"] = sum(d["tokens_per_question"] for d in ds) / len(ds)
        summary["calls_per_question"] = sum(d["calls_per_question"] for d in ds) / len(ds)
    summary["learn_usd"] = round(sum(d["learn_usage"]["usd"] for d in ds), 6)
    summary["real_usd_spent"] = round(
        sum(d["learn_usage"]["real_usd"] + d.get("predict_usage", {}).get("real_usd", 0.0) for d in ds), 6
    )
    summary["seconds"] = round(time.time() - t0, 1)
    write_json(out_dir / "eval.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--harness", required=True)
    p.add_argument("--split", choices=["dev", "val", "test"], default="val")
    p.add_argument("--datasets", required=True, help="comma-separated")
    p.add_argument("--out", required=True)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-eval", type=int)
    p.add_argument("--fake-llm", action="store_true")
    p.add_argument("--config")
    a = p.parse_args()
    load_dotenv()
    cfg = load_config(Path(a.config) if a.config else None)
    s = evaluate_harness(Path(a.harness), a.datasets.split(","), a.split, Path(a.out), cfg,
                         fake=a.fake_llm, limit_train=a.limit_train, limit_eval=a.limit_eval)
    acc = s.get("accuracy")
    print(f"status={s['status']} acc={acc if acc is None else round(acc, 4)} "
          f"usd/q={s.get('usd_per_question')} spent=${s.get('real_usd_spent')} {s.get('error', '')}")


if __name__ == "__main__":
    main()
