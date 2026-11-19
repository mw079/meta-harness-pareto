"""Offline tests (no API or agent calls): frontier and queue, static checks and audit, data splits,
isolation settings and agent views, cost metering, rollback and crash recovery, Pareto rounds,
prompt rendering, and an end-to-end evaluation with a fake model."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pareto.checks import audit_transcript, interface_check, static_check
from pareto.common import BASELINES_DIR, load_config
from pareto.evaluate import evaluate_harness
from pareto.frontier import ParentQueue, Point, compute_frontier, dominates, hypervolume

CFG = load_config()
RULES = CFG["harness_rules"]
AUDIT = CFG["isolation"]["audit_patterns"]


class TestFrontier(unittest.TestCase):
    def ids(self, pts, ae=0.02, ce=0.10):
        return [p.id for p in compute_frontier(pts, ae, ce)]

    def test_plain_dominance(self):
        pts = [Point("a", 0.5, 1.0, 0), Point("b", 0.6, 1.0, 1), Point("c", 0.4, 2.0, 2)]
        self.assertEqual(self.ids(pts, 0, 0), ["b"])

    def test_tradeoff_kept(self):
        pts = [Point("cheap", 0.5, 1.0, 0), Point("strong", 0.7, 5.0, 1)]
        self.assertEqual(self.ids(pts), ["cheap", "strong"])

    def test_noise_margin_removes_expensive_near_tie(self):
        # 'exp' is 1 point better (noise) but 3x the cost -> dominated by 'cheap'
        pts = [Point("cheap", 0.60, 1.0, 0), Point("exp", 0.61, 3.0, 1)]
        self.assertEqual(self.ids(pts), ["cheap"])

    def test_not_meaningfully_cheaper_keeps_both(self):
        # 5% cheaper (< cost_eps) and 1 point worse -> trade-off, both stay
        pts = [Point("a", 0.60, 0.95, 0), Point("b", 0.61, 1.0, 1)]
        self.assertEqual(self.ids(pts), ["a", "b"])

    def test_duplicates_older_wins(self):
        pts = [Point("old", 0.5, 1.0, 0), Point("new", 0.5, 1.0, 1)]
        self.assertEqual(self.ids(pts), ["old"])

    def test_antisymmetric(self):
        a, b = Point("a", 0.59, 1.0, 0), Point("b", 0.60, 2.0, 1)
        self.assertFalse(dominates(a, b) and dominates(b, a))

    def test_no_chain_loss(self):
        # A drops B (B within 2 pts, pricier); B would drop C pairwise, but C is 2.9 pts above A
        # (the last kept point) -> C must stay. Regression: this once hid the Stanford run's best point.
        pts = [Point("A", 0.649, 435, 0), Point("B", 0.665, 700, 1), Point("C", 0.678, 1321, 2)]
        self.assertEqual(self.ids(pts), ["A", "C"])

    def test_exact_mode_is_plain_pareto(self):
        pts = [Point("A", 0.649, 435, 0), Point("B", 0.665, 700, 1), Point("C", 0.678, 1321, 2),
               Point("D", 0.6, 900, 3)]
        self.assertEqual(self.ids(pts, 0, 0), ["A", "B", "C"])

    def test_hypervolume_grows_with_better_point(self):
        base = [Point("a", 0.5, 1.0), Point("b", 0.7, 3.0)]
        better = base + [Point("c", 0.65, 1.5)]
        self.assertGreater(hypervolume(better, ref_cost=4.0), hypervolume(base, ref_cost=4.0))

    def test_queue_round_robin(self):
        q = ParentQueue()
        q.sync(["a", "b"])
        self.assertEqual(q.pop(["a", "b"]), "a")
        q.sync(["a", "b", "c"])
        self.assertEqual(q.pop(["b", "c"]), "b")
        self.assertEqual(q.pop(["b", "c"]), "c")
        self.assertEqual(q.pop(["b", "c"]), "b")  # refill: new cycle
        self.assertEqual(q.pop(["c"]), "c")  # skips members no longer on frontier


class TestChecks(unittest.TestCase):
    def write(self, code: str) -> Path:
        d = Path(tempfile.mkdtemp())
        p = d / "candidate.py"
        p.write_text(code)
        return p

    def test_baselines_pass(self):
        for f in BASELINES_DIR.glob("*.py"):
            self.assertEqual(static_check(f, RULES), [], f.name)
            self.assertEqual(interface_check(f), [], f.name)

    def test_rejects_file_access_and_imports(self):
        src = (BASELINES_DIR / "no_memory.py").read_text()
        bad = src.replace("from typing import Any", "import os\nfrom typing import Any") + "\nX = open('x')\n"
        probs = static_check(self.write(bad), RULES)
        self.assertTrue(any("import not allowed: os" in p for p in probs))
        self.assertTrue(any("open" in p for p in probs))

    def test_rejects_dataset_names(self):
        src = (BASELINES_DIR / "no_memory.py").read_text() + "\n# tuned for LawBench\n"
        self.assertTrue(any("dataset-specific" in p for p in static_check(self.write(src), RULES)))

    def test_rejects_direct_llm_access(self):
        src = (BASELINES_DIR / "no_memory.py").read_text().replace("self.call_llm(", "self._llm(")
        self.assertTrue(any("_llm" in p for p in static_check(self.write(src), RULES)))

    def test_rejects_escape_routes_from_allowed_modules(self):
        """Harness code runs unsandboxed during evaluation: block routes to files/OS/interpreter."""
        base = (BASELINES_DIR / "no_memory.py").read_text()
        for payload in ("import threading\nX = threading._os\n",          # private module attribute
                        "import re\nX = re.enum.sys.modules\n",            # public chain to sys
                        "import typing\nX = typing.sys\n",
                        "X = __builtins__\n",
                        "import random\nX = getattr(random, '_os')\n"):    # getattr on a module
            self.assertTrue(static_check(self.write(base + payload), RULES), payload)
        # Private attributes on the harness itself (self._x) and harness_base imports stay allowed.
        ok = base.replace("    def get_state(self) -> str:",
                          "    def _cached(self, key):\n        self._cache = getattr(self, '_cache', {})\n"
                          "        return self._cache.get(key)\n\n    def get_state(self) -> str:")
        self.assertIn("self._cache", ok)
        self.assertEqual(static_check(self.write(ok), RULES), [])

    def test_audit(self):
        self.assertTrue(audit_transcript([{"name": "Bash", "input": {"command": "cat ../../.env"}}], AUDIT))
        self.assertTrue(audit_transcript([{"name": "Read", "input": {"file_path": "/x/third_party/a"}}], AUDIT))
        self.assertFalse(audit_transcript([{"name": "Read", "input": {"file_path": "archive/h0001/harness.py"}}], AUDIT))
        # Regression: os.environ in an analysis script is NOT the .env file.
        self.assertFalse(audit_transcript([{"name": "Bash", "input": {"command": "python3 -c 'import os; os.environ[\"TMPDIR\"]'"}}], AUDIT))
        self.assertTrue(audit_transcript([{"name": "Read", "input": {"file_path": "/a/b/.env"}}], AUDIT))
        self.assertTrue(audit_transcript([{"name": "Bash", "input": {"command": "cat .env | head"}}], AUDIT))


class TestSplits(unittest.TestCase):
    def test_learn_dev_sizes(self):
        from pareto.data import load_splits
        from pareto.evaluate import split_learn_dev

        expected = {"USPTO": (38, 12), "Symptom2Disease": (150, 50), "LawBench": (150, 50)}
        for name, (n_learn, n_dev) in expected.items():
            train, *_ = load_splits(name, CFG)
            learn, dev = split_learn_dev(train, CFG["evaluation"]["dev_fraction"])
            self.assertEqual((len(learn), len(dev)), (n_learn, n_dev), name)
            ids = {e["input"] for e in learn}
            self.assertFalse(any(e["input"] in ids for e in dev), f"{name}: learn/dev overlap")

    def test_dev_split_does_not_touch_val(self):
        from pareto.data import load_splits
        from pareto.evaluate import resolve_split

        _train, val, _test, _ = load_splits("LawBench", CFG)
        _learn, dev, _ = resolve_split("LawBench", "dev", CFG)
        val_inputs = {e["input"] for e in val}
        self.assertFalse(any(e["input"] in val_inputs for e in dev))

    def test_val_summary_has_no_examples(self):
        from pareto.run import val_summary

        ev = {"status": "crashed", "error": "LawBench: KeyError: 'some input text'", "accuracy": 0.5,
              "datasets": {"LawBench": {"status": "ok", "n": 50, "accuracy": 0.5, "errors": 1,
                                        "traceback": "secret", "usd_per_question": 0.001}}}
        s = val_summary(ev)
        self.assertNotIn("some input text", str(s))
        self.assertNotIn("secret", str(s))
        self.assertEqual(s["datasets"]["LawBench"]["accuracy"], 0.5)

    def test_audit_blocks_hidden_val(self):
        self.assertTrue(audit_transcript([{"name": "Bash", "input": {"command": "cat ../../hidden/h0001/val/eval.json"}}], AUDIT))


class TestIsolationAndViews(unittest.TestCase):
    """Isolation settings and frontier-only agent views (offline; the real-session check is
    tests/probe_isolation.py)."""

    def setUp(self):
        import secrets

        from pareto.common import RUNS_DIR

        self.run_dir = RUNS_DIR / f"unittest-{secrets.token_hex(3)}"
        for cid in ("h0001", "h0002", "h0003"):
            d = self.run_dir / "archive" / cid
            (d / "dev" / "traces").mkdir(parents=True)
            (d / "harness.py").write_text("# code\n")
            (d / "meta.json").write_text("{}")
            (d / "val_summary.json").write_text("{}")
            (d / "dev" / "traces" / "X.jsonl").write_text("{}\n")
            (d / "dev" / "stdout.log").write_text("log")
            (d / "smoke").mkdir()
        (self.run_dir / "hidden" / "h0001" / "val").mkdir(parents=True)
        (self.run_dir / "data" / "X").mkdir(parents=True)
        (self.run_dir / "data" / "X" / "learn.jsonl").write_text("{}\n")
        (self.run_dir / "history.jsonl").write_text("{}\n")
        (self.run_dir / "state.json").write_text("{}")
        for w in ("w1", "w2"):
            (self.run_dir / "workers" / w).mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_settings_hide_everything_but_own_dir_and_data(self):
        from pareto.common import ROOT, STANFORD_DIR
        from pareto.workers import build_settings

        s = build_settings(self.run_dir, self.run_dir / "workers" / "w1")
        deny = set(s["sandbox"]["filesystem"]["denyRead"])
        r = self.run_dir.resolve()
        third_party = (ROOT / "third_party").resolve()
        self.assertIn(third_party, STANFORD_DIR.resolve().parents)  # Stanford data lies inside it
        for must in (r / "hidden", r / "archive", r / "history.jsonl", r / "state.json",
                     r / "workers" / "w2", third_party):
            self.assertIn(str(must), deny)
        for allowed in (r / "workers" / "w1", r / "data", r / "workers"):
            self.assertNotIn(str(allowed), deny)
        self.assertTrue(s["sandbox"]["enabled"] and not s["sandbox"]["allowUnsandboxedCommands"])
        self.assertIn(f"Read(/{r / 'hidden'}/**)", s["permissions"]["deny"])

    def test_view_contains_only_given_ids_and_visible_files(self):
        from pareto.workers import build_view

        w = self.run_dir / "workers" / "w1"
        build_view(self.run_dir, w, ["h0001", "h0003"], {"frontier.json": []})
        self.assertEqual(sorted(p.name for p in (w / "archive").iterdir()), ["h0001", "h0003"])
        files = sorted(str(p.relative_to(w / "archive" / "h0001")) for p in (w / "archive" / "h0001").rglob("*")
                       if p.is_file())
        self.assertEqual(files, ["dev/traces/X.jsonl", "harness.py", "meta.json", "val_summary.json"])
        self.assertFalse((w / "history.jsonl").exists())
        self.assertTrue((w / "frontier.json").exists() and (w / "data" / "X" / "learn.jsonl").exists())


class TestCostMetering(unittest.TestCase):
    def test_cache_hits_are_counted_and_repriced_at_current_prices(self):
        """A cached call costs the same as a fresh one at the CURRENT config prices, even if the
        cache entry was written under different prices."""
        from unittest import mock

        from pareto import llm as L

        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        cfg = dict(CFG["harness_model"])
        usage = {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 500, "reasoning_tokens": 400}
        with mock.patch.object(L, "CACHE_DIR", d):
            old = dict(cfg, price_per_million={"input": 9.0, "cached_input": 9.0, "output": 9.0})
            first = L.HarnessLLM(old, L.UsageTracker())
            with mock.patch.object(L.HarnessLLM, "_complete", return_value=("ans", dict(usage, usd=0.0135))):
                self.assertEqual(first("q"), "ans")
            tracker = L.UsageTracker()
            second = L.HarnessLLM(cfg, tracker)  # current prices; must hit the cache, not the API
            with mock.patch.object(L.HarnessLLM, "_complete", side_effect=AssertionError("API called")):
                self.assertEqual(second("q"), "ans")
        p = cfg["price_per_million"]
        expected = (1000 * p["input"] + 500 * p["output"]) / 1e6
        u = tracker.phase_summary("learn")
        self.assertEqual((u["cache_hits"], u["real_usd"]), (1, 0.0))
        self.assertAlmostEqual(u["usd"], expected, places=9)


class TestRollbackAndLimits(unittest.TestCase):
    def test_rollback_removes_later_rounds_and_unfinished(self):
        import json
        import secrets

        from pareto.common import RUNS_DIR, write_json
        from pareto.run import Run, rollback_to

        name = f"unittest-rb-{secrets.token_hex(3)}"
        run = Run(name, CFG)
        try:
            run.archive.mkdir(parents=True)
            run.state = {"strategy": "pareto", "iteration": 3, "next_id": 4, "candidates": 4,
                         "spent": {"harness_usd": 1.0, "proposer_usd": 1.0}, "queue": {}}
            rows = [("b_x", 0, "baseline", 0.3, 0.001), ("h0001", 1, "cheaper", 0.4, 0.0009),
                    ("h0002", 2, "accurate", 0.5, 0.002), ("h0003", 3, "explore", 0.6, 0.003)]
            for cid, it, direction, acc, cost in rows:
                write_json(run.archive / cid / "meta.json", {"id": cid, "iteration": it, "direction": direction,
                                                             "status": "ok", "name": cid, "accuracy": acc,
                                                             "usd_per_question": cost})
            (run.archive / "h0004").mkdir()  # unfinished candidate (no meta.json)
            (run.hidden / "h0003" / "val").mkdir(parents=True)
            (run.dir / "workers" / "iter003_cheaper").mkdir(parents=True)
            (run.dir / "workers" / "iter002_cheaper").mkdir(parents=True)
            (run.dir / "history.jsonl").write_text("".join(json.dumps({"id": c}) + "\n" for c, *_ in rows))
            rollback_to(run, 2)
            self.assertEqual(sorted(p.name for p in run.archive.iterdir()), ["b_x", "h0001", "h0002"])
            self.assertFalse((run.hidden / "h0003").exists())
            self.assertFalse((run.dir / "workers" / "iter003_cheaper").exists())
            self.assertTrue((run.dir / "workers" / "iter002_cheaper").exists())
            self.assertEqual((run.state["iteration"], run.state["candidates"], run.state["next_id"]), (2, 2, 2))
            self.assertNotIn("h0003", (run.dir / "history.jsonl").read_text())
            self.assertEqual(run.state["spent"]["proposer_usd"], 1.0)  # money spent stays spent
        finally:
            shutil.rmtree(RUNS_DIR / name, ignore_errors=True)

    def test_usage_limit_detected_and_reviewer_without_verdict_fails(self):
        import json
        from unittest import mock

        from pareto import workers

        result_ev = {"type": "result", "is_error": True, "api_error_status": 429,
                     "api_error": "usage_limit_reached", "result": "You've hit your 5-hour limit"}
        d = Path(tempfile.mkdtemp())
        try:
            def fake_run(cmd, stdout, **_):
                stdout.write(json.dumps(result_ev) + "\n")
                return mock.Mock(returncode=1, stderr="")

            with mock.patch.object(workers.subprocess, "run", side_effect=fake_run):
                r = workers.run_claude("p", "s", d, {"model": "m"}, d / "log.jsonl", ["Read"], "low", 10, None)
                self.assertTrue(r.limited)
                with self.assertRaises(workers.AgentUnavailable):
                    workers.run_reviewer(d, d, d, {"proposer": {"model": "m"}})
            ok_ev = {"type": "result", "result": "I could not decide."}

            def fake_run_ok(cmd, stdout, **_):
                stdout.write(json.dumps(ok_ev) + "\n")
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(workers.subprocess, "run", side_effect=fake_run_ok), \
                    mock.patch.object(workers, "build_settings", return_value={}):
                verdict, _ = workers.run_reviewer(d, d, d, {"proposer": {"model": "m"}})
                self.assertEqual(verdict["verdict"], "fail")
        finally:
            shutil.rmtree(d)


class TestParetoRound(unittest.TestCase):
    """One Pareto Queue round with the agent and the candidate pipeline mocked: checks the
    1-agent and 3-agent modes route the right direction to the right folder, stage the 1-agent
    candidates, and update the queue."""

    def run_round(self, agents: int):
        import secrets
        from unittest import mock

        from pareto import run as R
        from pareto.common import RUNS_DIR, write_json
        from pareto.workers import AgentResult

        run = R.Run(f"unittest-round-{secrets.token_hex(3)}", CFG)
        self.addCleanup(shutil.rmtree, run.dir, True)
        run.archive.mkdir(parents=True)
        run.state = {"strategy": "pareto", "agents": agents, "iteration": 1, "next_id": 0, "candidates": 0,
                     "spent": {"harness_usd": 0.0, "proposer_usd": 0.0}, "queue": {"queue": ["b_x"], "seen": ["b_x"]}}
        write_json(run.archive / "b_x" / "meta.json", {"id": "b_x", "name": "x", "direction": "baseline", "iteration": 0,
                                                       "status": "ok", "accuracy": 0.5, "usd_per_question": 0.001})
        (run.archive / "b_x" / "harness.py").write_text("# parent\n")
        calls, processed = [], []

        def fake_agent(run_dir, wdir, iteration, parent, frontier, cfg, direction):
            calls.append(direction)
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "diagnosis.md").write_text("d")
            return AgentResult(0, "", 0.3, 1.0)

        def fake_process(_run, cdir, _sandbox_root, _agent, direction, _base, _it, *, proposer_usd, **_):
            processed.append((direction, cdir.relative_to(run.dir).as_posix(), round(proposer_usd, 2),
                              (cdir / "parent.py").exists(), (cdir / "diagnosis.md").exists()))
            n = len(processed)
            m = {"id": f"h{n:04d}", "name": direction, "direction": direction, "iteration": 1, "status": "ok",
                 "accuracy": 0.5 + 0.05 * n, "usd_per_question": 0.001 + 0.0001 * n}
            write_json(run.archive / m["id"] / "meta.json", m)
            return m

        with mock.patch.object(R, "run_pareto_agent", side_effect=fake_agent), \
                mock.patch.object(R, "process_candidate", side_effect=fake_process):
            R.iterate_pareto(run)
        self.assertTrue(str(run.dir).startswith(str(RUNS_DIR)))
        return calls, sorted(processed), run

    def test_three_agents(self):
        calls, processed, run = self.run_round(3)
        self.assertEqual(sorted(calls), ["accurate", "cheaper", "explore"])
        self.assertEqual([p[:3] for p in processed], [("accurate", "workers/iter001_accurate", 0.3),
                                                      ("cheaper", "workers/iter001_cheaper", 0.3),
                                                      ("explore", "workers/iter001_explore", 0.3)])
        self.assertIn("h0003", run.state["queue"]["seen"])  # new frontier members were enqueued

    def test_one_agent(self):
        calls, processed, _ = self.run_round(1)
        self.assertEqual(calls, [None])
        self.assertEqual(processed, [("accurate", "workers/iter001/c2", 0.1, True, True),
                                     ("cheaper", "workers/iter001/c1", 0.1, True, True),
                                     ("explore", "workers/iter001/c3", 0.1, True, True)])

    def test_crash_mid_round_discards_the_round(self):
        import secrets
        from unittest import mock

        from pareto import run as R
        from pareto.common import write_json

        run = R.Run(f"unittest-crash-{secrets.token_hex(3)}", CFG)
        self.addCleanup(shutil.rmtree, run.dir, True)
        run.archive.mkdir(parents=True)
        run.state = {"strategy": "pareto", "agents": 3, "iteration": 2, "next_id": 3, "candidates": 3,
                     "spent": {"harness_usd": 0.0, "proposer_usd": 0.0}, "queue": {}}
        for cid, it in (("b_x", 0), ("h0001", 1), ("h0002", 2), ("h0003", 2)):
            write_json(run.archive / cid / "meta.json", {"id": cid, "iteration": it, "status": "ok", "name": cid,
                                                         "direction": "baseline" if it == 0 else "cheaper",
                                                         "accuracy": 0.5, "usd_per_question": 0.001})

        def crash(r):
            r.new_id()  # the round already allocated an id and wrote a partial candidate
            (r.archive / "h0004").mkdir()
            raise RuntimeError("boom")

        with mock.patch.dict(R.ITERATE, {"pareto": crash}), self.assertRaises(RuntimeError):
            R.run_round(run)
        self.assertFalse((run.archive / "h0004").exists())
        self.assertEqual((run.state["iteration"], run.state["next_id"], run.state["candidates"]), (2, 3, 3))


class TestAgentsCannotLearnUnseenDatasets(unittest.TestCase):
    UNSEEN = [n.lower() for n in CFG["datasets"]["unseen"]] + ["aegis2"]

    def test_files_agents_can_read_never_name_unseen_datasets(self):
        """Agents can read pareto/ (code, prompts, baselines). config.yaml, README, results and
        tests are hidden. None of the readable files may name an unseen dataset."""
        from pareto.common import PKG

        for f in PKG.rglob("*"):
            if f.is_file() and f.suffix in (".py", ".md"):
                text = f.read_text().lower()
                for name in self.UNSEEN:
                    self.assertNotIn(name, text, f"{f.relative_to(PKG)} names unseen dataset '{name}'")

    def test_files_naming_unseen_datasets_are_hidden(self):
        from pareto.common import ROOT, RUNS_DIR
        from pareto.workers import hidden_paths

        run_dir = RUNS_DIR / "unittest-hidden"
        (run_dir / "workers" / "w1").mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, run_dir, True)
        hidden = {str(p) for p in hidden_paths(run_dir, run_dir / "workers" / "w1")}
        for name in ("config.yaml", "README.md", "results", "tests", "third_party", ".env"):
            if (ROOT / name).exists():
                self.assertIn(str((ROOT / name).resolve()), hidden, name)

    def test_agents_self_check_runs_from_rules_file_without_config(self):
        import json
        import subprocess
        import sys

        from pareto.common import ROOT
        from pareto.workers import SELF_CHECK_RULES, self_check_rules

        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        rules = self_check_rules(CFG)
        self.assertNotIn("dataset_patterns", rules)
        (d / SELF_CHECK_RULES).write_text(json.dumps(rules))
        shutil.copy(BASELINES_DIR / "fewshot_10.py", d / "candidate.py")
        p = subprocess.run([sys.executable, "-m", "pareto.checks", "--rules", SELF_CHECK_RULES, "candidate.py"],
                           cwd=d, env={"PYTHONPATH": str(ROOT)}, capture_output=True, text=True)
        self.assertIn("SELF-CHECK PASSED", p.stdout, p.stdout + p.stderr)


class TestPrompts(unittest.TestCase):
    def test_all_roles_render_completely_from_config(self):
        import re

        from pareto.workers import system_prompt

        for role, direction in (("pareto", "cheaper"), ("pareto", "accurate"), ("pareto", "explore"),
                                ("pareto_single", None), ("stanford", None)):
            text = system_prompt(role, CFG, direction)
            self.assertEqual(re.findall(r"\{[a-z_]+\}", text), [], role)
            self.assertIn(f"at most {RULES['max_lines']} lines", text)
            self.assertIn("--rules self_check_rules.json", text)


class TestFakeEvaluation(unittest.TestCase):
    def test_baselines_run_end_to_end_with_fake_model(self):
        out = Path(tempfile.mkdtemp())
        try:
            for name in ("no_memory", "fewshot_10"):
                s = evaluate_harness(BASELINES_DIR / f"{name}.py", ["USPTO", "LawBench"], "val",
                                     out / name, CFG, fake=True, limit_train=5, limit_eval=4)
                self.assertEqual(s["status"], "ok", s.get("error"))
                self.assertIn("accuracy", s)
                self.assertTrue((out / name / "traces" / "USPTO.jsonl").exists())
                calls = s["calls_per_question"]
                self.assertAlmostEqual(calls, 1.0)
        finally:
            shutil.rmtree(out)


if __name__ == "__main__":
    unittest.main()
