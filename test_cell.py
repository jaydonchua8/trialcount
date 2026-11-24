"""Tests for the parts added after the first version: exact sequential analysis,
session-aware intervals, drift detection, the episode log and the acceptance report."""
import contextlib
import io
import json
import math
import os
import random
import tempfile
import unittest

import cellrun
import robust
import simcell
import trialcount as tc

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = {"parts": ["PN-1041", "PN-2210"], "first_pass_yield": 0.95, "cycle_time_s": 12.0,
        "continuous_run_s": 14400, "changeover_s": 600}


def episodes(**kw):
    kw.setdefault("intervene_given_failure", 0.02)
    return [cellrun.parse_episode(r, "r") for r in simcell.generate(**kw)]


def verdicts(checks):
    return {(c.name, c.scope): c.verdict for c in checks}


def ep(i, t, part="A", session="S1", success=True, intervention=False, cycle=10.0):
    return cellrun.Episode(i, session, float(t), part, "p", success, intervention, cycle)


class Bounds(unittest.TestCase):
    def test_t_quantiles(self):
        self.assertAlmostEqual(tc.t_ppf(0.975, 1), 12.7062, places=3)
        self.assertAlmostEqual(tc.t_ppf(0.975, 4), 2.7764, places=3)
        self.assertAlmostEqual(tc.t_ppf(0.95, 30), 1.6973, places=3)
        self.assertAlmostEqual(tc.t_ppf(0.025, 9), -2.2622, places=3)
        self.assertAlmostEqual(tc.t_cdf(0.0, 7), 0.5)
        with self.assertRaises(ValueError):
            tc.t_ppf(1.0, 3)

    def test_one_sided_bounds_match_two_sided_at_double_alpha(self):
        for k, n in ((0, 10), (8, 10), (34, 40), (40, 40)):
            lo, hi = tc.clopper_pearson(k, n, conf=0.90)
            self.assertAlmostEqual(tc.lower_bound(k, n, 0.95), lo, places=9)
            self.assertAlmostEqual(tc.upper_bound(k, n, 0.95), hi, places=9)

    def test_zero_failure_rule(self):
        n = tc.zero_failure_trials(0.99)
        self.assertEqual(n, 299)
        self.assertGreaterEqual(tc.lower_bound(n, n), 0.99)
        self.assertLess(tc.lower_bound(n - 1, n - 1), 0.99)
        with self.assertRaises(ValueError):
            tc.zero_failure_trials(1.0)

    def test_mcnemar(self):
        self.assertAlmostEqual(tc.mcnemar_exact(1, 9), 22 / 1024, places=12)
        self.assertEqual(tc.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(tc.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(tc.mcnemar_exact(9, 1), tc.mcnemar_exact(1, 9))


class ExactSequential(unittest.TestCase):
    def test_exact_matches_simulation(self):
        exact = tc.sprt_exact(tc.SPRT(0.80, 0.90), 0.90)
        rng = random.Random(4)
        runs, acc, tot = 4000, 0, 0
        for _ in range(runs):
            s = tc.SPRT(0.80, 0.90)
            while s.update(rng.random() < 0.90) is None:
                pass
            acc += s.decision == "accept"
            tot += s.n
        self.assertAlmostEqual(acc / runs, exact["accept"], delta=0.012)
        self.assertAlmostEqual(tot / runs, exact["mean_trials"], delta=3.0)

    def test_accept_probability_rises_with_true_rate(self):
        t = tc.SPRT(0.80, 0.90, max_trials=135)
        acc = [tc.sprt_exact(t, p)["accept"] for p in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)]
        self.assertEqual(acc, sorted(acc))
        self.assertLess(acc[0], 1e-3)
        self.assertGreater(acc[-1], 0.999)

    def test_naive_cap_breaks_the_error_bound(self):
        t = tc.SPRT(0.80, 0.90, max_trials=135)
        self.assertGreater(tc.sprt_exact(t, 0.80)["accept"], 0.05)
        self.assertGreater(1 - tc.sprt_exact(t, 0.90)["accept"], 0.05)

    def test_capped_plan_meets_bounds_and_is_never_longer_than_cap(self):
        for cap in (135, 149):
            pl = tc.capped_plan(0.80, 0.90, max_trials=cap)
            t = tc.SPRT(0.80, 0.90, pl["alpha_inner"], pl["beta_inner"], pl["max_trials"], pl["pass_mark"])
            fp = tc.sprt_exact(t, 0.80)
            ff = tc.sprt_exact(t, 0.90)
            self.assertLessEqual(fp["accept"], 0.05)
            self.assertLessEqual(1 - ff["accept"], 0.05)
            self.assertAlmostEqual(fp["accept"], pl["false_pass"], places=12)
            self.assertLess(ff["mean_trials"], cap)
            rng = random.Random(cap)
            for _ in range(200):
                s = tc.SPRT(0.80, 0.90, pl["alpha_inner"], pl["beta_inner"], pl["max_trials"], pl["pass_mark"])
                while s.update(rng.random() < 0.85) is None:
                    pass
                self.assertLessEqual(s.n, cap)

    def test_capped_plan_refuses_an_impossible_cap(self):
        with self.assertRaises(ValueError):
            tc.capped_plan(0.80, 0.90, max_trials=50)

    def test_pass_mark_decides_a_forced_stop(self):
        s = tc.SPRT(0.80, 0.90, max_trials=10, pass_mark=10)
        for i in range(10):
            s.update(i != 0)
        self.assertEqual((s.decision, s.truncated), ("reject", True))

    def test_bad_settings(self):
        for kw in (dict(alpha=0.0), dict(beta=1.0), dict(alpha=0.6, beta=0.6),
                   dict(max_trials=0), dict(pass_mark=5)):
            with self.assertRaises(ValueError):
                tc.SPRT(0.8, 0.9, **kw)


class Sessions(unittest.TestCase):
    def test_design_effect(self):
        self.assertEqual(robust.design_effect([(50, 60)]), 1.0)
        self.assertEqual(robust.design_effect([(60, 60), (60, 60)]), 1.0)
        self.assertEqual(robust.design_effect([(54, 60)] * 5), 1.0)
        self.assertGreater(robust.design_effect([(60, 60), (45, 60), (58, 60), (40, 60)]), 3.0)

    def test_cluster_interval_never_narrower_than_wilson(self):
        rng = random.Random(2)
        for _ in range(200):
            cl = []
            for _ in range(rng.randint(1, 6)):
                n = rng.randint(1, 80)
                cl.append((rng.randint(0, n), n))
            lo, hi = robust.cluster_interval(cl)
            wlo, whi = tc.wilson(sum(k for k, _ in cl), sum(n for _, n in cl))
            self.assertLessEqual(lo, wlo + 1e-12)
            self.assertGreaterEqual(hi, whi - 1e-12)
            self.assertTrue(0.0 <= lo <= hi <= 1.0)
            blo, bhi = robust.cluster_bounds(cl)
            self.assertTrue(0.0 <= blo <= bhi <= 1.0)

    def test_disagreeing_sessions_widen_the_interval(self):
        same = robust.cluster_interval([(54, 60)] * 4)
        apart = robust.cluster_interval([(60, 60), (48, 60), (60, 60), (48, 60)])
        self.assertGreater(apart[1] - apart[0], 2 * (same[1] - same[0]))

    def test_one_session_falls_back_to_exact_bounds(self):
        self.assertEqual(robust.cluster_interval([(34, 40)]), tc.wilson(34, 40))
        self.assertEqual(robust.cluster_bounds([(34, 40)])[0], tc.lower_bound(34, 40))

    def test_bad_clusters(self):
        with self.assertRaises(ValueError):
            robust.cluster_interval([])
        with self.assertRaises(ValueError):
            robust.cluster_interval([(5, 4)])
        with self.assertRaises(ValueError):
            robust.design_effect([(0, 0)])


class Drift(unittest.TestCase):
    def test_finds_an_obvious_step(self):
        rng = random.Random(0)
        xs = [rng.random() < 0.97 for _ in range(300)] + [rng.random() < 0.75 for _ in range(300)]
        cp = robust.changepoint(xs, n_perm=199)
        self.assertLess(cp["p_value"], 0.01)
        self.assertLess(abs(cp["index"] - 300), 40)
        self.assertGreater(cp["before"], cp["after"])

    def test_quiet_on_a_steady_run(self):
        flagged = 0
        for seed in range(40):
            rng = random.Random(seed)
            xs = [rng.random() < 0.9 for _ in range(300)]
            flagged += robust.changepoint(xs, n_perm=99, seed=seed)["p_value"] < 0.05
        self.assertLessEqual(flagged, 6)

    def test_no_test_possible(self):
        for xs in ([True] * 200, [False] * 200, [True, False] * 10, []):
            cp = robust.changepoint(xs)
            self.assertEqual((cp["index"], cp["p_value"]), (None, 1.0))

    def test_grouping_ignores_a_bad_session_but_plain_test_flags_it(self):
        rng = random.Random(1)
        xs = [rng.random() < 0.97 for _ in range(300)] + [rng.random() < 0.80 for _ in range(300)]
        groups = [0] * 300 + [1] * 300
        self.assertLess(robust.changepoint(xs, n_perm=199)["p_value"], 0.01)
        self.assertGreater(robust.changepoint(xs, groups, n_perm=199)["p_value"], 0.05)

    def test_grouping_still_finds_a_change_inside_a_session(self):
        rng = random.Random(3)
        xs = ([rng.random() < 0.90 for _ in range(400)]
              + [rng.random() < 0.97 for _ in range(200)] + [rng.random() < 0.70 for _ in range(200)])
        groups = [0] * 400 + [1] * 400
        cp = robust.changepoint(xs, groups, n_perm=199)
        self.assertLess(cp["p_value"], 0.01)
        self.assertLess(abs(cp["index"] - 600), 40)

    def test_strata_stop_a_part_change_looking_like_drift(self):
        rng = random.Random(5)
        xs = [rng.random() < 0.98 for _ in range(300)] + [rng.random() < 0.80 for _ in range(300)]
        parts = ["easy"] * 300 + ["hard"] * 300
        self.assertLess(robust.changepoint(xs, n_perm=199)["p_value"], 0.01)
        self.assertGreater(robust.changepoint(xs, strata=parts, n_perm=199)["p_value"], 0.05)

    def test_deterministic_and_validated(self):
        rng = random.Random(9)
        xs = [rng.random() < 0.9 for _ in range(200)]
        self.assertEqual(robust.changepoint(xs, n_perm=50, seed=1), robust.changepoint(xs, n_perm=50, seed=1))
        with self.assertRaises(ValueError):
            robust.changepoint(xs, groups=[0])
        with self.assertRaises(ValueError):
            robust.changepoint(xs, min_seg=0)


class Log(unittest.TestCase):
    GOOD = {"episode": 1, "session": "S1", "t_start": 0.0, "part": "A", "success": True, "cycle_s": 9.5}

    def _write(self, lines):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        self.addCleanup(os.remove, path)
        return path

    def test_good_row_and_defaults(self):
        e = cellrun.parse_episode(self.GOOD, "x")
        self.assertEqual((e.intervention, e.policy, e.success), (False, "", True))

    def test_bad_rows(self):
        bad = [dict(self.GOOD, success="false"), dict(self.GOOD, success=1),
               dict(self.GOOD, intervention="no"), dict(self.GOOD, cycle_s=0),
               dict(self.GOOD, cycle_s=float("nan")), dict(self.GOOD, t_start="noon"),
               dict(self.GOOD, episode=1.5), dict(self.GOOD, episode=True),
               dict(self.GOOD, part=""), dict(self.GOOD, session=3), [1, 2]]
        for key in ("episode", "session", "t_start", "part", "success", "cycle_s"):
            bad.append({k: v for k, v in self.GOOD.items() if k != key})
        for row in bad:
            with self.assertRaises(cellrun.LogError, msg=repr(row)):
                cellrun.parse_episode(row, "x")

    def test_file_errors_name_the_line(self):
        path = self._write([json.dumps(self.GOOD), "{not json"])
        with self.assertRaisesRegex(cellrun.LogError, "line 2"):
            cellrun.load_log(path)
        with self.assertRaisesRegex(cellrun.LogError, "no episodes"):
            cellrun.load_log(self._write([""]))

    def test_duplicate_and_overlapping_episodes(self):
        with self.assertRaisesRegex(cellrun.LogError, "more than once"):
            cellrun.check_log([ep(1, 0), ep(1, 20)])
        with self.assertRaisesRegex(cellrun.LogError, "overlap"):
            cellrun.check_log([ep(1, 0), ep(2, 5)])
        # the same clock times in two different sessions (two cells) are fine
        self.assertEqual(len(cellrun.check_log([ep(1, 0), ep(2, 5, session="S2")])), 2)

    def test_log_is_sorted_by_time(self):
        out = cellrun.check_log([ep(2, 50), ep(1, 0)])
        self.assertEqual([e.episode for e in out], [1, 2])

    def test_old_loader_refuses_string_booleans(self):
        path = self._write(['{"policy": "a", "success": "false"}'])
        with self.assertRaises(ValueError):
            tc.load_episodes(path)

    def test_simulator_output_round_trips(self):
        recs = simcell.generate(seed=1, sessions=2, per_session=60, block=30)
        self.assertTrue(all(r["synthetic"] is True for r in recs))
        self.assertEqual(recs, simcell.generate(seed=1, sessions=2, per_session=60, block=30))
        path = self._write([json.dumps(r) for r in recs])
        self.assertEqual(len(cellrun.load_log(path)), 120)


class Spec(unittest.TestCase):
    def test_shipped_spec_loads(self):
        spec = cellrun.load_spec(os.path.join(HERE, "acceptance_spec.json"))
        self.assertEqual(spec["confidence"], 0.95)

    def test_bad_specs(self):
        for change in (dict(parts=[]), dict(parts=["A", "A"]), dict(parts="A"),
                       dict(first_pass_yield=1.0), dict(first_pass_yield=95),
                       dict(cycle_time_s=0), dict(continuous_run_s=None),
                       dict(changeover_s="fast"), dict(confidence=1.0)):
            with self.assertRaises(ValueError, msg=repr(change)):
                cellrun.load_spec(dict(SPEC, **change))


class LongestRun(unittest.TestCase):
    def test_intervention_and_idle_gap_break_a_run(self):
        eps = [ep(0, 0), ep(1, 12), ep(2, 24, intervention=True), ep(3, 36), ep(4, 48), ep(5, 60),
               ep(6, 500), ep(7, 512)]
        length, session, a, b = cellrun.longest_run(eps, changeover_s=600)
        self.assertEqual((a, b), (3, 5))
        self.assertAlmostEqual(length, 34.0)

    def test_changeover_gap_is_allowed_up_to_the_limit(self):
        eps = [ep(0, 0), ep(1, 12), ep(2, 322, part="B"), ep(3, 334, part="B")]
        self.assertAlmostEqual(cellrun.longest_run(eps, changeover_s=600)[0], 344.0)
        self.assertAlmostEqual(cellrun.longest_run(eps, changeover_s=100)[0], 22.0)

    def test_all_interventions(self):
        self.assertEqual(cellrun.longest_run([ep(0, 0, intervention=True)], 600)[0], 0.0)


class Acceptance(unittest.TestCase):
    def test_healthy_cell_passes(self):
        overall, checks = cellrun.evaluate(SPEC, episodes(seed=3), n_perm=199)
        self.assertEqual(overall, "PASS")
        self.assertEqual(set(verdicts(checks).values()), {"PASS"})

    def test_wear_inside_a_shift_is_reported_and_blocks_a_pass(self):
        wear = {"session": 1, "at": 0.6, "drop": 0.06, "shape": "step", "persist": True}
        overall, checks = cellrun.evaluate(SPEC, episodes(seed=3, drift=wear), n_perm=199)
        v = verdicts(checks)
        self.assertEqual(v[("yield stability", "cell")], "UNSTABLE")
        self.assertNotEqual(overall, "PASS")
        detail = [c.detail for c in checks if c.name == "yield stability"][0]
        self.assertIn("S02", detail)

    def test_too_few_trials_is_not_yet_and_says_how_many_more(self):
        overall, checks = cellrun.evaluate(
            SPEC, episodes(seed=3, sessions=1, per_session=150, block=75), n_perm=99)
        v = verdicts(checks)
        self.assertEqual(overall, "NOT YET")
        self.assertEqual(v[("first-pass yield", "PN-2210")], "NOT YET")
        self.assertEqual(v[("continuous run", "cell")], "NOT YET")
        detail = [c.detail for c in checks if c.scope == "PN-2210" and c.name == "first-pass yield"][0]
        self.assertRegex(detail, r"about \d+ more trials")

    def test_low_yield_fails(self):
        parts = {"PN-1041": (0.975, 10.8, 0.9), "PN-2210": (0.85, 11.4, 1.1)}
        overall, checks = cellrun.evaluate(SPEC, episodes(seed=2, parts=parts, sessions=1), n_perm=99)
        v = verdicts(checks)
        self.assertEqual(overall, "FAIL")
        self.assertEqual(v[("first-pass yield", "PN-2210")], "FAIL")
        self.assertEqual(v[("first-pass yield", "PN-1041")], "PASS")

    def test_slow_part_fails_cycle_time(self):
        parts = {"PN-1041": (0.975, 10.8, 0.9), "PN-2210": (0.975, 13.0, 1.1)}
        overall, checks = cellrun.evaluate(SPEC, episodes(seed=2, parts=parts, sessions=1), n_perm=99)
        self.assertEqual(verdicts(checks)[("cycle time", "PN-2210")], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_frequent_interventions_fail_the_continuous_run(self):
        overall, checks = cellrun.evaluate(
            SPEC, episodes(seed=3, sessions=1, intervene_given_failure=0.9), n_perm=99)
        self.assertEqual(verdicts(checks)[("continuous run", "cell")], "FAIL")

    def test_intervention_counts_against_first_pass_yield(self):
        eps = [ep(i, 12 * i, intervention=(i % 2 == 0)) for i in range(200)]
        spec = dict(SPEC, parts=["A"])
        c = cellrun.yield_check("A", eps, cellrun.load_spec(spec))
        self.assertEqual(c.verdict, "FAIL")
        self.assertIn("100/200", c.detail)

    def test_slow_changeover_fails(self):
        overall, checks = cellrun.evaluate(
            SPEC, episodes(seed=3, sessions=1, changeover_s=(900.0, 10.0)), n_perm=99)
        self.assertEqual(verdicts(checks)[("changeover", "cell")], "FAIL")

    def test_part_in_spec_with_no_episodes(self):
        spec = dict(SPEC, parts=SPEC["parts"] + ["PN-9999"])
        overall, checks = cellrun.evaluate(spec, episodes(seed=3), n_perm=99)
        self.assertEqual(verdicts(checks)[("first-pass yield", "PN-9999")], "NOT YET")
        self.assertEqual(overall, "NOT YET")

    def test_parts_outside_the_spec_are_ignored_and_reported(self):
        spec = dict(SPEC, parts=["PN-1041"])
        overall, checks = cellrun.evaluate(spec, episodes(seed=3), n_perm=99)
        scope = [c for c in checks if c.name == "scope"]
        self.assertEqual(len(scope), 1)
        self.assertIn("PN-2210", scope[0].detail)
        self.assertNotIn(("first-pass yield", "PN-2210"), verdicts(checks))

    def test_two_sessions_that_disagree_do_not_pass(self):
        # 97% pooled, comfortably above 95% on a binomial bound, but one session is at 94%
        eps = ([ep(i, 12 * i, session="S1") for i in range(1000)]
               + [ep(1000 + i, 12 * i, session="S2", success=(i % 50 >= 3)) for i in range(1000)])
        spec = cellrun.load_spec(dict(SPEC, parts=["A"]))
        self.assertGreaterEqual(tc.lower_bound(1940, 2000), 0.95)
        c = cellrun.yield_check("A", eps, spec)
        self.assertEqual(c.verdict, "NOT YET")
        self.assertIn("more sessions", c.detail)

    def test_trials_needed(self):
        n = cellrun.trials_needed(0.98, 0.95, 0.95)
        self.assertGreaterEqual(tc.lower_bound(math.floor(0.98 * n + 1e-9), n), 0.95)
        self.assertIsNone(cellrun.trials_needed(0.95, 0.95, 0.95))
        self.assertIsNone(cellrun.trials_needed(0.90, 0.95, 0.95))

    def test_cli_refuses_mixed_policies(self):
        recs = simcell.generate(seed=1, sessions=1, per_session=60, block=30)
        recs[5]["policy"] = "other"
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(json.dumps(r) for r in recs))
        self.addCleanup(os.remove, path)
        spec = os.path.join(HERE, "acceptance_spec.json")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cellrun.main(["cellrun.py", spec, path]), 2)
            self.assertEqual(cellrun.main(["cellrun.py", spec, path, "--policy=nobody"]), 2)
            self.assertEqual(cellrun.main(["cellrun.py", spec, "/no/such/file.jsonl"]), 2)
            self.assertEqual(cellrun.main(["cellrun.py", spec]), 2)
            self.assertEqual(cellrun.main(["cellrun.py", spec, path, "--policy=act_v3"]), 1)
        self.assertIn("mixes policies", out.getvalue())
        self.assertIn("overall: NOT YET", out.getvalue())


if __name__ == "__main__":
    unittest.main()
