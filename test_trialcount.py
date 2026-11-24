import math
import os
import random
import unittest

import trialcount as tc

HERE = os.path.dirname(os.path.abspath(__file__))


class Intervals(unittest.TestCase):
    def test_wilson_known_value(self):
        # 8/10 at 95%: 0.4902 to 0.9433 (standard worked example)
        lo, hi = tc.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)

    def test_clopper_pearson_known_value(self):
        # 8/10 at 95%: 0.4439 to 0.9748
        lo, hi = tc.clopper_pearson(8, 10)
        self.assertAlmostEqual(lo, 0.4439, places=3)
        self.assertAlmostEqual(hi, 0.9748, places=3)

    def test_rule_of_three(self):
        # 0 failures in n trials: upper bound on failure rate is close to 3/n (one-sided 95%)
        lo, hi = tc.clopper_pearson(100, 100, conf=0.90)
        self.assertAlmostEqual(1 - lo, 3 / 100, delta=0.002)
        self.assertEqual(hi, 1.0)

    def test_wald_collapses_at_perfect_score(self):
        self.assertEqual(tc.wald(40, 40), (1.0, 1.0))
        lo, hi = tc.wilson(40, 40)
        self.assertLess(lo, 0.92)

    def test_bounds_ordered_and_contain_estimate(self):
        for n in (1, 7, 40, 250):
            for k in range(n + 1):
                for fn in (tc.wilson, tc.clopper_pearson):
                    lo, hi = fn(k, n)
                    self.assertLessEqual(0.0, lo)
                    self.assertLessEqual(hi, 1.0)
                    self.assertLessEqual(lo, k / n + 1e-12)
                    self.assertLessEqual(k / n - 1e-12, hi)

    def test_clopper_pearson_contains_wilson_mostly_wider(self):
        lo_w, hi_w = tc.wilson(34, 40)
        lo_c, hi_c = tc.clopper_pearson(34, 40)
        self.assertGreater(hi_c - lo_c, hi_w - lo_w)


class Binomial(unittest.TestCase):
    def test_cdf_matches_direct_sum(self):
        n, p = 30, 0.37
        for k in range(n + 1):
            direct = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
            self.assertAlmostEqual(tc.binom_cdf(k, n, p), direct, places=10)
            self.assertAlmostEqual(tc.binom_cdf(k, n, p) + tc.binom_sf(k + 1, n, p), 1.0, places=10)


class Fisher(unittest.TestCase):
    def test_known_value(self):
        # tea tasting table [[3,1],[1,3]]: two-sided p = 34/70
        self.assertAlmostEqual(tc.fisher_exact(3, 4, 1, 4), 34 / 70, places=9)

    def test_symmetric_and_identical(self):
        self.assertAlmostEqual(tc.fisher_exact(30, 40, 36, 40), tc.fisher_exact(36, 40, 30, 40))
        self.assertAlmostEqual(tc.fisher_exact(20, 40, 20, 40), 1.0)

    def test_obvious_difference(self):
        self.assertLess(tc.fisher_exact(5, 40, 35, 40), 1e-9)


class Planning(unittest.TestCase):
    def test_compare_known_value(self):
        # 0.80 vs 0.90, alpha 0.05, power 0.80: 199 per group in standard tables
        self.assertEqual(tc.trials_to_compare(0.80, 0.90), 199)

    def test_compare_symmetric_and_monotone(self):
        self.assertEqual(tc.trials_to_compare(0.7, 0.9), tc.trials_to_compare(0.9, 0.7))
        self.assertGreater(tc.trials_to_compare(0.80, 0.85), tc.trials_to_compare(0.80, 0.90))

    def test_width(self):
        n = tc.trials_for_width(0.8, 0.05)
        lo, hi = tc.wilson(round(0.8 * n), n)
        self.assertLessEqual((hi - lo) / 2, 0.05)
        self.assertGreater(n, 200)

    def test_fixed_plan_meets_both_error_bounds(self):
        n, c = tc.fixed_plan(0.80, 0.90)
        self.assertLessEqual(tc.binom_sf(c, n, 0.80), 0.05)
        self.assertGreaterEqual(tc.binom_sf(c, n, 0.90), 0.95)
        # and one fewer trial does not have a pass mark that works
        m = n - 1
        ok = any(tc.binom_sf(k, m, 0.80) <= 0.05 and tc.binom_sf(k, m, 0.90) >= 0.95
                 for k in range(m + 2))
        self.assertFalse(ok)


class Sequential(unittest.TestCase):
    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            tc.SPRT(0.9, 0.8)

    def test_all_failures_rejects_fast(self):
        s = tc.SPRT(0.80, 0.90)
        while s.decision is None:
            s.update(False)
        self.assertEqual(s.decision, "reject")
        self.assertLessEqual(s.n, 5)

    def test_all_successes_accepts(self):
        s = tc.SPRT(0.80, 0.90)
        while s.decision is None:
            s.update(True)
        self.assertEqual(s.decision, "accept")
        self.assertEqual(s.n, math.ceil(math.log(19) / math.log(0.9 / 0.8)))

    def test_decision_is_sticky(self):
        s = tc.SPRT(0.5, 0.9)
        while s.decision is None:
            s.update(True)
        n = s.n
        s.update(False)
        self.assertEqual((s.decision, s.n), ("accept", n))

    def test_truncation(self):
        s = tc.SPRT(0.80, 0.90, max_trials=10)
        for i in range(10):
            s.update(i != 0)  # one failure then successes
        self.assertIsNotNone(s.decision)
        self.assertTrue(s.truncated)

    def test_error_rates_within_bounds(self):
        rng = random.Random(1)
        wrong_accept = wrong_reject = 0
        runs = 2000
        for _ in range(runs):
            s = tc.SPRT(0.80, 0.90)
            while s.update(rng.random() < 0.80) is None:
                pass
            wrong_accept += s.decision == "accept"
            s = tc.SPRT(0.80, 0.90)
            while s.update(rng.random() < 0.90) is None:
                pass
            wrong_reject += s.decision == "reject"
        self.assertLess(wrong_accept / runs, 0.07)
        self.assertLess(wrong_reject / runs, 0.07)


class Log(unittest.TestCase):
    def test_report_on_sample(self):
        eps = tc.load_episodes(os.path.join(HERE, "sample_episodes.jsonl"))
        self.assertEqual(sorted(eps), ["act_v1", "act_v2"])
        self.assertEqual(len(eps["act_v1"]), 40)
        text = tc.report(eps)
        self.assertIn("act_v1", text)
        self.assertIn("cannot tell them apart yet", text)

    def test_bad_line(self):
        path = os.path.join(HERE, "_bad.jsonl")
        with open(path, "w") as f:
            f.write('{"policy": "a"}\n')
        try:
            with self.assertRaises(ValueError):
                tc.load_episodes(path)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
