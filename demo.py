"""demo: python demo.py

Runs in well under a minute. All data is synthetic (see simcell.py).
"""
import json
import os
import random

import cellrun
import simcell
import trialcount as tc

HERE = os.path.dirname(os.path.abspath(__file__))


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. What 40 rollouts can say")
    lo, hi = tc.wilson(34, 40)
    print("34/40 successes: 85.0%%, 95%% interval %.1f%% to %.1f%%" % (100 * lo, 100 * hi))
    print("40/40 successes: the true rate is at least %.1f%% (95%% one-sided)"
          % (100 * tc.lower_bound(40, 40)))
    print("to show 99%% with zero failures you need %d trials in a row" % tc.zero_failure_trials(0.99))

    section("2. Certify 'at least 90%, not 80%' with a hard cap on trials")
    plan = tc.capped_plan(0.80, 0.90, max_trials=149)
    print("plan: stop early on clear evidence, never run more than %d trials, pass mark %d at the cap"
          % (plan["max_trials"], plan["pass_mark"]))
    print("exact error rates: passes an 80%% policy %.2f%%, fails a 90%% policy %.2f%% (fixed plan needs %d trials)"
          % (100 * plan["false_pass"], 100 * plan["false_fail"], plan["fixed_n"]))
    rng = random.Random(5)
    for true_rate in (0.95, 0.70):
        test = tc.SPRT(0.80, 0.90, plan["alpha_inner"], plan["beta_inner"],
                       plan["max_trials"], plan["pass_mark"])
        while test.update(rng.random() < true_rate) is None:
            pass
        print("simulated policy with true rate %.0f%%: %s after %d trials (%d successes)"
              % (100 * true_rate, test.decision, test.n, test.k))

    section("3. Acceptance report for a synthetic cell")
    spec = cellrun.load_spec(os.path.join(HERE, "acceptance_spec.json"))
    print("spec: " + json.dumps({k: spec[k] for k in sorted(spec)}))
    wear = {"session": 1, "at": 0.6, "drop": 0.06, "shape": "step", "persist": True}
    cases = (
        ("healthy cell, 3 shifts of 1400 parts", dict(seed=3)),
        ("same cell, gripper wear from 60% into shift 2 (6 point yield drop)",
         dict(seed=3, drift=wear)),
        ("healthy cell, but only 2 short sessions of 150 parts",
         dict(seed=3, sessions=2, per_session=150, block=75)),
    )
    for title, kw in cases:
        records = simcell.generate(**kw)
        episodes = [cellrun.parse_episode(r, "record %d" % i) for i, r in enumerate(records)]
        overall, checks = cellrun.evaluate(spec, episodes)
        print("\n[%s]" % title)
        print(cellrun.format_report(overall, checks))


if __name__ == "__main__":
    main()
