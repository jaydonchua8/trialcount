"""Numbers quoted in the README. Deterministic: fixed seeds, exact sums where possible.

A and B: independent trials (intervals, comparing two policies).
C: sequential certification, exact operating characteristics, capped designs.
D: intervals when trials cluster by session. Method chosen on DEV settings, then
   measured on HELD-OUT settings (different seed, session counts, sizes, rates, spread).
E: drift detection. min_seg chosen on DEV, then measured on HELD-OUT (different seed,
   session layout, base rate, and a ramp instead of a step).
All data in D and E is synthetic.
"""
import json
import math
import random
import time

import robust
import simcell
import trialcount as tc


def coverage(fn, n, p):
    """Exact probability that fn's 95% interval contains the true rate p."""
    tot = 0.0
    for k in range(n + 1):
        lo, hi = fn(k, n)
        if lo <= p <= hi:
            tot += math.exp(tc._logpmf(k, n, p))
    return tot


def part_a():
    grid = [x / 100 for x in range(50, 100)]
    rows = []
    print("A. Actual coverage of a nominal 95% interval, true rate from 0.50 to 0.99 (exact)")
    print("   n   | wald mean / worst | wilson mean / worst | clopper-pearson mean / worst | wilson width at 85%")
    for n in (10, 20, 40, 100, 400):
        cache = {name: [fn(k, n) for k in range(n + 1)]
                 for name, fn in (("wald", tc.wald), ("wilson", tc.wilson), ("cp", tc.clopper_pearson))}
        r = {"n": n}
        for name in cache:
            cov = [coverage(lambda k, _n, c=cache[name]: c[k], n, p) for p in grid]
            r[name + "_mean"] = sum(cov) / len(cov)
            r[name + "_worst"] = min(cov)
        lo, hi = tc.wilson(round(0.85 * n), n)
        r["wilson_halfwidth_at_85"] = (hi - lo) / 2
        rows.append(r)
        print("  %4d | %5.1f%% / %5.1f%%   | %5.1f%% / %5.1f%%     | %5.1f%% / %5.1f%%              | +/- %.1f points"
              % (n, 100 * r["wald_mean"], 100 * r["wald_worst"], 100 * r["wilson_mean"],
                 100 * r["wilson_worst"], 100 * r["cp_mean"], 100 * r["cp_worst"],
                 100 * r["wilson_halfwidth_at_85"]))
    return rows


def part_b():
    print("\nB. Comparing a policy with true rate 80% against one with true rate 90%")
    rows = []
    rng = random.Random(0)
    need = tc.trials_to_compare(0.80, 0.90)
    print("   planned trials per policy for 80%% power: %d" % need)
    print("   trials each | worse policy scores >= better | Fisher p < 0.05")
    for n in (20, 40, 100, need, 400):
        if n <= 100:
            pa = [math.exp(tc._logpmf(k, n, 0.80)) for k in range(n + 1)]
            pb = [math.exp(tc._logpmf(k, n, 0.90)) for k in range(n + 1)]
            wrong = sig = 0.0
            for a in range(n + 1):
                for b in range(n + 1):
                    w = pa[a] * pb[b]
                    if w < 1e-12:
                        continue
                    if a >= b:
                        wrong += w
                    if tc.fisher_exact(a, n, b, n) < 0.05:
                        sig += w
            how = "exact"
        else:
            runs = 3000
            wrong = sig = 0
            for _ in range(runs):
                a = sum(rng.random() < 0.80 for _ in range(n))
                b = sum(rng.random() < 0.90 for _ in range(n))
                wrong += a >= b
                sig += tc.fisher_exact(a, n, b, n) < 0.05
            wrong /= runs
            sig /= runs
            how = "%d simulated pairs" % runs
        rows.append({"n": n, "worse_ge_better": wrong, "fisher_sig": sig, "method": how})
        print("   %11d | %5.1f%%                        | %5.1f%%   (%s)" % (n, 100 * wrong, 100 * sig, how))
    return need, rows


def part_c():
    print("\nC. Certifying one policy: is it at most 80% or at least 90%? (5% error allowed each way)")
    n, c = tc.fixed_plan(0.80, 0.90)
    print("   fixed plan: run %d trials, pass at %d or more successes" % (n, c))
    print("   (fixed plan error: passes an 80%% policy %.1f%% of the time, fails a 90%% policy %.1f%%)"
          % (100 * tc.binom_sf(c, n, 0.80), 100 * (1 - tc.binom_sf(c, n, 0.90))))
    rates = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
    rows = []
    print("   uncapped sequential test, exact (sum over every path, no simulation):")
    print("   true rate | mean trials | 95th pct | accepts | vs fixed")
    for p in rates:
        r = tc.sprt_exact(tc.SPRT(0.80, 0.90), p)
        r["p"] = p
        rows.append(r)
        print("   %8.0f%% | %11.1f | %8d | %6.1f%% | %.0f%% fewer"
              % (100 * p, r["mean_trials"], r["p95_trials"], 100 * r["accept"],
                 100 * (1 - r["mean_trials"] / n)))

    print("\n   capping the test (worst case trials is what a cell schedule needs):")
    print("   design                      | cap | false pass at 80% | false fail at 90% | mean trials at 80 / 85 / 90 / 95%")
    designs = []

    def show(name, test):
        oc = {p: tc.sprt_exact(test, p) for p in (0.80, 0.85, 0.90, 0.95)}
        d = {"design": name, "cap": test.max_trials, "pass_mark": test.pass_mark,
             "false_pass": oc[0.80]["accept"], "false_fail": 1 - oc[0.90]["accept"],
             "mean_trials": {str(p): oc[p]["mean_trials"] for p in oc}}
        designs.append(d)
        print("   %-27s | %3d | %16.2f%% | %16.2f%% | %.1f / %.1f / %.1f / %.1f"
              % (name, test.max_trials, 100 * d["false_pass"], 100 * d["false_fail"],
                 *[oc[p]["mean_trials"] for p in (0.80, 0.85, 0.90, 0.95)]))

    show("naive: stop at %d, take lean" % n, tc.SPRT(0.80, 0.90, max_trials=n))
    for cap in (n, math.ceil(1.1 * n), math.ceil(1.5 * n)):
        pl = tc.capped_plan(0.80, 0.90, max_trials=cap)
        show("calibrated, cap %d" % cap,
             tc.SPRT(0.80, 0.90, pl["alpha_inner"], pl["beta_inner"], pl["max_trials"], pl["pass_mark"]))
        designs[-1]["alpha_inner"] = pl["alpha_inner"]
    return {"fixed_n": n, "fixed_c": c, "sprt_exact": rows, "capped": designs}


def _sessions(rng, m, sizes, p, sd):
    out = []
    for i in range(m):
        q = simcell._expit(simcell._logit(p) + rng.gauss(0.0, sd)) if sd > 0 else p
        n = sizes[i % len(sizes)]
        out.append((sum(rng.random() < q for _ in range(n)), n))
    return out


def _marginal_rate(p, sd):
    """The long-run rate over many sessions, which is what the interval should cover."""
    if sd == 0:
        return p
    rng = random.Random(12345)
    return sum(simcell._expit(simcell._logit(p) + rng.gauss(0.0, sd)) for _ in range(200000)) / 200000


METHODS = (("wilson", lambda c: tc.wilson(sum(k for k, _ in c), sum(n for _, n in c))),
           ("deff", robust.deff_interval),
           ("cluster_t", robust.cluster_interval))


def _coverage(seed, m, sizes, p, sd, runs=1500):
    rng = random.Random(seed)
    truth = _marginal_rate(p, sd)
    hit = {name: 0 for name, _ in METHODS}
    width = {name: 0.0 for name, _ in METHODS}
    for _ in range(runs):
        c = _sessions(rng, m, sizes, p, sd)
        for name, fn in METHODS:
            lo, hi = fn(c)
            hit[name] += lo <= truth <= hi
            width[name] += hi - lo
    return {"sessions": m, "sizes": sizes, "rate": p, "session_sd": sd, "runs": runs,
            "coverage": {k: v / runs for k, v in hit.items()},
            "mean_width": {k: v / runs for k, v in width.items()}}


def part_d():
    print("\nD. Trials grouped in sessions: real coverage of a nominal 95% interval (synthetic)")
    print("   session_sd is the spread of a session's yield on the logit scale; 0 means independent trials")
    out = {}
    for split, seed, configs in (
            ("dev", 11, [(6, [60], 0.90, 0.0), (6, [60], 0.90, 0.6), (3, [300], 0.95, 0.6)]),
            ("held-out", 2027, [(4, [40, 80, 120], 0.95, 0.0), (4, [40, 80, 120], 0.95, 0.9),
                                (10, [30], 0.85, 0.4), (3, [700], 0.97, 0.5), (2, [75], 0.97, 0.6)])):
        print("   %s (seed %d)" % (split, seed))
        print("   sessions x trials    | rate | session_sd | wilson cover / width | deff-wilson       | session-level t")
        rows = []
        for m, sizes, p, sd in configs:
            r = _coverage(seed, m, sizes, p, sd)
            rows.append(r)
            print("   %2d x %-15s | %3.0f%% | %10.1f | %s"
                  % (m, "/".join(map(str, sizes)), 100 * p, sd,
                     " | ".join("%5.1f%% / %4.1f pts   " % (100 * r["coverage"][k], 100 * r["mean_width"][k])
                                for k, _ in METHODS)))
        out[split] = rows
    return out


def _drift_run(rng, sessions, per, base, sd, drift):
    xs, groups = [], []
    for s in range(sessions):
        shift = rng.gauss(0.0, sd) if sd > 0 else 0.0
        for i in range(per):
            p = simcell._expit(simcell._logit(base) + shift)
            if drift is not None:
                frac = i / per
                if s == drift["session"] and frac >= drift["at"]:
                    p -= drift["drop"] * ((frac - drift["at"]) / (1 - drift["at"])
                                          if drift["shape"] == "ramp" else 1.0)
                elif s > drift["session"]:
                    p -= drift["drop"]
            xs.append(rng.random() < p)
            groups.append(s)
    return xs, groups


def _flag_rate(seed, layout, base, sd, drift, grouped, min_seg, runs, n_perm=199):
    rng = random.Random(seed)
    f05 = f01 = 0
    for i in range(runs):
        xs, groups = _drift_run(rng, layout[0], layout[1], base, sd, drift)
        p = robust.changepoint(xs, groups if grouped else None, n_perm=n_perm,
                               min_seg=min_seg, seed=i)["p_value"]
        f05 += p < 0.05
        f01 += p <= 0.01
    return f05 / runs, f01 / runs


def part_e(runs=150):
    print("\nE. Drift detection: share of %d synthetic runs flagged (199 shuffles each)" % runs)
    out = {"runs": runs}
    layout, base = (3, 200), 0.95
    step = {"session": 1, "at": 0.5, "drop": 0.10, "shape": "step"}
    print("   dev: 3 sessions x 200 trials at 95%, step drop of 10 points half way through session 2")
    cand = {}
    for ms in (10, 20, 40):
        a = _flag_rate(21, layout, base, 0.0, step, True, ms, runs)
        b = _flag_rate(22, layout, base, 0.6, step, True, ms, runs)
        cand[ms] = {"power_sd0": a[0], "power_sd0.6": b[0]}
        print("   grouped test, min_seg %2d: flagged (p < 0.05) %.1f%% with no session effect, %.1f%% with session_sd 0.6"
              % (ms, 100 * a[0], 100 * b[0]))
    best = max(cand, key=lambda ms: (cand[ms]["power_sd0"] + cand[ms]["power_sd0.6"], ms))
    out["dev_candidates"] = cand
    out["chosen_min_seg"] = best
    print("   chosen on dev: min_seg = %d" % best)

    def table(split, seed, layout, base, sd, drift, label):
        print("   %s: %s" % (split, label))
        print("   case                        | plain test p<0.05 / p<=0.01 | grouped test p<0.05 / p<=0.01")
        rows = []
        for name, s, d in (("no drift, independent", 0.0, None), ("no drift, session effect", sd, None),
                           ("drift, independent", 0.0, drift), ("drift, session effect", sd, drift)):
            pl = _flag_rate(seed, layout, base, s, d, False, best, runs)
            gr = _flag_rate(seed, layout, base, s, d, True, best, runs)
            rows.append({"case": name, "plain": pl, "grouped": gr})
            print("   %-27s | %10.1f%% / %5.1f%%       | %10.1f%% / %5.1f%%"
                  % (name, 100 * pl[0], 100 * pl[1], 100 * gr[0], 100 * gr[1]))
        return rows

    out["dev"] = table("dev", 23, layout, base, 0.6, step, "same settings, session_sd 0.6 where stated")
    print("   grouped test against session length (held-out seed 4052, same step, session_sd 0.6, 100 runs):")
    out["by_length"] = []
    for per in (200, 400, 800):
        r = _flag_rate(4052, (3, per), base, 0.6, step, True, best, 100)
        out["by_length"].append({"per_session": per, "flag_p05": r[0], "flag_p01": r[1]})
        print("   3 sessions x %3d trials: flagged %.0f%% at p < 0.05, %.0f%% at p <= 0.01"
              % (per, 100 * r[0], 100 * r[1]))
    out["held_out"] = table(
        "held-out", 4051, (4, 150), 0.92, 0.4,
        {"session": 2, "at": 0.4, "drop": 0.20, "shape": "ramp"},
        "4 sessions x 150 trials at 92%, yield ramps down 20 points over the last 60% of session 3, session_sd 0.4")
    return out


if __name__ == "__main__":
    t = time.time()
    a = part_a()
    need, b = part_b()
    c = part_c()
    d = part_d()
    e = part_e()
    with open("bench_results.json", "w") as f:
        json.dump({"coverage": a, "compare_need": need, "compare": b, "certify": c,
                   "sessions": d, "drift": e}, f, indent=1)
    print("\nwrote bench_results.json in %.1fs" % (time.time() - t))
