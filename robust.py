"""robust: what to do when evaluation trials are not independent coin flips.

trialcount.py assumes every trial succeeds with the same probability, independently.
On a real cell two things break that:

  1. Sessions. Trials run on the same day share a calibration, a fixture setup and
     lighting, so they succeed or fail together more than chance. 600 trials over 6
     sessions carry less information than 600 independent trials, and a binomial
     interval on them is too narrow.
  2. Drift. The success rate changes part way through a run (a gripper pad wears,
     a fixture shifts), so a single pooled rate describes nothing.

This module has an interval that accounts for sessions and a test for drift.
Standard library only.
"""
import math
import random

import trialcount as tc


def _totals(clusters):
    clusters = [(k, n) for k, n in clusters if n > 0]
    for k, n in clusters:
        if not 0 <= k <= n:
            raise ValueError("each cluster needs 0 <= successes <= trials")
    if not clusters:
        raise ValueError("no trials")
    return clusters, sum(k for k, _ in clusters), sum(n for _, n in clusters)


def design_effect(clusters):
    """How much the sessions inflate the variance of the pooled rate (1.0 = not at all).

    clusters is a list of (successes, trials), one pair per session. Uses the one-way
    ANOVA estimate of the intra-cluster correlation; negative estimates are clipped to 0.
    """
    clusters, K, N = _totals(clusters)
    m = len(clusters)
    p = K / N
    if m < 2 or N <= m or p in (0.0, 1.0):
        return 1.0
    msb = sum(n * (k / n - p) ** 2 for k, n in clusters) / (m - 1)
    msw = sum(k * (1 - k / n) for k, n in clusters) / (N - m)
    n0 = (N - sum(n * n for _, n in clusters) / N) / (m - 1)
    denom = msb + (n0 - 1) * msw
    icc = 0.0 if denom <= 0 else max(0.0, min(1.0, (msb - msw) / denom))
    size = sum(n * n for _, n in clusters) / N
    return 1.0 + (size - 1) * icc


def deff_interval(clusters, conf=0.95):
    """Wilson interval on the effective sample size N / design_effect."""
    clusters, K, N = _totals(clusters)
    n_eff = N / design_effect(clusters)
    return tc.wilson(K / N * n_eff, n_eff, conf)


def _ratio_se(clusters, K, N):
    m = len(clusters)
    p = K / N
    return math.sqrt(m / (m - 1) * sum((k - p * n) ** 2 for k, n in clusters)) / N


def cluster_interval(clusters, conf=0.95):
    """Two-sided interval for the success rate that treats sessions as the unit.

    The pooled rate gets a standard error from the spread between sessions and a
    Student t quantile with (sessions - 1) degrees of freedom. With few sessions that
    quantile is large, which is the point. The result is never narrower than the plain
    Wilson interval, so sessions that happen to agree cannot shrink it, and with one
    session it is the Wilson interval.
    """
    clusters, K, N = _totals(clusters)
    lo, hi = tc.wilson(K, N, conf)
    if len(clusters) >= 2:
        h = tc.t_ppf(1 - (1 - conf) / 2, len(clusters) - 1) * _ratio_se(clusters, K, N)
        lo, hi = min(lo, K / N - h), max(hi, K / N + h)
    return max(0.0, lo), min(1.0, hi)


def cluster_bounds(clusters, conf=0.95):
    """One-sided (lower, upper) bounds, each at confidence conf, session aware.

    Each is the more cautious of the exact binomial bound and the session-level t bound.
    """
    clusters, K, N = _totals(clusters)
    lo, hi = tc.lower_bound(K, N, conf), tc.upper_bound(K, N, conf)
    if len(clusters) >= 2:
        h = tc.t_ppf(conf, len(clusters) - 1) * _ratio_se(clusters, K, N)
        lo, hi = min(lo, K / N - h), max(hi, K / N + h)
    return max(0.0, lo), min(1.0, hi)


def _scan(xs, base, min_seg):
    """Largest standardized gap between a running success count and its expected value.

    base[t] is the expected number of successes in the first t trials.
    Returns (statistic, split index); the statistic is 0 if the run is too short.
    """
    n = len(xs)
    c = 0
    best = 0.0
    at = None
    for t in range(1, n - min_seg + 1):
        c += xs[t - 1]
        if t >= min_seg:
            d = c - base[t]
            d = d * d / (t * (n - t))
            if d > best:
                best, at = d, t
    return best, at


def changepoint(outcomes, groups=None, strata=None, n_perm=499, min_seg=40, seed=0):
    """Permutation test for a change in success rate part way through a run.

    outcomes is the success sequence in time order. The statistic is the largest
    standardized gap between the running success count and what a constant rate would
    give, over every split point at least min_seg trials from either end. Its null
    distribution comes from shuffling the outcomes, so the p-value needs no
    distributional assumption beyond exchangeability.

    groups (same length, for example the session of each trial) changes the question.
    Without it, any difference between sessions counts as a change. With it, each
    session is scanned against its own rate, the statistic is the largest gap found in
    any session, and outcomes are shuffled only inside their session. The test then
    looks for a change within a session, and a session that is simply better or worse
    than another does not trigger it. The cost is that a change that starts exactly at a
    session boundary is invisible to the grouped test; cluster_interval is what
    accounts for that kind of variation.

    strata (same length, for example the part number of each trial) lets trials with
    different expected rates share one scan: each trial is compared with the rate of its
    own (group, stratum) cell and shuffled only inside that cell, so switching from an
    easy part to a hard part is not reported as drift.

    Returns a dict with index (position in outcomes of the first trial after the
    change), the rate before and after it within that group, the statistic and p_value.
    p_value is 1.0 when no test is possible.
    """
    xs = [1 if x else 0 for x in outcomes]
    n = len(xs)
    groups = [0] * n if groups is None else list(groups)
    strata = [0] * n if strata is None else list(strata)
    if len(groups) != n or len(strata) != n:
        raise ValueError("groups and strata must be as long as outcomes")
    if min_seg < 1 or n_perm < 1:
        raise ValueError("min_seg and n_perm must be at least 1")
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    # a group can show a change only if it is long enough to split and has both outcomes
    runs = []
    for idx in members.values():
        seq = [xs[i] for i in idx]
        if len(seq) < 2 * min_seg or sum(seq) in (0, len(seq)):
            continue
        cells = {}
        for pos, i in enumerate(idx):
            cells.setdefault(strata[i], []).append(pos)
        base = [0.0] * (len(seq) + 1)
        rate = {c: sum(seq[p] for p in ps) / len(ps) for c, ps in cells.items()}
        for pos, i in enumerate(idx):
            base[pos + 1] = base[pos] + rate[strata[i]]
        shuffle = [(ps, sum(1 - seq[p] for p in ps)) for ps in cells.values()]
        runs.append((idx, seq, base, shuffle))
    best = (0.0, None, None)
    for r, (idx, seq, base, _) in enumerate(runs):
        stat, at = _scan(seq, base, min_seg)
        if stat > best[0]:
            best = (stat, r, at)
    stat, r, at = best
    if r is None or stat <= 1e-12:
        return {"index": None, "before": None, "after": None, "stat": 0.0, "p_value": 1.0}
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        top = 0.0
        for idx, seq, base, shuffle in runs:
            perm = [1] * len(seq)
            for ps, fails in shuffle:
                for p in rng.sample(ps, fails):
                    perm[p] = 0
            v = _scan(perm, base, min_seg)[0]
            if v > top:
                top = v
        if top >= stat - 1e-12:
            ge += 1
    idx, seq = runs[r][0], runs[r][1]
    return {"index": idx[at], "before": sum(seq[:at]) / at, "after": sum(seq[at:]) / (len(seq) - at),
            "stat": stat, "p_value": (1 + ge) / (n_perm + 1)}
