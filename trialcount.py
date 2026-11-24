"""trialcount: how many evaluation trials a success rate needs before you can trust it.

Standard library only. This module is the statistics core:
  1. Confidence intervals for a success rate (Wald, Wilson, Clopper-Pearson, one-sided bounds).
  2. Planning: trials needed to tell two rates apart, or to certify one rate.
  3. A sequential test (SPRT) that stops an evaluation as soon as the answer is clear,
     its exact operating characteristics, and a capped version with calibrated boundaries.
  4. Paired comparison (exact McNemar) and a Student t quantile for the other modules.

robust.py handles trials that are not independent (sessions, drift) and cellrun.py turns
an episode log plus an acceptance spec into a verdict.
"""
import json
import math
import sys
from statistics import NormalDist

_N = NormalDist()


def _z(conf):
    return _N.inv_cdf(1 - (1 - conf) / 2)


def _logpmf(k, n, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.exp(_logpmf(i, n, p)) for i in range(k + 1)))


def binom_sf(k, n, p):
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(math.exp(_logpmf(i, n, p)) for i in range(k, n + 1)))


def wald(k, n, conf=0.95):
    """The textbook p +/- z*sqrt(p(1-p)/n) interval. Included to show why not to use it."""
    p = k / n
    h = _z(conf) * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)


def wilson(k, n, conf=0.95):
    z = _z(conf)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _bisect(f, lo, hi, iters=50):
    # f is decreasing in p on [lo, hi]; find the root
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, conf=0.95):
    """Exact interval. Never under-covers, at the cost of being wider."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else _bisect(lambda p: a - binom_sf(k, n, p), 0.0, 1.0)
    hi = 1.0 if k == n else _bisect(lambda p: binom_cdf(k, n, p) - a, 0.0, 1.0)
    return lo, hi


def lower_bound(k, n, conf=0.95):
    """Exact one-sided lower bound: with confidence conf the true rate is at least this."""
    if k <= 0:
        return 0.0
    return _bisect(lambda p: (1 - conf) - binom_sf(k, n, p), 0.0, 1.0)


def upper_bound(k, n, conf=0.95):
    """Exact one-sided upper bound on the true rate."""
    if k >= n:
        return 1.0
    return _bisect(lambda p: binom_cdf(k, n, p) - (1 - conf), 0.0, 1.0)


def zero_failure_trials(target, conf=0.95):
    """Trials needed, with no failures at all, to show the rate is at least target.

    This is the exact form of the "3 / n" rule: zero_failure_trials(0.99) is 299.
    """
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1")
    return math.ceil(math.log(1 - conf) / math.log(target))


def mcnemar_exact(only_a, only_b):
    """Two-sided exact McNemar p-value for paired trials.

    only_a is the number of paired trials (same part, same start pose) where policy A
    succeeded and B failed, only_b the reverse. Pairs where both agree carry no information.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_cdf(min(only_a, only_b), n, 0.5))


def _betacf(a, b, x):
    # continued fraction for the incomplete beta function (modified Lentz)
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q, df):
    """Quantile of Student's t with df degrees of freedom (bisection on t_cdf)."""
    if not 0 < q < 1 or df <= 0:
        raise ValueError("need 0 < q < 1 and df > 0")
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > q:
        lo *= 2
    while t_cdf(hi, df) < q:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fisher_exact(k1, n1, k2, n2):
    """Two-sided Fisher exact p-value for k1/n1 vs k2/n2."""
    K = k1 + k2
    N = n1 + n2

    def lp(x):
        return (math.lgamma(n1 + 1) - math.lgamma(x + 1) - math.lgamma(n1 - x + 1)
                + math.lgamma(n2 + 1) - math.lgamma(K - x + 1) - math.lgamma(n2 - K + x + 1)
                - (math.lgamma(N + 1) - math.lgamma(K + 1) - math.lgamma(N - K + 1)))

    obs = lp(k1)
    total = 0.0
    for x in range(max(0, K - n2), min(n1, K) + 1):
        v = lp(x)
        if v <= obs + 1e-9:
            total += math.exp(v)
    return min(1.0, total)


def trials_to_compare(p_a, p_b, alpha=0.05, power=0.8):
    """Trials PER POLICY to detect p_a vs p_b with a two-sided test (normal approximation)."""
    za = _N.inv_cdf(1 - alpha / 2)
    zb = _N.inv_cdf(power)
    pbar = (p_a + p_b) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar))
           + zb * math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))) ** 2
    return math.ceil(num / (p_a - p_b) ** 2)


def trials_for_width(p, half_width, conf=0.95):
    """Smallest n whose Wilson interval at rate p is within +/- half_width."""
    n = 1
    while True:
        lo, hi = wilson(round(p * n), n, conf)
        if (hi - lo) / 2 <= half_width:
            return n
        n += max(1, n // 50)


def fixed_plan(p0, p1, alpha=0.05, beta=0.05, n_max=5000):
    """Smallest fixed n and pass mark c: P(pass | p0) <= alpha and P(pass | p1) >= 1 - beta.

    Returns (n, c). The policy passes if it succeeds at least c times out of n.
    """
    for n in range(1, n_max + 1):
        # smallest c with P(X >= c | p0) <= alpha, found by walking the upper tail
        tail = 0.0
        c = n + 1
        for k in range(n, -1, -1):
            t = tail + math.exp(_logpmf(k, n, p0))
            if t > alpha:
                break
            tail = t
            c = k
        if c <= n and binom_sf(c, n, p1) >= 1 - beta:
            return n, c
    raise ValueError("no plan within n_max")


class SPRT:
    """Wald's sequential probability ratio test for H0: p <= p0 against H1: p >= p1.

    Feed it one trial at a time. `decision` becomes "accept" (policy is at least p1),
    "reject" (policy is at most p0), or stays None while the evidence is unclear.
    If max_trials is reached without a decision the test is forced: it accepts when the
    success count is at least pass_mark, or, with no pass_mark, when the log ratio is
    positive. A forced stop changes the error rates; capped_plan() picks alpha, beta,
    max_trials and pass_mark so the exact error rates still meet the target.
    """

    def __init__(self, p0, p1, alpha=0.05, beta=0.05, max_trials=None, pass_mark=None):
        if not 0 < p0 < p1 < 1:
            raise ValueError("need 0 < p0 < p1 < 1")
        if not (0 < alpha < 1 and 0 < beta < 1 and alpha + beta < 1):
            raise ValueError("need alpha and beta in (0, 1) with alpha + beta < 1")
        if max_trials is not None and max_trials < 1:
            raise ValueError("max_trials must be at least 1")
        if pass_mark is not None and max_trials is None:
            raise ValueError("pass_mark only applies with max_trials")
        self.p0, self.p1 = p0, p1
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))
        self._win = math.log(p1 / p0)
        self._loss = math.log((1 - p1) / (1 - p0))
        self.max_trials = max_trials
        self.pass_mark = pass_mark
        self.llr = 0.0
        self.n = 0
        self.k = 0
        self.decision = None
        self.truncated = False

    def update(self, success):
        if self.decision is not None:
            return self.decision
        self.n += 1
        self.k += 1 if success else 0
        self.llr = self.k * self._win + (self.n - self.k) * self._loss
        if self.llr >= self.upper:
            self.decision = "accept"
        elif self.llr <= self.lower:
            self.decision = "reject"
        elif self.max_trials is not None and self.n >= self.max_trials:
            if self.pass_mark is not None:
                self.decision = "accept" if self.k >= self.pass_mark else "reject"
            else:
                self.decision = "accept" if self.llr > 0 else "reject"
            self.truncated = True
        return self.decision


def _sprt_dp(test, p, horizon):
    """Walk every undecided (n, k) state of `test` forward at true rate p.

    Returns (P(accept before a forced stop), P(reject before a forced stop),
    {k: P(undecided with k successes at max_trials)}, [P(stop at n)] indexed by n).
    Only a narrow band of k is ever undecided, so this is cheap.
    """
    cap = test.max_trials
    live = {0: 1.0}
    acc = rej = 0.0
    stop = [0.0]
    terminal = {}
    n = 0
    while live:
        n += 1
        if n > horizon:
            raise ValueError("test still undecided after %d trials" % horizon)
        nxt = {}
        for k, pr in live.items():
            for kk, w in ((k + 1, pr * p), (k, pr * (1 - p))):
                if w:
                    nxt[kk] = nxt.get(kk, 0.0) + w
        live = {}
        stopped = 0.0
        for k, pr in nxt.items():
            llr = k * test._win + (n - k) * test._loss
            if llr >= test.upper:
                acc += pr
                stopped += pr
            elif llr <= test.lower:
                rej += pr
                stopped += pr
            elif cap is not None and n >= cap:
                terminal[k] = pr
                stopped += pr
            else:
                live[k] = pr
        stop.append(stopped)
        if cap is None and sum(live.values()) < 1e-13:
            break
    return acc, rej, terminal, stop


def sprt_exact(test, p, horizon=200000):
    """Exact operating characteristics of an SPRT (capped or not) at true rate p.

    `test` is an SPRT instance used only for its settings. No simulation: the result
    is a sum over every path. Returns accept probability, mean and 95th percentile of
    the number of trials, the longest possible run, and the chance of a forced stop.
    """
    acc, rej, terminal, stop = _sprt_dp(test, p, horizon)
    forced = sum(terminal.values())
    for k, pr in terminal.items():
        if test.pass_mark is not None:
            ok = k >= test.pass_mark
        else:
            ok = k * test._win + (test.max_trials - k) * test._loss > 0
        acc += pr if ok else 0.0
    mean = sum(n * pr for n, pr in enumerate(stop))
    run = 0.0
    p95 = len(stop) - 1
    for n, pr in enumerate(stop):
        run += pr
        if run >= 0.95:
            p95 = n
            break
    return {"accept": acc, "mean_trials": mean, "p95_trials": p95,
            "max_trials": test.max_trials, "forced": forced}


def capped_plan(p0, p1, alpha=0.05, beta=0.05, max_trials=None, caps=(1.0, 1.1, 1.25, 1.5),
                shrink=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)):
    """A sequential test with a hard trial cap whose exact error rates meet alpha and beta.

    Stopping an SPRT at a cap and taking the leaning side breaks its error bound. This
    searches caps (max_trials if given, otherwise multiples of the fixed plan's n, since
    no test capped below that can meet both bounds) and tighter inner thresholds
    (alpha and beta scaled by `shrink`), and for each computes the exact error rates for
    every final pass mark. Among settings that meet both bounds it returns the one with
    the fewest mean trials averaged over p0 and p1.

    Returns a dict; build the test with
        SPRT(p0, p1, plan["alpha_inner"], plan["beta_inner"], plan["max_trials"], plan["pass_mark"])
    """
    n_fixed, _ = fixed_plan(p0, p1, alpha, beta)
    best = None
    cap_list = [max_trials] if max_trials is not None else [math.ceil(n_fixed * m) for m in caps]
    for cap in cap_list:
        for g in shrink:
            t = SPRT(p0, p1, alpha * g, beta * g, max_trials=cap)
            a0, _, term0, stop0 = _sprt_dp(t, p0, cap + 1)
            _, r1, term1, stop1 = _sprt_dp(t, p1, cap + 1)
            cost = (sum(n * w for n, w in enumerate(stop0))
                    + sum(n * w for n, w in enumerate(stop1))) / 2
            if best is not None and cost >= best["mean_trials_avg"]:
                continue
            ks = sorted(set(term0) | set(term1))
            pick = None
            for c in ks + [ks[-1] + 1] if ks else [0]:
                fp = a0 + sum(w for k, w in term0.items() if k >= c)
                ff = r1 + sum(w for k, w in term1.items() if k < c)
                if fp <= alpha and ff <= beta and (pick is None or max(fp, ff) < max(pick[1:])):
                    pick = (c, fp, ff)
            if pick is not None:
                best = {"p0": p0, "p1": p1, "alpha_inner": alpha * g, "beta_inner": beta * g,
                        "max_trials": cap, "pass_mark": pick[0], "false_pass": pick[1],
                        "false_fail": pick[2], "mean_trials_avg": cost, "fixed_n": n_fixed}
    if best is None:
        raise ValueError("no capped plan found; widen caps or shrink")
    return best


def load_episodes(path):
    """Read a JSONL episode log. Each line needs "policy" and "success" keys."""
    out = {}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError("line %d: not valid JSON" % i) from None
            if not isinstance(row, dict) or "policy" not in row or "success" not in row:
                raise ValueError("line %d: need 'policy' and 'success'" % i)
            if not isinstance(row["success"], bool):
                # bool("false") is True, so anything but a JSON boolean is refused
                raise ValueError("line %d: 'success' must be true or false, got %r"
                                 % (i, row["success"]))
            out.setdefault(row["policy"], []).append(row["success"])
    return out


def report(episodes, conf=0.95):
    lines = []
    names = sorted(episodes)
    for name in names:
        r = episodes[name]
        k, n = sum(r), len(r)
        lo, hi = wilson(k, n, conf)
        lines.append("%-12s %3d/%-3d = %5.1f%%   %d%% interval %.1f%% to %.1f%%"
                     % (name, k, n, 100 * k / n, round(conf * 100), 100 * lo, 100 * hi))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = episodes[names[i]], episodes[names[j]]
            p = fisher_exact(sum(a), len(a), sum(b), len(b))
            pa, pb = sum(a) / len(a), sum(b) / len(b)
            if p < 1 - conf:
                verdict = "difference is unlikely to be noise"
            elif pa == pb:
                verdict = "no difference observed"
            else:
                need = trials_to_compare(pa, pb)
                verdict = ("cannot tell them apart yet; if these rates are real, "
                           "about %d trials per policy would be needed" % need)
            lines.append("%s vs %s: Fisher p = %.3f, %s" % (names[i], names[j], p, verdict))
    return "\n".join(lines)


def main(argv):
    if len(argv) == 3 and argv[1] == "report":
        print(report(load_episodes(argv[2])))
    elif len(argv) == 4 and argv[1] == "compare":
        print(trials_to_compare(float(argv[2]), float(argv[3])), "trials per policy")
    elif len(argv) == 4 and argv[1] == "interval":
        k, n = int(argv[2]), int(argv[3])
        for name, fn in (("wald", wald), ("wilson", wilson), ("clopper-pearson", clopper_pearson)):
            lo, hi = fn(k, n)
            print("%-16s %.3f to %.3f" % (name, lo, hi))
    else:
        print("usage: trialcount.py report episodes.jsonl\n"
              "       trialcount.py interval <successes> <trials>\n"
              "       trialcount.py compare <rate_a> <rate_b>")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
