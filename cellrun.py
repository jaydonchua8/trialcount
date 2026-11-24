"""cellrun: decide whether a cell has met its acceptance numbers, from an episode log.

A spec states "done" as numbers, agreed before the run:

    {
      "parts": ["PN-1041", "PN-2210"],   part numbers in scope
      "first_pass_yield": 0.95,          minimum share of parts made right with no human help
      "cycle_time_s": 12.0,              maximum mean seconds per part
      "continuous_run_s": 14400,         the cell must run this long with no intervention
      "changeover_s": 600,               maximum seconds to switch between part numbers
      "confidence": 0.95                 optional, default 0.95
    }

The log is JSON Lines, one episode per line:

    {"episode": 17, "session": "S01", "t_start": 201.4, "part": "PN-1041",
     "policy": "act_v3", "success": true, "intervention": false, "cycle_s": 10.9}

episode is a unique integer, session names the shift or day, t_start is seconds on any
clock that is shared by the whole log, cycle_s is the episode duration. policy and
intervention are optional. Extra keys are ignored.

Each criterion gets one of four verdicts:
  PASS       the data shows the number is met at the stated confidence
  FAIL       the data shows it is not met
  NOT YET    the data cannot say either way; the report says how much more is needed
  UNSTABLE   the yield changed during a session, so no single yield number describes it
The overall verdict is FAIL if any criterion fails, PASS only if every one passes, and
NOT YET otherwise.

    python cellrun.py acceptance_spec.json cell_log.jsonl
"""
import json
import math
import sys
from collections import namedtuple

import robust
import trialcount as tc

Episode = namedtuple("Episode", "episode session t_start part policy success intervention cycle_s")
Check = namedtuple("Check", "name scope verdict detail")

PASS, FAIL, NOT_YET, UNSTABLE = "PASS", "FAIL", "NOT YET", "UNSTABLE"
DRIFT_ALPHA = 0.01
IDLE_GAP_S = 120.0


class LogError(ValueError):
    pass


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_episode(row, where):
    if not isinstance(row, dict):
        raise LogError("%s: expected a JSON object" % where)
    for key in ("episode", "session", "t_start", "part", "success", "cycle_s"):
        if key not in row:
            raise LogError("%s: missing '%s'" % (where, key))
    if not isinstance(row["episode"], int) or isinstance(row["episode"], bool):
        raise LogError("%s: 'episode' must be an integer" % where)
    for key in ("success", "intervention"):
        if not isinstance(row.get(key, False), bool):
            raise LogError("%s: '%s' must be true or false, got %r" % (where, key, row[key]))
    for key in ("session", "part"):
        if not isinstance(row[key], str) or not row[key]:
            raise LogError("%s: '%s' must be a non-empty string" % (where, key))
    if not _number(row["t_start"]):
        raise LogError("%s: 't_start' must be a number" % where)
    if not _number(row["cycle_s"]) or row["cycle_s"] <= 0:
        raise LogError("%s: 'cycle_s' must be a positive number" % where)
    policy = row.get("policy", "")
    if not isinstance(policy, str):
        raise LogError("%s: 'policy' must be a string" % where)
    return Episode(row["episode"], row["session"], float(row["t_start"]), row["part"], policy,
                   row["success"], row.get("intervention", False), float(row["cycle_s"]))


def check_log(episodes):
    """Refuse logs whose order or identity is ambiguous, then return them in time order."""
    seen = set()
    for e in episodes:
        if e.episode in seen:
            raise LogError("episode %d appears more than once" % e.episode)
        seen.add(e.episode)
    out = sorted(episodes, key=lambda e: (e.t_start, e.episode))
    last = {}
    for e in out:
        prev = last.get(e.session)
        if prev is not None and e.t_start < prev.t_start + prev.cycle_s - 1e-6:
            raise LogError("episodes %d and %d in session %s overlap in time"
                           % (prev.episode, e.episode, e.session))
        last[e.session] = e
    return out


def load_log(path):
    episodes = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise LogError("line %d: not valid JSON" % i) from None
            episodes.append(parse_episode(row, "line %d" % i))
    if not episodes:
        raise LogError("the log has no episodes")
    return check_log(episodes)


def load_spec(src):
    """src is a path or an already parsed dict. Returns a validated spec dict."""
    if isinstance(src, dict):
        spec = dict(src)
    else:
        with open(src) as f:
            spec = json.load(f)
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON object")
    parts = spec.get("parts")
    if not isinstance(parts, list) or not parts or not all(isinstance(p, str) and p for p in parts):
        raise ValueError("spec 'parts' must be a non-empty list of part numbers")
    if len(set(parts)) != len(parts):
        raise ValueError("spec 'parts' lists a part number twice")
    y = spec.get("first_pass_yield")
    if not _number(y) or not 0 < y < 1:
        raise ValueError("spec 'first_pass_yield' must be between 0 and 1")
    for key in ("cycle_time_s", "continuous_run_s", "changeover_s"):
        if not _number(spec.get(key)) or spec[key] <= 0:
            raise ValueError("spec '%s' must be a positive number" % key)
    conf = spec.setdefault("confidence", 0.95)
    if not _number(conf) or not 0.5 <= conf < 1:
        raise ValueError("spec 'confidence' must be at least 0.5 and below 1")
    return spec


def _by_session(eps):
    out = {}
    for e in eps:
        out.setdefault(e.session, []).append(e)
    return out


def trials_needed(rate, target, conf):
    """Smallest n whose exact lower bound reaches target if the observed rate holds."""
    if rate <= target:
        return None
    ok = lambda n: tc.lower_bound(math.floor(rate * n + 1e-9), n, conf) >= target
    hi = 8
    while not ok(hi):
        hi *= 2
        if hi > 200000:
            return None
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _first_pass(e):
    return e.success and not e.intervention


def yield_check(part, eps, spec):
    target, conf = spec["first_pass_yield"], spec["confidence"]
    if not eps:
        return Check("first-pass yield", part, NOT_YET, "no episodes for this part number")
    k, n = sum(_first_pass(e) for e in eps), len(eps)
    clusters = [(sum(_first_pass(e) for e in v), len(v)) for v in _by_session(eps).values()]
    lo, hi = robust.cluster_bounds(clusters, conf)
    head = ("%d/%d = %.1f%% over %d session%s, %d%% bounds %.1f%% to %.1f%%, target %.1f%%"
            % (k, n, 100 * k / n, len(clusters), "" if len(clusters) == 1 else "s",
               round(100 * conf), 100 * lo, 100 * hi, 100 * target))
    if lo >= target:
        return Check("first-pass yield", part, PASS, head)
    if hi < target:
        return Check("first-pass yield", part, FAIL, head)
    if tc.lower_bound(k, n, conf) >= target:
        more = ("the trial count would be enough for independent trials, but the spread between "
                "%d sessions leaves the bound below target; more sessions are needed, "
                "not more trials per session" % len(clusters))
    else:
        need = trials_needed(k / n, target, conf)
        if need is None:
            more = "the observed rate is not above the target, or only barely, so more trials alone will not pass it"
        else:
            more = "at this rate about %d more trials would be needed" % max(1, need - n)
    return Check("first-pass yield", part, NOT_YET, head + ". " + more)


def stability_check(eps, n_perm=499, seed=0):
    """Did first-pass yield change inside a session? Part numbers are compared separately."""
    cp = robust.changepoint([_first_pass(e) for e in eps], groups=[e.session for e in eps],
                            strata=[e.part for e in eps], n_perm=n_perm, seed=seed)
    if cp["index"] is None:
        return Check("yield stability", "cell", NOT_YET,
                     "no session is long enough, or has both good and bad parts, to test for drift")
    if cp["p_value"] < DRIFT_ALPHA:
        at = eps[cp["index"]]
        return Check("yield stability", "cell", UNSTABLE,
                     "yield changed inside session %s: %.1f%% before episode %d, %.1f%% after, "
                     "p = %.3f. A pooled yield number mixes the two; find the cause first."
                     % (at.session, 100 * cp["before"], at.episode, 100 * cp["after"], cp["p_value"]))
    return Check("yield stability", "cell", PASS,
                 "no change inside a session detected (smallest p = %.3f, flagged below %.2f)"
                 % (cp["p_value"], DRIFT_ALPHA))


def cycle_check(part, eps, spec):
    limit, conf = spec["cycle_time_s"], spec["confidence"]
    xs = [e.cycle_s for e in eps if not e.intervention]
    if len(xs) < 2:
        return Check("cycle time", part, NOT_YET, "fewer than 2 episodes without intervention")
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    h = tc.t_ppf(conf, n - 1) * sd / math.sqrt(n)
    p95 = sorted(xs)[min(n - 1, math.ceil(0.95 * n) - 1)]
    head = ("mean %.2f s over %d episodes (%d%% bounds %.2f to %.2f s, 95th percentile %.2f s), "
            "limit %.2f s" % (mean, n, round(100 * conf), mean - h, mean + h, p95, limit))
    if mean + h <= limit:
        return Check("cycle time", part, PASS, head)
    if mean - h > limit:
        return Check("cycle time", part, FAIL, head)
    return Check("cycle time", part, NOT_YET, head)


def longest_run(eps, changeover_s, idle_gap_s=IDLE_GAP_S):
    """Longest stretch in seconds with no intervention and no unexplained idle gap.

    A gap between episodes longer than idle_gap_s ends the stretch (the cell was not
    running), unless the part number changed, where up to changeover_s is allowed.
    Returns (seconds, session, first episode id, last episode id).
    """
    best = (0.0, None, None, None)
    for session, es in _by_session(eps).items():
        start = prev = None
        for e in es:
            if e.intervention:
                start = prev = None
                continue
            if prev is not None:
                gap = e.t_start - (prev.t_start + prev.cycle_s)
                if gap > (changeover_s if e.part != prev.part else idle_gap_s):
                    start = None
            if start is None:
                start = e
            prev = e
            length = e.t_start + e.cycle_s - start.t_start
            if length > best[0]:
                best = (length, session, start.episode, e.episode)
    return best


def run_check(eps, spec):
    need = spec["continuous_run_s"]
    length, session, a, b = longest_run(eps, spec["changeover_s"])
    span = max((v[-1].t_start + v[-1].cycle_s - v[0].t_start) for v in _by_session(eps).values())
    head = "longest run without intervention %.2f h" % (length / 3600)
    if session is not None:
        head += " (session %s, episodes %d to %d)" % (session, a, b)
    head += ", required %.2f h" % (need / 3600)
    if length >= need:
        return Check("continuous run", "cell", PASS, head)
    if span < need:
        return Check("continuous run", "cell", NOT_YET,
                     head + ". No session in the log lasts %.2f h, so it could not have been shown"
                     % (need / 3600))
    return Check("continuous run", "cell", FAIL, head)


def changeover_check(eps, spec):
    limit = spec["changeover_s"]
    gaps = []
    for es in _by_session(eps).values():
        for prev, e in zip(es, es[1:]):
            if e.part != prev.part:
                gaps.append(e.t_start - (prev.t_start + prev.cycle_s))
    if not gaps:
        return Check("changeover", "cell", NOT_YET, "the log contains no change of part number")
    head = ("%d changeover%s, longest %.0f s, limit %.0f s"
            % (len(gaps), "" if len(gaps) == 1 else "s", max(gaps), limit))
    return Check("changeover", "cell", PASS if max(gaps) <= limit else FAIL, head)


def evaluate(spec, episodes, n_perm=499, seed=0):
    """Return (overall verdict, [Check, ...]) for episodes against spec."""
    spec = load_spec(spec)
    episodes = check_log(list(episodes))
    checks = []
    in_scope = [e for e in episodes if e.part in spec["parts"]]
    extra = sorted({e.part for e in episodes} - set(spec["parts"]))
    for part in spec["parts"]:
        eps = [e for e in in_scope if e.part == part]
        checks.append(yield_check(part, eps, spec))
        if eps:
            checks.append(cycle_check(part, eps, spec))
    if in_scope:
        checks.append(stability_check(in_scope, n_perm, seed))
        checks.append(run_check(in_scope, spec))
        checks.append(changeover_check(in_scope, spec))
    if extra:
        checks.append(Check("scope", "cell", NOT_YET if not in_scope else PASS,
                            "ignored episodes for part numbers outside the spec: " + ", ".join(extra)))
    verdicts = {c.verdict for c in checks}
    if FAIL in verdicts:
        overall = FAIL
    elif verdicts == {PASS}:
        overall = PASS
    else:
        overall = NOT_YET
    return overall, checks


def format_report(overall, checks):
    lines = ["%-8s  %-16s %-8s  %s" % (c.verdict, c.name, c.scope, c.detail) for c in checks]
    lines.append("overall: " + overall)
    return "\n".join(lines)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--policy=")]
    policy = [a.split("=", 1)[1] for a in argv[1:] if a.startswith("--policy=")]
    if len(args) != 2:
        print("usage: cellrun.py spec.json log.jsonl [--policy=NAME]")
        return 2
    try:
        spec = load_spec(args[0])
        eps = load_log(args[1])
    except (OSError, ValueError) as err:
        print("error: %s" % err)
        return 2
    if policy:
        eps = [e for e in eps if e.policy == policy[0]]
        if not eps:
            print("error: no episodes for policy %s" % policy[0])
            return 2
    names = sorted({e.policy for e in eps})
    if len(names) > 1:
        print("error: the log mixes policies (%s); pick one with --policy=NAME" % ", ".join(names))
        return 2
    overall, checks = evaluate(spec, eps)
    print(format_report(overall, checks))
    return 0 if overall == PASS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
