"""simcell: a synthetic robot cell that writes episode logs. Everything here is made up.

I do not have logs from a real cell, so this generator produces the failure patterns I
want the evaluation code to handle, with the truth known:

  - several part numbers with different true first-pass yields and cycle times,
    run in blocks with a changeover between them
  - sessions (shifts) whose yield moves together, set by session_sd on the logit scale
  - drift: a drop in yield that starts part way through one session, as a step or a
    ramp, and either persists into later sessions or resets
  - failures that sometimes need a person to step in (2% of failures by default; the
    rest are treated as parts dropped in a reject bin), which breaks a continuous run

Every record carries "synthetic": true. The log format is documented in cellrun.py.

    python simcell.py --seed 3 --out cell_log.jsonl
"""
import argparse
import json
import math
import random
import sys

DEFAULT_PARTS = {
    # part number: (true first-pass yield, mean cycle seconds, sd of cycle seconds)
    "PN-1041": (0.975, 10.8, 0.9),
    "PN-2210": (0.965, 11.4, 1.1),
}


def _logit(p):
    return math.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + math.exp(-x))


def generate(seed=0, parts=None, sessions=3, per_session=1400, block=350, session_sd=0.0,
             drift=None, intervene_given_failure=0.02, changeover_s=(240.0, 40.0),
             policy="act_v3"):
    """Return a list of episode records in time order.

    drift is None or a dict: session (index where it starts), at (fraction of that
    session, 0 to 1), drop (yield lost, in probability), shape ("step" or "ramp"; a ramp
    reaches the full drop at the end of the starting session), persist (bool, carry the
    full drop into later sessions).
    """
    parts = dict(DEFAULT_PARTS if parts is None else parts)
    if sessions < 1 or per_session < 1 or block < 1:
        raise ValueError("sessions, per_session and block must be positive")
    for name, (y, mean, sd) in parts.items():
        if not 0 < y < 1 or mean <= 0 or sd < 0:
            raise ValueError("bad settings for part %s" % name)
    rng = random.Random(seed)
    names = sorted(parts)
    out = []
    ep = 0
    for s in range(sessions):
        t = s * 86400.0
        shift = rng.gauss(0.0, session_sd) if session_sd > 0 else 0.0
        prev_part = None
        for i in range(per_session):
            part = names[(i // block) % len(names)]
            y, mean, sd = parts[part]
            if prev_part is not None and part != prev_part:
                t += max(30.0, rng.gauss(*changeover_s))
            prev_part = part
            p = _expit(_logit(y) + shift)
            if drift is not None:
                frac = i / per_session
                if s == drift["session"] and frac >= drift["at"]:
                    if drift.get("shape", "step") == "ramp":
                        p -= drift["drop"] * (frac - drift["at"]) / max(1e-9, 1 - drift["at"])
                    else:
                        p -= drift["drop"]
                elif s > drift["session"] and drift.get("persist", True):
                    p -= drift["drop"]
            p = min(0.999, max(0.001, p))
            success = rng.random() < p
            intervention = (not success) and rng.random() < intervene_given_failure
            cycle = max(0.5 * mean, rng.gauss(mean, sd))
            if not success:
                cycle *= 1.0 + 0.5 * rng.random()
            out.append({"episode": ep, "session": "S%02d" % (s + 1), "t_start": round(t, 2),
                        "part": part, "policy": policy, "success": success,
                        "intervention": intervention, "cycle_s": round(cycle, 2),
                        "synthetic": True})
            ep += 1
            t += cycle + rng.uniform(1.0, 3.0)
            if intervention:
                t += rng.uniform(60.0, 180.0)
    return out


def write_log(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write a synthetic cell episode log (JSONL).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--per-session", type=int, default=1400)
    ap.add_argument("--session-sd", type=float, default=0.0)
    ap.add_argument("--wear", action="store_true",
                    help="add a 6 point yield drop starting 60%% into the second session")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    drift = {"session": 1, "at": 0.6, "drop": 0.06, "shape": "step", "persist": True} if a.wear else None
    recs = generate(a.seed, sessions=a.sessions, per_session=a.per_session,
                    session_sd=a.session_sd, drift=drift)
    write_log(recs, a.out)
    print("wrote %d synthetic episodes to %s" % (len(recs), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
