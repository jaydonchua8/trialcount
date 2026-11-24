# trialcount

An evaluation harness for robot policies and robot cells that answers one question: do I have enough trials, of the right kind, to trust this success rate?

This is an independent student prototype. It is not affiliated with, reviewed by, or endorsed by Relling. Every result below comes from synthetic data that I generated, and I say so again wherever a number appears.

## The problem

A robot trial is slow and usually needs a person next to the cell, so a policy is often judged on 20 to 40 rollouts. The first version of this repo showed what that many rollouts can and cannot tell you, assuming each trial is an independent coin flip with a fixed success rate.

That assumption is the weak part. On a real cell:

- Trials run in sessions. Trials from the same shift share a calibration, a fixture setup and lighting, so they pass or fail together more than chance allows. A binomial interval on 600 trials from 3 shifts is too narrow.
- The rate drifts. A gripper pad wears or a fixture shifts part way through a shift, and a single pooled success rate then describes neither the cell before the change nor the cell after it.
- "Done" is more than one success rate. An acceptance run also has a cycle time, a continuous run with no human help, a list of part numbers, and a changeover time.

The second version, which is what this README describes, handles those three things and replaces the simulated sequential-test numbers with exact ones.

## Why this matters to Relling

These are things I read on Relling's public pages, not things I know about how they work internally.

- Their Software Engineer Intern posting lists "training and evaluation infrastructure" as an area of work and says that without enough trials to trust a policy's success rate "we don't know anything". It also mentions deployment "without it falling over on hour four". (https://www.ycombinator.com/companies/relling/jobs/H005USc-software-engineer-intern)
- A blog post by their CTO, "Why most robot pilots never reach production", lists "Done was never a number" as a failure mode and proposes agreeing five numbers before the purchase order: cycle time, first-pass yield, a continuous run duration without human intervention, the part numbers in scope, and changeover time. `cellrun.py` takes exactly those five numbers as its spec. The same post uses a worn gripper pad "dropping one part in fifty" as its example of slow decay, which is the case the drift test is for. (https://rellingsystems.com/resources/blog/why-robot-pilots-stall)
- Their post on vision system validation uses the rule that zero misses in n presentations bounds the miss rate at about 3 / n, and that showing a rate below 1% needs about 300 presentations. `zero_failure_trials(0.99)` returns 299, the exact version of that rule. (https://rellingsystems.com/resources/blog/vision-system-validation)

I could not find any public source code or data format from Relling, so the episode log format here is my own and is documented in `cellrun.py`.

## What it does

1. **Intervals and planning for independent trials** (`trialcount.py`). Wilson and exact Clopper-Pearson intervals, one-sided bounds, Fisher's exact test for two policies, exact McNemar for paired trials, and sample size planning.
2. **Sequential certification with a hard cap** (`trialcount.py`). Wald's SPRT stops early when a policy is clearly good or bad. `sprt_exact` computes its exact error rates and trial counts by summing over every path, and `capped_plan` finds a version with a fixed maximum number of trials whose exact error rates still meet the target.
3. **Session-aware intervals** (`robust.py`). `cluster_interval` treats the session as the unit and uses a Student t quantile on the spread between sessions.
4. **Drift detection** (`robust.py`). `changepoint` is a permutation test for a change in success rate inside a session. It compares each part number against its own rate, so switching from an easy part to a hard one is not reported as drift.
5. **Acceptance report** (`cellrun.py`). Reads an episode log and a five-number spec and prints PASS, FAIL, NOT YET or UNSTABLE for each criterion, with how many more trials or sessions would be needed when the answer is NOT YET.
6. **A synthetic cell** (`simcell.py`) that writes episode logs with known truth: several part numbers, session effects, drift, interventions and changeovers.

## Architecture

```
trialcount.py   statistics core: intervals, bounds, tests, planning, SPRT, exact SPRT analysis, capped plans
robust.py       session-aware intervals and the drift test (imports trialcount)
cellrun.py      episode log parsing and validation, acceptance spec, per-criterion verdicts, CLI
simcell.py      synthetic episode log generator
bench.py        every number in this README; writes bench_results.json
demo.py         one minute tour
test_trialcount.py, test_cell.py
acceptance_spec.json, sample_episodes.jsonl
```

The dependency direction is one way: `cellrun` uses `robust` and `trialcount`, `robust` uses `trialcount`, and `trialcount` uses only the standard library. `simcell` is used only by the demo, the benchmark and the tests.

The episode log is JSON Lines, one episode per line:

```
{"episode": 17, "session": "S01", "t_start": 201.4, "part": "PN-1041", "policy": "act_v3", "success": true, "intervention": false, "cycle_s": 10.9}
```

## Key decisions and tradeoffs

**Exact computation instead of simulation for the sequential test.** Only a narrow band of (trials, successes) states is ever undecided in an SPRT, so walking every state forward is cheap and gives the error rates to many digits. The first version simulated 20,000 runs per row. The exact numbers agree with those simulations and showed me something the simulation could not search over: which capped designs actually meet the error bound.

**The session-level t interval is the default, and it is deliberately wide.** I compared three intervals on development settings and picked the one that held coverage when sessions differ. The price is width. With two or three sessions the t quantile is large, so the acceptance report often says NOT YET and asks for more sessions. I think that is the correct answer when three shifts are all the evidence there is, but it is a real cost and the table below shows it.

**The drift test shuffles outcomes only inside a session.** Without that, any session that is simply better or worse than another looks like drift (28% to 31% false alarms in the benchmark). With it, false alarms stay near the nominal rate, but the test can only use evidence from inside one session. A drop that persists into later sessions is indistinguishable from a bad session, so the grouped test is much less sensitive than the plain one. Long sessions are what give it power.

**First-pass yield counts an intervention as a miss.** An episode that succeeded only because a person stepped in is not a first-pass success.

**NOT YET and UNSTABLE are separate verdicts.** A report that can only say pass or fail hides the most common real outcome, which is that the data cannot decide yet.

**Strict log parsing.** The first version read `success` with `bool(...)`, which turns the string `"false"` into `True`. The loaders now accept only JSON booleans and refuse duplicate episode ids and episodes that overlap in time within a session.

## Install and usage

Python 3.8 or newer. Nothing to install.

```
python demo.py                                         # one minute tour
python trialcount.py interval 34 40                    # intervals for 34 successes in 40 trials
python trialcount.py report sample_episodes.jsonl      # compare two policies in a simple log
python simcell.py --seed 3 --wear --out cell_log.jsonl # write a synthetic cell log with gripper wear
python cellrun.py acceptance_spec.json cell_log.jsonl  # acceptance report (exit code 0 only on PASS)
python -m unittest                                     # all tests
python bench.py                                        # regenerate every number below
```

The demo and the tests each finish in under a minute. The benchmark used 66 seconds of CPU time when I ran it, and about 7 minutes of wall time on a shared, throttled machine.

A capped sequential test in code:

```python
import trialcount as tc

plan = tc.capped_plan(0.80, 0.90, max_trials=149)
test = tc.SPRT(0.80, 0.90, plan["alpha_inner"], plan["beta_inner"], plan["max_trials"], plan["pass_mark"])
for episode in run_rollouts(policy):          # your own loop
    if test.update(episode.success) is not None:
        break
print(test.decision, test.k, test.n)
```

## A real example

This is the actual output of the two commands above (`simcell.py --seed 3 --wear`, then `cellrun.py`). The log is synthetic: 3 shifts of 1400 parts, with a 6 point yield drop that starts 60% of the way through the second shift and persists. The spec asks for 95% first-pass yield, 12 s cycle time, a 4 hour continuous run and 600 s changeover.

```
NOT YET   first-pass yield PN-1041   2003/2100 = 95.4% over 3 sessions, 95% bounds 88.7% to 100.0%, target 95.0%. at this rate about 7030 more trials would be needed
PASS      cycle time       PN-1041   mean 10.94 s over 2097 episodes (95% bounds 10.90 to 10.98 s, 95th percentile 12.59 s), limit 12.00 s
NOT YET   first-pass yield PN-2210   1967/2100 = 93.7% over 3 sessions, 95% bounds 89.2% to 98.1%, target 95.0%. the observed rate is not above the target, or only barely, so more trials alone will not pass it
PASS      cycle time       PN-2210   mean 11.55 s over 2098 episodes (95% bounds 11.51 to 11.60 s, 95th percentile 13.65 s), limit 12.00 s
UNSTABLE  yield stability  cell      yield changed inside session S02: 98.2% before episode 2161, 90.3% after, p = 0.002. A pooled yield number mixes the two; find the cause first.
PASS      continuous run   cell      longest run without intervention 5.36 h (session S02, episodes 1400 to 2799), required 4.00 h
PASS      changeover       cell      9 changeovers, longest 281 s, limit 600 s
overall: NOT YET
```

The wear was injected at episode 2240. The test placed the change at episode 2161. Without `--wear` the same seed gives PASS on every line (`python demo.py` shows both, plus a log that is too short to decide).

## Tests and verified results

`python -m unittest` runs 70 tests (22 from the first version, which I did not edit, and 48 new). They also pass under `python -W error -m unittest`. The new tests cover failure cases: malformed log rows, string booleans, duplicate and overlapping episodes, invalid specs, impossible trial caps, a part number with no episodes, logs that mix policies, and sequences too short to test for drift.

All numbers below are printed by `python bench.py` and saved in `bench_results.json`. Sections D and E use synthetic data.

**A capped sequential test needs calibrated boundaries.** The question is whether a policy is at most 80% or at least 90%, with 5% error allowed each way. A fixed plan needs 135 trials. All values are exact.

| design | max trials | passes an 80% policy | fails a 90% policy | mean trials at 80% / 90% / 95% |
|---|---|---|---|---|
| fixed plan | 135 | 4.9% | 4.8% | 135 / 135 / 135 |
| uncapped SPRT | none | 4.8% | 3.9% | 65.5 / 75.0 / 38.4 |
| SPRT stopped at 135, take the leaning side | 135 | 6.68% | 5.91% | 61.1 / 69.8 / 38.4 |
| calibrated, cap 135 | 135 | 4.98% | 4.87% | 89.4 / 102.1 / 60.1 |
| calibrated, cap 149 | 149 | 4.93% | 4.65% | 73.4 / 84.3 / 45.9 |
| calibrated, cap 203 | 203 | 4.53% | 4.58% | 67.1 / 77.0 / 40.0 |

Stopping an SPRT at the fixed plan's trial count and taking the leaning side breaks both error bounds. The calibrated design with the same 135 trial worst case keeps both under 5% and still averages 89 to 102 trials near the decision boundary and 60 for a 95% policy. Allowing 10% more trials in the worst case (149) brings the mean for a 95% policy down to 46.

**Binomial intervals fail when trials come in sessions.** Real coverage of a nominal 95% interval over 1500 synthetic evaluations per row. `session_sd` is the spread of a session's yield on the logit scale. I chose the default method on the development rows and then ran the held-out rows, which use a different seed, session counts, session sizes, rates and spreads.

| split | sessions x trials | rate | session_sd | Wilson coverage / width | design-effect Wilson | session-level t (default) |
|---|---|---|---|---|---|---|
| dev | 6 x 60 | 90% | 0.0 | 95.5% / 6.2 pts | 96.5% / 6.8 pts | 97.8% / 8.2 pts |
| dev | 6 x 60 | 90% | 0.6 | 75.6% / 6.5 pts | 89.8% / 11.1 pts | 93.6% / 14.3 pts |
| dev | 3 x 300 | 95% | 0.6 | 53.4% / 3.0 pts | 81.3% / 7.2 pts | 92.8% / 12.6 pts |
| held-out | 4 x 40/80/120 | 95% | 0.0 | 96.0% / 5.2 pts | 96.9% / 6.0 pts | 98.3% / 7.6 pts |
| held-out | 4 x 40/80/120 | 95% | 0.9 | 63.0% / 5.8 pts | 85.4% / 12.5 pts | 88.9% / 15.1 pts |
| held-out | 10 x 30 | 85% | 0.4 | 86.9% / 8.2 pts | 92.5% / 10.2 pts | 94.0% / 11.7 pts |
| held-out | 3 x 700 | 97% | 0.5 | 55.1% / 1.5 pts | 83.2% / 3.7 pts | 93.7% / 6.7 pts |
| held-out | 2 x 75 | 97% | 0.6 | 88.4% / 6.1 pts | 93.0% / 8.6 pts | 97.2% / 23.9 pts |

With 2100 trials over 3 sessions, a Wilson interval that claims 95% covered the true rate 55.1% of the time. The session-level t interval covered 93.7%. It is not a guarantee: with a large spread and unequal session sizes it dropped to 88.9%. With independent trials it over-covers and is 30% to 45% wider than Wilson.

**Drift detection is honest about false alarms and weak on short sessions.** Share of 150 synthetic runs flagged at p < 0.05, 199 shuffles per test. I chose `min_seg = 40` on the development setting (out of 10, 20 and 40).

| split | case | plain test | grouped by session |
|---|---|---|---|
| dev | no drift, independent trials | 2.7% | 2.0% |
| dev | no drift, sessions differ | 30.7% | 2.0% |
| dev | 10 point step half way through session 2 of 3 x 200, sessions differ | 86.0% | 25.3% |
| held-out | no drift, independent trials | 3.3% | 5.3% |
| held-out | no drift, sessions differ | 28.0% | 3.3% |
| held-out | 20 point ramp over the last 60% of session 3 of 4 x 150, sessions differ | 99.3% | 22.0% |

The plain test finds almost every drift but also flags 28% to 31% of healthy logs whose sessions differ, so it cannot be used for a verdict. The grouped test keeps false alarms near 5% but catches only about a quarter of these drifts, because 100 trials or fewer on each side of the change inside one session is not much evidence. Power grows with session length: for the same 10 point step it flagged 33% of runs at 200 trials per session, 52% at 400 and 83% at 800 (100 runs each, held-out seed).

**Results carried over from the first version** (independent trials, exact unless stated): 40 trials per policy separate an 80% policy from a 90% policy with Fisher's exact test only 16.2% of the time, and the worse policy scores the same or higher 13.3% of the time. A 34/40 result has a Wilson interval of 70.9% to 92.9%.

## Limitations and next steps

- I have never run this on a real robot or a real policy. Every log is synthetic, and the generator encodes my guesses about how a cell misbehaves. The policy and part names in the logs are placeholders.
- The log format is mine. A real integration would read whatever the cell's episode store already writes.
- The grouped drift test has low power on sessions shorter than several hundred trials, as shown above, and cannot see a change that starts exactly at a session boundary. A one-sided test for drops only, or a model that shares a wear trend across sessions, would do better.
- The session-level interval needs at least two sessions to see any spread and is very wide with two or three. It also under-covered (88.9%) in the hardest held-out setting.
- The acceptance report checks each part number at the stated confidence with no correction for checking several parts, and the continuous run and changeover checks compare one observed number with the limit, with no interval.
- Success is pass or fail per episode. Partial progress and failure type are ignored.
- The capped plan searches a small grid of caps and thresholds. Some caps have no valid design on that grid (140 for the 80% versus 90% question), and the search reports that as an error instead of widening the grid.
- Paired comparison has an exact test (`mcnemar_exact`) but no planning function and no benchmark yet.
