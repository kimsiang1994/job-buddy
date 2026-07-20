"""Tune task_profiles.json from real usage recorded in usage_log.jsonl.

    py calibrate_budgets.py --report    # show stats, change nothing
    py calibrate_budgets.py             # retune profiles with enough samples
    py calibrate_budgets.py --dry-run

Two jobs:

  1. Budget tuning -- set each profile's max_tokens to p95(completion_tokens) x
     1.25, so budgets follow what the workload actually needs.

  2. Estimator accuracy -- compare estimated_prompt_tokens against the API's real
     prompt_tokens. The only published tokenizer is the *v3* one while we call
     *v4* models, so its accuracy is a measured number here rather than an
     assumption. If the error is small the dependency earns its place; if not,
     drop it and keep the char-ratio heuristic.

This is the only writer of task_profiles.json.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

from jobbuddy.deepseek import deepseek_common as common

# src/jobbuddy/deepseek/<file> -> four levels up is the repo root.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PROFILES_PATH = os.path.join(REPO_DIR, "config", "task_profiles.json")
USAGE_LOG = os.path.join(REPO_DIR, "usage_log.jsonl")

# Below this, p95 is noise rather than signal.
MIN_SAMPLES = 30
HEADROOM = 1.25


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_log():
    """Every usable row of usage_log.jsonl, saying how many were not usable.

    A torn line is skipped rather than aborting the run -- the log is appended
    to by concurrent workers, so a partial final line is normal. Skipping it
    SILENTLY is the part that was wrong: every number this module produces is a
    percentile over these rows, and percentiles do not announce that their
    sample shrank. A log half of which fails to parse yields budgets tuned on
    the other half and looks exactly like a clean run.

    `job_store.read_sightings` reads the same shape of file and already counts
    and reports its damaged lines; this now matches it.
    """
    if not os.path.exists(USAGE_LOG):
        return []
    rows = []
    damaged = 0
    with open(USAGE_LOG, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                damaged += 1
    if damaged:
        print(f"[calibrate_budgets] skipped {damaged} unparseable line(s) in "
              f"{os.path.basename(USAGE_LOG)}; every statistic below is over "
              f"the remaining {len(rows)}", file=sys.stderr)
    return rows


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def load_profiles():
    with open(PROFILES_PATH, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def write_profiles(data):
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp = PROFILES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
    os.replace(tmp, PROFILES_PATH)


def report(rows):
    print(f"usage log: {len(rows)} calls\n")
    if not rows:
        print("Nothing logged yet. Budgets stay at their seeded defaults until\n"
              "real traffic accumulates -- run some calls through deepseek_client.")
        return

    by_profile = defaultdict(list)
    for row in rows:
        by_profile[row.get("profile") or "?"].append(row)

    print(f"{'profile':12} {'n':>4} {'p50':>7} {'p95':>7} {'max':>7} "
          f"{'reason p95':>11} {'trunc':>6}")
    print("-" * 60)
    for name, entries in sorted(by_profile.items()):
        completions = [r["completion_tokens"] for r in entries
                       if isinstance(r.get("completion_tokens"), int)]
        reasoning = [r["reasoning_tokens"] for r in entries
                     if isinstance(r.get("reasoning_tokens"), int)]
        truncated = sum(1 for r in entries if r.get("finish_reason") == "length")
        if not completions:
            continue
        print(f"{name:12} {len(entries):>4} "
              f"{percentile(completions, 0.5):>7} "
              f"{percentile(completions, 0.95):>7} "
              f"{max(completions):>7} "
              f"{(percentile(reasoning, 0.95) if reasoning else '-'):>11} "
              f"{truncated:>6}")

    # Estimator accuracy: predicted prompt tokens vs what the API actually billed.
    paired = [(r["estimated_prompt_tokens"], r["prompt_tokens"], r.get("estimator"))
              for r in rows
              if isinstance(r.get("estimated_prompt_tokens"), int)
              and isinstance(r.get("prompt_tokens"), int) and r["prompt_tokens"] > 0]
    print("\nestimator accuracy (predicted vs actual prompt_tokens)")
    if not paired:
        print("  no paired samples yet")
        return
    by_backend = defaultdict(list)
    for predicted, actual, backend in paired:
        by_backend[backend or "?"].append(abs(predicted - actual) / actual)
    for backend, errors in sorted(by_backend.items()):
        mean_err = statistics.fmean(errors) * 100
        worst = max(errors) * 100
        print(f"  {backend:10} n={len(errors):<5} mean error {mean_err:5.1f}%  "
              f"worst {worst:5.1f}%")


def calibrate(rows, min_samples, dry_run):
    data = load_profiles()
    profiles = data.get("profiles") or {}
    by_profile = defaultdict(list)
    for row in rows:
        if isinstance(row.get("completion_tokens"), int):
            by_profile[row.get("profile")].append(row["completion_tokens"])

    changed = []
    for name, profile in profiles.items():
        samples = by_profile.get(name) or []
        profile["sample_count"] = len(samples)
        if len(samples) < min_samples:
            continue
        p95 = percentile(samples, 0.95)
        proposed = max(16, int(math.ceil(p95 * HEADROOM)))
        profile["observed_p95"] = p95
        if proposed != profile.get("max_tokens"):
            changed.append(f"{name}: max_tokens {profile.get('max_tokens')} "
                           f"-> {proposed} (p95={p95}, n={len(samples)})")
            profile["max_tokens"] = proposed
        profile["last_calibrated"] = now_iso()

    if not changed:
        under = [n for n in profiles
                 if len(by_profile.get(n) or []) < min_samples]
        print(f"No profile had enough samples to retune "
              f"(need {min_samples}). Waiting on: {', '.join(sorted(under))}")
        return 0

    print("Retuned:")
    for line in changed:
        print(f"  - {line}")

    if dry_run:
        print("\n(--dry-run: nothing written)")
        return 0

    data["last_calibrated"] = now_iso()
    write_profiles(data)
    print(f"\nwrote {PROFILES_PATH}")
    return 0


def main():
    common.enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true",
                        help="print stats and estimator accuracy; change nothing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    args = parser.parse_args()

    rows = read_log()
    if args.report:
        report(rows)
        return 0
    return calibrate(rows, args.min_samples, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
