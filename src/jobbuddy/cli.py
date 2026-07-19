"""CLI over the job pipeline. No LLM, no optional dependencies, no cost.

    py slice1.py --scope ai-engineer-sg
    py slice1.py --all --limit 40
    py slice1.py --explain mcf:59501ac0...

Everything here is argparse and printing. The run itself lives in `pipeline.py`
so that the notebook and this file share one implementation -- they used to have
one each, and the copies drifted.

Run it twice a day apart: `first_seen_at` must stay put, already-seen jobs must
not report as new, and the ranking must look sensible.

Exit codes follow the repo convention: 0 clean, 1 a source degraded, 2 action
needed.
"""

from __future__ import annotations

import argparse
import sys

from jobbuddy import job_store
from jobbuddy import pipeline
from jobbuddy import scoring

# `enable_utf8_stdout` lives in deepseek_common, which reads .env at import.
# Importing the LLM plumbing into the no-LLM spine would make it load-bearing
# here, so the four lines are inlined instead.
def _enable_utf8_stdout() -> None:
    """Stop Windows' cp1252 console mangling em-dashes in job titles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def print_result(result: pipeline.RunResult, top: int, had_prior_runs: bool) -> None:
    """Render a RunResult to stdout."""
    print(f"\n{len(result.jobs)} job(s) ranked  "
          f"({result.new_count} new, {result.returning_count} seen before, "
          f"{result.prior_run_count} prior run(s))\n")

    header = (f"{'#':>3} {'score':>5} {'title':42} {'company':20} "
              f"{'level':9} {'salary':13} {'apps':>5} {'conf':>5} {'src':<9}")
    print(header)
    print("-" * len(header))
    for index, job in enumerate(result.jobs[:top], start=1):
        salary = (f"{job['salary_min_sgd']}-{job['salary_max_sgd']}"
                  if job["salary_is_stated"] else "not stated")
        marker = "*" if job["job_key"] in result.new_keys else " "
        scores = job["scores"]
        source = (job.get("_source_adapter") or job.get("source") or "?")[:9]
        # Show the ADJUSTED score, because that is what the list is ordered by.
        # Printing the raw total next to a rank derived from something else
        # makes a correct ordering look arbitrary.
        print(f"{index:3d}{marker}{scores['adjusted']:5.0f} "
              f"{job['title'][:42]:42} {job['company'][:20]:20} "
              f"{str(job['seniority'] or '-'):9} {salary:13} "
              f"{str(job['applications'] or '-'):>5} "
              f"{scores['confidence']:>5.0%} {source:<9}")

    if had_prior_runs:
        print("\n  * = first seen this run")

    if result.absent:
        print(f"\n{len(result.absent)} tracked job(s) absent from the feed for "
              f"{job_store.ABSENT_RUNS_BEFORE_SUSPECT}+ runs -- candidates for a "
              f"liveness re-check (absence alone is not evidence of closure)")

    reasons = result.exclusion_reasons()
    if reasons:
        print(f"\n{len(result.excluded)} job(s) excluded by hard filters. Top reasons:")
        for reason, count in list(reasons.items())[:6]:
            print(f"    {count:4d}  {reason}")

    for path in result.written:
        try:
            print(f"\nwrote {path.relative_to(pipeline.REPO_DIR)}")
        except ValueError:
            print(f"\nwrote {path}")


def explain_job(job_key: str) -> int:
    """Dump everything known about one job from the sightings history."""
    history = job_store.JobHistory.load()
    rows = history.sightings_of(job_key)
    if not rows:
        print(f"no sightings recorded for {job_key}")
        return 0  # asking about an unseen job is a question, not a failure

    print(f"=== {job_key} ===")
    print(f"seen {len(rows)} time(s) across runs: "
          f"{sorted({r.get('run_id') for r in rows})}\n")
    print(f"{'when':22} {'apps':>5} {'views':>6} {'open':>6}")
    for row in rows:
        print(f"{row.get('ts', ''):22} {str(row.get('applications')):>5} "
              f"{str(row.get('views')):>6} {str(row.get('is_open')):>6}")

    first, last = rows[0].get("applications"), rows[-1].get("applications")
    if isinstance(first, int) and isinstance(last, int) and len(rows) > 1:
        print(f"\napplications while watching: {first} -> {last} ({last - first:+d})")

    print("\nlatest record")
    for key in sorted(rows[-1]):
        print(f"  {key:22} {rows[-1][key]}")
    return 0


def main() -> int:
    _enable_utf8_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scope", help="scope name from run_config.json")
    parser.add_argument("--all", action="store_true", help="run every configured scope")
    parser.add_argument("--limit", type=int, help="cap results per query (dev)")
    parser.add_argument("--explain", metavar="JOB_KEY", help="dump one job's history")
    parser.add_argument("--no-cache", action="store_true", help="bypass the HTTP cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="score and print, but write no state and no artefacts")
    parser.add_argument("--top", type=int, default=15, help="rows to print (default 15)")
    args = parser.parse_args()

    if args.explain:
        return explain_job(args.explain)

    config = scoring.load_config()
    scopes = config.get("scopes") or []
    if not scopes:
        print("run_config.json defines no scopes", file=sys.stderr)
        return 2

    if args.all:
        selected = scopes
    elif args.scope:
        selected = [s for s in scopes if s.get("name") == args.scope]
        if not selected:
            names = ", ".join(s.get("name", "?") for s in scopes)
            print(f"unknown scope {args.scope!r}. Available: {names}", file=sys.stderr)
            return 2
    else:
        selected = scopes[:1]
        print(f"no --scope given; using {selected[0]['name']!r}\n")

    history = job_store.JobHistory.load()
    had_prior_runs = history.run_count > 0

    for scope in selected:
        print(f"--- scope: {scope['name']} ---")

    result = pipeline.run_scopes(
        selected, config,
        limit=args.limit,
        cache_ttl_s=0.0 if args.no_cache else 900.0,
        dry_run=args.dry_run,
        history=history,
    )
    print(f"    {result.counters}")

    if args.dry_run:
        print("\n[dry run] no state written")

    if not result.jobs:
        print("\nno jobs survived filtering", file=sys.stderr)
        return 2

    print_result(result, args.top, had_prior_runs)
    return result.exit_code()


if __name__ == "__main__":
    sys.exit(main())
