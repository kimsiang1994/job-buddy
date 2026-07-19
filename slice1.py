"""Slice 1: the pipeline spine, with no LLM and no optional dependencies.

    py slice1.py --scope ai-engineer-sg
    py slice1.py --all --limit 40
    py slice1.py --explain mcf:59501ac0...

This exists to prove the hardest-to-change decisions -- the schema, the state
keys, the fold -- before anything is built on top of them. Run it twice a day
apart: `first_seen_at` must stay put, already-seen jobs must not report as new,
and the ranking must look sensible. If that holds, the architecture is sound.

Costs nothing and needs no API key. Exit codes follow the repo convention:
0 clean, 1 a source degraded, 2 action needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import deepseek_common
import job_schema
import job_store
import scoring
import source_mcf

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_DIR / "potential applications"

CSV_COLUMNS = (
    "rank", "total", "title", "company", "seniority", "salary_min_sgd",
    "salary_max_sgd", "applications", "views", "apps_per_view", "age_days",
    "vacancies", "is_agency", "reposted", "skill_matched", "skill_total",
    "location", "scope", "job_key", "url", "why",
)


def collect(
    scope: dict[str, Any],
    config: dict[str, Any],
    limit: int | None,
    cache_ttl_s: float,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    """Fetch and filter every query in a scope. Returns (jobs, counters, excluded)."""
    counters: dict[str, int] = {}
    seen: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []

    per_query = scope.get("max_results_per_query", 50)
    if limit:
        per_query = min(per_query, limit)

    for query in scope.get("queries", []):
        jobs, query_counters = source_mcf.fetch_jobs(
            query,
            max_results=per_query,
            singapore_only=(config.get("filters") or {}).get("singapore_only", True),
            open_only=(config.get("filters") or {}).get("open_only", True),
            cache_ttl_s=cache_ttl_s,
        )
        for key, value in query_counters.items():
            counters[key] = counters.get(key, 0) + value

        for job in jobs:
            job["scope"] = scope["name"]
            reason = scoring.check_filters(job, config)
            if reason:
                excluded.append({
                    "job_key": job["job_key"], "title": job["title"],
                    "company": job["company"], "reason": reason,
                })
                counters["filtered"] = counters.get("filtered", 0) + 1
                continue
            # The same role turns up under several queries; keep it once.
            if job["job_key"] in seen:
                counters["duplicate_job"] = counters.get("duplicate_job", 0) + 1
                continue
            seen[job["job_key"]] = job

    # Collapse agency duplicates of one requisition: identical content posted by
    # different companies. Prefer the direct employer, else the oldest sighting.
    by_content: dict[str, list[dict[str, Any]]] = {}
    for job in seen.values():
        by_content.setdefault(job["content_key"], []).append(job)

    deduped: list[dict[str, Any]] = []
    for group in by_content.values():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        group.sort(key=lambda j: (j["is_agency"], j.get("posted_at") or "9999"))
        winner = group[0]
        winner["duplicate_postings"] = [j["job_key"] for j in group[1:]]
        deduped.append(winner)
        counters["duplicate_content"] = counters.get("duplicate_content", 0) + len(group) - 1

    return deduped, counters, excluded


def write_csv(jobs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for index, job in enumerate(jobs, start=1):
            scores = job.get("scores") or {}
            skill = (scores.get("components") or {}).get("skill_match", {}).get("detail", {})
            writer.writerow({
                "rank": index,
                "total": scores.get("total"),
                "title": job["title"],
                "company": job["company"],
                "seniority": job["seniority"],
                "salary_min_sgd": job["salary_min_sgd"],
                "salary_max_sgd": job["salary_max_sgd"],
                "applications": job["applications"],
                "views": job["views"],
                "apps_per_view": job["apps_per_view"],
                "age_days": job["age_days"],
                "vacancies": job["vacancies"],
                "is_agency": job["is_agency"],
                "reposted": job["reposted"],
                "skill_matched": skill.get("matched_count"),
                "skill_total": skill.get("total_count"),
                "location": job["location"],
                "scope": job["scope"],
                "job_key": job["job_key"],
                "url": job["url"],
                "why": scores.get("explanation", ""),
            })


def explain_job(job_key: str) -> int:
    """Dump everything known about one job from the sightings history."""
    sightings = job_store.read_sightings()
    rows = [r for r in sightings if r.get("job_key") == job_key]
    if not rows:
        print(f"no sightings recorded for {job_key}")
        return 2
    print(f"=== {job_key} ===")
    print(f"seen {len(rows)} time(s) across runs: "
          f"{sorted({r.get('run_id') for r in rows})}")
    print()
    for row in rows:
        print(f"  {row.get('ts')}  apps={row.get('applications')} "
              f"views={row.get('views')} open={row.get('is_open')} "
              f"status={row.get('source_status')}")
    print()
    latest = rows[-1]
    for key in sorted(latest):
        print(f"  {key:20s} {latest[key]}")
    return 0


def main() -> int:
    deepseek_common.enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", help="scope name from run_config.json")
    parser.add_argument("--all", action="store_true", help="run every configured scope")
    parser.add_argument("--limit", type=int, help="cap results per query (dev)")
    parser.add_argument("--explain", metavar="JOB_KEY", help="dump one job's history")
    parser.add_argument("--no-cache", action="store_true", help="bypass the HTTP cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="score and print, but write no state and no CSV")
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

    cache_ttl = 0.0 if args.no_cache else 900.0
    run_id = job_store.new_run_id()

    # History BEFORE this run, so "new since last time" is answerable.
    prior_sightings = job_store.read_sightings()
    prior_history = job_store.fold(prior_sightings)
    prior_runs = job_store.all_run_ids(prior_sightings)
    velocity = job_store.company_velocity(prior_history)

    all_jobs: list[dict[str, Any]] = []
    all_counters: dict[str, int] = {}
    all_excluded: list[dict[str, Any]] = []

    for scope in selected:
        print(f"--- scope: {scope['name']} ---")
        jobs, counters, excluded = collect(scope, config, args.limit, cache_ttl)
        print(f"    {counters}")
        all_jobs.extend(jobs)
        all_excluded.extend(excluded)
        for key, value in counters.items():
            all_counters[key] = all_counters.get(key, 0) + value

    if not all_jobs:
        print("\nno jobs survived filtering", file=sys.stderr)
        return 2

    # Record first, then fold, so this run's observation is in the history that
    # first_seen_at is derived from.
    if not args.dry_run:
        written = job_store.record_sightings(all_jobs, run_id)
        print(f"\nrecorded {written} sighting(s) as run {run_id}")
        sightings = job_store.read_sightings()
    else:
        print("\n[dry run] no state written")
        sightings = prior_sightings

    history = job_store.fold(sightings)
    run_ids = job_store.all_run_ids(sightings)
    job_store.apply_history(all_jobs, history, run_id, run_ids)

    new_this_run = sum(
        1 for j in all_jobs if j["job_key"] not in prior_history
    )
    returning = len(all_jobs) - new_this_run

    for job in all_jobs:
        scoring.score_job(job, config, velocity)
    all_jobs.sort(key=lambda j: j["scores"]["total"], reverse=True)

    print(f"\n{len(all_jobs)} job(s) ranked  "
          f"({new_this_run} new, {returning} seen before, "
          f"{len(prior_runs)} prior run(s))\n")

    header = (f"{'#':>3} {'score':>5} {'title':42} {'company':20} "
              f"{'level':9} {'salary':13} {'apps':>5} {'age':>5}")
    print(header)
    print("-" * len(header))
    for index, job in enumerate(all_jobs[: args.top], start=1):
        salary = (f"{job['salary_min_sgd']}-{job['salary_max_sgd']}"
                  if job["salary_is_stated"] else "not stated")
        marker = "*" if job["job_key"] not in prior_history else " "
        print(f"{index:3d}{marker}{job['scores']['total']:5.0f} "
              f"{job['title'][:42]:42} {job['company'][:20]:20} "
              f"{str(job['seniority'] or '-'):9} {salary:13} "
              f"{str(job['applications'] or '-'):>5} "
              f"{('%.0f' % job['age_days']) if job['age_days'] is not None else '-':>5}")

    if prior_runs:
        print("\n  * = first seen this run")

    suspects = job_store.mark_absent(
        history, {j["job_key"] for j in all_jobs}, run_ids
    )
    if suspects:
        print(f"\n{len(suspects)} tracked job(s) absent from the feed for "
              f"{job_store.ABSENT_RUNS_BEFORE_SUSPECT}+ runs -- candidates for a "
              f"liveness re-check (absence alone is not evidence of closure)")

    if all_excluded:
        print(f"\n{len(all_excluded)} job(s) excluded by hard filters. Top reasons:")
        reasons: dict[str, int] = {}
        for item in all_excluded:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:6]:
            print(f"    {count:4d}  {reason}")

    if not args.dry_run:
        scope_label = selected[0]["name"] if len(selected) == 1 else "all-scopes"
        csv_path = OUTPUT_DIR / scope_label / run_id / "ranked.csv"
        write_csv(all_jobs, csv_path)
        job_store.write_snapshot(history, run_id)
        print(f"\nwrote {csv_path.relative_to(REPO_DIR)}")

        json_path = csv_path.with_name("ranked.json")
        try:
            json_path.write_text(
                json.dumps(all_jobs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"wrote {json_path.relative_to(REPO_DIR)}")
        except OSError as exc:
            print(f"could not write ranked.json ({exc})", file=sys.stderr)

    degraded = all_counters.get("invalid", 0) or all_counters.get("unusable", 0)
    return 1 if degraded else 0


if __name__ == "__main__":
    sys.exit(main())
