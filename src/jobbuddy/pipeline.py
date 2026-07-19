"""The run: fetch, filter, dedupe, record, score, rank.

This exists because the sequence below is real behaviour that previously had no
module to live in. It was written out longhand in `slice1.main()` and again in a
notebook cell, and the copies had already diverged in six ways -- the notebook
had silently dropped the absent-job re-check and the JSON output. Two
implementations of one pipeline, neither reachable by a test.

Now there is one interface:

    result = pipeline.run(scope, config)

`slice1.py` wraps it in argparse and printing. The notebook calls it directly.
Both get the same ranking, the same artefacts and the same history.

Deliberately free of I/O policy: no argparse, no print, no sys.exit. Callers
decide how to report. That is what makes the run testable -- its only impure
dependencies are `source_mcf.fetch_jobs` and the history log, and both are
substitutable through the arguments below.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jobbuddy import job_store
from jobbuddy import scoring
from jobbuddy import source_mcf

REPO_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_DIR / "potential applications"

CSV_COLUMNS = (
    "rank", "adjusted", "total", "confidence", "source", "title", "company", "seniority", "salary_min_sgd",
    "salary_max_sgd", "applications", "views", "apps_per_view", "age_days",
    "vacancies", "is_agency", "reposted", "skill_matched", "skill_total",
    "location", "scope", "job_key", "url", "why",
)


@dataclass
class RunResult:
    """Everything one run produced. No printing, no exit codes -- just facts."""

    run_id: str
    jobs: list[dict[str, Any]]
    excluded: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    absent: list[dict[str, Any]] = field(default_factory=list)
    new_keys: set[str] = field(default_factory=set)
    prior_run_count: int = 0
    written: list[Path] = field(default_factory=list)
    dry_run: bool = False

    @property
    def new_count(self) -> int:
        return len(self.new_keys)

    @property
    def returning_count(self) -> int:
        return len(self.jobs) - len(self.new_keys)

    @property
    def degraded(self) -> bool:
        """True when a source produced records we could not use."""
        return bool(self.counters.get("invalid") or self.counters.get("unusable"))

    def exclusion_reasons(self) -> dict[str, int]:
        """Filter reasons, most common first."""
        reasons: dict[str, int] = {}
        for item in self.excluded:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        return dict(sorted(reasons.items(), key=lambda kv: -kv[1]))

    def exit_code(self) -> int:
        """Repo convention: 0 clean, 1 degraded, 2 action needed."""
        if not self.jobs:
            return 2
        return 1 if self.degraded else 0


def _all_sources(enabled: list[str] | None = None) -> Callable[..., tuple[list[dict], dict]]:
    """Adapt the multi-source registry to the single-source fetch signature.

    Keeps `collect` unaware of how many sources exist. Per-source counters are
    flattened with a prefix so a run summary still shows where jobs came from
    and which adapter dropped what.
    """
    from jobbuddy import sources

    def fetch(query: str, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
        jobs, per_source = sources.fetch_all(
            query,
            max_results_per_source=kwargs.get("max_results", 60),
            singapore_only=kwargs.get("singapore_only", True),
            open_only=kwargs.get("open_only", True),
            cache_ttl_s=kwargs.get("cache_ttl_s", 900.0),
            enabled=enabled,
        )
        flat: dict[str, int] = {}
        for name, counts in per_source.items():
            flat[f"{name}_kept"] = counts.get("kept", 0)
            for key in ("invalid", "unusable", "error"):
                if counts.get(key):
                    flat[key] = flat.get(key, 0) + counts[key]
        flat["fetched"] = len(jobs)
        return jobs, flat

    return fetch


def collect(
    scope: dict[str, Any],
    config: dict[str, Any],
    limit: int | None = None,
    cache_ttl_s: float = 900.0,
    fetch_jobs: Callable[..., tuple[list[dict], dict[str, int]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    """Fetch and filter every query in a scope. Returns (jobs, counters, excluded).

    `fetch_jobs` is the seam for tests -- the one impure dependency, passed in
    rather than looked up, so the dedupe and filter logic can be exercised with
    no network.
    """
    # Default to every enabled source, not just MyCareersFuture. MCF covers the
    # middle of the Singapore market well but is not the whole of it -- its
    # advertising mandate only binds when a foreign work pass is involved, and
    # exempts anything paying over S$22,500/month, which is exactly the band a
    # senior search cares about.
    if fetch_jobs is None:
        enabled = (config.get("sources") or {}).get("enabled")
        fetch_jobs = _all_sources(enabled)
    filters = config.get("filters") or {}
    counters: dict[str, int] = {}
    seen: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []

    per_query = scope.get("max_results_per_query", 50)
    if limit:
        per_query = min(per_query, limit)

    for query in scope.get("queries", []):
        jobs, query_counters = fetch_jobs(
            query,
            max_results=per_query,
            singapore_only=filters.get("singapore_only", True),
            open_only=filters.get("open_only", True),
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

    deduped, collapsed = collapse_duplicate_postings(list(seen.values()))
    if collapsed:
        counters["duplicate_content"] = counters.get("duplicate_content", 0) + collapsed
    return deduped, counters, excluded


def collapse_duplicate_postings(
    jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Collapse one requisition advertised by several companies.

    In Singapore a large share of postings are agencies re-advertising the same
    role. Applying to one job through three agencies wastes everyone's time, so
    identical content collapses to a single entry.

    Which survives: the direct employer over an agency, then the oldest sighting.
    Applying direct beats applying through an intermediary, and the earliest
    posting is the one whose date is real.
    """
    by_content: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_content.setdefault(job["content_key"], []).append(job)

    kept: list[dict[str, Any]] = []
    collapsed = 0
    for group in by_content.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group.sort(key=lambda j: (bool(j.get("is_agency")), j.get("posted_at") or "9999"))
        winner = group[0]
        winner["duplicate_postings"] = [j["job_key"] for j in group[1:]]
        kept.append(winner)
        collapsed += len(group) - 1
    return kept, collapsed


def run(
    scope: dict[str, Any],
    config: dict[str, Any],
    *,
    limit: int | None = None,
    cache_ttl_s: float = 900.0,
    dry_run: bool = False,
    write_artefacts: bool = True,
    output_dir: Path | None = None,
    history: job_store.JobHistory | None = None,
    fetch_jobs: Callable[..., tuple[list[dict], dict[str, int]]] | None = None,
) -> RunResult:
    """Run one scope end to end.

    Ordering is not the caller's problem: `JobHistory.observe` owns the
    record-then-fold sequence, and velocity is captured from the prior log
    before this run's sightings land.
    """
    history = history or job_store.JobHistory.load()
    run_id = job_store.new_run_id()

    jobs, counters, excluded = collect(
        scope, config, limit=limit, cache_ttl_s=cache_ttl_s, fetch_jobs=fetch_jobs
    )

    observation = history.observe(jobs, run_id, record=not dry_run, snapshot=not dry_run)

    for job in jobs:
        scoring.score_job(job, config, observation.velocity)
    jobs.sort(key=lambda j: j["scores"]["adjusted"], reverse=True)

    result = RunResult(
        run_id=run_id,
        jobs=jobs,
        excluded=excluded,
        counters=counters,
        absent=observation.absent,
        new_keys=observation.new_keys,
        prior_run_count=observation.prior_run_count,
        dry_run=dry_run,
    )

    if jobs and write_artefacts and not dry_run:
        result.written = write_outputs(
            jobs, scope.get("name", "run"), run_id, output_dir=output_dir
        )
    return result


def run_scopes(
    scopes: list[dict[str, Any]],
    config: dict[str, Any],
    **kwargs: Any,
) -> RunResult:
    """Run several scopes as one observation.

    Not a loop over `run()`: that would record several runs and make each
    scope's velocity see the previous scope's sightings. One history, one
    run_id, one ranked list.
    """
    if len(scopes) == 1:
        return run(scopes[0], config, **kwargs)

    history = kwargs.pop("history", None) or job_store.JobHistory.load()
    dry_run = kwargs.get("dry_run", False)
    output_dir = kwargs.get("output_dir")
    run_id = job_store.new_run_id()

    all_jobs: list[dict[str, Any]] = []
    all_excluded: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    seen: set[str] = set()

    for scope in scopes:
        jobs, scope_counters, excluded = collect(
            scope, config,
            limit=kwargs.get("limit"),
            cache_ttl_s=kwargs.get("cache_ttl_s", 900.0),
            fetch_jobs=kwargs.get("fetch_jobs"),
        )
        for key, value in scope_counters.items():
            counters[key] = counters.get(key, 0) + value
        all_excluded.extend(excluded)
        for job in jobs:
            if job["job_key"] in seen:
                counters["duplicate_job"] = counters.get("duplicate_job", 0) + 1
                continue
            seen.add(job["job_key"])
            all_jobs.append(job)

    all_jobs, collapsed = collapse_duplicate_postings(all_jobs)
    if collapsed:
        counters["duplicate_content"] = counters.get("duplicate_content", 0) + collapsed

    observation = history.observe(all_jobs, run_id, record=not dry_run, snapshot=not dry_run)
    for job in all_jobs:
        scoring.score_job(job, config, observation.velocity)
    all_jobs.sort(key=lambda j: j["scores"]["adjusted"], reverse=True)

    result = RunResult(
        run_id=run_id, jobs=all_jobs, excluded=all_excluded, counters=counters,
        absent=observation.absent, new_keys=observation.new_keys,
        prior_run_count=observation.prior_run_count, dry_run=dry_run,
    )
    if all_jobs and kwargs.get("write_artefacts", True) and not dry_run:
        result.written = write_outputs(
            all_jobs, "all-scopes", run_id, output_dir=output_dir
        )
    return result


# --------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------

def write_outputs(
    jobs: list[dict[str, Any]],
    scope_label: str,
    run_id: str,
    output_dir: Path | None = None,
) -> list[Path]:
    """Write ranked.csv and ranked.json. Returns the paths written."""
    root = (output_dir or OUTPUT_DIR) / scope_label / run_id
    written: list[Path] = []

    csv_path = root / "ranked.csv"
    if write_csv(jobs, csv_path):
        written.append(csv_path)

    json_path = root / "ranked.json"
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, json_path)
        written.append(json_path)
    except OSError as exc:
        job_store._warn(f"could not write {json_path.name} ({exc})")

    return written


def csv_row(job: dict[str, Any], rank: int) -> dict[str, Any]:
    """One CSV row. Separate from write_csv so the column contract is testable."""
    scores = job.get("scores") or {}
    components = scores.get("components") or {}
    skill = (components.get("skill_match") or {}).get("detail") or {}
    return {
        "rank": rank,
        "adjusted": scores.get("adjusted"),
        "total": scores.get("total"),
        "confidence": scores.get("confidence"),
        "source": job.get("_source_adapter") or job.get("source"),
        "title": job.get("title"),
        "company": job.get("company"),
        "seniority": job.get("seniority"),
        "salary_min_sgd": job.get("salary_min_sgd"),
        "salary_max_sgd": job.get("salary_max_sgd"),
        "applications": job.get("applications"),
        "views": job.get("views"),
        "apps_per_view": job.get("apps_per_view"),
        "age_days": job.get("age_days"),
        "vacancies": job.get("vacancies"),
        "is_agency": job.get("is_agency"),
        "reposted": job.get("reposted"),
        "skill_matched": skill.get("matched_count"),
        "skill_total": skill.get("total_count"),
        "location": job.get("location"),
        "scope": job.get("scope"),
        "job_key": job.get("job_key"),
        "url": job.get("url"),
        "why": scores.get("explanation", ""),
    }


def write_csv(jobs: list[dict[str, Any]], path: Path) -> bool:
    """Write the ranked CSV. Never raises; returns whether it succeeded.

    utf-8-sig because Excel on Windows renders a plain-utf-8 CSV as mojibake,
    and this file is meant to be opened by a human.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            for index, job in enumerate(jobs, start=1):
                writer.writerow(csv_row(job, index))
        return True
    except OSError as exc:
        job_store._warn(f"could not write {path.name} ({exc})")
        return False
