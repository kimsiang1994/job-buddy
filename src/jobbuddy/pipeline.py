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
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jobbuddy import job_store
from jobbuddy import paths
from jobbuddy import scoring
from jobbuddy import source_mcf

REPO_DIR = paths.REPO_DIR
OUTPUT_DIR = paths.OUTPUT_DIR

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
    # None unless the tailoring stage was asked for. Every existing caller of
    # `run_scopes` therefore sees exactly the RunResult it saw before.
    tailoring: "TailorRun | None" = None

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
            if name.startswith("_"):
                # `_vocab` is bookkeeping from the skill harvest, not a source.
                continue
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

    # Companies seen become the discovery queue. Every search widens the
    # company-to-board map, so coverage compounds rather than staying flat.
    from jobbuddy import company_registry

    company_registry.observe(jobs, scope.get("name", ""))
    if (config.get("sources") or {}).get("discover_ats_boards", True) and not dry_run:
        company_registry.run_discovery(
            limit=(config.get("sources") or {}).get("discovery_limit_per_run", 12),
            scope=scope.get("name", ""))

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
    *,
    tailor_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> RunResult:
    """Search every scope, and optionally tailor the top of the result.

    `tailor_options` is the switch, and its default of None is the contract:
    without it this returns exactly what it always returned, at no cost and
    with no verified profile required. With it, the ranked list continues into
    `tailor_jobs` and the outcome lands on `result.tailoring`.

    Expected keys are `profile` (the verified profile dict, required) plus
    anything `tailor_jobs` takes: `top`, `max_pages`, `max_cost_usd`,
    `strategy_names`, `chat`.
    """
    result = _search_scopes(scopes, config, **kwargs)

    if tailor_options is None or result.dry_run or not result.jobs:
        return result

    options = dict(tailor_options)
    profile = options.pop("profile", None) or {}
    scope_label = (str(scopes[0].get("name") or "run") if len(scopes) == 1
                   else "all-scopes")
    # Reuse the run's own id rather than minting a second timestamp. Left to
    # itself `tailor_jobs` calls `paths.timestamp()`, which formats local time
    # as 2026-07-19_213859 while `write_outputs` uses the UTC run_id
    # 20260719T133510Z -- so a single run wrote ranked.csv into one directory
    # and every resume, report and workbook into another, hours apart by name.
    # One run, one directory.
    options.setdefault("stamp", result.run_id)
    result.tailoring = tailor_jobs(
        result.jobs, profile, scope_label,
        output_dir=kwargs.get("output_dir"), **options)
    return result


def _search_scopes(
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


# --------------------------------------------------------------------------
# The tailoring stage: ranked jobs in, a folder of deliverables out.
#
# Off unless asked for. It is the only part of this module that costs money and
# the only part that needs a verified profile, so it cannot be something a
# caller switches on by accident.
#
# Three properties are load-bearing, each because of a specific way this stage
# fails badly rather than visibly:
#
#   **Failure isolation.** Job seven raising must not lose jobs eight through
#   twenty. Every stage is caught, recorded as `FAILED_AT_<stage>`, and the job
#   continues to the workbook carrying its reason. A job that vanishes silently
#   is the worst outcome available here -- the user reads a workbook of
#   nineteen rows and never learns which one was dropped, or why.
#
#   **A budget ceiling.** Nothing about "tailor the top N" bounds spend when a
#   retry or a long JD triples the token count. Cost is summed from the tailor
#   results and checked between waves; once it is over, the remaining jobs are
#   marked `skipped_budget` and the run states how many were done and how many
#   were not. A partially-complete run that says so beats a surprise bill.
#
#   **Serial writes.** The per-job compute -- the model call, the rules check --
#   runs concurrently, because it is network-bound and there are N of them. It
#   returns data and touches no file. Every write happens in the caller, in
#   ranked order, single-threaded. Same shape as `sources.fetch_all`.
#
# And one hard gate: if `resume_rules.check` returns errors, the render does not
# happen. Errors there are personal-data leaks and hidden text, both harms that
# survive publication. There is deliberately no override.
# --------------------------------------------------------------------------

# Wave size for the concurrent compute phase. Also the budget granularity: the
# ceiling is checked between waves, so a run can overshoot by at most one
# wave's cost. Small enough to bound that, large enough to be worth the pool.
TAILOR_WAVE = 4

# Requirements handed to `tailor()` per job: the posting's own skill tags, which
# is what `experiment.py` already does. One convention, not two.
REQUIREMENTS_PER_JOB = 10


@dataclass
class TailorOutcome:
    """What happened to one job. Every job attempted gets one of these.

    `status` is the contract with the caller and with the workbook:

        ok                  rendered
        blocked             resume_rules returned errors; deliberately not rendered
        skipped_budget      the ceiling was reached before this job was
        FAILED_AT_<stage>   something raised; `reason` names it
    """

    job_key: str
    title: str
    company: str
    status: str
    reason: str = ""
    # The traceback, when something raised. `reason` names the exception but
    # not where it came from, and an intermittent failure cannot be reproduced
    # on demand to recover the rest. Kept off `reason` so the workbook column
    # stays readable.
    detail: str = ""
    cost_usd: float = 0.0
    directory: Path | None = None
    written: list[Path] = field(default_factory=list)
    rules: dict[str, Any] = field(default_factory=dict)
    page_one: dict[str, Any] | None = None
    pages: int | None = None
    degraded: list[str] = field(default_factory=list)

    @property
    def rendered(self) -> bool:
        return self.status == "ok"

    def note(self) -> str:
        """One line for the workbook cell. Never quotes the offending text.

        `resume_rules` withholds matched values by design -- a report that
        echoes the NRIC it found has leaked it -- and this inherits that.
        """
        return f"{self.status}: {self.reason}" if self.reason else self.status


@dataclass
class TailorRun:
    """The stage's whole result. Facts only; the CLI decides how to say them."""

    root: Path | None = None
    outcomes: list[TailorOutcome] = field(default_factory=list)
    workbook: Path | None = None
    written: list[Path] = field(default_factory=list)
    cost_usd: float = 0.0
    max_cost_usd: float = 0.0
    budget_exceeded: bool = False

    @property
    def rendered(self) -> list[TailorOutcome]:
        return [o for o in self.outcomes if o.status == "ok"]

    @property
    def blocked(self) -> list[TailorOutcome]:
        return [o for o in self.outcomes if o.status == "blocked"]

    @property
    def failed(self) -> list[TailorOutcome]:
        return [o for o in self.outcomes if o.status.startswith("FAILED_AT_")]

    @property
    def skipped(self) -> list[TailorOutcome]:
        return [o for o in self.outcomes if o.status == "skipped_budget"]

    def summary(self) -> str:
        """Honest one-liner, including the part that did not happen."""
        parts = [f"{len(self.rendered)} tailored"]
        if self.blocked:
            parts.append(f"{len(self.blocked)} blocked by house rules")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped (budget)")
        line = ", ".join(parts) + f"; ${self.cost_usd:.4f} spent"
        if self.budget_exceeded:
            line += (f" -- stopped at the ${self.max_cost_usd:.2f} ceiling; "
                     "this run is INCOMPLETE")
        return line


def _requirements_for(job: dict[str, Any]) -> list[str]:
    return [str(s) for s in (job.get("skills") or [])][:REQUIREMENTS_PER_JOB]


def _prepare_job(job: dict[str, Any], profile: dict[str, Any],
                 chat: Callable[..., dict[str, Any]] | None,
                 strategy_names: list[str] | None,
                 max_bullets: int) -> dict[str, Any]:
    """The concurrent half: select facts, build the model, check the rules.

    Returns data and nothing else. Opens no file, creates no directory, mutates
    no shared state -- which is what makes running N of these at once safe.
    Every failure is returned rather than raised, so one job cannot take the
    pool down with it.

    Every catch here records `traceback` alongside `error`, without exception.
    Isolation is what keeps one bad job from losing nineteen good ones; it is
    not a licence to throw the evidence away. A report that says only
    "AttributeError: 'NoneType' object has no attribute 'get'" names no file and
    no line, and an intermittent failure cannot be re-run on demand to recover
    what was discarded -- two wrong diagnoses in this codebase were bought
    exactly that way. `error` is the readable one-liner for the workbook;
    `traceback` is the evidence, and it lands on `TailorOutcome.detail`.
    """
    from jobbuddy import render_resume, resume_rules
    from jobbuddy import tailor as tailor_module

    try:
        tailored = tailor_module.tailor(
            profile, job, _requirements_for(job), chat=chat,
            max_bullets=max_bullets, strategy_names=strategy_names)
    except Exception as exc:  # one job's model call must not lose the others
        return {"stage": "tailor", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}

    raw_cost = tailored.get("cost_usd")
    cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else 0.0
    if not tailored.get("ok"):
        return {"stage": "tailor", "cost_usd": cost,
                "error": str(tailored.get("error") or "selection failed")}

    try:
        model = render_resume.build_model(profile, tailored)
    except Exception as exc:
        return {"stage": "build_model", "cost_usd": cost,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}

    try:
        report = resume_rules.check(model)
    except Exception as exc:
        return {"stage": "rules", "cost_usd": cost,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}

    return {"ok": True, "cost_usd": cost, "tailored": tailored,
            "model": model, "report": report}


def _write_job(job: dict[str, Any], prepared: dict[str, Any],
               profile: dict[str, Any], directory: Path, max_pages: int,
               population: list[Any]) -> TailorOutcome:
    """The serial half: render and write. Single-threaded by construction.

    Called only from `tailor_jobs`, one job at a time, because these are the
    only calls in the stage that touch the filesystem.

    Same evidence rule as `_prepare_job`: every catch sets `outcome.detail` to
    the traceback. These are the harder failures of the two to diagnose -- a
    render or a PDF read fails against one specific job's data, on one machine,
    and "KeyError: 'org'" without a frame does not say whose data or which
    layer. `reason` stays the readable workbook line; `detail` carries the
    frames.
    """
    from jobbuddy import render_report, render_resume, resume_rules

    outcome = TailorOutcome(
        job_key=str(job.get("job_key") or ""),
        title=str(job.get("title") or ""),
        company=str(job.get("company") or ""),
        status="ok",
        cost_usd=prepared["cost_usd"],
        directory=directory,
        rules=resume_rules.summarise(prepared["report"]),
    )

    # The gate. Errors here are personal-data leaks and hidden text: harms that
    # cannot be withdrawn once the document is sent. No override exists, and
    # the reason is recorded so the block is never silent.
    report = prepared["report"]
    if report.errors:
        broken = ", ".join(sorted({v.rule for v in report.errors}))
        outcome.status = "blocked"
        outcome.reason = (f"{len(report.errors)} blocking rule violation(s) "
                          f"[{broken}] -- not rendered")
        return outcome

    paths.ensure_dir(directory)

    try:
        rendered = render_resume.render(prepared["model"], directory,
                                        stem="resume", max_pages=max_pages)
    except Exception as exc:
        outcome.status = "FAILED_AT_render_resume"
        outcome.reason = f"{type(exc).__name__}: {exc}"
        outcome.detail = traceback.format_exc()
        return outcome

    outcome.pages = rendered.get("pages")
    outcome.degraded = list(rendered.get("degraded") or [])
    for key in ("pdf", "docx"):
        path = (rendered.get(key) or {}).get("path")
        if path:
            outcome.written.append(Path(path))

    # Page two does not get read. Only answerable against a real PDF, so a
    # degraded render says so rather than reporting a check it never ran.
    pdf_path = (rendered.get("pdf") or {}).get("path")
    if pdf_path and Path(pdf_path).suffix.lower() == ".pdf":
        try:
            outcome.page_one = render_resume.page_one_sufficiency(
                Path(pdf_path), prepared["model"])
        except Exception as exc:
            outcome.status = "FAILED_AT_page_one_sufficiency"
            outcome.reason = f"{type(exc).__name__}: {exc}"
            outcome.detail = traceback.format_exc()
            return outcome
    else:
        outcome.page_one = {"ok": None, "missing": [],
                            "note": "no PDF rendered; page-1 check not run"}

    try:
        analysis = render_report.build_model(
            job, prepared["tailored"], profile, rendered, population)
        written = render_report.render(analysis, directory / "report.pdf")
    except Exception as exc:
        outcome.status = "FAILED_AT_render_report"
        outcome.reason = f"{type(exc).__name__}: {exc}"
        outcome.detail = traceback.format_exc()
        return outcome

    if written.get("path"):
        outcome.written.append(Path(written["path"]))
    outcome.written.extend(Path(p) for p in (written.get("charts") or []))
    if written.get("degraded"):
        outcome.degraded.append(str(written["degraded"]))

    return outcome


def tailor_jobs(jobs: list[dict[str, Any]],
                profile: dict[str, Any],
                scope_label: str = "run",
                *,
                top: int = 5,
                max_pages: int = 1,
                max_cost_usd: float = 1.0,
                strategy_names: list[str] | None = None,
                max_bullets: int = 14,
                output_dir: Path | None = None,
                stamp: str | None = None,
                chat: Callable[..., dict[str, Any]] | None = None,
                wave: int = TAILOR_WAVE) -> TailorRun:
    """Tailor the top `top` jobs, write the deliverables, return what happened.

    Never raises. `jobs` must already be ranked -- this takes the front of the
    list and does not sort, so a caller cannot accidentally tailor an arbitrary
    five and believe they were the best five.

    `chat` is the seam: the only impure dependency besides the filesystem, and
    it is passed in rather than looked up, so the whole stage runs offline in
    tests at no cost.
    """
    stamp = stamp or paths.timestamp()
    root = paths.run_root(scope_label, stamp, output_dir)
    outcome_run = TailorRun(root=root, max_cost_usd=float(max_cost_usd))

    selected = list(jobs or [])[:max(0, int(top))]
    taken: set[str] = set()
    population = [(j.get("scores") or {}).get("adjusted") for j in (jobs or [])]

    index = 0
    while index < len(selected):
        # Checked BEFORE spending, not after. Checking after means the first
        # refusal happens one job past the limit, which is the bug `quota.py`
        # exists to avoid.
        if outcome_run.cost_usd >= outcome_run.max_cost_usd:
            outcome_run.budget_exceeded = True
            break

        batch = selected[index:index + max(1, int(wave))]
        index += len(batch)

        prepared_by_position: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {
                pool.submit(_prepare_job, job, profile, chat, strategy_names,
                            max_bullets): position
                for position, job in enumerate(batch)
            }
            for future in as_completed(futures):
                position = futures[future]
                try:
                    prepared_by_position[position] = future.result()
                except Exception as exc:  # belt and braces; _prepare_job catches
                    # The traceback here is the ONLY evidence available: this
                    # path is reached when something escaped `_prepare_job`
                    # entirely, so by definition nothing inside it recorded a
                    # frame. `format_exc()` is called while handling the
                    # exception, so it formats this one and not some earlier
                    # one on the pool thread.
                    prepared_by_position[position] = {
                        "stage": "tailor",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}

        # Writes happen HERE, serially, in ranked order -- never in the pool.
        for position, job in enumerate(batch):
            prepared = prepared_by_position.get(position) or {
                "stage": "tailor", "error": "no result returned"}
            outcome_run.cost_usd += float(prepared.get("cost_usd") or 0.0)

            if not prepared.get("ok"):
                outcome_run.outcomes.append(TailorOutcome(
                    job_key=str(job.get("job_key") or ""),
                    title=str(job.get("title") or ""),
                    company=str(job.get("company") or ""),
                    status=f"FAILED_AT_{prepared.get('stage') or 'unknown'}",
                    reason=str(prepared.get("error") or ""),
                    detail=str(prepared.get("traceback") or ""),
                    cost_usd=float(prepared.get("cost_usd") or 0.0),
                ))
                continue

            directory = root / paths.job_component(
                paths.job_label(job), root, taken)
            try:
                outcome = _write_job(job, prepared, profile, directory,
                                     max_pages, population)
            except Exception as exc:  # a write failure loses one job, not the run
                outcome = TailorOutcome(
                    job_key=str(job.get("job_key") or ""),
                    title=str(job.get("title") or ""),
                    company=str(job.get("company") or ""),
                    status="FAILED_AT_write",
                    reason=f"{type(exc).__name__}: {exc}",
                    detail=traceback.format_exc(),
                    cost_usd=float(prepared.get("cost_usd") or 0.0),
                    directory=directory,
                )
            outcome_run.outcomes.append(outcome)
            outcome_run.written.extend(outcome.written)

    # Everything the ceiling cost, named. Silence here would read as "the top
    # five were tailored" when two of them never were.
    for job in selected[index:]:
        outcome_run.budget_exceeded = True
        outcome_run.outcomes.append(TailorOutcome(
            job_key=str(job.get("job_key") or ""),
            title=str(job.get("title") or ""),
            company=str(job.get("company") or ""),
            status="skipped_budget",
            reason=(f"${outcome_run.cost_usd:.4f} of the "
                    f"${outcome_run.max_cost_usd:.2f} ceiling was already spent"),
        ))

    outcome_run.workbook = _write_run_workbook(
        jobs or [], outcome_run, scope_label, root)
    if outcome_run.workbook:
        outcome_run.written.append(outcome_run.workbook)
    return outcome_run


def _write_run_workbook(jobs: list[dict[str, Any]], outcome_run: TailorRun,
                        scope_label: str, root: Path) -> Path | None:
    """One workbook for the whole run, every job in it, tailoring status and all.

    Every attempted job appears with its outcome -- including the ones that
    failed and the ones the budget cut. A workbook listing only the successes
    lies by omission, and the omission is invisible.
    """
    from jobbuddy import render_excel

    by_key = {o.job_key: o for o in outcome_run.outcomes if o.job_key}
    for job in jobs:
        outcome = by_key.get(str(job.get("job_key") or ""))
        if outcome is not None:
            job["tailoring"] = outcome.note()

    out_path = root / "ranked.xlsx"
    try:
        result = render_excel.write_workbook({scope_label: jobs}, out_path)
    except (OSError, ValueError) as exc:
        job_store._warn(f"could not write the run workbook ({exc})")
        return None
    if not result.get("ok"):
        # `write_workbook` reports partial writes rather than raising. Returning
        # a path for a scope that never reached the disk would have the run
        # summary point the user at a file that is not there.
        failed = ", ".join(sorted(result.get("failed") or {})) or "unknown"
        job_store._warn(f"the run workbook is incomplete -- scope(s) not "
                        f"written: {failed}")
        return None
    return Path(result.get("path") or out_path)
