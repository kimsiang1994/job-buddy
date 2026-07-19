"""The source registry. One call fans out across every configured adapter.

Adding a source is adding a row to SOURCES. Everything downstream -- filtering,
dedupe, scoring, history -- already works on the canonical Job shape, so a new
board does not touch the pipeline.

Sources differ in what they can tell you, and the scorer is built for that:
a component with no data returns None and its weight leaves the denominator.

    source        salary    applications   skills    reaches
    ----------------------------------------------------------------------
    mcf           always    YES (real)     yes       SG middle market
    workable      no        no             no        employers off MCF
    ats           Ashby     no             no        direct, freshest
                  only
    hn            no        no             no        senior remote
    aggregator    sometimes no             no        LinkedIn, Indeed,
                                                     Glassdoor, JobStreet

MyCareersFuture is the only one publishing a real application count, which is
why competition scoring degrades to an age proxy elsewhere. It is also the only
one that mandates salary -- and it covers the middle of the market rather than
the top, because MOM exempts roles paying over S$22,500/month from the
advertising requirement entirely.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from jobbuddy import (
    source_aggregator,
    source_ats,
    source_hn,
    source_mcf,
    source_workable,
)


class FetchJobs(Protocol):
    def __call__(self, query: str, max_results: int = ..., singapore_only: bool = ...,
                 open_only: bool = ..., cache_ttl_s: float = ...
                 ) -> tuple[list[dict[str, Any]], dict[str, int]]: ...


@dataclass
class Source:
    """One adapter, plus what a caller needs to know before relying on it."""

    name: str
    fetch: FetchJobs
    enabled_by_default: bool = True
    needs_key: bool = False
    costs_money: bool = False
    note: str = ""

    def available(self) -> tuple[bool, str]:
        """(usable now, why not). Never raises."""
        if self.name == "aggregator":
            configured = source_aggregator.available()
            if not any(configured.values()):
                return False, "no JSEARCH_API_KEY or ADZUNA_APP_ID/KEY in .env"
            return True, ", ".join(k for k, v in configured.items() if v)
        if self.name == "ats":
            boards = [b for b in source_ats.load_boards().values() if b.get("vendor")]
            if not boards:
                return False, "no boards discovered yet -- run discovery first"
            return True, f"{len(boards)} board(s)"
        return True, "keyless"


SOURCES: list[Source] = [
    Source("mcf", source_mcf.fetch_jobs,
           note="MyCareersFuture -- real application counts, mandatory salary"),
    Source("workable", source_workable.fetch_jobs,
           note="Workable global search -- breadth, no salary"),
    Source("ats", source_ats.fetch_jobs,
           note="direct employer boards -- freshest, needs discovery"),
    Source("hn", source_hn.fetch_jobs, enabled_by_default=False,
           note="HN Who is Hiring -- senior remote, monthly thread"),
    Source("aggregator", source_aggregator.fetch_jobs, enabled_by_default=False,
           needs_key=True, costs_money=True,
           note="JSearch/Adzuna -- reaches LinkedIn, Indeed, Glassdoor, JobStreet"),
]

BY_NAME = {s.name: s for s in SOURCES}


def status() -> list[dict[str, Any]]:
    """What each source can do right now. For the notebook and --sources."""
    rows = []
    for source in SOURCES:
        ok, why = source.available()
        rows.append({
            "source": source.name, "available": ok, "detail": why,
            "default_on": source.enabled_by_default,
            "costs_money": source.costs_money, "note": source.note,
        })
    return rows


def fetch_all(
    query: str,
    max_results_per_source: int = 60,
    singapore_only: bool = True,
    open_only: bool = True,
    cache_ttl_s: float = 900.0,
    enabled: list[str] | None = None,
    max_workers: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Query every enabled source concurrently. Returns (jobs, per-source counters).

    One source failing must not lose the others -- a board going down should
    cost you that board's listings, not the run. Each adapter is therefore
    isolated, and its exception recorded as a counter rather than raised.
    """
    if enabled is None:
        chosen = [s for s in SOURCES if s.enabled_by_default]
    else:
        chosen = [BY_NAME[n] for n in enabled if n in BY_NAME]

    jobs: list[dict[str, Any]] = []
    counters: dict[str, dict[str, int]] = {}

    def run(source: Source) -> tuple[str, list[dict], dict[str, int]]:
        ok, why = source.available()
        if not ok:
            return source.name, [], {"skipped": 1, "reason_unavailable": 1}
        try:
            found, count = source.fetch(
                query, max_results=max_results_per_source,
                singapore_only=singapore_only, open_only=open_only,
                cache_ttl_s=cache_ttl_s,
            )
            return source.name, found, count
        except Exception as exc:  # one bad source must not lose the run
            return source.name, [], {"error": 1, "message": str(exc)[:120]}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run, s): s for s in chosen}
        for future in as_completed(futures):
            name, found, count = future.result()
            counters[name] = count
            for job in found:
                job["_source_adapter"] = name
            jobs.extend(found)

    return jobs, counters


def discover_boards_from(jobs: list[dict[str, Any]], limit: int = 20) -> int:
    """Widen the ATS registry using company websites seen this run.

    Workable hands out a company website with every result, which is the seed
    the board lookup needs and which no other free source provides in bulk. So
    the more you search, the more direct boards you gain -- coverage compounds
    instead of staying flat.
    """
    websites = source_workable.known_company_websites()
    if not websites:
        return 0
    before = len([b for b in source_ats.load_boards().values() if b.get("vendor")])
    source_ats.discover_from_websites(websites, limit=limit)
    after = len([b for b in source_ats.load_boards().values() if b.get("vendor")])
    return after - before
