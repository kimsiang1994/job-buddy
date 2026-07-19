"""Workable's global job search. Keyless, one endpoint, no per-company setup.

    GET https://jobs.workable.com/api/v1/jobs?query=...&location=Singapore

Workable hosts careers pages for thousands of companies and exposes a single
search across all of them. That makes it the cheapest breadth win available:
one adapter, no board tokens to discover, and it reaches employers who never
post to MyCareersFuture -- which matters more than it sounds, because MCF's
advertising mandate only binds when a foreign work pass is involved and exempts
anything paying over S$22,500/month.

What it does NOT carry, and MCF does:

    salary          absent entirely -- Workable does not require it
    applications    no equivalent of MCF's totalNumberJobApplication

So `comp_signal` and `competition` score None for these jobs and their weight
leaves the denominator. That is the correct behaviour, not a degradation: a job
with no salary data should not be ranked as though its pay were average.

Pagination is cursor-based via `nextPageToken`, not offsets.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Iterator

from jobbuddy import html_text, job_schema, net

BASE = "https://jobs.workable.com/api/v1/jobs"
SOURCE = "workable"

# Observed page size; the endpoint does not document a limit parameter.
PAGE_HINT = 100


def search(
    query: str,
    location: str = "Singapore",
    max_results: int = 100,
    cache_ttl_s: float = 900.0,
) -> Iterator[dict[str, Any]]:
    """Yield raw Workable job records, following the cursor until exhausted."""
    fetched = 0
    token: str | None = None

    while fetched < max_results:
        params = {"query": query, "location": location}
        if token:
            params["pageToken"] = token
        url = f"{BASE}?{urllib.parse.urlencode(params)}"

        data, result = net.get_json(url, cache_ttl_s=cache_ttl_s)
        if not result.ok or not isinstance(data, dict):
            net._warn(f"workable: {query!r} failed ({result.error or 'bad payload'})")
            return

        jobs = data.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            return

        for record in jobs:
            yield record
            fetched += 1
            if fetched >= max_results:
                return

        token = data.get("nextPageToken")
        if not token:
            return


def _location(record: dict[str, Any]) -> tuple[str, bool]:
    """(human label, is_overseas)."""
    loc = record.get("location") or {}
    parts = [loc.get("city"), loc.get("subregion"), loc.get("countryName")]
    label = ", ".join(dict.fromkeys(p for p in parts if p)) or "Unknown"
    country = job_schema.norm_text(loc.get("countryName")).lower()
    # Empty country is ambiguous; the search was location-scoped, so trust it
    # rather than dropping the job on a missing field.
    return label, bool(country) and country != "singapore"


def to_job(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map one Workable record onto the canonical Job. None if unusable."""
    job_id = job_schema.norm_text(record.get("id"))
    if not job_id:
        return None

    job = job_schema.new_job(SOURCE, job_id)
    company = record.get("company") or {}

    job["title"] = job_schema.norm_text(record.get("title"))
    job["company"] = job_schema.norm_text(company.get("title"))
    job["url"] = job_schema.norm_text(record.get("url"))
    job["is_agency"] = job_schema.looks_like_agency(job["company"])

    html = record.get("description") or ""
    job["jd_html"] = html or None
    # requirementsSection carries the "must have" list on some postings and is
    # where the skill terms live; losing it would gut the skill match.
    extra = " ".join(str(record.get(k) or "") for k in
                     ("requirementsSection", "benefitsSection"))
    job["jd_text"] = job_schema.norm_text(html_text.flatten_html(html + " " + extra))

    job["location"], job["is_overseas"] = _location(record)

    workplace = job_schema.norm_text(record.get("workplace")).lower()
    if workplace:
        job["is_remote"] = workplace == "remote"

    employment = job_schema.norm_text(record.get("employmentType"))
    job["employment_types"] = [employment] if employment else []

    job["posted_at"] = job_schema.parse_date(record.get("created"))
    job["source_status"] = job_schema.norm_text(record.get("state")) or None
    job["is_open"] = job["source_status"].lower() != "closed" if job["source_status"] else True
    job["liveness"] = "ALIVE" if job["is_open"] else "LIKELY_DEAD"

    department = job_schema.norm_text(record.get("department"))
    job["categories"] = [department] if department else []

    # Workable publishes no structured skills. The scorer reads skills_raw, so
    # leaving it empty makes skill_match return None rather than a false zero --
    # which is why the JD text above matters as the fallback signal.
    job["skills_raw"] = []

    job["_provenance"] = {
        "source": SOURCE,
        "fetched_at": job["_normalised_at"],
        "company_website": company.get("website"),
        "salary": "absent -- Workable does not publish it",
        "applications": "absent",
    }

    return job_schema.finalise(job)


def company_website(record: dict[str, Any]) -> str | None:
    """The employer's own site, used by ATS board discovery.

    This is the quiet reason Workable is worth adapting first: every result
    hands you a company website, which is the seed the Greenhouse/Lever/Ashby
    discovery needs and which no other free source provides in bulk.
    """
    site = ((record.get("company") or {}).get("website") or "").strip()
    return site or None


def fetch_jobs(
    query: str,
    max_results: int = 100,
    singapore_only: bool = True,
    open_only: bool = True,
    cache_ttl_s: float = 900.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Search, normalise and filter. Same contract as source_mcf.fetch_jobs."""
    counters = {"fetched": 0, "unusable": 0, "dropped_overseas": 0,
                "dropped_closed": 0, "invalid": 0, "kept": 0}
    jobs: list[dict[str, Any]] = []
    websites: dict[str, str] = {}

    location = "Singapore" if singapore_only else ""
    for record in search(query, location=location, max_results=max_results,
                         cache_ttl_s=cache_ttl_s):
        counters["fetched"] += 1
        job = to_job(record)
        if job is None:
            counters["unusable"] += 1
            continue
        if singapore_only and job["is_overseas"]:
            counters["dropped_overseas"] += 1
            continue
        if open_only and not job["is_open"]:
            counters["dropped_closed"] += 1
            continue
        problems = job_schema.validate_job(job)
        if problems:
            counters["invalid"] += 1
            net._warn(f"workable: {job['job_key']} rejected -- {'; '.join(problems)}")
            continue

        site = company_website(record)
        if site:
            websites[job["company_norm"]] = site
        jobs.append(job)
        counters["kept"] += 1

    if websites:
        counters["company_sites"] = len(websites)
        _remember_websites(websites)

    return jobs, counters


# Company website cache, feeding ATS discovery. Written here because this is
# the only source that hands them out; read by source_ats.
_WEBSITE_CACHE: dict[str, str] = {}


def _remember_websites(sites: dict[str, str]) -> None:
    _WEBSITE_CACHE.update(sites)


def known_company_websites() -> dict[str, str]:
    """company_norm -> website, from everything seen this process."""
    return dict(_WEBSITE_CACHE)
