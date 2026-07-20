"""MyCareersFuture -- the primary source.

The best single source for Singapore, but NOT the complete market -- a belief
this module was originally written under, and which is wrong in a way that
matters.

The Fair Consideration Framework is a precondition on *foreign work-pass
applications*, not a general posting mandate. A role the employer expects to
fill with a Singaporean or PR need never appear here, at any salary. And MOM
exempts jobs paying >= S$22,500/month fixed, and companies with fewer than 10
employees -- so the senior band and the startups are both systematically
under-represented. Treat MCF as broad coverage of the middle of the market, and
lean on the ATS adapters for the top of it.

What MCF does carry that no other board publishes:

  - salary, structured and mandatory (legally required, not optional metadata)
  - `totalNumberJobApplication`  -- the REAL number of applications submitted
  - `totalNumberOfView`          -- so you get a conversion ratio too
  - `repostCount` / `editCount`  -- churn signals, given not derived
  - `ssocCode`                   -- joins to MOM wage tables
  - `uen`                        -- joins to the ACRA company registry

The API is undocumented (it is the site's own frontend API) and unversioned in
any contract sense, so everything here is defensive: shapes are probed, not
assumed, and a field that moves degrades that one job rather than the run.

robots.txt at www.mycareersfuture.gov.sg is `Disallow:` (empty) -- nothing is
disallowed. We still throttle in net.py; it is a public service.
"""

from __future__ import annotations

import urllib.parse
from datetime import date
from typing import Any, Iterator

from jobbuddy import job_schema
from jobbuddy import net
from jobbuddy import html_text

BASE = "https://api.mycareersfuture.gov.sg/v2/jobs"
JOB_URL = "https://www.mycareersfuture.gov.sg/job/{uuid}"

# Verified live: limit=100 returns 100, limit=200 returns HTTP 400.
MAX_LIMIT = 100

# status.jobStatus values that mean "you can still apply".
OPEN_STATUSES = frozenset({"open", "re-open", "reopen"})


def _get(record: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Walk a nested dict without raising when a level is missing or None."""
    node: Any = record
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def search_page(
    query: str,
    page: int = 0,
    limit: int = MAX_LIMIT,
    cache_ttl_s: float = 900.0,
) -> tuple[list[dict[str, Any]], int, net.FetchResult]:
    """Fetch one page. Returns (raw_records, total_available, fetch_result)."""
    limit = max(1, min(int(limit), MAX_LIMIT))
    url = (
        f"{BASE}?search={urllib.parse.quote(query)}"
        f"&limit={limit}&page={int(page)}"
    )
    data, result = net.get_json(url, cache_ttl_s=cache_ttl_s)
    if data is None:
        return [], 0, result
    if not isinstance(data, dict):
        return [], 0, result
    results = data.get("results")
    if not isinstance(results, list):
        return [], 0, result
    total = data.get("total")
    return results, int(total) if isinstance(total, int) else 0, result


def search(
    query: str,
    max_results: int = 100,
    cache_ttl_s: float = 900.0,
) -> Iterator[dict[str, Any]]:
    """Yield raw MCF records for a query, paginating until max_results.

    Yields the vendor shape verbatim -- normalisation happens in `to_job`. That
    separation is what lets a captured fixture replay exactly.
    """
    fetched = 0
    page = 0
    while fetched < max_results:
        want = min(MAX_LIMIT, max_results - fetched)
        records, total, result = search_page(query, page, want, cache_ttl_s)
        if not result.ok:
            net._warn(f"mcf: page {page} of {query!r} failed ({result.error})")
            return
        if not records:
            return
        for record in records:
            yield record
            fetched += 1
            if fetched >= max_results:
                return
        page += 1
        if total and fetched >= total:
            return


def is_open(record: dict[str, Any]) -> bool:
    """True when the posting still accepts applications.

    Checks three independent things, because any one of them can be the reason
    a job is not applicable: the status label, a soft-delete, and the expiry
    date. MCF leaves expired postings in the index.
    """
    status = job_schema.norm_text(_get(record, "status", "jobStatus")).lower()
    if status and status not in OPEN_STATUSES:
        return False
    if _get(record, "metadata", "deletedAt"):
        return False
    expiry = job_schema.parse_date(_get(record, "metadata", "expiryDate"))
    if expiry:
        try:
            if date.fromisoformat(expiry) < date.today():
                return False
        except ValueError:
            pass  # unparseable expiry is not evidence of closure
    return True


def is_singapore(record: dict[str, Any]) -> bool:
    """True when the role is located in Singapore.

    MCF is the SG portal but does carry overseas postings, flagged on the
    address object.
    """
    if _get(record, "address", "isOverseas") is True:
        return False
    if job_schema.norm_text(_get(record, "address", "overseasCountry")):
        return False
    return True


def _location(record: dict[str, Any]) -> str:
    """Human-readable location, preferring the specific over the generic."""
    address = record.get("address") or {}
    parts = [
        job_schema.norm_text(address.get("building")),
        job_schema.norm_text(address.get("street")),
    ]
    districts = address.get("districts") or []
    if isinstance(districts, list):
        for district in districts:
            if isinstance(district, dict):
                label = job_schema.norm_text(district.get("location"))
                if label:
                    parts.append(label)
                    break
    located = ", ".join(p for p in parts if p)
    return located or "Singapore"


def to_job(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map one raw MCF record onto the canonical Job. None if unusable."""
    uuid = job_schema.norm_text(record.get("uuid"))
    if not uuid:
        return None

    job = job_schema.new_job("mcf", uuid)
    meta = record.get("metadata") or {}
    posted = record.get("postedCompany") or {}
    hiring = record.get("hiringCompany") or {}

    job["url"] = job_schema.norm_text(meta.get("jobDetailsUrl")) or JOB_URL.format(uuid=uuid)
    job["api_url"] = f"{BASE}/{uuid}"
    job["title"] = job_schema.norm_text(record.get("title"))

    # hiringCompany is the real employer when an agency posts on behalf; it is
    # frequently null (and can be deliberately hidden), so postedCompany is the
    # fallback rather than the other way round.
    hiring_name = job_schema.norm_text(hiring.get("name"))
    posted_name = job_schema.norm_text(posted.get("name"))
    job["company"] = hiring_name or posted_name
    job["company_uen"] = job_schema.norm_text(hiring.get("uen") or posted.get("uen")) or None
    job["is_agency"] = job_schema.looks_like_agency(
        posted_name, bool(meta.get("isPostedOnBehalf"))
    )

    html = record.get("description")
    job["jd_html"] = html if isinstance(html, str) else None
    try:
        job["jd_text"] = job_schema.norm_jd_text(html_text.flatten_html(html or ""))
    except Exception as exc:  # a malformed JD must not kill the record
        net._warn(f"mcf: could not flatten description for {uuid} ({exc})")
        job["jd_text"] = job_schema.norm_jd_text(html or "")

    job["location"] = _location(record)
    job["is_overseas"] = not is_singapore(record)

    arrangements = record.get("flexibleWorkArrangements") or []
    if isinstance(arrangements, list):
        blob = " ".join(
            job_schema.norm_text(a.get("flexibleWorkArrangement") if isinstance(a, dict) else a)
            for a in arrangements
        ).lower()
        if blob:
            job["is_remote"] = "work from home" in blob or "telecommut" in blob or "remote" in blob

    employment = record.get("employmentTypes") or []
    if isinstance(employment, list):
        job["employment_types"] = [
            job_schema.norm_text(e.get("employmentType") if isinstance(e, dict) else e)
            for e in employment
        ]

    levels = record.get("positionLevels") or []
    if isinstance(levels, list) and levels:
        first = levels[0]
        job["seniority_raw"] = job_schema.norm_text(
            first.get("position") if isinstance(first, dict) else first
        )

    years = record.get("minimumYearsExperience")
    job["min_years_exp"] = years if isinstance(years, int) and years >= 0 else None

    salary = record.get("salary") or {}
    period = _get(salary, "type", "salaryType", default="Monthly")
    job["salary_period_raw"] = job_schema.norm_text(period) or None
    job["salary_min_sgd"] = job_schema.to_monthly_sgd(salary.get("minimum"), period)
    job["salary_max_sgd"] = job_schema.to_monthly_sgd(salary.get("maximum"), period)
    job["salary_is_stated"] = (
        not bool(meta.get("isHideSalary"))
        and job["salary_min_sgd"] is not None
    )

    skills = record.get("skills") or []
    if isinstance(skills, list):
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            name = job_schema.norm_text(skill.get("skill"))
            if not name:
                continue
            job["skills_raw"].append(name)
            if skill.get("isKeySkill"):
                job["skills_key"].append(name)

    categories = record.get("categories") or []
    if isinstance(categories, list):
        job["categories"] = [
            job_schema.norm_text(c.get("category") if isinstance(c, dict) else c)
            for c in categories
        ]

    job["ssoc_code"] = job_schema.norm_text(record.get("ssocCode")) or None

    # originalPostingDate is the true first publication; newPostingDate resets
    # on a repost, so using it would make an old role look fresh.
    job["posted_at"] = (
        job_schema.parse_date(meta.get("originalPostingDate"))
        or job_schema.parse_date(meta.get("newPostingDate"))
        or job_schema.parse_date(meta.get("createdAt"))
    )
    job["expires_at"] = job_schema.parse_date(meta.get("expiryDate"))
    job["source_status"] = job_schema.norm_text(_get(record, "status", "jobStatus")) or None
    job["is_open"] = is_open(record)
    job["liveness"] = "ALIVE" if job["is_open"] else "LIKELY_DEAD"

    applications = meta.get("totalNumberJobApplication")
    views = meta.get("totalNumberOfView")
    job["applications"] = applications if isinstance(applications, int) else None
    job["views"] = views if isinstance(views, int) else None
    reposts = meta.get("repostCount")
    job["repost_count"] = reposts if isinstance(reposts, int) else 0
    edits = meta.get("editCount")
    job["edit_count"] = edits if isinstance(edits, int) else None
    vacancies = record.get("numberOfVacancies")
    job["vacancies"] = vacancies if isinstance(vacancies, int) else None

    job["_provenance"] = {
        "source": "mcf",
        "job_post_id": job_schema.norm_text(meta.get("jobPostId")) or None,
        "fetched_at": job["_normalised_at"],
        "salary": (
            "period_corrected" if job_schema.salary_was_adjusted(salary.get("maximum"), period)
            # An unrecognised period is inferred from magnitude. Defensible, but
            # it feeds the pay score, so a reader must be able to tell an
            # inferred figure from a stated one.
            else "period_guessed" if job_schema.salary_period_was_guessed(
                salary.get("maximum"), period)
            else "asserted" if job["salary_is_stated"] else "hidden"
        ),
        "applications": "asserted" if job["applications"] is not None else "absent",
        "company": "hiring" if hiring_name else "posted",
    }

    return job_schema.finalise(job)


def fetch_jobs(
    query: str,
    max_results: int = 100,
    singapore_only: bool = True,
    open_only: bool = True,
    cache_ttl_s: float = 900.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Search, normalise and filter. Returns (jobs, counters).

    The two filters are applied here, before anything downstream spends money
    on a job that is closed or in another country.
    """
    counters = {
        "fetched": 0, "unusable": 0, "dropped_overseas": 0,
        "dropped_closed": 0, "invalid": 0, "kept": 0,
    }
    jobs: list[dict[str, Any]] = []

    for record in search(query, max_results=max_results, cache_ttl_s=cache_ttl_s):
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
            net._warn(f"mcf: {job['job_key']} rejected -- {'; '.join(problems)}")
            continue
        job["scope"] = query
        jobs.append(job)
        counters["kept"] += 1

    return jobs, counters
