"""Commercial job-data APIs. The legitimate route to the walled sites.

Glints, NodeFlair, JobStreet, FastJobs, Indeed and Glassdoor all serve a
Cloudflare challenge or CAPTCHA to a real browser -- measured, not assumed.
Getting past that means fingerprint spoofing, residential proxy rotation or
CAPTCHA solving, none of which belongs in a personal tool.

Buying the data is the honest workaround, and it is cheap:

    JSearch     $25/mo for 10,000 requests, or 200/mo free.
                Singapore explicitly supported. Carries LinkedIn, Indeed,
                Glassdoor and ZipRecruiter listings -- sourced from Google for
                Jobs, so the provenance chain is publishers submitting
                structured JobPosting data rather than a scraper hitting
                LinkedIn.

    Adzuna      2,500 hits/mo free, `sg` supported. No LinkedIn.
                Attribution is MANDATORY under their terms -- a "Jobs by
                Adzuna" badge linking back -- and the data may not be
                redistributed. Fine for personal search; not for a product.

Both are optional. With no keys configured this module reports itself
unavailable and the pipeline runs on the free sources, rather than failing.

Configure in .env:

    JSEARCH_API_KEY=...
    ADZUNA_APP_ID=...
    ADZUNA_APP_KEY=...
"""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

from jobbuddy import html_text, job_schema, net, quota

# Endpoint and header taken from OpenWeb Ninja's own snippet. The path is
# /jsearch/search-v2 with an `X-API-Key` header -- not the /v1/ path or the
# lowercase header this originally guessed at.
JSEARCH_URL = "https://api.openwebninja.com/jsearch/search-v2"
JSEARCH_RAPIDAPI_URL = "https://jsearch.p.rapidapi.com/search"
ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/sg/search"

# Adzuna's terms require this wherever their data is shown. Surfaced as a
# constant so it cannot be quietly dropped when the output format changes.
ADZUNA_ATTRIBUTION = "Jobs by Adzuna — https://www.adzuna.sg/"


def available() -> dict[str, bool]:
    """Which paid sources are configured."""
    return {
        "jsearch": bool(os.environ.get("JSEARCH_API_KEY", "").strip()),
        "adzuna": bool(os.environ.get("ADZUNA_APP_ID", "").strip()
                       and os.environ.get("ADZUNA_APP_KEY", "").strip()),
    }


# --------------------------------------------------------------------------
# JSearch
# --------------------------------------------------------------------------

def _jsearch_records(query: str, pages: int, cache_ttl_s: float) -> list[dict[str, Any]]:
    key = os.environ.get("JSEARCH_API_KEY", "").strip()
    if not key:
        return []

    records: list[dict[str, Any]] = []
    for page in range(1, max(1, pages) + 1):
        params = urllib.parse.urlencode({
            "query": f"{query} in Singapore",
            "page": page,
            "num_pages": 1,
            "country": "sg",
            "date_posted": "month",
        })
        # Route by key PREFIX, not length. OpenWeb Ninja issues `ak_...` keys
        # and RapidAPI issues bare hex; a length test sent a 50-character
        # `ak_` key to RapidAPI, which answered 403.
        if not key.startswith("ak_") and len(key) > 45:
            url, headers = (f"{JSEARCH_RAPIDAPI_URL}?{params}",
                            {"x-rapidapi-key": key,
                             "x-rapidapi-host": "jsearch.p.rapidapi.com"})
        else:
            url, headers = f"{JSEARCH_URL}?{params}", {"X-API-Key": key}

        if not quota.can_spend("jsearch"):
            net._warn(f"jsearch: monthly budget spent "
                      f"({quota.used('jsearch')}/{quota.limit_for('jsearch')}); skipping")
            break
        data, result = net.get_json(url, headers=headers, cache_ttl_s=cache_ttl_s)
        if not result.from_cache:
            quota.spend("jsearch")
        if not result.ok or not isinstance(data, dict):
            net._warn(f"jsearch: page {page} failed ({result.error})")
            break
        # search-v2 nests results under data.jobs and paginates by cursor.
        # The v1 shape put them directly in `data`, so both are accepted --
        # reading the wrong one yielded a list of strings and an AttributeError
        # deep in the mapper rather than an obvious failure at the boundary.
        payload = data.get("data")
        if isinstance(payload, dict):
            batch = payload.get("jobs") or []
        elif isinstance(payload, list):
            batch = payload
        else:
            batch = []
        batch = [r for r in batch if isinstance(r, dict)]
        if not batch:
            break
        records.extend(batch)
    return records


def _jsearch_to_job(record: dict[str, Any]) -> dict[str, Any] | None:
    job_id = job_schema.norm_text(record.get("job_id"))
    if not job_id:
        return None

    job = job_schema.new_job("jsearch", job_id)
    job["title"] = job_schema.norm_text(record.get("job_title"))
    job["company"] = job_schema.norm_text(record.get("employer_name"))
    job["url"] = job_schema.norm_text(
        record.get("job_apply_link") or record.get("job_google_link"))
    job["is_agency"] = job_schema.looks_like_agency(job["company"])

    description = record.get("job_description") or ""
    job["jd_html"] = None
    job["jd_text"] = job_schema.norm_jd_text(html_text.flatten_html(description))

    city = job_schema.norm_text(record.get("job_city"))
    country = job_schema.norm_text(record.get("job_country"))
    job["location"] = ", ".join(p for p in (city, country) if p) or "Singapore"
    job["is_overseas"] = bool(country) and country.lower() not in ("sg", "singapore")
    job["is_remote"] = bool(record.get("job_is_remote"))

    employment = job_schema.norm_text(record.get("job_employment_type"))
    job["employment_types"] = [employment] if employment else []
    job["posted_at"] = job_schema.parse_date(record.get("job_posted_at_datetime_utc"))
    job["expires_at"] = job_schema.parse_date(record.get("job_offer_expiration_datetime_utc"))

    # JSearch sometimes carries a salary range; the period varies by publisher.
    lo, hi = record.get("job_min_salary"), record.get("job_max_salary")
    period = job_schema.norm_text(record.get("job_salary_period")) or "Yearly"
    if lo or hi:
        job["salary_min_sgd"] = job_schema.to_monthly_sgd(lo, period)
        job["salary_max_sgd"] = job_schema.to_monthly_sgd(hi, period)
        job["salary_is_stated"] = job["salary_min_sgd"] is not None
        job["salary_period_raw"] = f"{period} (JSearch)"

    job["is_open"] = True
    job["liveness"] = "ALIVE"
    job["_provenance"] = {
        "source": "jsearch",
        "publisher": job_schema.norm_text(record.get("job_publisher")),
        "fetched_at": job["_normalised_at"],
        "note": "aggregated via Google for Jobs; publisher names the original board",
        "applications": "absent",
    }
    return job_schema.finalise(job)


# --------------------------------------------------------------------------
# Adzuna
# --------------------------------------------------------------------------

def _adzuna_records(query: str, pages: int, cache_ttl_s: float) -> list[dict[str, Any]]:
    app_id = os.environ.get("ADZUNA_APP_ID", "").strip()
    app_key = os.environ.get("ADZUNA_APP_KEY", "").strip()
    if not (app_id and app_key):
        return []

    records: list[dict[str, Any]] = []
    for page in range(1, max(1, pages) + 1):
        params = urllib.parse.urlencode({
            "app_id": app_id, "app_key": app_key,
            "what": query, "results_per_page": 50,
            "content-type": "application/json",
        })
        if not quota.can_spend("adzuna"):
            net._warn("adzuna: monthly budget spent; skipping")
            break
        data, result = net.get_json(f"{ADZUNA_URL}/{page}?{params}",
                                    cache_ttl_s=cache_ttl_s)
        if not result.from_cache:
            quota.spend("adzuna")
        if not result.ok or not isinstance(data, dict):
            net._warn(f"adzuna: page {page} failed ({result.error})")
            break
        batch = data.get("results") or []
        if not batch:
            break
        records.extend(batch)
    return records


def _adzuna_to_job(record: dict[str, Any]) -> dict[str, Any] | None:
    job_id = job_schema.norm_text(record.get("id"))
    if not job_id:
        return None

    job = job_schema.new_job("adzuna", job_id)
    job["title"] = job_schema.norm_text(record.get("title"))
    job["company"] = job_schema.norm_text((record.get("company") or {}).get("display_name"))
    job["url"] = job_schema.norm_text(record.get("redirect_url"))
    job["is_agency"] = job_schema.looks_like_agency(job["company"])
    job["jd_text"] = job_schema.norm_jd_text(html_text.flatten_html(record.get("description") or ""))
    job["location"] = job_schema.norm_text(
        (record.get("location") or {}).get("display_name")) or "Singapore"
    job["is_overseas"] = "singapore" not in job["location"].lower()

    # Adzuna quotes annual SGD; the plausibility guard catches the conversion.
    lo, hi = record.get("salary_min"), record.get("salary_max")
    if lo or hi:
        job["salary_min_sgd"] = job_schema.to_monthly_sgd(lo, "Annually")
        job["salary_max_sgd"] = job_schema.to_monthly_sgd(hi, "Annually")
        # `salary_is_predicted == "1"` means Adzuna estimated it, not the
        # employer. Presenting a guess as a stated range would be a lie.
        predicted = str(record.get("salary_is_predicted", "0")) == "1"
        job["salary_is_stated"] = not predicted and job["salary_min_sgd"] is not None
        job["salary_period_raw"] = "Annually (Adzuna, predicted)" if predicted else "Annually (Adzuna)"

    job["posted_at"] = job_schema.parse_date(record.get("created"))
    job["categories"] = [job_schema.norm_text((record.get("category") or {}).get("label"))]
    job["is_open"] = True
    job["liveness"] = "ALIVE"
    job["_provenance"] = {
        "source": "adzuna", "fetched_at": job["_normalised_at"],
        "attribution_required": ADZUNA_ATTRIBUTION,
        "terms": "personal use only; redistribution prohibited without licence",
        "applications": "absent",
    }
    return job_schema.finalise(job)


# --------------------------------------------------------------------------
# Common entry point
# --------------------------------------------------------------------------

def fetch_jobs(
    query: str,
    max_results: int = 100,
    singapore_only: bool = True,
    open_only: bool = True,
    cache_ttl_s: float = 3600.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pull from whichever paid sources are configured. Same contract as the rest."""
    counters = {"fetched": 0, "unusable": 0, "dropped_overseas": 0,
                "invalid": 0, "kept": 0}
    configured = available()
    if not any(configured.values()):
        counters["skipped_no_key"] = 1
        return [], counters

    pages = max(1, min(3, (max_results // 40) + 1))
    records: list[tuple[str, dict[str, Any]]] = []
    if configured["jsearch"]:
        records += [("jsearch", r) for r in _jsearch_records(query, pages, cache_ttl_s)]
    if configured["adzuna"]:
        records += [("adzuna", r) for r in _adzuna_records(query, pages, cache_ttl_s)]

    mappers = {"jsearch": _jsearch_to_job, "adzuna": _adzuna_to_job}
    jobs: list[dict[str, Any]] = []
    for vendor, record in records:
        counters["fetched"] += 1
        job = mappers[vendor](record)
        if job is None:
            counters["unusable"] += 1
            continue
        if singapore_only and job["is_overseas"]:
            counters["dropped_overseas"] += 1
            continue
        problems = job_schema.validate_job(job)
        if problems:
            # The counter alone says how many were dropped but not why, which
            # is the half of the answer you need when a vendor renames a field
            # and every record starts failing on the same missing key.
            counters["invalid"] += 1
            net._warn(f"{vendor}: {job['job_key']} rejected -- {'; '.join(problems)}")
            continue
        job["scope"] = query
        jobs.append(job)
        counters["kept"] += 1
        if len(jobs) >= max_results:
            break

    return jobs, counters


def attribution_notices(jobs: list[dict[str, Any]]) -> list[str]:
    """Attribution any displayed result set is contractually required to carry.

    Adzuna's terms make the badge mandatory. Returning it from the data layer
    means the obligation travels with the jobs instead of living in someone's
    memory.
    """
    notices = []
    if any(j.get("source") == "adzuna" for j in jobs):
        notices.append(ADZUNA_ATTRIBUTION)
    return notices
