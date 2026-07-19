"""Careerjet and Jooble: documented partner APIs, free keys, SG coverage.

Both are aggregators that index many boards, and both run an affiliate model --
they *want* you consuming their API, because traffic back is the point. That
makes them the cleanest broad-coverage sources available: no scraping question
to weigh, no Cloudflare, no per-company token discovery.

    Careerjet   https://www.careerjet.com/partners/api/
                Free partner key. Basic auth, key as username, blank password.
                careerjet.com.sg locale covers Singapore. Official Python
                client exists; this uses the documented HTTP endpoint directly
                to avoid the dependency.

    Jooble      https://jooble.org/api/about
                Free key, GUID, no card. POST to /api/{key} with a JSON body.
                sg.jooble.org covers Singapore.

Neither publishes structured skills, and salary arrives as free text when it
arrives at all -- so `comp_signal` and `competition` will score None for these
and their weight leaves the denominator, exactly as for the ATS boards.

Both are optional. Without keys this module reports itself unavailable and the
pipeline runs on the keyless sources.

Configure in .env:

    CAREERJET_API_KEY=...
    JOOBLE_API_KEY=...
"""

from __future__ import annotations

import base64
import os
import re
import urllib.parse
from typing import Any

from jobbuddy import html_text, job_schema, net, quota

CAREERJET_URL = "http://public.api.careerjet.net/search"
# The apex domain, NOT sg.jooble.org. The country subdomains serve the consumer
# site and return 403 HTML to an API call; the country is selected by the
# `location` field in the request body instead. Verified: jooble.org returns
# 144 Singapore results for the same query that sg.jooble.org refuses.
JOOBLE_URL = "https://jooble.org/api"

# Careerjet wants a caller identity; sending a real one is both required by
# their docs and the polite thing to do.
CAREERJET_AGENT = "job-buddy/1.0"


def available() -> dict[str, bool]:
    return {
        "careerjet": bool(os.environ.get("CAREERJET_API_KEY", "").strip()),
        "jooble": bool(os.environ.get("JOOBLE_API_KEY", "").strip()),
    }


# --------------------------------------------------------------------------
# Careerjet
# --------------------------------------------------------------------------

def _careerjet_records(query: str, max_results: int,
                       cache_ttl_s: float) -> list[dict[str, Any]]:
    key = os.environ.get("CAREERJET_API_KEY", "").strip()
    if not key:
        return []

    records: list[dict[str, Any]] = []
    pages = max(1, min(3, (max_results // 50) + 1))
    for page in range(1, pages + 1):
        params = urllib.parse.urlencode({
            "keywords": query,
            "location": "Singapore",
            "locale_code": "en_SG",
            "affid": key,
            "pagesize": 50,
            "page": page,
            "user_ip": "127.0.0.1",
            "user_agent": CAREERJET_AGENT,
            "sort": "date",
        })
        if not quota.can_spend("careerjet"):
            net._warn("careerjet: monthly budget spent; skipping")
            break
        data, result = net.get_json(f"{CAREERJET_URL}?{params}", cache_ttl_s=cache_ttl_s)
        if not result.from_cache:
            quota.spend("careerjet")
        if not result.ok or not isinstance(data, dict):
            net._warn(f"careerjet: page {page} failed ({result.error})")
            break
        if data.get("type") == "ERROR":
            net._warn(f"careerjet: {data.get('error', 'rejected the request')}")
            break
        batch = data.get("jobs") or []
        if not batch:
            break
        records.extend(batch)
    return records


_SALARY_RE = re.compile(r"([\d,]+(?:\.\d+)?)")


def _careerjet_salary(record: dict[str, Any]) -> tuple[int | None, int | None, bool]:
    """Careerjet gives salary_min/max as strings plus a free-text `salary`."""
    period = (record.get("salary_type") or "").lower()
    # Their `salary_type` is 'Y' yearly, 'M' monthly, 'D' daily, 'H' hourly.
    period_name = {"y": "Annually", "m": "Monthly",
                   "d": "Daily", "h": "Hourly"}.get(period[:1], "Monthly")

    values = []
    for field in ("salary_min", "salary_max"):
        raw = str(record.get(field) or "").replace(",", "")
        if raw and raw.replace(".", "").isdigit():
            values.append(float(raw))
    if len(values) < 2:
        found = _SALARY_RE.findall(str(record.get("salary") or ""))
        values = [float(f.replace(",", "")) for f in found[:2]]
    if not values:
        return None, None, False

    low = job_schema.to_monthly_sgd(min(values), period_name)
    high = job_schema.to_monthly_sgd(max(values), period_name)
    return low, high, low is not None


def _careerjet_to_job(record: dict[str, Any]) -> dict[str, Any] | None:
    url = job_schema.norm_text(record.get("url"))
    if not url:
        return None
    # Careerjet has no stable id field; the redirect URL is the identity.
    import hashlib

    job_id = hashlib.sha256(url.encode()).hexdigest()[:16]

    job = job_schema.new_job("careerjet", job_id)
    job["title"] = job_schema.norm_text(record.get("title"))
    job["company"] = job_schema.norm_text(record.get("company"))
    job["url"] = url
    job["is_agency"] = job_schema.looks_like_agency(job["company"])
    job["jd_text"] = job_schema.norm_jd_text(
        html_text.flatten_html(record.get("description") or ""))
    job["location"] = job_schema.norm_text(record.get("locations")) or "Singapore"
    job["is_overseas"] = "singapore" not in job["location"].lower()
    job["posted_at"] = job_schema.parse_date(record.get("date"))

    low, high, stated = _careerjet_salary(record)
    job["salary_min_sgd"], job["salary_max_sgd"] = low, high
    job["salary_is_stated"] = stated
    if stated:
        job["salary_period_raw"] = "Careerjet"

    job["is_open"] = True
    job["liveness"] = "ALIVE"
    job["_provenance"] = {
        "source": "careerjet", "fetched_at": job["_normalised_at"],
        "site": job_schema.norm_text(record.get("site")),
        "note": "aggregated; `site` names the board it came from",
        "applications": "absent",
    }
    return job_schema.finalise(job)


# --------------------------------------------------------------------------
# Jooble
# --------------------------------------------------------------------------

def _jooble_records(query: str, max_results: int,
                    cache_ttl_s: float) -> list[dict[str, Any]]:
    key = os.environ.get("JOOBLE_API_KEY", "").strip()
    if not key:
        return []

    records: list[dict[str, Any]] = []
    pages = max(1, min(3, (max_results // 20) + 1))
    for page in range(1, pages + 1):
        if not quota.can_spend("jooble"):
            net._warn("jooble: monthly budget spent; skipping")
            break
        data, result = net.get_json(
            f"{JOOBLE_URL}/{key}", method="POST",
            payload={"keywords": query, "location": "Singapore", "page": page},
            cache_ttl_s=cache_ttl_s,
        )
        if not result.from_cache:
            quota.spend("jooble")
        if not result.ok or not isinstance(data, dict):
            net._warn(f"jooble: page {page} failed ({result.error})")
            break
        batch = data.get("jobs") or []
        if not batch:
            break
        records.extend(batch)
    return records


def _jooble_to_job(record: dict[str, Any]) -> dict[str, Any] | None:
    job_id = job_schema.norm_text(record.get("id"))
    url = job_schema.norm_text(record.get("link"))
    if not (job_id or url):
        return None
    if not job_id:
        import hashlib

        job_id = hashlib.sha256(url.encode()).hexdigest()[:16]

    job = job_schema.new_job("jooble", job_id)
    job["title"] = job_schema.norm_text(record.get("title"))
    job["company"] = job_schema.norm_text(record.get("company"))
    job["url"] = url
    job["is_agency"] = job_schema.looks_like_agency(job["company"])
    job["jd_text"] = job_schema.norm_jd_text(
        html_text.flatten_html(record.get("snippet") or ""))
    job["location"] = job_schema.norm_text(record.get("location")) or "Singapore"
    job["is_overseas"] = bool(job["location"]) and "singapore" not in job["location"].lower()
    job["posted_at"] = job_schema.parse_date(record.get("updated"))

    # Jooble's salary is free text like "SGD 8,000 - 12,000 per month".
    salary_text = str(record.get("salary") or "")
    if salary_text:
        found = [float(f.replace(",", "")) for f in _SALARY_RE.findall(salary_text)[:2]]
        period = "Annually" if re.search(r"year|annum|p\.?a\.?", salary_text, re.I) else "Monthly"
        if len(found) >= 2:
            job["salary_min_sgd"] = job_schema.to_monthly_sgd(min(found), period)
            job["salary_max_sgd"] = job_schema.to_monthly_sgd(max(found), period)
            job["salary_is_stated"] = job["salary_min_sgd"] is not None
            job["salary_period_raw"] = f"{period} (Jooble, parsed from text)"

    job["is_open"] = True
    job["liveness"] = "ALIVE"
    job["_provenance"] = {
        "source": "jooble", "fetched_at": job["_normalised_at"],
        "origin_board": job_schema.norm_text(record.get("source")),
        "note": "aggregated; snippet only, not the full description",
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
    """Pull from whichever partner APIs are configured."""
    counters = {"fetched": 0, "unusable": 0, "dropped_overseas": 0,
                "invalid": 0, "kept": 0}
    configured = available()
    if not any(configured.values()):
        counters["skipped_no_key"] = 1
        return [], counters

    records: list[tuple[str, dict[str, Any]]] = []
    if configured["careerjet"]:
        records += [("careerjet", r)
                    for r in _careerjet_records(query, max_results, cache_ttl_s)]
    if configured["jooble"]:
        records += [("jooble", r)
                    for r in _jooble_records(query, max_results, cache_ttl_s)]

    mappers = {"careerjet": _careerjet_to_job, "jooble": _jooble_to_job}
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
        if job_schema.validate_job(job):
            counters["invalid"] += 1
            continue
        job["scope"] = query
        jobs.append(job)
        counters["kept"] += 1
        if len(jobs) >= max_results:
            break

    return jobs, counters
