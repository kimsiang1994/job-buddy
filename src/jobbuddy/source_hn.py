"""Hacker News "Who is Hiring", via the Algolia search API. Keyless.

    https://hn.algolia.com/api/v1/search_by_date?query="Ask HN: Who is hiring"
    https://hn.algolia.com/api/v1/items/<story_id>

One thread per month, one top-level comment per role, written by the hiring
manager rather than a recruiter. Small volume, but a different population from
every other source here: senior and staff engineering roles at companies that
never post to a Singapore job board, often remote-eligible.

The awkward part is that a comment is prose, not a record. There is no title
field, no company field, no location field -- just a paragraph that conventionally
opens with something like:

    Acme Corp | Senior ML Engineer | Singapore or Remote (APAC) | $180-250k

That convention holds often enough to parse deterministically, and when it does
not, the job is skipped rather than guessed at. No LLM here: this stage runs
before any API key exists, and a wrong guess about who is hiring for what is
worse than a missing row.
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterator

from jobbuddy import html_text, job_schema, net

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
ITEM_URL = "https://hn.algolia.com/api/v1/items/{story_id}"
SOURCE = "hn"

# The pipe-delimited convention. Requires at least three fields so a plain
# sentence containing a pipe cannot masquerade as a posting.
PIPE_RE = re.compile(r"^\s*([^|\n]{2,60})\|([^|\n]{2,80})\|(.{2,200})$", re.M)

REMOTE_RE = re.compile(r"\bremote\b", re.I)
SG_RE = re.compile(r"\b(singapore|sg|apac|asia[- ]pacific|sea)\b", re.I)


def find_threads(limit: int = 3, cache_ttl_s: float = 86400.0) -> list[dict[str, Any]]:
    """The most recent 'Who is hiring' stories, newest first."""
    url = (f"{SEARCH_URL}?query=%22Ask+HN%3A+Who+is+hiring%22"
           f"&tags=story&hitsPerPage={max(1, limit)}")
    data, result = net.get_json(url, cache_ttl_s=cache_ttl_s)
    if not result.ok or not isinstance(data, dict):
        net._warn(f"hn: thread search failed ({result.error})")
        return []
    hits = data.get("hits") or []
    return [h for h in hits if "who is hiring" in (h.get("title") or "").lower()]


def thread_comments(story_id: Any, cache_ttl_s: float = 86400.0) -> Iterator[dict[str, Any]]:
    """Top-level comments of one thread. Replies are discussion, not postings."""
    data, result = net.get_json(ITEM_URL.format(story_id=story_id),
                                cache_ttl_s=cache_ttl_s)
    if not result.ok or not isinstance(data, dict):
        net._warn(f"hn: thread {story_id} failed ({result.error})")
        return
    for child in data.get("children") or []:
        if isinstance(child, dict) and child.get("text") and not child.get("deleted"):
            yield child


def parse_posting(text: str) -> dict[str, str] | None:
    """Pull (company, title, rest) out of a comment. None when it does not fit.

    Deliberately strict. A comment that does not follow the convention is
    skipped -- inferring a company name from free prose produces confident
    nonsense, and a job list with wrong employers on it is worse than a shorter
    one.
    """
    plain = html.unescape(html_text.flatten_html(text or ""))
    first_block = plain[:400]

    match = PIPE_RE.search(first_block)
    if not match:
        # Some posters use the same convention on one line without newlines.
        parts = [p.strip() for p in first_block.split("|")]
        if len(parts) >= 3 and all(2 <= len(p) <= 200 for p in parts[:3]):
            company, title, rest = parts[0], parts[1], " | ".join(parts[2:])
        else:
            return None
    else:
        company, title, rest = (g.strip() for g in match.groups())

    if not company or not title:
        return None
    return {"company": company, "title": title, "rest": rest, "full": plain}


def to_job(comment: dict[str, Any]) -> dict[str, Any] | None:
    parsed = parse_posting(comment.get("text") or "")
    if not parsed:
        return None

    comment_id = job_schema.norm_text(comment.get("id"))
    if not comment_id:
        return None

    job = job_schema.new_job(SOURCE, comment_id)
    job["title"] = job_schema.norm_text(parsed["title"])[:140]
    job["company"] = job_schema.norm_text(parsed["company"])[:100]
    job["url"] = f"https://news.ycombinator.com/item?id={comment_id}"
    job["jd_text"] = job_schema.norm_text(parsed["full"])
    job["is_agency"] = job_schema.looks_like_agency(job["company"])

    context = f"{parsed['rest']} {parsed['full'][:600]}"
    job["location"] = job_schema.norm_text(parsed["rest"])[:80] or "see posting"
    job["is_remote"] = bool(REMOTE_RE.search(context))
    # A thread this size cannot be filtered by country reliably, so the test is
    # "does it mention SG or APAC, or is it remote" -- anything else is elsewhere.
    job["is_overseas"] = not (SG_RE.search(context) or job["is_remote"])

    job["posted_at"] = job_schema.parse_date(comment.get("created_at"))
    job["is_open"] = True
    job["liveness"] = "ALIVE"
    job["_provenance"] = {
        "source": SOURCE, "fetched_at": job["_normalised_at"],
        "parsed_by": "pipe convention, no LLM",
        "salary": "absent", "applications": "absent",
    }
    return job_schema.finalise(job)


def fetch_jobs(
    query: str,
    max_results: int = 60,
    singapore_only: bool = True,
    open_only: bool = True,
    cache_ttl_s: float = 86400.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Search recent Who-is-hiring threads. Same contract as the other sources."""
    counters = {"threads": 0, "comments": 0, "unparseable": 0,
                "dropped_overseas": 0, "dropped_offtopic": 0, "invalid": 0,
                "kept": 0}
    terms = [t for t in re.split(r"\s+", query.lower()) if len(t) > 2]
    jobs: list[dict[str, Any]] = []

    for thread in find_threads(limit=2, cache_ttl_s=cache_ttl_s):
        counters["threads"] += 1
        for comment in thread_comments(thread.get("objectID"), cache_ttl_s):
            counters["comments"] += 1
            job = to_job(comment)
            if job is None:
                counters["unparseable"] += 1
                continue
            if singapore_only and job["is_overseas"]:
                counters["dropped_overseas"] += 1
                continue
            haystack = f"{job['title']} {job['jd_text'][:1500]}".lower()
            if terms and not any(t in haystack for t in terms):
                counters["dropped_offtopic"] += 1
                continue
            problems = job_schema.validate_job(job)
            if problems:
                # `parse_posting` already rejects a comment with no company or
                # no title, so anything failing HERE is this mapper losing a
                # field -- the shape that breaks every record at once. Dropping
                # those with no counter and no message made "the mapper is
                # broken" look identical to "no SG roles in this thread". MCF
                # and Workable both name the problems; this did not.
                counters["invalid"] += 1
                net._warn(f"hn: {job['job_key']} rejected -- {'; '.join(problems)}")
                continue
            job["scope"] = query
            jobs.append(job)
            counters["kept"] += 1
            if len(jobs) >= max_results:
                return jobs, counters
    return jobs, counters
