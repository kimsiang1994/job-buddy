"""Persistent job history. The sole writer of state/sightings.jsonl.

Competition signals are the reason this exists. Most of them are only visible
across time:

  first_seen_at  when *we* first saw it -- the only date nobody else controls
  reposted       same role, new posting id, after a gap
  absent_runs    stopped showing up in the feed
  velocity       how many reqs this employer opened lately

Append-only event log, folded on read. Not SQLite, deliberately: at ~18k rows a
year the fold is milliseconds, and a JSONL file can be inspected, diffed and
hand-repaired at 11pm. A torn final line (interrupted run) costs one record and
is skipped with a warning, which matches the repo's "read path cannot raise".

Writes go through a lock and an atomic replace, because the pipeline fans out
over threads and Windows has no atomic append guarantee.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import job_schema

REPO_DIR = Path(__file__).resolve().parent
STATE_DIR = REPO_DIR / "state"
SIGHTINGS_PATH = STATE_DIR / "sightings.jsonl"
JOB_STATE_PATH = STATE_DIR / "job_state.json"

# A job must be missing from this many consecutive runs before we will even
# call it "probably gone". One absence is almost always the search feed
# reshuffling, not the job closing.
ABSENT_RUNS_BEFORE_SUSPECT = 2

# Only jobs seen within this window stay in the compacted snapshot.
TRACKING_WINDOW_DAYS = 90

# Fields copied from a Job into a sighting. Kept deliberately small -- the JD
# lives in the run output tree, not in the event log.
_SIGHTING_FIELDS = (
    "job_key", "content_key", "source", "url", "title_norm", "company_norm",
    "company_uen", "is_agency", "seniority", "salary_min_sgd", "salary_max_sgd",
    "salary_is_stated", "ssoc_code", "posted_at", "expires_at", "source_status",
    "is_open", "liveness", "applications", "views", "apps_per_view",
    "repost_count", "edit_count", "vacancies", "scope",
)

_write_lock = threading.Lock()
_warned: set[str] = set()


def _warn(message: str) -> None:
    if message in _warned:
        return
    _warned.add(message)
    try:
        import sys

        print(f"job_store: {message}", file=sys.stderr)
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_sightings(path: Path = SIGHTINGS_PATH) -> list[dict[str, Any]]:
    """Read every sighting. Skips damaged lines rather than raising.

    A run interrupted mid-write leaves at most one torn line; losing one
    observation is acceptable, refusing to start is not.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    damaged = 0
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    damaged += 1
                    continue
                if isinstance(row, dict) and row.get("job_key"):
                    rows.append(row)
    except OSError as exc:
        _warn(f"could not read {path.name} ({exc}); treating history as empty")
        return []
    if damaged:
        _warn(f"skipped {damaged} damaged line(s) in {path.name}")
    return rows


def record_sightings(
    jobs: Iterable[dict[str, Any]],
    run_id: str,
    path: Path = SIGHTINGS_PATH,
) -> int:
    """Append one sighting per job. Returns the count written.

    Called from a single-threaded pipeline stage; the lock is belt-and-braces
    for anyone who calls it from a worker later.
    """
    timestamp = _now_iso()
    lines: list[str] = []
    for job in jobs:
        row: dict[str, Any] = {"ts": timestamp, "run_id": run_id}
        for field in _SIGHTING_FIELDS:
            row[field] = job.get(field)
        lines.append(json.dumps(row, ensure_ascii=False))

    if not lines:
        return 0

    with _write_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            _warn(f"could not append to {path.name} ({exc}); history not updated")
            return 0
    return len(lines)


def fold(sightings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse the event log into per-job history.

    Returns {job_key: {first_seen_at, last_seen_at, seen_count, runs, ...}}.
    """
    by_key: dict[str, dict[str, Any]] = {}
    for row in sightings:
        key = row.get("job_key")
        if not key:
            continue
        entry = by_key.get(key)
        if entry is None:
            entry = {
                "job_key": key,
                "content_key": row.get("content_key"),
                "source": row.get("source"),
                "company_norm": row.get("company_norm"),
                "title_norm": row.get("title_norm"),
                "first_seen_at": row.get("ts"),
                "last_seen_at": row.get("ts"),
                "seen_count": 0,
                "runs": [],
                "last_applications": None,
                "first_applications": None,
                "last_is_open": True,
            }
            by_key[key] = entry

        timestamp = row.get("ts") or ""
        if timestamp and timestamp < (entry["first_seen_at"] or timestamp):
            entry["first_seen_at"] = timestamp
        if timestamp and timestamp > (entry["last_seen_at"] or ""):
            entry["last_seen_at"] = timestamp
        entry["seen_count"] += 1
        run_id = row.get("run_id")
        if run_id and run_id not in entry["runs"]:
            entry["runs"].append(run_id)

        applications = row.get("applications")
        if isinstance(applications, int):
            if entry["first_applications"] is None:
                entry["first_applications"] = applications
            entry["last_applications"] = applications
        if row.get("is_open") is not None:
            entry["last_is_open"] = bool(row.get("is_open"))

    for entry in by_key.values():
        entry["runs"].sort()
    return by_key


def _repost_index(history: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Map content_key -> job_keys that have carried it, oldest first."""
    groups: dict[str, list[str]] = defaultdict(list)
    for key, entry in history.items():
        content = entry.get("content_key")
        if content:
            groups[content].append(key)
    for content, keys in groups.items():
        keys.sort(key=lambda k: history[k].get("first_seen_at") or "")
    return groups


def apply_history(
    jobs: list[dict[str, Any]],
    history: dict[str, dict[str, Any]],
    run_id: str,
    all_run_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Attach history-derived fields to freshly-fetched jobs. Mutates in place.

    Called *after* this run's sightings have been recorded, so `history`
    already includes the current observation.
    """
    groups = _repost_index(history)
    run_ids = sorted(all_run_ids or [])

    for job in jobs:
        entry = history.get(job["job_key"])
        if entry is None:
            # First time we have ever seen it.
            job["first_seen_at"] = _now_iso()
            job["last_seen_at"] = job["first_seen_at"]
            job["seen_count"] = 1
            continue

        job["first_seen_at"] = entry["first_seen_at"]
        job["last_seen_at"] = entry["last_seen_at"]
        job["seen_count"] = entry["seen_count"]

        # Reposts: the same role advertised again under a new posting id. MCF
        # gives us repostCount directly; for other sources this is how we get it.
        siblings = [k for k in groups.get(job["content_key"], []) if k != job["job_key"]]
        if siblings:
            job["reposted"] = True
            job["repost_of"] = siblings
            job["repost_count"] = max(job.get("repost_count") or 0, len(siblings))

        # How many consecutive recent runs did this job miss? Only meaningful
        # once there are runs to have missed.
        if run_ids:
            seen_runs = set(entry["runs"])
            absent = 0
            for candidate in reversed(run_ids):
                if candidate in seen_runs:
                    break
                absent += 1
            job["absent_runs"] = absent

        # Applications accrued while we have been watching -- a direct read on
        # how fast competition is arriving, independent of the absolute count.
        first_apps = entry.get("first_applications")
        last_apps = entry.get("last_applications")
        if isinstance(first_apps, int) and isinstance(last_apps, int):
            observed = job_schema.days_between(entry["first_seen_at"], entry["last_seen_at"])
            if observed and observed >= 1:
                job["apps_per_day_observed"] = round((last_apps - first_apps) / observed, 3)

    return jobs


def mark_absent(
    history: dict[str, dict[str, Any]],
    seen_this_run: set[str],
    run_ids: list[str],
) -> list[dict[str, Any]]:
    """Report tracked jobs that did not appear in this run.

    Explicitly does NOT conclude they are closed. Feed absence is not evidence:
    search relevance shifts and pagination truncates. These are candidates for
    the re-check stage, which asks the job's own endpoint.
    """
    suspects: list[dict[str, Any]] = []
    if len(run_ids) <= ABSENT_RUNS_BEFORE_SUSPECT:
        return suspects

    for key, entry in history.items():
        if key in seen_this_run:
            continue
        seen_runs = set(entry["runs"])
        absent = 0
        for candidate in reversed(sorted(run_ids)):
            if candidate in seen_runs:
                break
            absent += 1
        if absent >= ABSENT_RUNS_BEFORE_SUSPECT:
            suspects.append({
                "job_key": key,
                "absent_runs": absent,
                "last_seen_at": entry["last_seen_at"],
                "first_seen_at": entry["first_seen_at"],
                "needs_recheck": True,
            })
    return suspects


def company_velocity(
    history: dict[str, dict[str, Any]],
    window_days: int = 30,
) -> dict[str, dict[str, Any]]:
    """Reqs opened per company in the trailing window.

    Returns {company_norm: {open_reqs, new_in_window, history_days, sufficient}}.
    `sufficient` is False until we have watched long enough for the number to
    mean anything -- the report must say "insufficient history", never impute.
    """
    if not history:
        return {}

    timestamps = [e["first_seen_at"] for e in history.values() if e.get("first_seen_at")]
    if not timestamps:
        return {}
    earliest = min(timestamps)
    history_days = job_schema.days_between(earliest[:10]) or 0.0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"open_reqs": 0, "new_in_window": 0}
    )
    for entry in history.values():
        company = entry.get("company_norm")
        if not company:
            continue
        stats[company]["open_reqs"] += 1
        if (entry.get("first_seen_at") or "") >= cutoff:
            stats[company]["new_in_window"] += 1

    sufficient = history_days >= window_days
    for company in stats:
        stats[company]["history_days"] = round(history_days, 1)
        stats[company]["sufficient"] = sufficient
    return dict(stats)


def write_snapshot(
    history: dict[str, dict[str, Any]],
    run_id: str,
    path: Path = JOB_STATE_PATH,
) -> bool:
    """Write the compacted fold. Atomic; sole writer is this module."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=TRACKING_WINDOW_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    tracked = {
        key: entry
        for key, entry in history.items()
        if (entry.get("last_seen_at") or "") >= cutoff
    }
    payload = {
        "_written_by": "job_store.py",
        "_written_at": _now_iso(),
        "run_id": run_id,
        "tracked_jobs": len(tracked),
        "total_jobs_ever": len(history),
        "jobs": tracked,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        _warn(f"could not write {path.name} ({exc}); snapshot skipped")
        return False


def load_history(path: Path = SIGHTINGS_PATH) -> dict[str, dict[str, Any]]:
    """Convenience: read and fold in one call."""
    return fold(read_sightings(path))


def all_run_ids(sightings: list[dict[str, Any]]) -> list[str]:
    return sorted({row["run_id"] for row in sightings if row.get("run_id")})
