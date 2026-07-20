"""Monthly request budget for paid APIs. Stops a loop from spending the plan.

JSearch bills per request against a monthly allowance. Nothing in a search
pipeline naturally bounds how many calls it makes -- add a scope, widen a
query, run it twice, and the number moves without anyone deciding it should.
A cap that lives next to the caller is the only kind that gets respected.

Counts reset on the calendar month, matching how the plans are sold. The
counter is advisory in the sense that it cannot un-send a request, but it is
checked BEFORE each call, so the first refusal happens at the limit rather
than after it.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobbuddy import net

REPO_DIR = Path(__file__).resolve().parents[2]
USAGE_PATH = REPO_DIR / "state" / "api_usage.json"

# Default ceilings per provider per calendar month. Set below the plan you
# bought, not at it -- the point is to notice before the vendor does.
DEFAULT_LIMITS = {
    "jsearch": 10000,
    "adzuna": 2500,
    "careerjet": 5000,
    "jooble": 500,      # their stated default; they raise it on request
}

_lock = threading.Lock()


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _load() -> dict[str, Any]:
    """This month's counts. Never raises, but never resets silently.

    A fresh dict is the right fallback -- refusing to run because a counter
    file is malformed would be worse than the overspend it guards against --
    but it is NOT a harmless one. Returning `{"counts": {}}` says "you have
    spent nothing this month", which re-arms the whole monthly allowance on
    every caller, and the next `spend()` writes that fiction back to disk. The
    failure mode is a bill, so the fallback has to announce itself.
    """
    if not USAGE_PATH.is_file():
        return {"month": _month(), "counts": {}}
    fresh = {"month": _month(), "counts": {}}
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        net._warn(f"quota: could not read {USAGE_PATH.name} ({exc}); treating "
                  f"this month's spend as ZERO -- paid sources are unguarded "
                  f"until the file is repaired")
        return fresh
    if not isinstance(data, dict) or not isinstance(data.get("counts"), dict):
        net._warn(f"quota: {USAGE_PATH.name} is not a usage record; treating "
                  f"this month's spend as ZERO -- paid sources are unguarded")
        return fresh
    if data.get("month") != _month():
        return fresh                                   # new month, fresh budget
    return data


def _save(data: dict[str, Any]) -> None:
    """Persist the counts. A failure here is an overspend, so it must be loud.

    `spend()` ignores this -- there is nothing useful it could do -- so silence
    meant every request this run counted against a budget that was never
    written down, and the next process started the month over.
    """
    try:
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = USAGE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, USAGE_PATH)
    except OSError as exc:
        net._warn(f"quota: could not write {USAGE_PATH.name} ({exc}); spend is "
                  f"NOT being recorded, so the monthly cap is not enforced")


def limit_for(provider: str) -> int:
    """Configured ceiling. `JSEARCH_MONTHLY_LIMIT` etc. override the default."""
    env_name = f"{provider.upper()}_MONTHLY_LIMIT"
    raw = os.environ.get(env_name, "").strip()
    if raw.isdigit():
        return int(raw)
    return DEFAULT_LIMITS.get(provider, 1000)


def used(provider: str) -> int:
    return int(_load()["counts"].get(provider, 0))


def remaining(provider: str) -> int:
    return max(0, limit_for(provider) - used(provider))


def can_spend(provider: str, count: int = 1) -> bool:
    return remaining(provider) >= count


def spend(provider: str, count: int = 1) -> int:
    """Record `count` requests. Returns the new total."""
    with _lock:
        data = _load()
        total = int(data["counts"].get(provider, 0)) + count
        data["counts"][provider] = total
        _save(data)
        return total


def report() -> list[dict[str, Any]]:
    """Usage this month, for the CLI and the notebook."""
    data = _load()
    rows = []
    for provider in sorted(set(DEFAULT_LIMITS) | set(data["counts"])):
        cap = limit_for(provider)
        spent = int(data["counts"].get(provider, 0))
        rows.append({
            "provider": provider, "used": spent, "limit": cap,
            "remaining": max(0, cap - spent),
            "pct": round(100.0 * spent / cap, 1) if cap else 0.0,
            "month": data["month"],
        })
    return rows
