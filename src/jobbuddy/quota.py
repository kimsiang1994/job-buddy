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
    if not USAGE_PATH.is_file():
        return {"month": _month(), "counts": {}}
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8-sig"))
        if data.get("month") != _month():
            return {"month": _month(), "counts": {}}   # new month, fresh budget
        return data
    except (OSError, ValueError):
        return {"month": _month(), "counts": {}}


def _save(data: dict[str, Any]) -> None:
    try:
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = USAGE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, USAGE_PATH)
    except OSError:
        pass


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
