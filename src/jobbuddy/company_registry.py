"""Which companies hire for a scope, and how to reach their boards directly.

Portals are a sampling of the market. Going direct to an employer's own board
gets the roles that never reach a portal at all -- which in Singapore is
structurally the top of the market, since MOM exempts anything paying over
S$22,500/month from the advertising requirement.

Two things that look like one, and must not be run as one:

    discovery    company -> ATS board. Two HTTP fetches per company, and a
                 given company resolves ONCE, EVER. Doing this for 500
                 companies inside a search means fifteen minutes of crawling
                 before the first result appears.

    fetching     poll the boards you already know. Fast, parallel, on demand.

So discovery is amortised -- a slice of the unresolved queue each run -- and
the registry it fills compounds. Run the tool for a fortnight and you own a
company-to-board map nobody publishes, which is the actual moat here.

The company list bootstraps from the jobs already fetched. Companies posting
most often for "machine learning engineer" ARE the top companies for that
scope; no external ranking is needed, and the ranking sharpens as you search.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jobbuddy import job_schema, source_ats, source_workable

REPO_DIR = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_DIR / "config" / "companies.json"

# Resolved per call, not at import, so a test run can redirect it. No test
# imports this module directly -- the pipeline reaches it several layers down --
# so the suite was quietly writing "acme"/"test-scope" into the real shipped
# config, where it survived until someone noticed the dirty working tree.
_PATH_ENV = "JB_COMPANY_REGISTRY"


def registry_path() -> Path:
    override = os.environ.get(_PATH_ENV, "").strip()
    return Path(override) if override else REGISTRY_PATH

# How many unresolved companies to look up per run. Discovery costs two fetches
# each and is polite-rate-limited, so this trades a slow first fortnight for a
# search that never stalls. Raise it if you want the registry faster.
DISCOVERY_PER_RUN = 12

# Companies whose board resolved to nothing are retried, but rarely -- a firm
# with no public ATS today may adopt one, and re-crawling every run is rude.
RETRY_MISSES_AFTER_DAYS = 30


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load() -> dict[str, dict[str, Any]]:
    """company_norm -> record. Never raises."""
    path = registry_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data.get("companies", {}) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(companies: dict[str, dict[str, Any]]) -> bool:
    """Atomic write. Sole writer of config/companies.json."""
    resolved = sum(1 for c in companies.values() if c.get("board_vendor"))
    payload = {
        "_written_by": "company_registry.py",
        "_comment": "Companies seen while searching, ranked by how often they "
                    "post for each scope, plus their ATS board once resolved. "
                    "Discovery is amortised across runs -- a company resolves "
                    "once and then stays resolved.",
        "companies_known": len(companies),
        "boards_resolved": resolved,
        "companies": dict(sorted(companies.items(),
                                 key=lambda kv: -kv[1].get("seen", 0))),
    }
    path = registry_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def observe(jobs: Iterable[dict[str, Any]], scope: str = "") -> int:
    """Record every company these jobs came from. Returns newly-seen count.

    Called after each fetch. Agencies are recorded but flagged -- they post for
    other people, so their own board is rarely worth polling.
    """
    companies = load()
    new = 0

    for job in jobs:
        key = job.get("company_norm")
        if not key:
            continue
        record = companies.get(key)
        if record is None:
            new += 1
            record = {
                "name": job.get("company"),
                "seen": 0,
                "scopes": {},
                "is_agency": bool(job.get("is_agency")),
                "uen": job.get("company_uen"),
                "website": None,
                "board_vendor": None,
                "board_token": None,
                "discovered_at": None,
                "first_seen": _now(),
            }
            companies[key] = record

        record["seen"] = record.get("seen", 0) + 1
        record["last_seen"] = _now()
        if scope:
            record["scopes"][scope] = record["scopes"].get(scope, 0) + 1
        if job.get("company_uen") and not record.get("uen"):
            record["uen"] = job["company_uen"]

        # Read the website off the job itself, not a process-local cache.
        # Workable stamps it into provenance, and reading only the in-memory
        # cache meant a fresh process saw no websites at all -- so the
        # discovery queue was permanently empty and the registry could never
        # grow past what had already been resolved.
        if not record.get("website"):
            site = (job.get("_provenance") or {}).get("company_website")
            if site:
                record["website"] = site

    # Anything this process fetched, as well.
    for company, site in source_workable.known_company_websites().items():
        if company in companies and not companies[company].get("website"):
            companies[company]["website"] = site

    # Boards already resolved elsewhere fold in, so the two registries agree.
    for company, board in source_ats.load_boards().items():
        record = companies.get(company)
        if record and board.get("vendor") and not record.get("board_vendor"):
            record["board_vendor"] = board["vendor"]
            record["board_token"] = board["token"]
            record["discovered_at"] = _now()

    save(companies)
    return new


def top_companies(scope: str = "", limit: int = 500,
                  include_agencies: bool = False) -> list[tuple[str, dict[str, Any]]]:
    """Companies ranked by how often they post for a scope.

    No external ranking needed: a company that keeps appearing in your search
    results is, by construction, a company that hires for what you search for.
    """
    companies = load()
    scored = []
    for key, record in companies.items():
        if record.get("is_agency") and not include_agencies:
            continue
        weight = record["scopes"].get(scope, 0) if scope else record.get("seen", 0)
        if weight:
            scored.append((weight, key, record))
    scored.sort(key=lambda x: -x[0])
    return [(key, record) for _, key, record in scored[:limit]]


def discovery_queue(limit: int = DISCOVERY_PER_RUN,
                    scope: str = "") -> list[tuple[str, dict[str, Any]]]:
    """The next companies worth resolving a board for.

    Ordered by how often they post for the scope, so the discovery budget goes
    to the employers you would actually apply to rather than alphabetically.
    """
    queue = []
    for key, record in top_companies(scope, limit=2000):
        if record.get("board_vendor"):
            continue        # already resolved
        if not record.get("website"):
            continue        # nothing to crawl from
        attempted = record.get("discovery_attempted_at")
        if attempted:
            age = job_schema.days_between(attempted[:10])
            if age is not None and age < RETRY_MISSES_AFTER_DAYS:
                continue    # missed recently; do not re-crawl
        queue.append((key, record))
        if len(queue) >= limit:
            break
    return queue


def run_discovery(limit: int = DISCOVERY_PER_RUN, scope: str = "") -> dict[str, int]:
    """Resolve boards for the next slice of the queue. Returns counters.

    Deliberately bounded. The whole point of amortising is that no single run
    pays the cost of the whole registry.
    """
    queue = discovery_queue(limit, scope)
    counters = {"attempted": 0, "resolved": 0, "missed": 0}
    if not queue:
        return counters

    companies = load()
    boards = source_ats.load_boards()

    for key, record in queue:
        counters["attempted"] += 1
        found = source_ats.discover_board(record["website"])
        entry = companies.setdefault(key, record)
        entry["discovery_attempted_at"] = _now()

        if found:
            vendor, token = found
            entry["board_vendor"], entry["board_token"] = vendor, token
            entry["discovered_at"] = _now()
            boards[key] = {"vendor": vendor, "token": token,
                           "website": record["website"], "via": "company registry"}
            counters["resolved"] += 1
        else:
            counters["missed"] += 1

    save(companies)
    source_ats.save_boards(boards)
    return counters


def seed_from_websites(websites: dict[str, str]) -> int:
    """Add companies by name -> website, for hand-curated seed lists."""
    companies = load()
    added = 0
    for name, site in websites.items():
        key = job_schema.norm_company(name)
        if not key:
            continue
        record = companies.setdefault(key, {
            "name": name, "seen": 0, "scopes": {}, "is_agency": False,
            "uen": None, "board_vendor": None, "board_token": None,
            "discovered_at": None, "first_seen": _now(),
        })
        if not record.get("website"):
            record["website"] = site
            added += 1
    save(companies)
    return added


def summary() -> dict[str, Any]:
    """Registry state, for the notebook and the CLI."""
    companies = load()
    resolved = [c for c in companies.values() if c.get("board_vendor")]
    with_site = [c for c in companies.values() if c.get("website")]
    pending = [c for c in companies.values()
               if c.get("website") and not c.get("board_vendor")]
    vendors: dict[str, int] = defaultdict(int)
    for record in resolved:
        vendors[record["board_vendor"]] += 1
    return {
        "companies_known": len(companies),
        "with_website": len(with_site),
        "boards_resolved": len(resolved),
        "pending_discovery": len(pending),
        "by_vendor": dict(vendors),
        "runs_to_drain_queue": (len(pending) + DISCOVERY_PER_RUN - 1) // max(DISCOVERY_PER_RUN, 1),
    }
