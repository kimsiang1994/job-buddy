"""Direct-from-employer job boards, and the discovery that finds them.

Ten ATS vendors host most companies' careers pages, and nearly all expose a
public JSON endpoint per company -- unauthenticated, because the whole point is
for the jobs to be read. Going direct gets you:

  - roles that never reach MyCareersFuture (no work-pass involved, or the
    salary is above the S$22,500 advertising exemption)
  - the posting before an aggregator indexes it
  - Ashby boards optionally publish compensation, which restores the salary
    signal Workable lacks

The bottleneck is not fetching, it is knowing a company's board token. No
public registry maps company -> token, so `discover_board` does what a human
would: fetch the company's site, find the careers link, and read the ATS out of
the URL it points at. Results are cached to config/ats_boards.json so the crawl
happens once per company, not once per run.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any, Callable

from jobbuddy import html_text, job_schema, net

REPO_DIR = Path(__file__).resolve().parents[2]
BOARDS_PATH = REPO_DIR / "config" / "ats_boards.json"

# Vendor -> how to spot it in a URL. Ordered: first match wins, so put the
# more specific patterns first (myworkdayjobs before the generic ones).
ATS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("workday", re.compile(r"https?://([\w-]+)\.(wd\d+)\.myworkdayjobs\.com/(?:[\w-]+/)?([\w-]+)", re.I)),
    ("greenhouse", re.compile(r"(?:boards|job-boards)\.greenhouse\.io/(?:embed/job_board\?for=)?([\w.-]+)", re.I)),
    ("lever", re.compile(r"jobs\.lever\.co/([\w.-]+)", re.I)),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([\w.-]+)", re.I)),
    ("smartrecruiters", re.compile(r"(?:jobs|careers)\.smartrecruiters\.com/([\w.-]+)", re.I)),
    ("recruitee", re.compile(r"([\w-]+)\.recruitee\.com", re.I)),
    ("personio", re.compile(r"([\w-]+)\.jobs\.personio\.(?:de|com)", re.I)),
    ("teamtailor", re.compile(r"([\w-]+)\.teamtailor\.com", re.I)),
    ("breezy", re.compile(r"([\w-]+)\.breezy\.hr", re.I)),
    ("rippling", re.compile(r"ats\.rippling\.com/([\w-]+)", re.I)),
    ("workable", re.compile(r"apply\.workable\.com/([\w-]+)", re.I)),
)

CAREERS_LINK_RE = re.compile(
    r'href=["\']([^"\']*(?:career|jobs?|join-us|work-with-us|opportunit|'
    r'vacanc|hiring|employment|life-at)[^"\']*)["\']', re.I)


# --------------------------------------------------------------------------
# Board registry -- sole writer of config/ats_boards.json
# --------------------------------------------------------------------------

def load_boards() -> dict[str, dict[str, Any]]:
    """company_norm -> {vendor, token, ...}. Never raises."""
    if not BOARDS_PATH.is_file():
        return {}
    try:
        data = json.loads(BOARDS_PATH.read_text(encoding="utf-8-sig"))
        return data.get("boards", {}) if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        net._warn(f"ats: could not read {BOARDS_PATH.name} ({exc}); starting empty")
        return {}


def save_boards(boards: dict[str, dict[str, Any]]) -> bool:
    """Atomic write. Sole writer of the registry."""
    payload = {
        "_written_by": "source_ats.py",
        "_comment": "company_norm -> ATS board. Built by discovery; edit freely, "
                    "hand-added entries are never overwritten by a re-crawl.",
        "boards": boards,
    }
    try:
        BOARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = BOARDS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, BOARDS_PATH)
        return True
    except OSError as exc:
        net._warn(f"ats: could not write {BOARDS_PATH.name} ({exc})")
        return False


def detect_ats(url: str) -> tuple[str, str] | None:
    """(vendor, token) from a careers URL, or None.

    Workday is special: its token is tenant + datacentre + site, so it is
    packed into one string and unpacked at fetch time.
    """
    for vendor, pattern in ATS_PATTERNS:
        match = pattern.search(url or "")
        if not match:
            continue
        if vendor == "workday":
            tenant, datacentre, site = match.groups()
            return vendor, f"{tenant}|{datacentre}|{site}"
        token = match.group(1)
        # Greenhouse embeds sometimes capture a trailing path fragment.
        return vendor, token.strip("/.")
    return None


def discover_board(company_website: str, timeout: int = 20) -> tuple[str, str] | None:
    """Find a company's ATS board by reading their careers page.

    Two hops at most: the homepage, then whichever careers link it exposes.
    Deliberately shallow -- a deep crawl of an employer's marketing site to
    save one manual lookup is not a good trade, and it is rude.
    """
    if not company_website:
        return None

    # The homepage often links straight to the ATS.
    result = net.fetch(company_website, accept="text/html", timeout=timeout,
                       cache_ttl_s=86400.0)
    if not result.ok:
        return None
    html = result.text()

    found = detect_ats(html)
    if found:
        return found

    # Otherwise follow the first careers-looking link.
    for match in CAREERS_LINK_RE.finditer(html[:400000]):
        href = urllib.parse.urljoin(result.url, match.group(1))
        if urllib.parse.urlparse(href).netloc == urllib.parse.urlparse(result.url).netloc:
            page = net.fetch(href, accept="text/html", timeout=timeout,
                             cache_ttl_s=86400.0)
            if page.ok:
                found = detect_ats(page.text()) or detect_ats(page.url)
                if found:
                    return found
        else:
            # An off-site careers link is usually the ATS itself.
            found = detect_ats(href)
            if found:
                return found
        break   # one careers link is enough; more is a crawl, not a lookup
    return None


def discover_from_websites(
    websites: dict[str, str],
    limit: int = 25,
) -> dict[str, dict[str, Any]]:
    """Resolve boards for companies we have websites for. Caches as it goes."""
    boards = load_boards()
    attempted = 0

    for company, site in websites.items():
        if company in boards or attempted >= limit:
            continue
        attempted += 1
        found = discover_board(site)
        if found:
            vendor, token = found
            boards[company] = {"vendor": vendor, "token": token,
                               "website": site, "via": "discovery"}
            net._warn(f"ats: discovered {company} -> {vendor}:{token}")
        else:
            # Record the miss so the next run does not re-crawl the same site.
            boards[company] = {"vendor": None, "token": None,
                               "website": site, "via": "discovery",
                               "note": "no ATS found"}
    if attempted:
        save_boards(boards)
    return boards


# --------------------------------------------------------------------------
# Vendor adapters -- each returns a list of raw records
# --------------------------------------------------------------------------

def _greenhouse(token: str, ttl: float) -> list[dict[str, Any]]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
    data, _ = net.get_json(url, cache_ttl_s=ttl)
    return (data or {}).get("jobs", []) if isinstance(data, dict) else []


def _lever(token: str, ttl: float) -> list[dict[str, Any]]:
    # Lever tokens are case-sensitive -- 'Coda' works, 'coda' 404s.
    url = f"https://api.lever.co/v0/postings/{token}?mode=json"
    data, _ = net.get_json(url, cache_ttl_s=ttl)
    return data if isinstance(data, list) else []


def _ashby(token: str, ttl: float) -> list[dict[str, Any]]:
    url = (f"https://api.ashbyhq.com/posting-api/job-board/{token}"
           f"?includeCompensation=true")
    data, _ = net.get_json(url, cache_ttl_s=ttl)
    return (data or {}).get("jobs", []) if isinstance(data, dict) else []


def _smartrecruiters(token: str, ttl: float) -> list[dict[str, Any]]:
    url = (f"https://api.smartrecruiters.com/v1/companies/{token}"
           f"/postings?limit=100&country=sg")
    data, _ = net.get_json(url, cache_ttl_s=ttl)
    return (data or {}).get("content", []) if isinstance(data, dict) else []


def _workday(token: str, ttl: float) -> list[dict[str, Any]]:
    try:
        tenant, datacentre, site = token.split("|")
    except ValueError:
        return []
    url = (f"https://{tenant}.{datacentre}.myworkdayjobs.com"
           f"/wday/cxs/{tenant}/{site}/jobs")
    # Workday needs a POST body, and returns HTML if Accept is not set.
    data, _ = net.get_json(url, method="POST",
                           payload={"appliedFacets": {}, "limit": 20,
                                    "offset": 0, "searchText": ""},
                           cache_ttl_s=ttl)
    return (data or {}).get("jobPostings", []) if isinstance(data, dict) else []


def _recruitee(token: str, ttl: float) -> list[dict[str, Any]]:
    data, _ = net.get_json(f"https://{token}.recruitee.com/api/offers/", cache_ttl_s=ttl)
    return (data or {}).get("offers", []) if isinstance(data, dict) else []


def _personio(token: str, ttl: float) -> list[dict[str, Any]]:
    data, _ = net.get_json(f"https://{token}.jobs.personio.de/search.json", cache_ttl_s=ttl)
    return data if isinstance(data, list) else []


def _teamtailor(token: str, ttl: float) -> list[dict[str, Any]]:
    data, _ = net.get_json(f"https://{token}.teamtailor.com/jobs.json", cache_ttl_s=ttl)
    return (data or {}).get("items", []) if isinstance(data, dict) else []


def _breezy(token: str, ttl: float) -> list[dict[str, Any]]:
    data, _ = net.get_json(f"https://{token}.breezy.hr/json", cache_ttl_s=ttl)
    return data if isinstance(data, list) else []


def _rippling(token: str, ttl: float) -> list[dict[str, Any]]:
    url = f"https://api.rippling.com/platform/api/ats/v1/board/{token}/jobs"
    data, _ = net.get_json(url, cache_ttl_s=ttl)
    return data if isinstance(data, list) else []


FETCHERS: dict[str, Callable[[str, float], list[dict[str, Any]]]] = {
    "greenhouse": _greenhouse, "lever": _lever, "ashby": _ashby,
    "smartrecruiters": _smartrecruiters, "workday": _workday,
    "recruitee": _recruitee, "personio": _personio,
    "teamtailor": _teamtailor, "breezy": _breezy, "rippling": _rippling,
}


# --------------------------------------------------------------------------
# Normalisation -- each vendor names everything differently
# --------------------------------------------------------------------------

def _first(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _text_of(value: Any) -> str:
    """Location fields arrive as a string, a dict, or a list of dicts."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [value.get(k) for k in ("city", "name", "text", "location",
                                        "region", "country", "countryName")]
        joined = ", ".join(job_schema.norm_text(p) for p in parts if p)
        return joined or job_schema.norm_text(value.get("locationsText"))
    if isinstance(value, list) and value:
        return "; ".join(filter(None, (_text_of(v) for v in value[:3])))
    return ""


# Enough non-SG markers to classify the boards we actually pull. Deliberately
# not a world gazetteer -- an unrecognised place stays "unknown" and is kept,
# because dropping a job you could not classify is worse than showing one.
_NON_SG = (
    "united states", "usa", "u.s.", "canada", "united kingdom", "london",
    "san francisco", "new york", "nyc", "seattle", "chicago", "austin",
    "boston", "los angeles", "toronto", "vancouver", "dublin", "berlin",
    "amsterdam", "paris", "madrid", "barcelona", "munich", "zurich",
    "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "pune", "chennai",
    "tokyo", "osaka", "seoul", "beijing", "shanghai", "shenzhen", "hong kong",
    "taipei", "sydney", "melbourne", "auckland", "dubai", "tel aviv",
    "sao paulo", "mexico city", "bogota", "buenos aires", "warsaw", "krakow",
    "lisbon", "stockholm", "copenhagen", "oslo", "helsinki", "milan", "rome",
    "kuala lumpur", "jakarta", "bangkok", "manila", "hanoi", "ho chi minh",
    "brazil", "india", "japan", "china", "australia", "germany", "france",
    "spain", "italy", "netherlands", "poland", "ireland", "israel",
)
_SG_MARKERS = ("singapore", "sgp", " sg ", ",sg", "sg,")


def classify_location(text: str) -> tuple[bool, str]:
    """(is_overseas, basis). Unknown locations are NOT called overseas.

    The first version read `is_overseas = ... and "remote" not in lowered`,
    which meant any US posting saying "US-Remote" was classified Singapore.
    'Remote' says nothing about country on its own -- it has to be paired with
    a place before it means anything.
    """
    lowered = f" {job_schema.norm_text(text).lower()} "
    if not lowered.strip():
        return False, "no location given"

    if any(marker in lowered for marker in _SG_MARKERS):
        return False, "names Singapore"

    hits = [place for place in _NON_SG if place in lowered]
    if hits:
        return True, f"names {hits[0]}"

    # Region-scoped remote. 'US-Remote' is not a Singapore job, but a bare 'us'
    # token is too dangerous to match on -- it appears inside ordinary words and
    # in prose. Only the compound forms are safe.
    for scope in ("us-remote", "us remote", "usa-remote", "uk-remote",
                  "uk remote", "emea", "latam", "amer remote", "nam remote",
                  "europe remote", "remote - us", "remote (us", "remote, us"):
        if scope in lowered:
            return True, f"region-scoped remote ({scope.strip()})"

    if "remote" in lowered:
        # Remote with no region named. Could be APAC-eligible; keep it and let
        # the reader judge rather than guessing either way.
        return False, "remote, region unstated"

    return False, "unrecognised location, kept"


def _public_url(record: dict[str, Any], vendor: str, token: str) -> str:
    """The URL a human applies at. Several vendors publish only an API ref."""
    url = job_schema.norm_text(
        _first(record, "absolute_url", "hostedUrl", "jobUrl", "url",
               "applyUrl", "careers_url", "shareLink", default=""))
    if url and url.startswith("http") and "api." not in url:
        return url

    job_id = job_schema.norm_text(_first(record, "id", "uuid", "shortcode",
                                         "friendly_id", default=""))
    if vendor == "smartrecruiters":
        # `ref` is an API endpoint, not an apply page -- using it sends the
        # reader to raw JSON.
        return f"https://jobs.smartrecruiters.com/{token}/{job_id}"
    if vendor == "workday":
        path = job_schema.norm_text(record.get("externalPath"))
        parts = token.split("|")
        if path and len(parts) == 3:
            return f"https://{parts[0]}.{parts[1]}.myworkdayjobs.com{path}"
    if vendor == "greenhouse":
        return f"https://boards.greenhouse.io/{token}/jobs/{job_id}"
    if vendor == "lever":
        return f"https://jobs.lever.co/{token}/{job_id}"
    if vendor == "ashby":
        return f"https://jobs.ashbyhq.com/{token}/{job_id}"
    if vendor == "personio":
        return f"https://{token}.jobs.personio.de/job/{job_id}"
    if vendor == "recruitee":
        return f"https://{token}.recruitee.com/o/{job_id}"
    if vendor == "breezy":
        return f"https://{token}.breezy.hr/p/{job_id}"
    return url


def to_job(record: dict[str, Any], vendor: str, company: str,
           token: str) -> dict[str, Any] | None:
    """Map any vendor's record onto the canonical Job."""
    job_id = job_schema.norm_text(
        _first(record, "id", "uuid", "jobId", "shortcode", "friendly_id", "bulletFields"))
    if not job_id:
        return None

    job = job_schema.new_job(f"ats-{vendor}", job_id)
    job["title"] = job_schema.norm_text(
        _first(record, "title", "text", "name", "jobTitle", default=""))
    job["company"] = job_schema.norm_text(company)
    job["is_agency"] = job_schema.looks_like_agency(job["company"])

    job["url"] = _public_url(record, vendor, token)

    html = _first(record, "content", "description", "descriptionHtml",
                  "jobDescription", "publicDescription", default="") or ""
    if isinstance(html, dict):
        html = json.dumps(html)
    job["jd_html"] = html or None
    job["jd_text"] = job_schema.norm_text(html_text.flatten_html(str(html)))

    location = _text_of(_first(record, "location", "locations", "offices",
                               "locationsText", "city", default=""))
    if not location:
        location = _text_of(record.get("categories"))
    job["location"] = location or "Unknown"
    job["is_overseas"], location_basis = classify_location(location)
    job["is_remote"] = True if "remote" in location.lower() else None

    job["posted_at"] = job_schema.parse_date(
        _first(record, "first_published", "publishedAt", "createdAt",
               "published_at", "postedDate", "updated_at", "updatedAt"))

    # Ashby publishes compensation when the board opts in -- the only ATS here
    # that restores the salary signal MCF gives for free.
    comp = record.get("compensation") or {}
    if isinstance(comp, dict):
        tiers = comp.get("compensationTierSummary") or comp.get("summaryComponents")
        amounts = re.findall(r"[\d,]{4,}", str(tiers or ""))
        values = sorted({int(a.replace(",", "")) for a in amounts if a.replace(",", "").isdigit()})
        if len(values) >= 2:
            # Ashby quotes annual figures; to_monthly_sgd's plausibility guard
            # catches the conversion.
            job["salary_min_sgd"] = job_schema.to_monthly_sgd(values[0], "Annually")
            job["salary_max_sgd"] = job_schema.to_monthly_sgd(values[-1], "Annually")
            job["salary_is_stated"] = True
            job["salary_period_raw"] = "Annually (Ashby)"

    job["is_open"] = True   # a board listing it is the liveness signal
    job["liveness"] = "ALIVE"
    job["_provenance"] = {"source": f"ats-{vendor}", "token": token,
                          "fetched_at": job["_normalised_at"],
                          "location_basis": location_basis,
                          "applications": "absent"}
    return job_schema.finalise(job)


def fetch_board(company: str, vendor: str, token: str,
                cache_ttl_s: float = 900.0) -> list[dict[str, Any]]:
    """Every current posting on one company's board, normalised."""
    fetcher = FETCHERS.get(vendor)
    if not fetcher or not token:
        return []
    try:
        records = fetcher(token, cache_ttl_s)
    except Exception as exc:
        net._warn(f"ats: {vendor}:{token} raised ({exc})")
        return []

    jobs = []
    for record in records:
        if not isinstance(record, dict):
            continue
        job = to_job(record, vendor, company, token)
        if job and not job_schema.validate_job(job):
            jobs.append(job)
    return jobs


def fetch_jobs(
    query: str,
    max_results: int = 200,
    singapore_only: bool = True,
    open_only: bool = True,
    cache_ttl_s: float = 900.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Search every known board. Same contract as the other sources.

    Boards have no search endpoint, so this pulls each board whole and filters
    on the title and JD locally.
    """
    counters = {"boards": 0, "fetched": 0, "dropped_overseas": 0,
                "dropped_offtopic": 0, "kept": 0}
    terms = [t for t in re.split(r"\s+", query.lower()) if len(t) > 2]

    # Collect per board first, then interleave. Draining boards in order let
    # the first one consume the entire budget -- Stripe's 524 listings meant
    # three other boards were never reached, and the run looked like it had
    # searched everything.
    per_board: list[list[dict[str, Any]]] = []
    for company, board in load_boards().items():
        vendor, token = board.get("vendor"), board.get("token")
        if not vendor or not token:
            continue
        counters["boards"] += 1
        matched: list[dict[str, Any]] = []
        for job in fetch_board(company, vendor, token, cache_ttl_s):
            counters["fetched"] += 1
            if singapore_only and job["is_overseas"]:
                counters["dropped_overseas"] += 1
                continue
            # Boards have no search endpoint, so relevance is decided here --
            # and matching on JD prose is far too loose. 'Company Strategy &
            # Operations' at Stripe matched "machine learning engineer" because
            # the words appeared somewhere in a long description. The title is
            # what the role IS; the body merely mentions things.
            title = job["title"].lower()
            if terms and not any(t in title for t in terms):
                # Allow a body match only when most of the query is present,
                # which catches "ML Platform Engineer" for "machine learning".
                body = job["jd_text"][:3000].lower()
                hits = sum(1 for t in terms if t in body)
                if hits < max(2, len(terms) - 1):
                    counters["dropped_offtopic"] += 1
                    continue
            job["scope"] = query
            matched.append(job)
        if matched:
            per_board.append(matched)

    # Round-robin, so every board contributes before any board contributes twice.
    jobs: list[dict[str, Any]] = []
    index = 0
    while per_board and len(jobs) < max_results:
        progressed = False
        for board_jobs in per_board:
            if index < len(board_jobs):
                jobs.append(board_jobs[index])
                progressed = True
                if len(jobs) >= max_results:
                    break
        if not progressed:
            break
        index += 1

    counters["kept"] = len(jobs)
    return jobs, counters
