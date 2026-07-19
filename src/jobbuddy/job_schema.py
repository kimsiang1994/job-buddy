"""The canonical Job record, and the deterministic normalisers that build it.

Every source maps into one dict shape so that scoring, storage and rendering
never need to know where a job came from. Nothing in this module calls an LLM
or the network -- it is all pure functions over already-fetched data, which is
what makes it testable offline with captured fixtures.

Two identity fields, deliberately distinct:

  job_key      "{source}:{id}"  -- this exact posting
  content_key  hash of title+company+jd  -- this *role*, across repostings

Repost detection is exactly "same content_key, different job_key". Collapsing
them into one id would make that undetectable.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Any

SCHEMA_VERSION = 1

# Ordered worst-to-best. Index distance is the seniority gap, so order matters.
SENIORITY_LADDER = (
    "intern",
    "junior",
    "mid",
    "senior",
    "lead",
    "principal",
    "manager",
    "director",
    "executive",
)

# MCF's positionLevels vocabulary, plus the free-text forms other boards use.
_SENIORITY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bintern(ship)?\b|\btrainee\b|\bapprentice\b", "intern"),
    (r"\bfresh grad|\bgraduate\b|\bjunior\b|\bentry[- ]level\b|\bassociate\b", "junior"),
    (r"\bvice president\b|\bvp\b|\bchief\b|\bhead of\b|\bc[teoi]o\b", "executive"),
    (r"\bdirector\b", "director"),
    (r"\bmanager\b|\bmanagement\b", "manager"),
    (r"\bprincipal\b|\bdistinguished\b|\bfellow\b", "principal"),
    (r"\blead\b|\bstaff\b|\bteam lead\b", "lead"),
    (r"\bsenior\b|\bsnr\b|\bsr\.?\b|\bexperienced\b", "senior"),
    (r"\bprofessional\b|\bexecutive\b|\bmid[- ]level\b|\bofficer\b", "mid"),
)

# MCF salary.type.salaryType -> multiplier to monthly SGD.
_SALARY_PERIOD_TO_MONTHLY = {
    "monthly": 1.0,
    "annually": 1.0 / 12.0,
    "annual": 1.0 / 12.0,
    "yearly": 1.0 / 12.0,
    "weekly": 52.0 / 12.0,
    "daily": 21.0,        # working days per month
    "hourly": 21.0 * 8.0,
}

# Agencies re-post the same requisition under their own name. Applying to one
# role through three agencies looks desperate and wastes everybody's time, so
# this feeds both dedupe and the application_friction score.
_AGENCY_NAME_PATTERNS = (
    # Prefixes, not whole words: 'TALENTSIS' and 'Talent Pulse' are both
    # agencies, and \btalent\b matches neither.
    r"\brecruit", r"\bstaffing\b", r"\bmanpower\b", r"\bconsultan", r"\btalent",
    r"\bsearch\b", r"\bheadhunt", r"\bhr\s+solutions\b", r"\bemployment\b",
    r"\boutsourc", r"\bresourc(e|ing)\b", r"\bpersonnel\b",
    r"\bmichael page\b", r"\brobert walters\b", r"\brandstad\b", r"\badecco\b",
    r"\bkelly services\b", r"\bhays\b", r"\bmanpowergroup\b", r"\bpersolkelly\b",
    r"\bflintex\b", r"\bgeco\b", r"\bscientec\b", r"\btrust recruit\b",
)
_AGENCY_RE = re.compile("|".join(_AGENCY_NAME_PATTERNS), re.I)

# Suffixes that are noise when comparing company identity.
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(pte\.?|private|ltd\.?|limited|llp|llc|inc\.?|corp\.?|corporation|"
    r"holdings?|group|company|co\.?|sg|singapore|asia|international|"
    r"technologies|technology)\b",
    re.I,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_text(value: Any) -> str:
    """Collapse whitespace and normalise unicode. Never raises."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def norm_jd_text(value: Any) -> str:
    """Normalise a job description while keeping its line structure.

    `norm_text` collapses every run of whitespace, which is right for a title
    and wrong for a description: it welds `Nice to have` onto the end of the
    previous sentence, and the skill extractor then reads every optional skill
    as mandatory. Spaces collapse, line breaks survive.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def norm_title(title: Any) -> str:
    """Lowercase a job title and strip the decoration boards add.

    Titles arrive as 'Senior ML Engineer (AI Platform) - Up to $12k!' and the
    bracketed/after-dash parts are almost always location, salary or urgency
    rather than the role. Stripping them makes content_key stable across a
    repost that only changed the salary teaser.
    """
    text = norm_text(title).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[\[\{][^\]\}]*[\]\}]", " ", text)
    text = re.split(r"\s+[-–—|/]\s+", text)[0]
    text = re.sub(r"[^a-z0-9+#. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_company(name: Any) -> str:
    """Lowercase a company name and drop legal-entity noise.

    'TIKTOK PTE. LTD.' and 'TikTok Singapore' must collapse to the same key or
    company-level metrics (open reqs, velocity) count the same employer twice.
    """
    text = norm_text(name).lower()
    text = re.sub(r"[^a-z0-9& ]+", " ", text)
    text = _COMPANY_SUFFIX_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_agency(company_name: Any, posted_on_behalf: bool | None = None) -> bool:
    """True when a posting is probably a recruiter rather than the employer."""
    if posted_on_behalf:
        return True
    return bool(_AGENCY_RE.search(norm_text(company_name)))


def normalise_seniority(*hints: Any) -> str | None:
    """Map free text onto SENIORITY_LADDER. First match wins, most senior first.

    Order inside _SENIORITY_PATTERNS matters: 'Senior Engineering Manager' must
    resolve to manager, not senior, so manager is tested first.
    """
    blob = " ".join(norm_text(h).lower() for h in hints if h)
    if not blob:
        return None
    for pattern, level in _SENIORITY_PATTERNS:
        if re.search(pattern, blob):
            return level
    return None


def step_up(level: str | None, steps: int = 1) -> str | None:
    """The level `steps` above `level`, clamped to the top of the ladder.

    Job searches are aspirational. Someone at mid is looking for senior, so the
    thing being matched against is the level they want, not the one they hold.
    """
    if level not in SENIORITY_LADDER:
        return None
    index = min(SENIORITY_LADDER.index(level) + steps, len(SENIORITY_LADDER) - 1)
    return SENIORITY_LADDER[index]


def seniority_from_years(years: Any) -> str | None:
    """Infer a level from required years of experience.

    Zero is treated as *unknown*, not as "no experience needed". MCF stores 0
    for a great many postings that plainly want a senior hire -- it is the
    default when the employer left the field alone.
    """
    if not isinstance(years, int) or years <= 0:
        return None
    if years <= 1:
        return "junior"
    if years <= 4:
        return "mid"
    if years <= 8:
        return "senior"
    return "lead"


def resolve_seniority(
    title: Any = None,
    years: Any = None,
    position_level: Any = None,
) -> tuple[str | None, str]:
    """Best-effort seniority, plus the basis it was decided on.

    Precedence is title, then years, then the board's own level field -- learned
    the hard way from live MCF data, where `positionLevels` gave 'Manager' for a
    role wanting 0 years and 'Professional' for one wanting 10. The title is
    written by the hiring manager and says what they mean; the level field is a
    dropdown someone clicked through.

    Returns (level, basis) so scoring can discount a weak inference and
    `--explain` can show its working.
    """
    from_title = normalise_seniority(title)
    if from_title:
        return from_title, "title"

    from_years = seniority_from_years(years)
    if from_years:
        return from_years, "years"

    from_level = normalise_seniority(position_level)
    if from_level:
        return from_level, "position_level"

    return None, "unknown"


def seniority_distance(a: str | None, b: str | None) -> int | None:
    """Signed ladder distance, b relative to a. None when either is unknown."""
    if a not in SENIORITY_LADDER or b not in SENIORITY_LADDER:
        return None
    return SENIORITY_LADDER.index(b) - SENIORITY_LADDER.index(a)


# Above this, a figure labelled "monthly" is almost certainly annual data that
# the employer typed into the wrong field. Seen live: a Coupang posting listing
# 200000-300000 "Monthly" (i.e. SGD 2.4-3.6M/year for an IC role). Left alone it
# maxes out the pay score and takes the top of the ranking.
#
# The trade-off is explicit: a genuine SGD 65k/month package would be divided
# wrongly. That is rarer than the data-entry error, and the failure is visible
# in the output rather than silently distorting every rank.
MONTHLY_PLAUSIBILITY_CEILING_SGD = 60000


def to_monthly_sgd(amount: Any, period: Any) -> int | None:
    """Convert a salary figure to monthly SGD. None when it cannot be trusted.

    MCF states salary in SGD by law, so no FX is applied here; a source that
    quotes another currency must convert before calling this.
    """
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    key = norm_text(period).lower()
    multiplier = _SALARY_PERIOD_TO_MONTHLY.get(key)
    if multiplier is None:
        # Unknown period: guess from magnitude rather than silently returning a
        # number that is 12x wrong. Monthly SG tech salaries do not reach 100k.
        multiplier = 1.0 / 12.0 if value >= 100000 else 1.0

    monthly = value * multiplier
    if monthly > MONTHLY_PLAUSIBILITY_CEILING_SGD:
        monthly = monthly / 12.0
    return int(round(monthly))


def salary_was_adjusted(amount: Any, period: Any) -> bool:
    """True when to_monthly_sgd had to override the stated period."""
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return False
    multiplier = _SALARY_PERIOD_TO_MONTHLY.get(norm_text(period).lower(), 1.0)
    return value * multiplier > MONTHLY_PLAUSIBILITY_CEILING_SGD


def parse_date(value: Any) -> str | None:
    """Parse the date shapes seen across sources into ISO 8601 date, or None."""
    text = norm_text(value)
    if not text:
        return None
    # Trim an ISO timestamp down to the date part.
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%b %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def days_between(start: str | None, end: str | None = None) -> float | None:
    """Whole days from `start` to `end` (default today). None if unparseable."""
    start_date = parse_date(start)
    if not start_date:
        return None
    end_date = parse_date(end) or date.today().isoformat()
    try:
        d0 = date.fromisoformat(start_date)
        d1 = date.fromisoformat(end_date)
    except ValueError:
        return None
    return float((d1 - d0).days)


def job_key(source: str, source_job_id: Any) -> str:
    """Stable primary key for one posting."""
    return f"{norm_text(source).lower()}:{norm_text(source_job_id)}"


def content_key(title: Any, company: Any, jd_text: Any) -> str:
    """Stable key for one *role*, so a repost under a new id still matches.

    Only the first 4000 characters of the JD are hashed: boards append
    boilerplate (EEO statements, cookie notices) that changes independently of
    the role, and a full-text hash would treat that as a different job.
    """
    basis = "|".join([
        norm_title(title),
        norm_company(company),
        norm_text(jd_text)[:4000].lower(),
    ])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def new_job(source: str, source_job_id: Any) -> dict[str, Any]:
    """An empty canonical Job with every field present.

    Fields always exist, so downstream code uses `job["x"]` rather than
    `.get("x")` and a typo raises instead of silently reading None.
    """
    return {
        "job_key": job_key(source, source_job_id),
        "content_key": "",
        "source": norm_text(source).lower(),
        "source_job_id": norm_text(source_job_id),
        "url": "",
        "api_url": None,

        "title": "",
        "title_norm": "",
        "company": "",
        "company_norm": "",
        "company_uen": None,
        "is_agency": False,
        "jd_text": "",
        "jd_html": None,

        "location": "",
        "is_overseas": False,
        "is_remote": None,
        "employment_types": [],
        "seniority": None,
        "seniority_raw": None,
        "seniority_basis": "unknown",
        "min_years_exp": None,

        "salary_min_sgd": None,
        "salary_max_sgd": None,
        "salary_is_stated": False,
        "salary_period_raw": None,

        "skills_raw": [],
        "skills_key": [],
        "categories": [],
        "ssoc_code": None,

        "posted_at": None,
        "expires_at": None,
        "source_status": None,
        "is_open": True,

        # Competition. MCF publishes real counts; other sources leave these None
        # and fall back to the age/repost proxies.
        "applications": None,
        "views": None,
        "apps_per_view": None,
        "apps_per_day": None,
        "repost_count": 0,
        "edit_count": None,
        "vacancies": None,

        # Owned by job_store, never by a source.
        "first_seen_at": None,
        "last_seen_at": None,
        "seen_count": 0,
        "age_days": None,
        "reposted": False,
        "repost_of": [],
        "absent_runs": 0,
        "liveness": "UNKNOWN",

        "scores": {},
        "scope": None,

        "_provenance": {},
        "_schema_version": SCHEMA_VERSION,
        "_normalised_at": _now_iso(),
    }


def finalise(job: dict[str, Any]) -> dict[str, Any]:
    """Fill derived fields once a source has populated the raw ones.

    Idempotent: safe to call twice.
    """
    job["title_norm"] = norm_title(job["title"])
    job["company_norm"] = norm_company(job["company"])
    if not job["content_key"]:
        job["content_key"] = content_key(job["title"], job["company"], job["jd_text"])

    if job["seniority"] is None:
        job["seniority"], job["seniority_basis"] = resolve_seniority(
            title=job["title"],
            years=job["min_years_exp"],
            position_level=job["seniority_raw"],
        )

    # Competition ratios. Derived here rather than at scoring time so they are
    # recorded in the sightings log and stay comparable over time.
    apps, views = job["applications"], job["views"]
    if isinstance(apps, int) and isinstance(views, int) and views > 0:
        job["apps_per_view"] = round(apps / views, 4)
    age = days_between(job["posted_at"])
    if age is not None:
        job["age_days"] = age
        if isinstance(apps, int):
            job["apps_per_day"] = round(apps / max(age, 1.0), 3)

    return job


def validate_job(job: dict[str, Any]) -> list[str]:
    """Return a list of problems. Empty means the record is usable.

    Used as a gate before a job enters the store, so that one malformed record
    cannot poison the history that competition metrics are folded from.
    """
    problems: list[str] = []
    if not job.get("job_key") or job["job_key"].endswith(":"):
        problems.append("job_key is empty or has no source id")
    if not norm_text(job.get("title")):
        problems.append("title is empty")
    if not norm_text(job.get("company")):
        problems.append("company is empty")
    if not job.get("url"):
        problems.append("url is empty")
    if not job.get("content_key"):
        problems.append("content_key was not computed (call finalise)")

    lo, hi = job.get("salary_min_sgd"), job.get("salary_max_sgd")
    if isinstance(lo, int) and isinstance(hi, int) and lo > hi:
        problems.append(f"salary min {lo} exceeds max {hi}")

    seniority = job.get("seniority")
    if seniority is not None and seniority not in SENIORITY_LADDER:
        problems.append(f"seniority {seniority!r} is not on the ladder")

    return problems
