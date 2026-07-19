"""User intake: resume ingest, profile validation, derivation, and logging.

The notebook is a thin shell over this module. Logic does not live in .ipynb
files -- they cannot be diffed properly, cannot be unit tested, and encourage
copy-paste drift between cells. Everything here is importable and testable.

Three responsibilities:

  1. Read a resume (PDF/DOCX/TXT/MD) into text.
  2. Derive what can be derived, so the user types as little as possible --
     name, email, phone, years of experience and skills all come out of the
     resume itself. Every derived value is marked as derived, and the user can
     override any of them.
  3. Log every submission to intake/, so there is a history of what was asked
     for and which resume version it was asked with.

Deliberately no LLM. Derivation here is regex over resume text: it is free,
instant, offline, and its failures are visible rather than plausible. The LLM
extraction that builds the verified fact store is a separate, later step.

**Sole writer of intake/.** That directory is gitignored -- it holds a real
name, phone number and salary expectations, and this repo is public.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parent
INTAKE_DIR = REPO_DIR / "intake"
RESUME_STORE = INTAKE_DIR / "resumes"
SUBMISSIONS_LOG = INTAKE_DIR / "submissions.jsonl"
CURRENT_PROFILE = INTAKE_DIR / "current_profile.json"

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")

# Compulsory. Everything else is optional or derived.
REQUIRED_FIELDS = ("full_name", "resume_path", "target_roles")

_warned: set[str] = set()


def _warn(message: str) -> None:
    if message in _warned:
        return
    _warned.add(message)
    try:
        import sys

        print(f"user_input: {message}", file=sys.stderr)
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Resume reading
# --------------------------------------------------------------------------

def read_resume_text(path: str | Path) -> tuple[str, str]:
    """Extract text from a resume. Returns (text, how).

    Never raises -- a failed read returns ("", reason) so the caller can show a
    useful message instead of a stack trace in a notebook cell.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Format is checked before existence on purpose. Someone who exports a
    # .pages or .odt file needs to be told the format is wrong, not sent
    # hunting for a missing file when the real problem is the export.
    if suffix not in SUPPORTED_SUFFIXES:
        return "", (f"unsupported file type {suffix!r} -- "
                    f"use {', '.join(SUPPORTED_SUFFIXES)}")
    if not path.is_file():
        return "", f"file not found: {path}"

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", "pypdf is not installed -- run: py -m pip install pypdf"
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n".join(pages).strip()
            if not text:
                return "", ("no text layer in this PDF -- it is probably a scan. "
                            "Export a text PDF from Word/Docs, or supply a .docx")
            return text, f"pypdf ({len(reader.pages)} page(s))"
        except Exception as exc:
            return "", f"could not read PDF: {exc}"

    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            return "", "python-docx is not installed -- run: py -m pip install python-docx"
        try:
            document = docx.Document(str(path))
            return "\n".join(p.text for p in document.paragraphs).strip(), "python-docx"
        except Exception as exc:
            return "", f"could not read DOCX: {exc}"

    try:
        return path.read_text(encoding="utf-8-sig").strip(), "plain text"
    except OSError as exc:
        return "", f"could not read file: {exc}"


def resume_fingerprint(path: str | Path) -> str:
    """Content hash, so an unchanged resume is not stored twice."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Derivation from resume text
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Singapore mobile numbers are 8 digits starting 8 or 9, often with +65.
_PHONE_RE = re.compile(r"(?:\+65[\s-]?)?[89]\d{3}[\s-]?\d{4}\b")
_LINKEDIN_RE = re.compile(r"(?:linkedin\.com/in/)[\w-]+", re.I)
_GITHUB_RE = re.compile(r"(?:github\.com/)[\w-]+", re.I)

# Date ranges like "Feb 2026 - Present", "Aug 2023 – Nov 2024", "2021-2024".
_MONTHS = ("jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec")
_RANGE_RE = re.compile(
    rf"(?:({_MONTHS})[a-z]*\.?\s+)?(\d{{4}})\s*[-–—to]+\s*"
    rf"(?:(?:({_MONTHS})[a-z]*\.?\s+)?(\d{{4}})|present|current|now)",
    re.I,
)


def derive_contact(text: str) -> dict[str, Any]:
    """Pull contact details out of resume text."""
    out: dict[str, Any] = {}
    email = _EMAIL_RE.search(text)
    if email:
        out["email"] = email.group(0)
    phone = _PHONE_RE.search(text)
    if phone:
        out["phone"] = re.sub(r"\s+", " ", phone.group(0)).strip()
    linkedin = _LINKEDIN_RE.search(text)
    if linkedin:
        out["linkedin"] = "https://www." + linkedin.group(0).lstrip("/")
    github = _GITHUB_RE.search(text)
    if github:
        out["github"] = "https://" + github.group(0)
    return out


def derive_name(text: str) -> str | None:
    """Guess the candidate's name from the top of the resume.

    Heuristic and easily wrong, which is exactly why `full_name` is compulsory:
    this only ever pre-fills a field the user confirms.
    """
    for line in text.splitlines()[:6]:
        candidate = line.strip()
        if not candidate or len(candidate) > 60:
            continue
        if _EMAIL_RE.search(candidate) or _PHONE_RE.search(candidate):
            continue
        if any(ch.isdigit() for ch in candidate):
            continue
        words = candidate.split()
        if 2 <= len(words) <= 5 and all(w[:1].isalpha() for w in words):
            return candidate.title() if candidate.isupper() else candidate
    return None


# Section headings, used to read dates from the right part of the document.
_EXPERIENCE_HEADING_RE = re.compile(
    r"^\s*(work\s+)?(experience|employment|professional\s+experience|"
    r"career\s+history|work\s+history)\s*:?\s*$",
    re.I | re.M,
)
_OTHER_HEADING_RE = re.compile(
    r"^\s*(education|academic|qualifications?|technical\s+skills?|skills?|"
    r"languages?|certifications?|projects?|publications?|references?|"
    r"interests?|awards?|volunteer\w*)\s*:?\s*$",
    re.I | re.M,
)


def experience_section(text: str) -> tuple[str, bool]:
    """Slice out the work-experience part of a resume. Returns (text, found).

    Dates under EDUCATION are degree dates, not employment. Counting them
    inflated a 5-year career to 11.3 years on a real resume, which would then
    push every seniority match two levels too senior.
    """
    start_match = _EXPERIENCE_HEADING_RE.search(text)
    if not start_match:
        return text, False

    start = start_match.end()
    tail = text[start:]
    end_match = _OTHER_HEADING_RE.search(tail)
    section = tail[: end_match.start()] if end_match else tail
    return section, True


def derive_years_experience(text: str, exclude_before: int = 2000) -> tuple[float | None, dict]:
    """Estimate total professional experience from date ranges in the resume.

    Reads only the experience section when one can be found, merges overlapping
    ranges rather than summing them (two concurrent roles are not two careers),
    and returns evidence so the notebook can show its working.
    """
    section, found_section = experience_section(text)
    text = section
    month_index = {m: i + 1 for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"])}
    today = datetime.now(timezone.utc)

    spans: list[tuple[float, float]] = []
    evidence: list[str] = []

    for match in _RANGE_RE.finditer(text):
        start_month, start_year, end_month, end_year = match.groups()
        try:
            sy = int(start_year)
        except (TypeError, ValueError):
            continue
        if sy < exclude_before or sy > today.year:
            continue
        sm = month_index.get((start_month or "jan").lower()[:3], 1)

        if end_year:
            ey, em = int(end_year), month_index.get((end_month or "dec").lower()[:3], 12)
        else:
            ey, em = today.year, today.month  # "Present"

        start = sy + (sm - 1) / 12.0
        end = ey + (em - 1) / 12.0
        if end < start or end - start > 50:
            continue
        spans.append((start, end))
        evidence.append(match.group(0).strip())

    if not spans:
        return None, {"reason": "no date ranges found", "spans": [],
                      "read_experience_section_only": found_section}

    # Merge overlaps so concurrent roles are counted once.
    spans.sort()
    merged: list[list[float]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    total = sum(end - start for start, end in merged)
    return round(total, 1), {
        "spans_found": evidence[:12],
        "merged_ranges": len(merged),
        "read_experience_section_only": found_section,
        "note": ("overlapping roles merged, not summed"
                 + ("" if found_section else
                    "; NO experience heading found, so education dates may be "
                    "included -- check this number")),
    }


def derive_skills(text: str, vocabulary: dict[str, float] | None = None) -> list[str]:
    """Find known skill terms present in the resume.

    Matches against the taxonomy's alias table, so 'LLMs' in the resume finds
    'large language model'. Only reports skills that literally appear -- it
    never adds a skill the resume does not evidence.
    """
    try:
        import skills_taxonomy
    except ImportError:
        return []

    lowered = " " + re.sub(r"[^a-z0-9+#]+", " ", text.lower()) + " "
    found: dict[str, None] = {}
    for surface, canonical in skills_taxonomy.ALIASES.items():
        if len(surface) < 2:
            continue
        # Word-boundary search so 'ai' does not match 'available'.
        if re.search(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])", lowered):
            found[canonical] = None
    return sorted(found)


def _looks_like_role_line(line: str) -> bool:
    """True for 'TikTok, AI Engineer, Global Marketing Science  Feb 2026 - Present'.

    A role line names an employer and a title, usually with a date. Bullet
    points describing the work are not role lines, which matters because the
    bullets are full of words like 'CEO' and 'Director' that belong to other
    people.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 140:
        return False
    if stripped[:1] in "-•*•●":
        return False
    if "," not in stripped and "|" not in stripped and "\t" not in stripped:
        return False
    # Sentences are prose; role lines are fragments.
    if stripped.endswith(".") and len(stripped.split()) > 12:
        return False
    return True


def derive_seniority(text: str, years: float | None) -> tuple[str | None, str]:
    """Best-guess current seniority. Returns (level, basis).

    Reads only role lines in the experience section. Feeding a blob of resume
    prose to a title matcher produced 'executive' on a real resume, because a
    bullet said 'Direct report to CEO' -- describing someone else's job.
    """
    try:
        import job_schema
    except ImportError:
        return None, "unavailable"

    section, found = experience_section(text)
    role_lines = [ln for ln in section.splitlines() if _looks_like_role_line(ln)]

    # The most recent role is listed first on a reverse-chronological resume.
    for line in role_lines[:3]:
        level = job_schema.normalise_seniority(line)
        if level:
            return level, f"role line: {line.strip()[:60]}"

    if years:
        level = job_schema.seniority_from_years(int(years))
        if level:
            return level, f"inferred from {years} years of experience"

    return None, "could not determine -- please set it"


# --------------------------------------------------------------------------
# The profile
# --------------------------------------------------------------------------

@dataclass
class IntakeProfile:
    """One user's inputs. Compulsory fields are checked by `validate`."""

    full_name: str = ""
    resume_path: str = ""
    target_roles: list[str] = field(default_factory=list)

    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None

    years_experience: float | None = None
    current_seniority: str | None = None
    target_seniority: str | None = None

    current_salary_sgd_monthly: int | None = None
    desired_salary_sgd_monthly: int | None = None
    min_salary_sgd_monthly: int | None = None

    location: str = "Singapore"
    open_to_remote: bool = True
    work_authorization: str = "Singapore Citizen / PR"
    notice_period_weeks: int | None = None

    exclude_companies: list[str] = field(default_factory=list)
    exclude_agencies: bool = False
    skills: list[str] = field(default_factory=list)
    notes: str = ""

    resume_text: str = ""
    resume_hash: str = ""
    derived: dict[str, Any] = field(default_factory=dict)
    submitted_at: str = ""

    def to_dict(self, include_resume_text: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_resume_text:
            data["resume_text"] = f"<{len(self.resume_text)} chars, not stored inline>"
        return data


def build_profile(
    full_name: str = "",
    resume_path: str | Path = "",
    target_roles: Any = None,
    **overrides: Any,
) -> IntakeProfile:
    """Read the resume, derive what can be derived, apply explicit overrides.

    Precedence is always: what the user typed beats what we derived. A derived
    value is only ever a default.
    """
    profile = IntakeProfile()
    profile.full_name = str(full_name or "").strip()
    profile.resume_path = str(resume_path or "").strip()

    if isinstance(target_roles, str):
        profile.target_roles = [r.strip() for r in re.split(r"[,\n;]", target_roles) if r.strip()]
    elif target_roles:
        profile.target_roles = [str(r).strip() for r in target_roles if str(r).strip()]

    if profile.resume_path:
        text, how = read_resume_text(profile.resume_path)
        profile.resume_text = text
        profile.resume_hash = resume_fingerprint(profile.resume_path)
        profile.derived["resume_read"] = how
        if not text:
            profile.derived["resume_error"] = how

    if profile.resume_text:
        contact = derive_contact(profile.resume_text)
        for key, value in contact.items():
            setattr(profile, key, value)
        profile.derived["contact"] = sorted(contact)

        if not profile.full_name:
            guessed = derive_name(profile.resume_text)
            if guessed:
                profile.full_name = guessed
                profile.derived["full_name"] = "guessed from resume -- please confirm"

        years, evidence = derive_years_experience(profile.resume_text)
        profile.years_experience = years
        profile.derived["years_experience"] = evidence

        profile.skills = derive_skills(profile.resume_text)
        profile.derived["skills"] = f"{len(profile.skills)} matched against the taxonomy"

        profile.current_seniority, seniority_basis = derive_seniority(
            profile.resume_text, years
        )
        profile.target_seniority = profile.current_seniority
        profile.derived["current_seniority"] = seniority_basis

    # Explicit values win over everything derived.
    for key, value in overrides.items():
        if value in (None, "", [], {}):
            continue
        if not hasattr(profile, key):
            _warn(f"ignoring unknown field {key!r}")
            continue
        setattr(profile, key, value)
        profile.derived.pop(key, None)

    # A salary floor the user did not set: 90% of current pay, so the search is
    # not cluttered with roles that would be a pay cut.
    if profile.min_salary_sgd_monthly is None:
        anchor = profile.desired_salary_sgd_monthly or profile.current_salary_sgd_monthly
        if anchor:
            profile.min_salary_sgd_monthly = int(anchor * 0.9)
            profile.derived["min_salary_sgd_monthly"] = "90% of current/desired pay"

    return profile


def validate(profile: IntakeProfile) -> list[str]:
    """Return blocking problems. Empty list means good to run."""
    problems: list[str] = []

    if not profile.full_name.strip():
        problems.append("full_name is required")
    if not profile.resume_path.strip():
        problems.append("resume_path is required")
    elif not Path(profile.resume_path).is_file():
        problems.append(f"resume not found at {profile.resume_path}")
    elif not profile.resume_text:
        reason = profile.derived.get("resume_error", "no text extracted")
        problems.append(f"could not read resume: {reason}")
    if not profile.target_roles:
        problems.append("target_roles is required -- what are you applying for?")

    for name in ("current_salary_sgd_monthly", "desired_salary_sgd_monthly",
                 "min_salary_sgd_monthly"):
        value = getattr(profile, name)
        if value is not None and (not isinstance(value, int) or value <= 0):
            problems.append(f"{name} must be a positive whole number of SGD per month")

    current = profile.current_salary_sgd_monthly
    desired = profile.desired_salary_sgd_monthly
    if current and desired and desired < current:
        problems.append(
            f"desired pay ({desired}) is below current pay ({current}) -- "
            "set it deliberately or leave it blank"
        )

    if profile.years_experience is not None and not (0 <= profile.years_experience <= 60):
        problems.append(f"years_experience {profile.years_experience} is implausible")

    return problems


def warnings_for(profile: IntakeProfile) -> list[str]:
    """Non-blocking things the user should know before spending money."""
    notes: list[str] = []
    if profile.years_experience is None:
        notes.append("years of experience could not be derived -- set it manually, "
                     "it drives the seniority match")
    if not profile.current_salary_sgd_monthly and not profile.desired_salary_sgd_monthly:
        notes.append("no salary given -- pay scoring will use a generic anchor "
                     "rather than your actual position")
    if len(profile.skills) < 5:
        notes.append(f"only {len(profile.skills)} skills matched the taxonomy -- "
                     "skill scoring will be weak. Add them manually if the resume "
                     "words them unusually")
    if "full_name" in profile.derived:
        notes.append(f"name was guessed as {profile.full_name!r} -- confirm it")

    basis = str(profile.derived.get("current_seniority", ""))
    if basis.startswith("inferred from"):
        notes.append(
            f"seniority was inferred as {profile.current_seniority!r} from tenure alone, "
            "because no job title on the resume carries a level word. Titles like "
            "'AI Engineer' say nothing about level -- set target_seniority yourself"
        )
    if not profile.derived.get("years_experience", {}).get(
            "read_experience_section_only", True):
        notes.append("no EXPERIENCE heading was found, so years of experience may "
                     "include education dates -- check it")
    return notes


# --------------------------------------------------------------------------
# Logging. Sole writer of intake/.
# --------------------------------------------------------------------------

def archive_resume(profile: IntakeProfile) -> Path | None:
    """Copy the resume into intake/resumes/ keyed by content hash.

    Keeps a versioned history: when you tweak your resume and re-run, both
    versions are kept, so a past application can be reproduced with the exact
    document it was generated from.
    """
    if not profile.resume_path or not profile.resume_hash:
        return None
    source = Path(profile.resume_path)
    if not source.is_file():
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    target = RESUME_STORE / f"{stamp}_{profile.resume_hash}{source.suffix.lower()}"
    if target.exists():
        return target  # identical content already archived
    try:
        RESUME_STORE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
    except OSError as exc:
        _warn(f"could not archive resume ({exc}); continuing")
        return None


def log_submission(profile: IntakeProfile, extra: dict[str, Any] | None = None) -> str:
    """Append the submission to intake/submissions.jsonl. Returns submission id.

    The log answers "what did I ask for, with which resume, and when" months
    later, when a recruiter finally replies.
    """
    profile.submitted_at = _now_iso()
    submission_id = hashlib.sha256(
        f"{profile.submitted_at}{profile.full_name}{profile.resume_hash}".encode()
    ).hexdigest()[:12]

    archived = archive_resume(profile)
    record = {
        "submission_id": submission_id,
        "submitted_at": profile.submitted_at,
        "resume_archived_as": archived.name if archived else None,
        "profile": profile.to_dict(include_resume_text=False),
    }
    if extra:
        record["extra"] = extra

    try:
        INTAKE_DIR.mkdir(parents=True, exist_ok=True)
        with open(SUBMISSIONS_LOG, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        _warn(f"could not write submission log ({exc})")

    save_current(profile)
    return submission_id


def save_current(profile: IntakeProfile) -> bool:
    """Write the active profile, atomically. This is what the pipeline reads."""
    try:
        INTAKE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CURRENT_PROFILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(profile.to_dict(include_resume_text=False), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, CURRENT_PROFILE)
        return True
    except OSError as exc:
        _warn(f"could not save current profile ({exc})")
        return False


def load_current() -> dict[str, Any] | None:
    """Read the active profile, or None. Never raises."""
    if not CURRENT_PROFILE.is_file():
        return None
    try:
        return json.loads(CURRENT_PROFILE.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        _warn(f"could not read current profile ({exc})")
        return None


def submission_history(limit: int = 20) -> list[dict[str, Any]]:
    """Recent submissions, newest first. Skips damaged lines."""
    if not SUBMISSIONS_LOG.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(SUBMISSIONS_LOG, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError as exc:
        _warn(f"could not read submission log ({exc})")
        return []
    return list(reversed(rows))[:limit]


# --------------------------------------------------------------------------
# Handing off to the pipeline
# --------------------------------------------------------------------------

def to_run_config(profile: IntakeProfile, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Translate an intake profile into run_config.json shape.

    Skills are placed in the `working` tier by default. Tiering is a judgement
    the user must make -- claiming `expert` in everything the resume mentions is
    how a tailored resume ends up overstating.
    """
    import copy

    config = copy.deepcopy(base) if base else {}
    config.setdefault("filters", {})
    config.setdefault("profile", {})
    config.setdefault("weights", {})
    config.setdefault("scopes", [])

    filters = config["filters"]
    filters["singapore_only"] = profile.location.strip().lower() == "singapore"
    filters["open_only"] = True
    if profile.min_salary_sgd_monthly:
        filters["min_salary_sgd_monthly"] = profile.min_salary_sgd_monthly
    if profile.exclude_companies:
        filters["exclude_companies"] = [c.lower() for c in profile.exclude_companies]
    filters["exclude_agencies"] = profile.exclude_agencies

    target = config["profile"]
    target["target_seniority"] = profile.target_seniority or profile.current_seniority
    target["years_experience"] = profile.years_experience
    target["current_salary_sgd_monthly"] = profile.current_salary_sgd_monthly
    existing = target.get("skills") or {}
    if profile.skills and not existing:
        target["skills"] = {"expert": [], "working": profile.skills, "familiar": []}
    else:
        target["skills"] = existing

    if profile.target_roles:
        config["scopes"] = [{
            "name": "intake-" + re.sub(r"[^a-z0-9]+", "-",
                                       (profile.target_roles[0] or "roles").lower()).strip("-"),
            "queries": profile.target_roles,
            "max_results_per_query": 50,
        }]

    return config
