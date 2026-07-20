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

## Confidential fields

Salary is the most sensitive thing here. It is used only for local arithmetic --
comparing a listing's range against your position -- and it must never be sent
anywhere. `redact_for_llm()` strips it, along with every other direct
identifier, and `test_pipeline.py` asserts that. Use it for anything crossing
the process boundary: an API call, a log line, a shared file.

The salary never needs to leave this machine. Scoring turns it into a ratio
before anything else sees it, and a ratio is not a salary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[2]
INTAKE_DIR = REPO_DIR / "intake"
RESUME_STORE = INTAKE_DIR / "resumes"
SUBMISSIONS_LOG = INTAKE_DIR / "submissions.jsonl"
CURRENT_PROFILE = INTAKE_DIR / "current_profile.json"

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")

# Compulsory. Everything else is optional or derived.
REQUIRED_FIELDS = ("full_name", "resume_path", "target_roles")

_warned: set[str] = set()

# One writer per file, per the repo rule. See save_current() for the two
# distinct races this closes -- a shared temp name, and concurrent os.replace
# onto the same destination, which on Windows fails outright.
_save_lock = threading.Lock()


def _warn(message: str) -> None:
    """Warn to stderr once per process.

    The catch is narrow on purpose. This is the module's whole reporting
    channel, so a blanket `except Exception: pass` here does not degrade one
    warning -- it silently disables every degraded-path message in the file,
    including the ones that explain why a resume came back empty. Only a
    detached, closed or unencodable stderr can realistically fail, and those
    are OSError/ValueError; anything else is a bug that should surface.

    `_warned.add` also moved AFTER the print. Marking first meant a message
    that failed to print was recorded as already-warned, so the retry that
    would have worked never printed either.
    """
    import sys

    if message in _warned:
        return
    try:
        print(f"user_input: {message}", file=sys.stderr)
    except (OSError, ValueError):
        # stderr is gone. Nothing to report it to, by definition -- but do not
        # mark the message as warned, so a later call with a working stream
        # still gets a chance to say it.
        return
    _warned.add(message)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Resume reading
# --------------------------------------------------------------------------

def read_resume_text(path: str | Path) -> tuple[str, str]:
    """Extract text from a resume. Returns (text, how).

    Never raises -- a failed read returns ("", reason) so the caller can show a
    useful message instead of a stack trace in a notebook cell. That reason is
    the ONLY thing the user ever sees about the failure, which is why its
    precision is load-bearing rather than cosmetic.

    **One try block per operation, and every reason names the exception type.**
    This used to be two blanket catches, each spanning four operations, each
    formatting only `{exc}`. Opening a shredded file, an encrypted file, and a
    page-30 content stream that pypdf chokes on all arrived at the user as
    "could not read PDF: <some message>" with no type and no operation -- so a
    bug in extraction was indistinguishable from a corrupt document, and the
    only two available fixes ("re-export the PDF" and "report the pypdf bug")
    could not be told apart. The blocks below are split so the reason says
    which step failed, and the page number, because a resume whose first
    twenty-nine pages extract fine is not a corrupt file.

    A failing page aborts the read rather than being skipped. Partial resume
    text is worse than none: `derive_years_experience` and `derive_skills` would
    both run on it and quietly under-report, and the user would see a plausible
    profile rather than an error.

    Each block ends with a last-resort catch. That is deliberate and is not the
    old blanket: it names the operation AND `type(exc).__name__`, so an
    unforeseen failure is still diagnosable. It exists because pypdf's
    extraction walks attacker-shaped content streams and raises well outside its
    own exception hierarchy (KeyError, struct.error, RecursionError have all
    been seen), and the repo rule that this function cannot raise outranks the
    rule that catches are narrow.
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
            from pypdf import errors as pypdf_errors
        except ImportError:
            return "", "pypdf is not installed -- run: py -m pip install pypdf"

        # 1. Open and read the page tree. A truncated header, an empty file, a
        #    password-protected document and a broken xref all fail HERE, and
        #    none of them is an extraction problem.
        try:
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
        except pypdf_errors.FileNotDecryptedError as exc:
            return "", (f"could not open PDF: {type(exc).__name__}: {exc} -- "
                        "it is password-protected; save an unprotected copy")
        except pypdf_errors.DependencyError as exc:
            return "", (f"could not open PDF: {type(exc).__name__}: {exc} -- "
                        "it uses a cipher pypdf needs a helper for: "
                        "py -m pip install cryptography")
        except pypdf_errors.PyPdfError as exc:
            # EmptyFileError, PdfReadError, PdfStreamError, ParseError.
            return "", (f"could not open PDF: {type(exc).__name__}: {exc} -- "
                        "the file itself is damaged, not just one page")
        except (OSError, ValueError) as exc:
            return "", f"could not open PDF: {type(exc).__name__}: {exc}"
        except Exception as exc:
            return "", (f"could not open PDF: unexpected "
                        f"{type(exc).__name__}: {exc}")

        # 2. Extract, one page at a time. The page number is the whole point:
        #    "page 30 of 31 raised KeyError" is a bug report, while "could not
        #    read PDF" is a shrug.
        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except pypdf_errors.PyPdfError as exc:
                return "", (f"could not extract text from page {number} of "
                            f"{page_count}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                # pypdf raises outside its own hierarchy on malformed content
                # streams. Named, not swallowed.
                return "", (f"could not extract text from page {number} of "
                            f"{page_count}: unexpected "
                            f"{type(exc).__name__}: {exc}")

        # 3. Join and judge. Nothing here can fail on a list of str, so it is
        #    outside the catches rather than hidden inside them.
        text = "\n".join(pages).strip()
        if not text:
            return "", (f"no text layer in this PDF ({page_count} page(s) "
                        "extracted, all empty) -- it is probably a scan. "
                        "Export a text PDF from Word/Docs, or supply a .docx")
        return text, f"pypdf ({page_count} page(s))"

    if suffix == ".docx":
        try:
            import docx  # python-docx
            from docx.opc.exceptions import OpcError
        except ImportError:
            return "", "python-docx is not installed -- run: py -m pip install python-docx"

        # Opening a .docx is a zip open plus an XML parse; reading paragraphs is
        # a separate walk over the document body. A file that is not really a
        # .docx (a renamed .doc, most often) fails at the first and needs a
        # different answer from one whose body will not walk.
        try:
            document = docx.Document(str(path))
        except OpcError as exc:
            # PackageNotFoundError is the common one: not a zip, or not OOXML.
            return "", (f"could not open DOCX: {type(exc).__name__}: {exc} -- "
                        "this is probably not really a .docx (an old .doc "
                        "renamed?); re-save it as .docx from Word")
        except (OSError, ValueError, KeyError) as exc:
            return "", f"could not open DOCX: {type(exc).__name__}: {exc}"
        except Exception as exc:
            return "", (f"could not open DOCX: unexpected "
                        f"{type(exc).__name__}: {exc}")

        try:
            paragraphs = [p.text for p in document.paragraphs]
        except Exception as exc:
            return "", (f"could not read the paragraphs of this DOCX: "
                        f"{type(exc).__name__}: {exc}")
        return "\n".join(paragraphs).strip(), "python-docx"

    # .txt / .md. `UnicodeDecodeError` is caught alongside OSError because it is
    # not an OSError -- a CV saved as UTF-16 or Latin-1 raised straight out of
    # this function, which is exactly the stack-trace-in-a-notebook-cell the
    # contract above promises never happens.
    try:
        return path.read_text(encoding="utf-8-sig").strip(), "plain text"
    except UnicodeDecodeError as exc:
        return "", (f"could not decode this file as UTF-8 "
                    f"({type(exc).__name__}: {exc}) -- re-save it as UTF-8")
    except OSError as exc:
        return "", f"could not read file: {type(exc).__name__}: {exc}"


def resume_fingerprint(path: str | Path) -> str:
    """Content hash, so an unchanged resume is not stored twice.

    Returns "" if the file cannot be read, and says so. The empty string is not
    inert: `archive_resume` treats a missing hash as "nothing to archive" and
    returns None, so a permission error here silently costs the user their
    resume archive and `log_submission` records `resume_archived_as: null` with
    no explanation anywhere in the run. The archive-write failure warns; the
    fingerprint failure that causes the same outcome used to be silent.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError as exc:
        _warn(f"could not fingerprint {path} ({exc}) - the resume will not be "
              "archived for this submission")
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
        from jobbuddy import skills_taxonomy
    except ImportError as exc:
        # Say so. An empty list here is indistinguishable downstream from "this
        # resume matched no skills", and `validate` turns that into the note
        # "only 0 skills matched the taxonomy" -- blaming the user's resume for
        # a missing module. `derive_seniority` already reports its equivalent
        # import failure as "unavailable"; this one used to be silent.
        _warn(f"skills_taxonomy unavailable ({exc}) - no skills could be "
              "derived; this is a missing module, not an empty resume")
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
    """True for 'Northwind Labs, AI Engineer, Platform  Feb 2026 - Present'.

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
        from jobbuddy import job_schema
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
    # How far up the ladder to aim, relative to current. 0 = same level,
    # 1 = the usual next step, 2 = a deliberate stretch.
    seniority_ambition: int = 1

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

    # Aim one level up unless told otherwise. Nobody runs a job search to move
    # sideways -- the point is better pay, better rank, or both. Defaulting the
    # target to the current level made the scorer rank staying-put roles top.
    if not profile.target_seniority and profile.current_seniority:
        from jobbuddy import job_schema

        stepped = job_schema.step_up(profile.current_seniority, profile.seniority_ambition)
        profile.target_seniority = stepped or profile.current_seniority
        profile.derived["target_seniority"] = (
            f"{profile.current_seniority} + {profile.seniority_ambition} "
            f"= {profile.target_seniority} (aiming up; set it yourself to change)"
        )

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
            f"current level read as {profile.current_seniority!r} from tenure alone, "
            "because no job title on the resume carries a level word ('AI Engineer' "
            f"says nothing about level). You are being matched against "
            f"{profile.target_seniority!r} -- adjust Ambition, or set the level "
            "exactly, if that is off"
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
    """Write the active profile, atomically. This is what the pipeline reads.

    Two separate races had to be closed, and neither was visible single-threaded.

    1. The temp file is per-call, not a fixed `current_profile.tmp`. os.replace
       is atomic, but only for a writer that owns its source file: with one
       shared temp name, writer B overwrites A's temp between A's write and A's
       replace (publishing B's bytes under A's call, or a half-written mix),
       and whichever replaces second hits FileNotFoundError because the first
       already consumed the file.

    2. Unique temp files alone are NOT enough on Windows. Concurrent
       os.replace() calls onto the same destination fail with
       `[WinError 5] Access is denied` -- the replace itself contends, not just
       the source. This was measured, not assumed: the regression test failed
       exactly this way with unique temps and no lock.

    So the write is serialised outright, which is the repo's "one writer per
    file" rule applied literally. The lock is per-process; that is the scope
    the thread pool needs.
    """
    payload = json.dumps(profile.to_dict(include_resume_text=False),
                         indent=2, ensure_ascii=False)
    with _save_lock:
        tmp_path = None
        try:
            INTAKE_DIR.mkdir(parents=True, exist_ok=True)
            # Same directory, so os.replace stays on one filesystem.
            handle, tmp_name = tempfile.mkstemp(dir=str(INTAKE_DIR),
                                                prefix="current_profile.",
                                                suffix=".tmp")
            tmp_path = Path(tmp_name)
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_path, CURRENT_PROFILE)
            return True
        except OSError as exc:
            _warn(f"could not save current profile ({exc})")
            if tmp_path is not None:
                # Do not leave the intake directory filling with orphaned temps.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    _warn(f"could not remove temp file {tmp_path} "
                          f"({cleanup_exc})")
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
# Confidentiality
# --------------------------------------------------------------------------

# Never leaves this machine. Salary is the obvious one, but a phone number or
# home address in a prompt is just as bad, and prompts get logged by providers.
CONFIDENTIAL_FIELDS = frozenset({
    "current_salary_sgd_monthly",
    "desired_salary_sgd_monthly",
    "min_salary_sgd_monthly",
    "phone",
    "email",
    "linkedin",
    "github",
    "notes",
})

# Kept, because tailoring genuinely needs them. `full_name` is on the resume
# being written anyway; the rest are job-matching inputs, not identifiers.
LLM_SAFE_FIELDS = frozenset({
    "full_name", "target_roles", "years_experience", "current_seniority",
    "target_seniority", "location", "open_to_remote", "work_authorization",
    "skills",
})


def redact_for_llm(profile: "IntakeProfile | dict[str, Any]") -> dict[str, Any]:
    """The subset of a profile that may cross the process boundary.

    Allowlist, not denylist. A new confidential field added to the dataclass is
    excluded by default rather than leaking until someone remembers to add it
    to a blocklist -- the failure mode of a denylist is silent and permanent,
    because prompts are retained by providers.

    Salary is deliberately absent. Pay scoring happens locally and produces a
    ratio; a ratio is not a salary, and the ratio is all any downstream stage
    needs.
    """
    data = profile if isinstance(profile, dict) else profile.to_dict()
    return {key: value for key, value in data.items() if key in LLM_SAFE_FIELDS}


def confidentiality_report(profile: "IntakeProfile") -> list[str]:
    """Plain statement of what is stored and what can leave. For the notebook."""
    stored = [f for f in sorted(CONFIDENTIAL_FIELDS)
              if getattr(profile, f, None) not in (None, "", [], 0)]
    lines = [
        f"stored locally in {INTAKE_DIR.name}/ (gitignored, never committed)",
    ]
    if stored:
        lines.append(f"confidential fields held: {', '.join(stored)}")
    lines.append("none of the above is sent to any API -- salary is converted "
                 "to a ratio locally and only the ratio is used")
    return lines


# --------------------------------------------------------------------------
# Handing off to the pipeline
# --------------------------------------------------------------------------

def to_run_config(profile: IntakeProfile, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Translate an intake profile into run_config.json shape.

    `base` supplies DEFAULTS ONLY -- weights, exclusion patterns, anything the
    profile has no opinion about. Where both have a view, the profile wins,
    because the profile came from the user's actual resume and the JSON is a
    checked-in file they may never have opened.

    That precedence was inverted once, with a silent and expensive result: the
    guard read `if profile.skills and not existing`, and `existing` was never
    falsy because run_config.json always ships a populated skills block. Every
    resume-derived skill was discarded, so the heaviest scoring component ran
    entirely on the checked-in defaults while the notebook told the user
    otherwise. Precedence is now unconditional and tested.
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
        filters["exclude_companies"] = list(profile.exclude_companies)
    filters["exclude_agencies"] = profile.exclude_agencies

    target = config["profile"]
    target["target_seniority"] = profile.target_seniority or profile.current_seniority
    target["current_seniority"] = profile.current_seniority
    target["years_experience"] = profile.years_experience
    target["current_salary_sgd_monthly"] = profile.current_salary_sgd_monthly
    target["skills"] = merge_skill_tiers(profile.skills, target.get("skills"))

    if profile.target_roles:
        config["scopes"] = [{
            "name": "intake-" + re.sub(r"[^a-z0-9]+", "-",
                                       (profile.target_roles[0] or "roles").lower()).strip("-"),
            "queries": profile.target_roles,
            "max_results_per_query": 50,
        }]

    return config


def merge_skill_tiers(
    derived: list[str] | None,
    configured: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    """Combine resume-derived skills with hand-tiered ones from run_config.json.

    Both matter, and neither should silently win:

      - The configured tiers carry a judgement only the user can make. Claiming
        `expert` in everything a resume mentions is how a tailored resume ends
        up overstating, so a skill the user has deliberately tiered keeps that
        tier.
      - A skill the resume evidences but the user never tiered still belongs in
        the profile. Dropping it means the job asking for it reads as a gap.

    So: configured tiers win for skills they mention; everything else derived
    lands in `working`. Never `expert` -- that is the user's call to promote.
    """
    from jobbuddy import skills_taxonomy

    tiers: dict[str, list[str]] = {"expert": [], "working": [], "familiar": []}
    for tier in tiers:
        for skill in (configured or {}).get(tier, []) or []:
            if skill not in tiers[tier]:
                tiers[tier].append(skill)

    already = {skills_taxonomy.canon(s) for group in tiers.values() for s in group}
    for skill in derived or []:
        if skills_taxonomy.canon(skill) not in already:
            tiers["working"].append(skill)
            already.add(skills_taxonomy.canon(skill))

    return tiers
