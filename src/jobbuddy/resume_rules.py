"""Deterministic house rules for a rendered resume. No model runs here.

`fact_guard` answers "is this line true?". This module answers a different and
narrower question: "should this line be on a resume at all?" -- and it answers
it with regexes and arithmetic, never with an LLM. A style rule that needs a
model to evaluate it is a rule nobody can reproduce, argue with, or test.

Every rule below carries the grade of the evidence behind it, because a resume
checker is exactly the kind of tool that silently accumulates folklore. The
grades are:

  A  government or regulator guidance
  B  practitioner consensus across independent sources
  C  folklore -- traced to a vendor blog, a defunct statistic, or nothing

**Grade C rules are deliberately absent, and the ones that were considered are
named at the bottom of this docstring so nobody re-adds them by reflex.**

What is enforced:

  personal data   (A)  Singapore TAFEP/PDPC actively list these for removal:
                       photo, NRIC/FIN, DOB/age, gender, race, religion,
                       marital status, nationality, NS liability, salary.
  structure       (B)  bullet length, past-tense openers, no first-person,
                       conventional headings, full job titles, no hobbies,
                       contact details present.
  inflated verbs  (B)  spearheaded, orchestrated, leveraged, ...
  LLM vocabulary  (B)  delve, underscore, meticulous, crucial -- and only those.
  metric density  (B)  a cap, not a target. See `metric_density`.
  hidden text     (B)  the pipeline must never EMIT invisible text. See
                       `check_no_hidden_text` -- this one protects the user.

Severity routing is the contract with the renderer: `error` blocks the render
(personal-data leaks, hidden text -- both are harms that survive publication),
`warning` is reported and rendered anyway (style, which is a judgement call the
user is allowed to lose).

One deliberate omission in the reporting: no violation, and nothing
`summarise` returns, ever quotes the offending source text. That would defeat
the personal-data rules -- a report that echoes the NRIC it just found has
leaked it into logs, run reports and `--explain` output. Violations carry a
rule, a reason and a location, and the caller looks at the document itself.

NOT IMPLEMENTED, on purpose (all Grade C):
  - keyword density / keyword stuffing objectives
  - font restrictions, or avoiding bold/italics/bullets. Oracle states styling
    does not affect parsing.
  - blanket table avoidance
  - anything premised on "75% of resumes are auto-rejected by an ATS" -- that
    number traces to a vendor that has not existed since 2013
  - AI-detector evasion of any kind
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Personal data (Grade A -- TAFEP/PDPC guidance; the highest-confidence rules
# in this module, and the only ones sourced from a regulator rather than from
# practitioner consensus.)
#
# Note on what is NOT here: the "Personal Particulars" block that Singapore
# resume-SEO sites recommend. The convention is real in application *forms*;
# the leap to CVs is invented and has no recruiter support. Adding it would
# have this module require the exact fields the regulator asks people to strip.
# --------------------------------------------------------------------------

# Singapore NRIC/FIN: prefix letter, 7 digits, checksum letter.
#
# The match is on FORMAT, not on checksum. A mistyped or partially redacted
# NRIC is still a disclosed NRIC, and failing it open to protect against a
# false positive gets the trade-off backwards -- the cost of a false positive
# here is one warning a human dismisses in three seconds.
NRIC_RE = re.compile(r"\b([STFGMstfgm])(\d{7})([A-Za-z])\b")

# Documented checksum tables for the S/T (resident) and F/G (foreigner)
# series. The M series is newer and its algorithm is not reliably documented,
# so it is reported as unknown rather than guessed at -- an assertion this
# module cannot back up is worse than an admission that it cannot check.
_ST_CHECKSUM = "JZIHGFEDCBA"
_FG_CHECKSUM = "XWUTRQPNMLK"
_NRIC_WEIGHTS = (2, 7, 6, 5, 4, 3, 2)

# Labelled particulars. Anchored on the label rather than on the value,
# because the values ("Male", "Single", "Chinese") are ordinary English words
# that appear legitimately -- "Chinese" is a language skill far more often
# than it is a declared race, and flagging it bare would train users to ignore
# this module.
_LABELLED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("date_of_birth", r"\b(?:date\s+of\s+birth|d\.?\s?o\.?\s?b\.?|birthdate|"
                      r"birthday|born\s+on)\b\s*[:\-]?"),
    ("age", r"\bage\s*[:\-]\s*\d{1,2}\b"),
    ("age", r"\b\d{1,2}\s+years?\s+old\b"),
    ("gender", r"\b(?:gender|sex)\s*[:\-]"),
    ("race", r"\b(?:race|ethnicity|dialect\s+group)\s*[:\-]"),
    ("religion", r"\breligion\s*[:\-]"),
    ("marital_status", r"\bmarital\s+status\b\s*[:\-]?"),
    ("nationality", r"\b(?:nationality|citizenship)\s*[:\-]"),
    ("nationality", r"\b(?:singapore(?:an)?\s+citizen|singapore\s+p\.?r\.?|"
                    r"permanent\s+resident)\b"),
    ("national_service", r"\b(?:national\s+service|ns\s+liability|nsf\b|"
                         r"ns(?:men|man)\b|reservist)\b"),
    ("salary", r"\b(?:expected|current|last[-\s]drawn|desired)\s+"
               r"(?:salary|pay|remuneration|package)\b"),
    ("salary", r"\bsalary\s+(?:expectation|expected|history)\b"),
    ("salary", r"\bsalary\s*[:\-]\s*\$?\d"),
    ("photo", r"<img\b|\bdata:image/|\[\s*(?:photo|photograph|headshot)\s*\]"),
)

# A bare value on a line of its own IS a declared particular -- "Male" alone on
# a line is not prose, it is a form field with the label stripped.
_STANDALONE_VALUES: dict[str, str] = {
    "male": "gender", "female": "gender",
    "single": "marital_status", "married": "marital_status",
    "divorced": "marital_status", "widowed": "marital_status",
    "chinese": "race", "malay": "race", "indian": "race", "eurasian": "race",
    "buddhist": "religion", "christian": "religion", "muslim": "religion",
    "hindu": "religion", "catholic": "religion",
}

# Model keys that carry a particular directly, so a structured model is checked
# even when it renders no text.
_PERSONAL_MODEL_KEYS: dict[str, str] = {
    "photo": "photo", "photo_url": "photo", "avatar": "photo", "image": "photo",
    "nric": "nric", "fin": "nric", "ic": "nric",
    "date_of_birth": "date_of_birth", "dob": "date_of_birth", "age": "age",
    "gender": "gender", "sex": "gender", "race": "race",
    "ethnicity": "race", "religion": "religion",
    "marital_status": "marital_status", "nationality": "nationality",
    "citizenship": "nationality", "national_service": "national_service",
    "salary_expectation": "salary", "expected_salary": "salary",
}

# --------------------------------------------------------------------------
# Hidden text and prompt injection (Grade B *against* -- this rule exists to
# stop the pipeline doing something, not to help it.)
# --------------------------------------------------------------------------

_HIDDEN_STYLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("white_text", r"color\s*:\s*(?:#f{3}\b|#f{6}\b|white\b|"
                   r"rgba?\(\s*255\s*,\s*255\s*,\s*255)"),
    ("zero_size", r"font-size\s*:\s*0(?:\.0+)?\s*(?:px|pt|em|rem|%)?\b"),
    ("zero_size", r"font-size\s*:\s*[01](?:\.\d+)?\s*px\b"),
    ("invisible", r"opacity\s*:\s*0(?:\.0+)?\b"),
    ("invisible", r"visibility\s*:\s*hidden\b"),
    ("invisible", r"display\s*:\s*none\b"),
    ("off_canvas", r"(?:left|top|right|bottom|text-indent|margin-left)\s*:\s*-\d{3,}"),
    ("off_canvas", r"position\s*:\s*absolute[^;\"']*?-\d{3,}\s*(?:px|pt)"),
)

# Strings whose only purpose is to address an LLM reading the document.
_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|the\s+above|preceding)\s+"
    r"(?:instruction|prompt|direction)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|the\s+above)",
    r"\byou\s+are\s+(?:an?\s+)?(?:ai|llm|language\s+model|ats|screener|recruiter)\b",
    r"\bas\s+an\s+ai\s+(?:language\s+)?model\b",
    r"\b(?:system|assistant)\s*:\s*",
    r"(?:rate|score|rank|recommend|advance|shortlist)\s+this\s+candidate",
    r"\bthis\s+(?:candidate|applicant)\s+is\s+(?:a\s+)?(?:perfect|ideal|"
    r"the\s+best)\s+(?:match|fit|candidate)\b",
    r"<\s*/?\s*(?:system|instruction|prompt)\s*>",
)

# --------------------------------------------------------------------------
# Structure (Grade B -- practitioner consensus)
# --------------------------------------------------------------------------

# Case is spelled out rather than using re.I, because "us" and "me" are only
# pronouns in lower case: IGNORECASE turns "US markets" and "ME region" into
# first-person violations, and a rule with obvious false positives is a rule
# the user switches off. Sentence-initial capitals still have to match, which
# is the bug this replaced.
_PRONOUN_RE = re.compile(
    r"\b(I|I'm|I've|[Mm]y|[Mm]ine|[Ww]e|[Ww]e're|[Oo]ur|[Oo]urs|me|us)\b")

# Section headings a human reviewer expects. Matching is per-vendor and
# unconventional headings are the documented failure mode -- "Where I've Been"
# parses as a heading for nobody. Common formal variants are accepted because
# they are conventional, not because they are identical.
CANONICAL_HEADINGS: dict[str, str] = {
    "experience": "Experience",
    "work experience": "Experience",
    "professional experience": "Experience",
    "employment": "Experience",
    "employment history": "Experience",
    "work history": "Experience",
    "education": "Education",
    "skills": "Skills",
    "technical skills": "Skills",
    "core skills": "Skills",
    "projects": "Projects",
    "personal projects": "Projects",
    "summary": "Summary",
    "professional summary": "Summary",
    "certifications": "Certifications",
}

_HOBBY_HEADINGS = frozenset({
    "hobbies", "interests", "personal interests", "hobbies and interests",
    "pastimes", "outside work", "activities", "personal", "about me",
})

# Deliberately small. Each entry has an unambiguous expansion; "QA", "IT" and
# "VP" are absent because they ARE the conventional written form of the title.
TITLE_ABBREVIATIONS: dict[str, str] = {
    "sr": "Senior", "snr": "Senior", "jr": "Junior", "jnr": "Junior",
    "mgr": "Manager", "eng": "Engineer", "engr": "Engineer",
    "dev": "Developer", "dir": "Director", "asst": "Assistant",
    "assoc": "Associate", "exec": "Executive", "admin": "Administrator",
    "coord": "Coordinator", "swe": "Software Engineer",
    "sde": "Software Development Engineer", "ba": "Business Analyst",
    "sysadmin": "Systems Administrator",
}

# Past-tense openers that no "-ed" test will catch.
_IRREGULAR_PAST = frozenset("""
led ran built drove grew cut won made wrote sold taught took held kept set
spent began brought chose drew found gave met oversaw rebuilt rewrote sent
spoke stood put read rose broke came did got knew left paid ran saw shot
sought stood struck swept undertook wound wrote
""".split())

# Base forms accepted only in the CURRENT role, where present tense is correct.
_PRESENT_ACTION_VERBS = frozenset("""
lead run build drive grow cut win make write own manage design develop deliver
automate migrate launch ship scale define implement establish introduce
maintain support analyse analyze coach train partner present report oversee
coordinate monitor review test operate mentor negotiate forecast
""".split())

# --------------------------------------------------------------------------
# Word deny-lists (Grade B)
# --------------------------------------------------------------------------

# Inflated writing, nothing more.
#
# CORRECTION, and it matters: an earlier draft of the research behind this
# module claimed this list doubles as a tell for LLM authorship. That claim was
# WITHDRAWN -- "spearheaded" as an AI marker is folklore, and there is no
# evidence for it. It is implemented here purely as an inflated-writing filter,
# which is what the evidence actually supports. Do not re-justify it as an
# authorship signal; that reasoning was checked and failed.
INFLATED_VERBS: tuple[str, ...] = (
    "spearheaded", "orchestrated", "leveraged", "crafted", "pioneered",
    "enhanced", "revolutionized", "revolutionised", "transformed",
    "utilized", "utilised",
)

# Four words. Deliberately four.
#
# Dozens more circulate in blog posts ("tapestry", "testament", "realm",
# "elevate", ...) on the strength of nothing but someone's impression. These
# four are the ones with evidence behind them. Padding the list would turn a
# rule with a basis into a rule with a vibe, and every added word costs a false
# positive against somebody's legitimate sentence.
LLM_VOCABULARY: tuple[str, ...] = ("delve", "underscore", "meticulous", "crucial")

_INFLATED_RE = re.compile(r"\b(" + "|".join(INFLATED_VERBS) + r")\b", re.I)
_LLM_VOCAB_RE = re.compile(
    r"\b(" + "|".join(f"{w}\\w*" for w in LLM_VOCABULARY) + r")\b", re.I)

_NUMBER_RE = re.compile(r"\d")

# --------------------------------------------------------------------------
# Defaults. All thresholds are parameters -- a hard-coded threshold is a rule
# nobody can disagree with, and these are judgement calls.
# --------------------------------------------------------------------------

DEFAULT_CHARS_PER_LINE = 95      # ~a one-page A4 body column at 10-11pt
DEFAULT_MAX_BULLET_LINES = 2
DEFAULT_MAX_METRIC_DENSITY = 0.6


@dataclass
class Violation:
    """One rule failure, in terms a human can act on.

    `location` is a coordinate into the model ("bullet[3]", "document offset
    412"), never a quotation of the text. The personal-data rules are the
    reason: a violation that echoes the value it found has published it.
    """

    rule: str
    severity: str            # "error" blocks the render; "warning" does not
    detail: str
    location: str = ""

    def __str__(self) -> str:
        where = f" [{self.location}]" if self.location else ""
        return f"{self.severity}: {self.rule}: {self.detail}{where}"


@dataclass
class Report:
    """Outcome for one resume."""

    violations: list[Violation] = field(default_factory=list)
    bullets_checked: int = 0
    density: float = 0.0

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when nothing blocks the render. Warnings do not block."""
        return not self.errors

    @property
    def reasons(self) -> list[str]:
        return [str(v) for v in self.violations]


# --------------------------------------------------------------------------
# Model access. The read path cannot raise: this module is a gate, and a gate
# that crashes on a malformed model fails open on the render it was meant to
# stop.
# --------------------------------------------------------------------------

def _as_dict(model: Any) -> dict[str, Any]:
    return model if isinstance(model, dict) else {}


def _bullets(model: Any) -> list[dict[str, Any]]:
    """Normalise bullets to dicts. Accepts `tailor.tailor()` output shape,
    plain strings, or anything list-like that mixes the two."""
    out: list[dict[str, Any]] = []
    raw = _as_dict(model).get("bullets")
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        return out
    for item in raw:
        if isinstance(item, dict):
            text = str(item.get("text") or "")
            entry = dict(item)
            entry["text"] = text
        elif isinstance(item, str):
            entry = {"text": item}
        else:
            continue
        out.append(entry)
    return out


def _is_current(bullet: dict[str, Any]) -> bool:
    """A bullet belongs to the current role when it says so, or when its fact
    has no end date -- which is how `tailor` represents an ongoing role."""
    if bullet.get("current") is True:
        return True
    end = bullet.get("end")
    if "end" in bullet and (end is None or not str(end).strip()):
        return True
    return str(end or "").strip().lower() in {"present", "current", "now"}


def _sections(model: Any) -> list[str]:
    raw = _as_dict(model).get("sections")
    if isinstance(raw, dict):
        return [str(k) for k in raw]
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(str(item.get("heading") or item.get("title") or ""))
        else:
            out.append(str(item))
    return [s for s in out if s.strip()]


def document_text(model: Any) -> str:
    """The whole document as text.

    Uses `model["text"]` when the caller supplies the rendered document, and
    otherwise reconstructs from the parts. Reconstruction is the fallback and
    it is stated as such: a rule scanning reconstructed text cannot see
    anything the renderer adds afterwards, which is precisely where hidden
    text would be injected.
    """
    data = _as_dict(model)
    explicit = data.get("text") or data.get("html") or data.get("rendered")
    if isinstance(explicit, str) and explicit.strip():
        return explicit

    parts: list[str] = []
    for key in ("name", "headline", "summary"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    parts.extend(_contact_values(data.get("contact")))
    parts.extend(_sections(model))
    for bullet in _bullets(model):
        for key in ("role", "org"):
            value = bullet.get(key)
            if value:
                parts.append(str(value))
        parts.append(bullet["text"])
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Rule 1: personal data (Grade A)
# --------------------------------------------------------------------------

def nric_checksum_state(prefix: str, digits: str, suffix: str) -> str:
    """"valid", "invalid" or "unknown" for an NRIC/FIN-shaped string.

    Advisory only -- `check_personal_data` flags on format regardless. The
    caller never needs this; it exists so a report can say how confident the
    match is without ever handling the value itself.
    """
    prefix, suffix = prefix.upper(), suffix.upper()
    try:
        total = sum(w * int(d) for w, d in zip(_NRIC_WEIGHTS, digits))
    except (TypeError, ValueError):
        return "unknown"
    if prefix in ("S", "T"):
        table, offset = _ST_CHECKSUM, (4 if prefix == "T" else 0)
    elif prefix in ("F", "G"):
        table, offset = _FG_CHECKSUM, (4 if prefix == "G" else 0)
    else:
        # M series (2022 onward). The algorithm is not reliably documented and
        # guessing it would put a wrong assertion in a report.
        return "unknown"
    return "valid" if table[(total + offset) % 11] == suffix else "invalid"


def check_personal_data(text: str, model: Any = None) -> list[Violation]:
    """Flag personal particulars Singapore guidance says to strip from a CV.

    TAFEP and the PDPC actively list these: photo, NRIC/FIN, date of birth or
    age, gender, race, religion, marital status, nationality, National Service
    liability, salary expectations. Carrying them is a discrimination-risk
    disclosure the candidate gains nothing from.

    Every violation here is an **error**. A leaked NRIC cannot be withdrawn
    once the document is sent, so this blocks the render rather than warning.

    No matched value is ever placed in a violation, a log line or a summary --
    only the rule that fired and the offset it fired at. A checker that quotes
    the NRIC it found has recreated the leak it exists to prevent.
    """
    body = text if isinstance(text, str) else ""
    violations: list[Violation] = []

    for match in NRIC_RE.finditer(body):
        state = nric_checksum_state(match.group(1), match.group(2), match.group(3))
        violations.append(Violation(
            "personal_data.nric", "error",
            f"NRIC/FIN-format identifier found (9 chars, checksum {state}); "
            "value withheld from this report by design",
            f"document offset {match.start()}"))

    for rule, pattern in _LABELLED_PATTERNS:
        for match in re.finditer(pattern, body, re.I):
            violations.append(Violation(
                f"personal_data.{rule}", "error",
                f"{rule.replace('_', ' ')} is disclosed; "
                "Singapore guidance is to remove it from the CV",
                f"document offset {match.start()}"))

    offset = 0
    for line in body.splitlines():
        stripped = line.strip().strip(".,;:").lower()
        rule = _STANDALONE_VALUES.get(stripped)
        if rule:
            violations.append(Violation(
                f"personal_data.{rule}", "error",
                f"a bare {rule.replace('_', ' ')} value stands on its own line, "
                "which is a personal-particulars field",
                f"document offset {offset}"))
        offset += len(line) + 1

    data = _as_dict(model)
    for key, rule in _PERSONAL_MODEL_KEYS.items():
        if data.get(key):
            violations.append(Violation(
                f"personal_data.{rule}", "error",
                f"the model carries a {rule.replace('_', ' ')} field "
                f"({key!r}); it must not reach the renderer",
                f"model.{key}"))

    return violations


# --------------------------------------------------------------------------
# Rule 6: hidden text (Grade B against)
# --------------------------------------------------------------------------

def check_no_hidden_text(model: Any) -> list[Violation]:
    """Guarantee the pipeline never emits invisible text or prompt injection.

    **This is not an evasion feature. It is the opposite of one.** It exists so
    this pipeline can never produce white-on-white keywords, zero-size fonts,
    off-canvas text, or a string addressed to an LLM screener. Four independent
    reasons, all of which have to be wrong before the trick is worth anything:

      1. It does not work on the systems it targets. An ATS product manager
         states they OCR the document, so text invisible to a human is not
         read as text at all.
      2. It does not work on LLM screeners either. Independent testing against
         GPT-4o showed embedded prompt injection had zero effect on the
         outcome.
      3. It is trivially visible to any recruiter who presses Ctrl-A, and
         reads as an attempt to cheat when found.
      4. It is deceptive toward an employer. That alone settles it, and the
         first three only matter because someone will argue about the fourth.

    All four reasons are written down here so this rule is never "optimised
    away" as dead weight by someone who only knows reason 1.

    Every violation is an **error**: a document containing one must not render.
    """
    violations: list[Violation] = []
    body = document_text(model)
    data = _as_dict(model)

    # Styling embedded in the document text or supplied markup.
    haystacks = [("document", body)]
    for key in ("html", "css", "styles", "raw"):
        value = data.get(key)
        if isinstance(value, str) and value and value != body:
            haystacks.append((f"model.{key}", value))

    for where, haystack in haystacks:
        for rule, pattern in _HIDDEN_STYLE_PATTERNS:
            for match in re.finditer(pattern, haystack, re.I):
                violations.append(Violation(
                    f"hidden_text.{rule}", "error",
                    "styling that renders text invisible to a human reader; "
                    "the pipeline must never emit this",
                    f"{where} offset {match.start()}"))
        for pattern in _INJECTION_PATTERNS:
            for match in re.finditer(pattern, haystack, re.I):
                violations.append(Violation(
                    "hidden_text.prompt_injection", "error",
                    "text addressed to an automated screener rather than to a "
                    "human reader; deceptive and, per testing, ineffective",
                    f"{where} offset {match.start()}"))

    # Structured spans, for a renderer that carries styling as data rather
    # than as CSS text.
    spans = data.get("spans")
    if isinstance(spans, Iterable) and not isinstance(spans, (str, bytes)):
        for index, span in enumerate(spans):
            if not isinstance(span, dict):
                continue
            location = f"spans[{index}]"
            colour = str(span.get("color") or "").strip().lower()
            background = str(span.get("background")
                             or span.get("background_color") or "#ffffff")
            if colour and colour.replace("#fff", "#ffffff") == \
                    str(background).strip().lower().replace("#fff", "#ffffff"):
                violations.append(Violation(
                    "hidden_text.white_text", "error",
                    "text colour matches its background, making it invisible",
                    location))
            size = span.get("font_size")
            if isinstance(size, (int, float)) and size <= 1:
                violations.append(Violation(
                    "hidden_text.zero_size", "error",
                    f"font size {size} is unreadable by design", location))
            opacity = span.get("opacity")
            if isinstance(opacity, (int, float)) and opacity <= 0:
                violations.append(Violation(
                    "hidden_text.invisible", "error",
                    "zero opacity hides the text from a human reader", location))
            for axis in ("x", "y", "left", "top"):
                value = span.get(axis)
                if isinstance(value, (int, float)) and value <= -500:
                    violations.append(Violation(
                        "hidden_text.off_canvas", "error",
                        f"positioned off the page ({axis}={value})", location))
            if span.get("hidden") is True:
                violations.append(Violation(
                    "hidden_text.invisible", "error",
                    "span is explicitly hidden", location))

    return violations


# --------------------------------------------------------------------------
# Rule 5: metric density (Grade B)
# --------------------------------------------------------------------------

def metric_density(bullets: Iterable[Any]) -> float:
    """Fraction of bullets containing a number. 0.0 for an empty resume.

    Read the direction of this carefully, because it is the opposite of the
    advice industry's. Every resume guide pushes toward quantifying 100% of
    bullets. Screeners report that at mid-level, near-total quantification
    reads as padding -- invented denominators, percentages of nothing, a metric
    bolted onto a duty that never had one. The research suggests 30-40% as the
    target and the cap here is deliberately looser than that, at ~0.6, so the
    rule only fires on resumes that are clearly stuffed.

    So this is a CAP, not a target. There is intentionally no rule telling
    anyone to add more numbers.
    """
    items = [b for b in bullets if b is not None]
    if not items:
        return 0.0
    hits = 0
    for bullet in items:
        text = bullet.get("text", "") if isinstance(bullet, dict) else str(bullet)
        if _NUMBER_RE.search(str(text)):
            hits += 1
    return hits / len(items)


# --------------------------------------------------------------------------
# Rule 2/3/4: structure, inflated verbs, LLM vocabulary
# --------------------------------------------------------------------------

def _check_bullet_text(text: str, location: str, current: bool,
                       chars_per_line: int, max_lines: int) -> list[Violation]:
    violations: list[Violation] = []
    stripped = text.strip()
    if not stripped:
        return violations

    # Length. Measured in wrapped lines rather than characters, because the
    # thing a reader notices is the wrap. chars_per_line is a parameter: it
    # depends on the template's column width and font size, and guessing it
    # would make this rule wrong for every template but one.
    if chars_per_line > 0:
        lines = math.ceil(len(stripped) / chars_per_line)
        if lines > max_lines:
            violations.append(Violation(
                "structure.bullet_length", "warning",
                f"wraps to ~{lines} lines at {chars_per_line} chars/line "
                f"(max {max_lines})", location))

    # First-person pronouns. A resume is written in an implied first person;
    # spelling it out costs words and reads as a cover letter.
    for match in _PRONOUN_RE.finditer(stripped):
        violations.append(Violation(
            "structure.first_person", "warning",
            f"first-person pronoun {match.group(1)!r}",
            f"{location} offset {match.start()}"))

    # Opening verb tense.
    opener = re.sub(r"^[^A-Za-z]+", "", stripped).split(" ", 1)[0].lower()
    opener = opener.strip(".,;:")
    if opener:
        past = opener.endswith("ed") or opener in _IRREGULAR_PAST
        present = opener in _PRESENT_ACTION_VERBS
        if current and not (past or present):
            violations.append(Violation(
                "structure.opening_verb", "warning",
                f"current-role bullet opens with {opener!r}, which is not a "
                "recognised action verb in past or present tense", location))
        elif not current and not past:
            violations.append(Violation(
                "structure.opening_verb", "warning",
                f"opens with {opener!r}; a past role takes a past-tense verb",
                location))

    for match in _INFLATED_RE.finditer(stripped):
        violations.append(Violation(
            "language.inflated_verb", "warning",
            f"{match.group(1).lower()!r} inflates the claim without adding "
            "information",
            f"{location} offset {match.start()}"))

    for match in _LLM_VOCAB_RE.finditer(stripped):
        violations.append(Violation(
            "language.llm_vocabulary", "warning",
            f"{match.group(1).lower()!r} is one of four words with evidence "
            "behind them as generated-text vocabulary",
            f"{location} offset {match.start()}"))

    return violations


def _check_headings(model: Any) -> list[Violation]:
    violations: list[Violation] = []
    for index, heading in enumerate(_sections(model)):
        key = heading.strip().strip(":").lower()
        location = f"sections[{index}]"
        if key in _HOBBY_HEADINGS:
            violations.append(Violation(
                "structure.hobbies_section", "warning",
                f"{heading!r} spends one-page space on material no screen "
                "reads for", location))
        elif key not in CANONICAL_HEADINGS:
            violations.append(Violation(
                "structure.unconventional_heading", "warning",
                f"{heading!r} is not a conventional heading; heading matching "
                "is per-vendor and this is the documented failure mode",
                location))
    return violations


def _check_titles(model: Any) -> list[Violation]:
    violations: list[Violation] = []
    seen: set[str] = set()
    for index, bullet in enumerate(_bullets(model)):
        title = str(bullet.get("role") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        for word in re.split(r"[\s/]+", title):
            key = word.strip(".,").lower()
            expansion = TITLE_ABBREVIATIONS.get(key)
            if expansion:
                violations.append(Violation(
                    "structure.abbreviated_title", "warning",
                    f"job title abbreviates {word!r}; write {expansion!r} in full",
                    f"bullet[{index}].role"))
    return violations


def _contact_values(contact: Any) -> list[str]:
    """Contact details, whatever container they arrived in.

    `render_resume.build_model` emits a LIST; this module originally handled
    only a dict and a string, so a perfectly good set of contact details was
    invisible and every render raised "no contact details found". A false
    warning is worse than a missing rule, because it teaches the reader to
    skim past the warnings that are real.
    """
    if isinstance(contact, dict):
        return [str(v) for v in contact.values() if str(v).strip()]
    if isinstance(contact, (list, tuple, set)):
        return [str(v) for v in contact if str(v).strip()]
    if isinstance(contact, str) and contact.strip():
        return [contact]
    return []


def _check_contact(model: Any, text: str) -> list[Violation]:
    """Contact details must be in the document BODY.

    Headers and footers are the failure mode this guards: several parsers read
    the body only, and a phone number that lives in a header is a phone number
    nobody can call.

    A warning rather than an error, to keep the severity contract simple --
    errors are reserved for harms that cannot be undone after sending. Missing
    contact details are obvious to the user the moment they look at the page.
    """
    if any(str(v).strip() for v in _contact_values(_as_dict(model).get("contact"))):
        return []
    has_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    has_phone = re.search(r"(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){7,}\d", text)
    if has_email or has_phone:
        return []
    return [Violation("structure.no_contact_details", "warning",
                      "no email or phone number found in the document body",
                      "document")]


def check(model: Any,
          chars_per_line: int = DEFAULT_CHARS_PER_LINE,
          max_bullet_lines: int = DEFAULT_MAX_BULLET_LINES,
          max_metric_density: float = DEFAULT_MAX_METRIC_DENSITY) -> Report:
    """Run every rule over one resume model. Deterministic; no model is called.

    Accepts `tailor.tailor()` output, optionally extended with `sections`,
    `contact`, `text`/`html` and `spans`. Missing keys are skipped rather than
    assumed -- a rule that fires because a field is absent teaches the caller
    to ignore this module.
    """
    bullets = _bullets(model)
    text = document_text(model)

    violations: list[Violation] = []
    violations += check_personal_data(text, model)
    violations += check_no_hidden_text(model)

    for index, bullet in enumerate(bullets):
        violations += _check_bullet_text(
            bullet["text"], f"bullet[{index}]", _is_current(bullet),
            chars_per_line, max_bullet_lines)

    violations += _check_headings(model)
    violations += _check_titles(model)
    violations += _check_contact(model, text)

    density = metric_density(bullets)
    if bullets and density > max_metric_density:
        violations.append(Violation(
            "language.metric_density", "warning",
            f"{density:.0%} of bullets carry a number (cap {max_metric_density:.0%}); "
            "near-total quantification reads as padding, not as evidence",
            "document"))

    return Report(violations=violations, bullets_checked=len(bullets),
                  density=density)


def summarise(report: Report) -> dict[str, Any]:
    """What happened, for the run report and `--explain`.

    Reports rules, counts and locations -- never the offending text. See the
    module docstring: echoing the match would defeat the personal-data rules,
    and a summary is exactly the thing that ends up in a log file.
    """
    by_rule: dict[str, int] = {}
    for violation in report.violations:
        by_rule[violation.rule] = by_rule.get(violation.rule, 0) + 1
    errors = report.errors
    return {
        "ok": report.ok,
        "bullets": report.bullets_checked,
        "errors": len(errors),
        "warnings": len(report.warnings),
        "metric_density": round(report.density, 3),
        "by_rule": by_rule,
        "blocking": [{"rule": v.rule, "detail": v.detail, "location": v.location}
                     for v in errors[:5]],
    }
