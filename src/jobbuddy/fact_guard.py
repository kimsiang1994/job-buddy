"""Rejects any generated resume line that asserts something unverified.

This is the most important module here, and it exists because of one decision:
**tailoring is a selection problem, not a generation problem.** The moment you
ask a model to "write a bullet about their experience with Kubernetes", you
have authorised it to invent one. Prompting alone cannot close that -- the
model has no way to know which numbers are real.

So generation is followed by a deterministic check that has nothing to do with
the model. Every bullet must cite a fact_id, and everything it asserts must
trace back to that fact:

  numbers      every digit token must appear in the fact's `numbers`
  entities     every proper noun must appear in its `entities`/`skills`/`org`
  durations    any "N years" must be computable from the fact's dates
  denylist     nothing from `constraints.never_claim`
  citation     an uncited bullet is rejected outright, no exceptions

A rejected bullet falls back to the fact's own approved phrasing. If that also
fails, the profile itself is wrong and the job hard-fails loudly -- the
pipeline may produce a blander resume, never a false one.

Built and tested BEFORE tailoring exists, deliberately. A gate added after the
thing it guards is a gate that never gets added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

# Numbers, including percentages, currency, ranges and magnitude suffixes.
NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%|k\b|m\b|bn?\b)?", re.I)

# Capitalised tokens that could be a company, product or technology. Sentence
# openers are handled by the stopword list rather than by position, because a
# bullet may legitimately begin with a proper noun.
PROPER_NOUN_RE = re.compile(r"\b([A-Z][A-Za-z0-9+#.\-]{1,})\b")

DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b", re.I)

# Words that are capitalised for grammar, not because they name anything.
_STOPWORDS = frozenset("""
A An The And But Or For Nor So Yet At By In On To Of With From Into Over Under
Built Led Designed Developed Created Delivered Drove Owned Ran Managed Cut
Reduced Increased Improved Automated Migrated Launched Shipped Scaled Defined
Architected Implemented Established Introduced Rolled Grew Saved Enabled
Analysed Analyzed Presented Coached Trained Partnered Worked Used Applied
I My We Our This That These Those It Its As Is Was Were Be Been Being Have Has
Had Do Does Did Will Would Can Could Should May Might Must
Team Teams Data Report Reports Project Projects Product Products System Systems
Platform Pipeline Pipelines Model Models Service Services Process Processes
Senior Junior Lead Staff Principal Manager Director Engineer Analyst Scientist
Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec Present Current
""".split())


@dataclass
class Violation:
    """One reason a bullet was rejected, in terms a human can act on."""

    kind: str
    detail: str
    token: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass
class Verdict:
    """Outcome for one bullet."""

    ok: bool
    bullet: str
    fact_id: str = ""
    violations: list[Violation] = field(default_factory=list)
    fallback_used: bool = False

    @property
    def reasons(self) -> list[str]:
        return [str(v) for v in self.violations]


def _normalise_number(raw: str, suffix: str | None) -> set[str]:
    """Every string form one number could legitimately take.

    '10' in a fact should satisfy '10', '10%', '10x' and '10,000' written as
    '10000'. Being strict about formatting would reject true statements, and a
    guard that rejects true statements gets switched off.
    """
    cleaned = raw.replace(",", "").rstrip(".")
    forms = {cleaned, raw, raw.replace(",", "")}
    if cleaned.endswith(".0"):
        forms.add(cleaned[:-2])
    try:
        value = float(cleaned)
        if value.is_integer():
            forms.add(str(int(value)))
            forms.add(f"{int(value):,}")
    except ValueError:
        pass
    if suffix:
        forms.add(f"{cleaned}{suffix.lower()}")
    return {f for f in forms if f}


def _fact_numbers(fact: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for raw in fact.get("numbers") or []:
        for match in NUMBER_RE.finditer(str(raw)):
            allowed |= _normalise_number(match.group(1), match.group(2))
        allowed.add(str(raw).strip())
    # Dates in the fact are legitimate sources of a year.
    for key in ("start", "end"):
        value = fact.get(key)
        if value:
            allowed.add(str(value)[:4])
    return allowed


def _singular(token: str) -> str:
    """Crude depluralisation, enough to stop 'APIs' failing against 'API'."""
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 2 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 2 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _fact_entities(fact: dict[str, Any]) -> set[str]:
    words: set[str] = set()

    def add(text: str) -> None:
        for token in re.split(r"[^\w+#.]+", str(text or "")):
            if token:
                words.add(token.lower())
                words.add(_singular(token.lower()))

    for key in ("entities", "skills"):
        for item in fact.get(key) or []:
            add(item)
    for key in ("org", "role", "team"):
        add(fact.get(key))

    # The fact's own verbatim text. A word standing in the resume line this
    # fact came from is traceable to the resume BY DEFINITION -- it cannot be
    # something the model invented. Without this the guard rejected true
    # bullets for saying "APIs" where the entity list said "API", and for
    # "ROI", a generic business term no extractor thinks to list as an entity.
    #
    # Worse, it rejected the approved phrasings too, so the fallback had
    # nothing to fall back to and the bullet vanished silently. A guard that
    # rejects the very text a human verified is not being strict, it is broken.
    for phrasing in fact.get("phrasings") or []:
        add(phrasing)
    add(fact.get("source_span"))

    return words


def _year_month(value: Any) -> tuple[int, int] | None:
    """(year, month) from a 'YYYY-MM' style date, or None if it is not one.

    The range checks are the point, not decoration. `int()` accepts '99' as
    happily as '09', so an `end` of '2019-99' used to parse as month
    ninety-nine and add just over eight years to the span the guard measures a
    claim against -- a bullet asserting "6 years" of something passed against a
    fact covering ten months. Same fail-open as the '20I9' case in the
    docstring below, reached through a date that is numeric rather than
    obviously mistyped, which is why it survived the first fix.

    A year is bounded too: '0000-01' and '99999-01' are not dates a resume
    means, and both distort the span rather than being rejected by it.
    """
    try:
        parts = str(value)[:7].split("-")[:2]
        year, month = int(parts[0]), int(parts[1])
    except (ValueError, TypeError, IndexError):
        return None
    if not (1900 <= year <= 2200) or not (1 <= month <= 12):
        return None
    return year, month


def _years_between(start: str | None, end: str | None) -> float | None:
    """Years spanned by a fact's dates, or None if they cannot be read.

    None means "unknown", and every unparseable date must map to it, because
    this number is the ceiling the duration check measures a claim against.

    An earlier version fell back to `date.today()` when `end` failed to parse,
    treating a malformed end date exactly like an absent one ("still employed").
    That is the wrong direction for a guard: an `end` of "2019" mistyped as
    "20I9" stopped meaning 2019 and started meaning now, inflating the supported
    span by years, so a bullet claiming "8 years of Kubernetes" against a
    two-year fact passed. The guard failed OPEN, silently, on the one input it
    exists to catch -- and it did so only for `end`, since an unparseable
    `start` already returned None.

    Both ends now fail closed. The caller reports it: `check_bullet` turns a
    None into an "unsupported duration" violation reading "fact supports
    unknown", so the absence is stated rather than imputed.
    """
    if not start:
        return None
    parsed_start = _year_month(start)
    if parsed_start is None:
        return None
    start_year, start_month = parsed_start
    if end:
        parsed_end = _year_month(end)
        if parsed_end is None:
            # Unknown, not "today". See the docstring.
            return None
        end_year, end_month = parsed_end
    else:
        # Absent end genuinely does mean "current role".
        today = date.today()
        end_year, end_month = today.year, today.month
    return (end_year - start_year) + (end_month - start_month) / 12.0


def check_bullet(bullet: str, fact: dict[str, Any] | None,
                 profile: dict[str, Any] | None = None,
                 total_years: float | None = None) -> Verdict:
    """Validate one generated bullet against the fact it claims to come from."""
    profile = profile or {}
    text = (bullet or "").strip()
    violations: list[Violation] = []

    if not text:
        return Verdict(False, text, violations=[Violation("empty", "no text")])

    if fact is None:
        # No citation, no bullet. Unconditional, because an uncited claim is
        # exactly the shape a hallucination arrives in.
        return Verdict(False, text,
                       violations=[Violation("uncited",
                                             "bullet cites no fact_id")])

    fact_id = str(fact.get("fact_id") or "")

    # Spans already accounted for by the duration check. Without this, "1 year
    # of ETL work" was rejected because the 1 is not in the fact's `numbers` --
    # a true statement failing on a technicality, which is how a guard loses
    # the trust that makes it useful.
    duration_spans = [m.span(1) for m in DURATION_RE.finditer(text)]

    # --- numbers ----------------------------------------------------------
    allowed_numbers = _fact_numbers(fact)
    for match in NUMBER_RE.finditer(text):
        if any(start <= match.start(1) < end for start, end in duration_spans):
            continue
        forms = _normalise_number(match.group(1), match.group(2))
        if not (forms & allowed_numbers):
            violations.append(Violation(
                "invented number",
                f"{match.group(0).strip()!r} is not in fact {fact_id}",
                match.group(1)))

    # --- entities ---------------------------------------------------------
    allowed_entities = _fact_entities(fact)
    allowed_entities |= {w.lower() for w in (profile.get("entity_allowlist") or [])}
    for match in PROPER_NOUN_RE.finditer(text):
        token = match.group(1)
        if token in _STOPWORDS or token.lower() in _STOPWORDS:
            continue
        if len(token) < 2:
            continue

        # A capitalised word opening the bullet is grammar, not a claim.
        # 'Processed 4,200,000 records' was rejected for naming a company
        # called Processed -- no stopword list can enumerate every verb, so
        # position carries the weight instead. A genuine name in that slot is
        # still caught when it carries an internal capital or digit (PySpark,
        # AWS, C++, GPT4), which real product names almost always do.
        if match.start(1) == 0 and not re.search(r"[A-Z0-9+#.]", token[1:]):
            continue

        lowered = token.lower()
        if lowered not in allowed_entities and _singular(lowered) not in allowed_entities:
            violations.append(Violation(
                "unverified entity",
                f"{token!r} does not appear in fact {fact_id}", token))

    # --- durations --------------------------------------------------------
    for match in DURATION_RE.finditer(text):
        claimed = float(match.group(1))
        supported = _years_between(fact.get("start"), fact.get("end"))
        if supported is None:
            supported = total_years
        if supported is None or claimed > supported + 0.5:
            violations.append(Violation(
                "unsupported duration",
                f"claims {claimed:g} years; fact supports "
                f"{'unknown' if supported is None else format(supported, '.1f')}",
                match.group(0)))

    # --- denylist ---------------------------------------------------------
    for phrase in (profile.get("constraints") or {}).get("never_claim") or []:
        if re.search(re.escape(str(phrase)), text, re.I):
            violations.append(Violation(
                "denylisted claim", f"matches never_claim {phrase!r}", str(phrase)))

    return Verdict(not violations, text, fact_id, violations)


def guard(bullets: Iterable[dict[str, Any]], facts: dict[str, dict[str, Any]],
          profile: dict[str, Any] | None = None) -> tuple[list[str], list[Verdict]]:
    """Validate generated bullets. Returns (safe_bullets, all_verdicts).

    Each input bullet is {"text": ..., "fact_id": ...}. A rejected bullet falls
    back to its fact's first approved phrasing; if that also fails, the bullet
    is dropped and the verdict records why. Nothing unverified is ever emitted.
    """
    profile = profile or {}
    total_years = profile.get("years_experience")
    safe: list[str] = []
    verdicts: list[Verdict] = []

    for item in bullets:
        text = (item or {}).get("text", "")
        fact_id = (item or {}).get("fact_id", "")
        fact = facts.get(fact_id) if fact_id else None

        verdict = check_bullet(text, fact, profile, total_years)
        if verdict.ok:
            safe.append(text)
            verdicts.append(verdict)
            continue

        # Fall back to what the human already approved.
        phrasings = (fact or {}).get("phrasings") or []
        if phrasings:
            fallback = str(phrasings[0])
            fallback_verdict = check_bullet(fallback, fact, profile, total_years)
            if fallback_verdict.ok:
                safe.append(fallback)
                verdict.fallback_used = True
                verdicts.append(verdict)
                continue

        # Neither the generated line nor the approved one survives. That is a
        # problem with the profile, not the model, and it must be visible.
        verdicts.append(verdict)

    return safe, verdicts


def summarise(verdicts: list[Verdict]) -> dict[str, Any]:
    """What happened, for the run report and `--explain`."""
    rejected = [v for v in verdicts if not v.ok]
    kinds: dict[str, int] = {}
    for verdict in rejected:
        for violation in verdict.violations:
            kinds[violation.kind] = kinds.get(violation.kind, 0) + 1
    return {
        "bullets": len(verdicts),
        "passed": sum(1 for v in verdicts if v.ok),
        "rejected": len(rejected),
        "fell_back": sum(1 for v in verdicts if v.fallback_used),
        "by_kind": kinds,
        "examples": [{"bullet": v.bullet[:110], "reasons": v.reasons[:3]}
                     for v in rejected[:5]],
    }
