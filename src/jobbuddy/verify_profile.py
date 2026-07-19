"""Mechanically verifies as much of the draft profile as can be checked.

The point of this module is to make the manual gate small enough that it
actually gets done. Reading forty extracted facts is a chore that gets skipped;
reading the six that could not be checked automatically is a five-minute job.

**What auto-verification actually proves.** That a fact was COPIED from the
resume rather than invented by the model. Its `source_span` appears in the PDF
text verbatim, and every number and entity it claims appears inside that span.
That is a string comparison, so it is trustworthy in a way no model output is.

**What it does not prove.** That the resume is true. If the resume overstates
something, this promotes the overstatement faithfully. No amount of automation
closes that, because the only source of truth is the person who lived it -- so
a fact reaching `verified: true` here means "the tool did not make this up",
never "this is accurate".

Facts that fail land in front of the user with a specific reason, which is the
correct outcome rather than a failure. The usual causes, in frequency order:

  paraphrase   the model tidied the wording, so the span is not literal
  merge        two resume bullets folded into one fact
  inference    a number in `numbers` that appears nowhere in the span
  layout       the PDF extractor broke a line mid-phrase

Only the last is a false alarm, and normalisation below removes most of those.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from jobbuddy import fact_guard
from jobbuddy.import_resume import DRAFT_PATH, VERIFIED_PATH

# Punctuation a PDF renders one way and a model reproduces another. Mapped
# rather than stripped, so lengths stay comparable and quotes stay readable.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "­": "",
    " ": " ", "•": " ", "ﬁ": "fi", "ﬂ": "fl",
}


def normalise(text: str) -> str:
    """Collapse the differences a PDF extractor introduces but a human ignores.

    Line breaks mid-sentence, non-breaking spaces, ligatures and smart quotes
    are all layout artefacts. Comparing raw strings rejects true spans on those
    alone, and a verifier that cries wolf gets bypassed -- which costs more than
    it ever saved.
    """
    text = unicodedata.normalize("NFKC", str(text or ""))
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _span_in_resume(span: str, resume_norm: str) -> bool:
    if len(span) < 12:
        # A span too short to be distinctive matches by accident. Requiring
        # length is cheaper than reasoning about which short spans are safe.
        return False
    return normalise(span) in resume_norm


def check_fact(fact: dict[str, Any], resume_norm: str) -> list[str]:
    """Every reason this fact cannot be auto-verified. Empty list means clean."""
    problems: list[str] = []

    span = str(fact.get("source_span") or "")
    if not span.strip():
        problems.append("no source_span -- nothing to check it against")
        return problems

    if not _span_in_resume(span, resume_norm):
        if len(normalise(span)) < 12:
            problems.append("source_span is too short to be distinctive")
        else:
            problems.append("source_span is not literal text from the resume "
                            "(likely paraphrased or merged from two bullets)")
        # Without a trustworthy span there is nothing to check the rest
        # against, so the remaining checks would be theatre.
        return problems

    span_norm = normalise(span)

    # Numbers must come from the span, not from the model's sense of what a
    # good bullet sounds like. Reuses fact_guard's forms so the two modules
    # cannot disagree about whether "4,200,000" and "4200000" are the same.
    for raw in fact.get("numbers") or []:
        match = fact_guard.NUMBER_RE.search(str(raw))
        if not match:
            continue
        forms = fact_guard._normalise_number(match.group(1), match.group(2))
        if not any(re.search(rf"(?<![\w.]){re.escape(f)}(?![\w])", span_norm)
                   for f in forms):
            problems.append(f"number {raw!r} does not appear in source_span")

    # Entities likewise. Substring rather than word-boundary here because
    # "PySpark" legitimately appears inside "PySpark-based".
    for entity in fact.get("entities") or []:
        token = normalise(entity)
        if token and token not in span_norm and token not in resume_norm:
            problems.append(f"entity {entity!r} appears nowhere in the resume")

    for key in ("org", "role"):
        value = normalise(fact.get(key) or "")
        if value and value not in resume_norm:
            problems.append(f"{key} {fact.get(key)!r} does not appear in the resume")

    # Dates are the one thing a resume states in a format the model rewrites
    # ("Aug 2023" -> "2023-08"), so only the year is checkable.
    for key in ("start", "end"):
        value = str(fact.get(key) or "")
        if value and value[:4].isdigit() and value[:4] not in resume_norm:
            problems.append(f"{key} year {value[:4]} does not appear in the resume")

    return problems


def auto_verify(draft: dict[str, Any], resume_text: str) -> dict[str, Any]:
    """Mark every mechanically-checkable fact verified. Returns a new draft.

    Does not mutate its input -- the caller may want to diff before and after,
    and an in-place edit makes that impossible.
    """
    resume_norm = normalise(resume_text)
    out = json.loads(json.dumps(draft))  # deep copy, stdlib, no dependency

    for fact in out.get("facts") or []:
        problems = check_fact(fact, resume_norm)
        if problems:
            fact["verified"] = False
            fact["verification"] = {"method": "auto", "result": "needs_review",
                                    "problems": problems}
        else:
            fact["verified"] = True
            fact["verification"] = {
                "method": "auto",
                "result": "span_matched",
                "proves": "copied from the resume, not invented by the model",
                "does_not_prove": "that the resume itself is accurate",
            }
    return out


def summarise(draft: dict[str, Any]) -> dict[str, Any]:
    """What the user needs to look at, and nothing else."""
    facts = draft.get("facts") or []
    needs_review = [f for f in facts if not f.get("verified")]
    kinds: dict[str, int] = {}
    for fact in needs_review:
        for problem in (fact.get("verification") or {}).get("problems") or []:
            key = problem.split(" -- ")[0].split(" (")[0]
            kinds[key] = kinds.get(key, 0) + 1
    return {
        "facts": len(facts),
        "auto_verified": len(facts) - len(needs_review),
        "needs_review": len(needs_review),
        "by_problem": kinds,
        "review": [{"fact_id": f.get("fact_id"),
                    "span": str(f.get("source_span") or "")[:120],
                    "problems": (f.get("verification") or {}).get("problems") or []}
                   for f in needs_review],
    }


def promote(draft: dict[str, Any], out_path: Path | None = None,
            allow_unverified: bool = False) -> Path:
    """Write the verified profile. Refuses while any fact is unverified.

    `allow_unverified` drops the failing facts rather than promoting them --
    a smaller true profile, never a larger doubtful one. There is deliberately
    no flag that promotes an unverified fact as verified.
    """
    facts = draft.get("facts") or []
    unverified = [f for f in facts if not f.get("verified")]

    if unverified and not allow_unverified:
        ids = ", ".join(str(f.get("fact_id")) for f in unverified[:5])
        raise ValueError(
            f"{len(unverified)} fact(s) still need review: {ids}"
            f"{' ...' if len(unverified) > 5 else ''}. "
            "Fix the source_span in the draft, or call promote(allow_unverified=True) "
            "to drop them.")

    out = json.loads(json.dumps(draft))
    out["facts"] = [f for f in facts if f.get("verified")]
    if not out["facts"]:
        raise ValueError("nothing verified -- refusing to write an empty profile")

    out["_status"] = "VERIFIED"
    out["_dropped_unverified"] = len(unverified)
    path = (out_path or VERIFIED_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_draft(path: Path | None = None) -> dict[str, Any]:
    path = path or DRAFT_PATH
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # The read path cannot raise, per the repo convention.
        return {}
