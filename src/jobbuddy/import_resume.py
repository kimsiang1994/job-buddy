"""Turns a resume PDF into a draft profile of atomic, citable facts.

This writes `profile/master_profile.draft.json` and **never** writes the
verified `master_profile.json`. That separation is the whole point: an LLM read
the PDF, so nothing it produced can be trusted until something else has checked
it. `verify_profile.py` does the mechanical part of that checking.

Every extracted fact carries a `source_span` -- a verbatim quote from the PDF
text. The extractor is instructed to copy, not paraphrase, precisely so the
span can be tested as a literal substring later. A model that paraphrases
produces a fact that fails verification and lands in front of the user, which
is the correct outcome rather than a failure.

Deliberately generic. Nothing here keys off section headings, fonts or
ordering, because a parser fitted to one resume's layout breaks on the next
one and fails quietly when it does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

REPO_DIR = Path(__file__).resolve().parents[2]
DRAFT_PATH = REPO_DIR / "profile" / "master_profile.draft.json"
VERIFIED_PATH = REPO_DIR / "profile" / "master_profile.json"

# The extractor is asked for these keys and the draft is rejected without them.
REQUIRED_FACT_KEYS = ("fact_id", "source_span")

SYSTEM_PROMPT = """You extract atomic, verifiable facts from a resume.

Return JSON: {"facts": [...], "skills_declared": {...}, "identity": {...}}

Each fact is ONE accomplishment or role, shaped as:
  fact_id      short dotted slug, e.g. "citibank.etl.automation"
  org          employer name exactly as written in the resume
  role         job title exactly as written
  start, end   "YYYY-MM"; end omitted if current
  source_span  THE EXACT TEXT FROM THE RESUME, COPIED CHARACTER FOR CHARACTER
  numbers      every numeric token appearing in source_span, as strings
  entities     proper nouns in source_span: companies, products, technologies
  skills       lowercase skill slugs the span demonstrates
  phrasings    [source_span] -- start with the resume's own wording, unchanged

RULES, in order of importance:

1. source_span MUST be copied verbatim from the resume. Do not fix typos, do
   not expand abbreviations, do not merge two bullets, do not tidy grammar.
   It is checked as a literal substring afterwards and a paraphrase fails.
2. Never introduce a number that is not in source_span.
3. Never infer an achievement the resume does not state. Omitting a real fact
   is recoverable; inventing one is not.
4. One bullet in, one fact out. Do not summarise across bullets.

identity: {"name", "email", "phone", "location", "links"} -- copied verbatim.
skills_declared: {"expert": [], "working": [], "familiar": []} -- assign a tier
only from evidence in the resume; when unsure use "familiar"."""


def read_pdf_text(pdf_path: Path) -> str:
    """Extract text from the resume PDF.

    Raises ImportError with an actionable message if pypdf is absent, because
    unlike the search pipeline there is no sensible degraded mode here -- a
    resume that cannot be read cannot be tailored.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading a resume needs pypdf: py -m pip install -e .[tailoring]"
        ) from exc

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def find_resume(input_dir: Path | None = None) -> Path | None:
    """Newest PDF in input/. Nothing clever -- the folder holds one file."""
    input_dir = input_dir or (REPO_DIR / "input")
    if not input_dir.is_dir():
        return None
    pdfs = sorted(input_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pdfs[0] if pdfs else None


def extract_facts(resume_text: str,
                  chat: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Ask the model for facts. Returns the draft payload, or an error record.

    `chat` is injectable so the whole path is testable offline. It defaults to
    the real client, imported lazily so importing this module costs nothing.
    """
    if chat is None:
        from jobbuddy.deepseek.deepseek_client import json_chat as chat

    # The `extract` profile budgets 512 tokens, sized for pulling a few fields
    # out of a job ad. A whole profile of structured facts is far larger, and
    # the JSON expands well past the prose it came from. Left to the default
    # this truncated and then paid for three doubling retries every run, so the
    # starting budget is sized from the input instead.
    result = chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": resume_text}],
        schema_keys=("facts",),
        profile="extract",
        tier="quality",
        max_tokens=min(8192, 2048 + len(resume_text)),
    )
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error") or "extraction failed",
                "facts": []}

    data = result.get("data") or {}
    facts = [f for f in (data.get("facts") or []) if isinstance(f, dict)]
    return {
        "ok": True,
        "facts": facts,
        "skills_declared": data.get("skills_declared") or {},
        "identity": data.get("identity") or {},
        "repaired": bool(result.get("repaired")),
    }


def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", str(value or "").lower()).strip(".")
    return slug or fallback


def build_draft(extracted: dict[str, Any], resume_text: str,
                source_path: Path | None = None) -> dict[str, Any]:
    """Assemble the draft. Every fact starts `verified: false`, no exceptions.

    Also de-duplicates fact_ids. A model asked for slugs will occasionally emit
    the same one twice, and a dict keyed by fact_id would silently drop the
    second -- losing a real accomplishment with no error anywhere.
    """
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, raw in enumerate(extracted.get("facts") or []):
        fact = dict(raw)
        fact_id = _slugify(fact.get("fact_id"), f"fact.{index}")
        if fact_id in seen:
            fact_id = f"{fact_id}.{index}"
        seen.add(fact_id)

        fact["fact_id"] = fact_id
        fact["numbers"] = [str(n) for n in (fact.get("numbers") or [])]
        fact["entities"] = [str(e) for e in (fact.get("entities") or [])]
        fact["skills"] = [str(s).lower() for s in (fact.get("skills") or [])]
        span = str(fact.get("source_span") or "")
        fact["source_span"] = span
        if not fact.get("phrasings"):
            fact["phrasings"] = [span] if span else []
        # Set here rather than trusted from the model, so a model that helpfully
        # returns verified:true cannot promote its own output.
        fact["verified"] = False
        fact["verification"] = None
        facts.append(fact)

    return {
        "_schema": "master_profile/1",
        "_status": "DRAFT -- not usable for tailoring until verified",
        "_source_pdf": str(source_path.name) if source_path else None,
        "_resume_text_chars": len(resume_text),
        "identity": extracted.get("identity") or {},
        "skills_declared": extracted.get("skills_declared") or {},
        # Seeded empty so the file shows the user the shape they may fill in.
        # fact_guard reads never_claim; an absent key is an empty denylist,
        # which is exactly the wrong default to leave implicit.
        "constraints": {"never_claim": [], "entity_allowlist": []},
        "facts": facts,
    }


def write_draft(draft: dict[str, Any], path: Path | None = None) -> Path:
    """Write the draft. Refuses to touch the verified file.

    The check is on the resolved path rather than the argument, so a relative
    path or a symlink cannot route a draft over hand-verified work.
    """
    path = (path or DRAFT_PATH).resolve()
    if path == VERIFIED_PATH.resolve():
        raise ValueError(
            "import_resume never writes master_profile.json -- only the user "
            "and verify_profile.promote() may write the verified file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def import_resume(pdf_path: Path | None = None,
                  chat: Callable[..., dict[str, Any]] | None = None,
                  out_path: Path | None = None) -> dict[str, Any]:
    """PDF -> draft profile. Returns a summary; writes nothing on failure."""
    pdf_path = pdf_path or find_resume()
    if pdf_path is None or not Path(pdf_path).is_file():
        return {"ok": False, "error": "no resume PDF found in input/"}

    resume_text = read_pdf_text(Path(pdf_path))
    if len(resume_text.strip()) < 200:
        # A scanned or image-only resume extracts to almost nothing. Failing
        # loudly beats handing the model 40 characters and extracting nonsense.
        return {"ok": False, "error":
                f"extracted only {len(resume_text.strip())} characters -- "
                "the PDF may be a scan with no text layer"}

    extracted = extract_facts(resume_text, chat=chat)
    if not extracted.get("ok"):
        return {"ok": False, "error": extracted.get("error")}

    draft = build_draft(extracted, resume_text, Path(pdf_path))
    written = write_draft(draft, out_path)
    return {"ok": True, "path": written, "facts": len(draft["facts"]),
            "resume_text": resume_text, "draft": draft}
