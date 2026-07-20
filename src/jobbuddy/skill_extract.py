"""Read skills out of a job description, for the sources that publish none.

MyCareersFuture tags every posting with structured skills. Workable, the ATS
boards and HN publish none at all -- so 71% of jobs on a live run scored nothing
on skill match, the heaviest component at weight 30, while carrying ~4,100
characters of description nobody read. Their confidence sat at 30-45% purely
because of that gap.

The obvious approach is a trap. If you scan a description for the skills the
candidate HAS, every job matches perfectly and the score means nothing -- you
cannot detect a gap by searching only for things you own. Extraction has to run
against a vocabulary that knows about skills the candidate lacks.

That vocabulary comes free from MCF. Its extractor has already tagged hundreds
of real Singapore tech postings, and those tags accumulate in the run history:
772 usable terms after a handful of runs, growing every time. One source's
structured data enriching another's prose, with no LLM and no API key.

Section awareness matters as much as the matching. A description usually splits
into what is required and what is merely nice to have, and treating those the
same makes an optional Kubernetes mention weigh as much as a mandatory Python
one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from jobbuddy import net, skills_taxonomy

REPO_DIR = Path(__file__).resolve().parents[2]
VOCAB_PATH = REPO_DIR / "config" / "skill_vocab.json"

# A term shorter than this matches too much prose to be worth the false hits.
# 'ai' and 'ml' survive via the alias table, which maps them to longer forms.
MIN_TERM_LENGTH = 2

# How many times a term must have been seen before it joins the vocabulary.
# MCF's extractor occasionally emits one-off garbage; requiring a repeat is a
# cheap filter that costs nothing but a run or two of patience.
MIN_SIGHTINGS = 1

# Headings that flip a description into "these are demands" or "these are
# wishes". Ordered longest-first so 'nice to have' wins over 'have'.
_REQUIRED_HEADS = (
    "requirements", "required", "must have", "must-have", "you must",
    "qualifications", "what you need", "what you'll need", "minimum",
    "essential", "we require", "you have", "you bring", "who you are",
    "skills and experience", "your profile", "key skills",
)
_OPTIONAL_HEADS = (
    "nice to have", "nice-to-have", "preferred", "bonus", "plus points",
    "desirable", "advantageous", "good to have", "a plus", "optional",
    "would be great", "icing on the cake", "not required",
)
_IRRELEVANT_HEADS = (
    "benefits", "perks", "what we offer", "about us", "about the company",
    "our values", "equal opportunity", "eeo", "diversity", "how to apply",
    "compensation", "why join",
)

_HEADING_RE = re.compile(
    r"(?:^|[.\n•·\-–—:]|\s{2,})\s*([A-Za-z][A-Za-z' /&\-]{2,44})\s*:?\s*(?=$|[\n•·])",
    re.M)


# How much a requirement counts, by where it appears.
#
# A job description is not a flat list. "Senior Machine Learning Engineer"
# wanting Python, ML and AWS, with twenty other things under 'nice to have', is
# mostly about the first three -- but scoring every term equally made a
# candidate who met all three and none of the twenty look like a 13% match.
#
# So the long tail is priced as a long tail. Missing 'Customer Segmentation' on
# a role whose title is Machine Learning Engineer should barely register;
# missing Python should be disqualifying.
WEIGHT_IN_TITLE = 4.0        # named in the job title: this IS the job
WEIGHT_REQUIRED = 1.0        # under Requirements / Must have
WEIGHT_OPTIONAL = 0.25       # under Nice to have / Preferred
WEIGHT_REPEAT_BONUS = 0.25   # per extra mention, capped
MAX_REPEAT_BONUS = 3


@dataclass
class ExtractedSkill:
    """One skill found in a description, and how strongly it was asked for."""

    term: str
    canonical: str
    required: bool
    evidence: str
    weight: float = 1.0
    mentions: int = 1

    def __hash__(self) -> int:
        return hash(self.canonical)


# --------------------------------------------------------------------------
# Vocabulary -- sole writer of config/skill_vocab.json
# --------------------------------------------------------------------------

# A term appearing in most job ads tells you nothing about any of them.
# 'Design' turned up in 35 of ~50 descriptions, 'Communication' in 32 -- so
# every candidate 'lacked' them and every score was dragged down by words that
# are not skills. Rather than maintain a blocklist by hand, the vocabulary
# measures how common each term is and drops the ones that fail to
# discriminate. Self-tuning, and it sharpens as the corpus grows.
MAX_DOCUMENT_FREQUENCY = 0.35

# Below this many observed documents, frequency is noise rather than evidence,
# so the filter stays off and everything is allowed through.
MIN_DOCS_FOR_FREQUENCY_FILTER = 25


def load_vocab(path: Path = VOCAB_PATH) -> dict[str, int]:
    """term -> times seen. Never raises."""
    return _load_full(path)["terms"]


def _load_full(path: Path = VOCAB_PATH) -> dict[str, Any]:
    """The whole vocabulary record: terms, document frequency, corpus size.

    Never raises, and never returns an empty vocabulary without saying so. A
    silent fallback here is expensive twice over:

      - an empty vocabulary makes `extract` find nothing, so `skill_match` --
        the heaviest component at weight 30 -- returns None for every job from
        every source that publishes no structured skills. The run still
        completes and the scores still look plausible;
      - `harvest()` writes back whatever this returned, so one unreadable read
        REPLACES a 772-term vocabulary with an empty one on the very next fetch.

    Both used to happen without a single line of output.
    """
    empty: dict[str, Any] = {"terms": {}, "doc_freq": {}, "documents": 0}
    if not path.is_file():
        return empty
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        net._warn(f"skills: could not read {path.name} ({exc}); "
                  f"extracting against an EMPTY vocabulary this run")
        return empty
    try:
        data = json.loads(raw)
    except ValueError as exc:
        net._warn(f"skills: {path.name} is not valid JSON ({exc}); "
                  f"extracting against an EMPTY vocabulary -- the next harvest "
                  f"will overwrite the file, so move it aside to keep it")
        return empty
    if not isinstance(data, dict):
        net._warn(f"skills: {path.name} is a {type(data).__name__}, not an "
                  f"object; extracting against an empty vocabulary")
        return empty
    terms = data.get("terms")
    doc_freq = data.get("doc_freq")
    return {
        "terms": {k: int(v) for k, v in (terms or {}).items()
                  if isinstance(v, (int, float))} if isinstance(terms, dict) else {},
        "doc_freq": {k: int(v) for k, v in (doc_freq or {}).items()
                     if isinstance(v, (int, float))} if isinstance(doc_freq, dict) else {},
        "documents": int(data["documents"]) if isinstance(
            data.get("documents"), (int, float)) else 0,
    }


def save_vocab(terms: dict[str, int], path: Path = VOCAB_PATH,
               doc_freq: dict[str, int] | None = None,
               documents: int = 0) -> bool:
    """Atomic write. Sole writer of the vocabulary file."""
    doc_freq = doc_freq or {}
    payload = {
        "_written_by": "skill_extract.py",
        "_comment": "Skill vocabulary harvested from MyCareersFuture's structured "
                    "tags, used to read skills out of descriptions from sources "
                    "that publish none. `doc_freq` counts how many descriptions "
                    "each term appeared in -- terms common enough to be filler "
                    "are filtered out at extraction time. Safe to delete; it "
                    "rebuilds from the next fetch.",
        "term_count": len(terms),
        "documents": documents,
        "terms": dict(sorted(terms.items(), key=lambda kv: (-kv[1], kv[0]))),
        "doc_freq": dict(sorted(doc_freq.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        # `harvest` discards this return value, so a silent False meant the
        # vocabulary stopped growing and nothing ever mentioned it.
        net._warn(f"skills: could not write {path.name} ({exc}); "
                  f"{len(terms)} term(s) harvested this run were not saved")
        return False


def harvest(jobs: Iterable[dict[str, Any]], path: Path = VOCAB_PATH) -> int:
    """Add structured skills to the vocabulary, and count where terms appear.

    Called after every fetch. Sources that publish structured skills pay for
    the sources that do not -- and every description scanned sharpens the
    frequency filter, whether or not that source tagged anything.
    """
    record = _load_full(path)
    terms, doc_freq = record["terms"], record["doc_freq"]
    documents = record["documents"]
    added = 0

    for job in jobs:
        for term in job.get("skills_raw") or []:
            text = str(term or "").strip()
            if len(text) < MIN_TERM_LENGTH or skills_taxonomy.is_noise(text):
                continue
            if text not in terms:
                added += 1
            terms[text] = terms.get(text, 0) + 1

        # Document frequency is counted over descriptions, not over tag lists:
        # the question is "how ordinary is this word in a job ad", and only the
        # prose can answer that.
        #
        # Word boundaries, not substring. Counting `"pos" in text` scored POS at
        # 69% of all ads -- because it matches inside "position". Every short
        # term was inflated the same way, and the filter then removed real
        # skills on the strength of numbers that measured nothing.
        text_blob = (job.get("jd_text") or "").lower()
        if len(text_blob) > 200:
            documents += 1
            for term in terms:
                if _boundary_pattern(term).search(text_blob):
                    doc_freq[term] = doc_freq.get(term, 0) + 1

    save_vocab(terms, path, doc_freq, documents)
    return added


@lru_cache(maxsize=4096)
def _boundary_pattern(term: str) -> re.Pattern[str]:
    """Word-boundary matcher for one term, tolerant of internal whitespace."""
    body = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.I)


def is_discriminative(term: str, record: dict[str, Any] | None = None,
                      path: Path = VOCAB_PATH) -> bool:
    """False for terms so common in job ads that they carry no signal.

    Frequency alone is the wrong test in a domain corpus, and applying it alone
    was actively harmful: Python appeared in 73% of these ads, AWS in 45%, API
    in 62% -- and the filter cheerfully discarded all three. The most important
    skills in a field are the most common ones in that field's job ads.

    So a term the curated taxonomy already recognises is kept whatever its
    frequency. The frequency test only judges terms nobody vouched for, which
    is exactly where the filler lives -- Design, Development, Communication,
    Management, all emitted by MCF's extractor and none of them a skill.
    """
    if skills_taxonomy.canon(term) in skills_taxonomy.ALIASES:
        return True     # a known skill; commonness is not evidence against it

    record = record or _load_full(path)
    documents = record.get("documents", 0)
    if documents < MIN_DOCS_FOR_FREQUENCY_FILTER:
        return True     # too little evidence to call anything filler
    frequency = record.get("doc_freq", {}).get(term, 0) / documents
    return frequency <= MAX_DOCUMENT_FREQUENCY


_compiled_cache: tuple[int, list[tuple[str, re.Pattern[str]]]] | None = None


def compiled_vocab(path: Path = VOCAB_PATH) -> list[tuple[str, re.Pattern[str]]]:
    """Vocabulary as (term, word-boundary regex), longest term first.

    Longest-first is what stops 'machine learning engineer' being counted as
    three separate hits, and stops the short generic terms swallowing the
    specific ones they sit inside.
    """
    global _compiled_cache
    record = _load_full(path)
    terms = record["terms"]
    if _compiled_cache and _compiled_cache[0] == len(terms):
        return _compiled_cache[1]

    usable = [t for t, n in terms.items() if n >= MIN_SIGHTINGS
              and len(t) >= MIN_TERM_LENGTH
              and not skills_taxonomy.is_noise(t)
              and is_discriminative(t, record)]
    usable.sort(key=len, reverse=True)

    compiled = []
    for term in usable:
        # Escape the term but let internal whitespace match any run of space,
        # so 'Fine Tuning' still finds 'fine  tuning' across a line break.
        pattern = r"\s+".join(re.escape(part) for part in term.split())
        compiled.append((term, re.compile(rf"(?<![\w-]){pattern}(?![\w-])", re.I)))
    _compiled_cache = (len(terms), compiled)
    return compiled


def reload_vocab() -> None:
    global _compiled_cache
    _compiled_cache = None


# --------------------------------------------------------------------------
# Section detection
# --------------------------------------------------------------------------

def section_map(text: str) -> list[tuple[int, str]]:
    """[(character offset, 'required'|'optional'|'irrelevant')], in order.

    Descriptions are prose with headings, not structured documents, so this
    reads the headings and marks where each region starts. Anything before the
    first heading is unclassified and treated as required -- the opening
    paragraph usually describes the actual job.
    """
    marks: list[tuple[int, str]] = []
    lowered = text.lower()
    for match in _HEADING_RE.finditer(lowered):
        heading = match.group(1).strip()
        if len(heading) > 46:
            continue
        for phrase in _OPTIONAL_HEADS:
            if phrase in heading:
                marks.append((match.start(1), "optional"))
                break
        else:
            for phrase in _IRRELEVANT_HEADS:
                if phrase in heading:
                    marks.append((match.start(1), "irrelevant"))
                    break
            else:
                for phrase in _REQUIRED_HEADS:
                    if phrase in heading:
                        marks.append((match.start(1), "required"))
                        break
    marks.sort()
    return marks


def classify_position(position: int, marks: list[tuple[int, str]]) -> str:
    """Which section a character offset falls in."""
    current = "required"       # before any heading: the role description itself
    for offset, kind in marks:
        if offset <= position:
            current = kind
        else:
            break
    return current


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract(text: str, max_skills: int = 40,
            path: Path = VOCAB_PATH,
            title: str = "") -> list[ExtractedSkill]:
    """Find vocabulary skills in a job description.

    Independent of the candidate's profile on purpose. Searching a description
    for skills the candidate already has would make every job a perfect match
    and hide every gap -- the score has to be able to come out low.
    """
    if not text or len(text) < 40:
        return []

    marks = section_map(text)
    found: dict[str, ExtractedSkill] = {}
    consumed: list[tuple[int, int]] = []

    # Canonical forms named in the title. Compared canonically, not literally:
    # a title reading "AI Engineer" is about `artificial intelligence`, and a
    # literal search for that phrase in that title finds nothing.
    title_canon: set[str] = set()
    if title:
        words = re.split(r"[^\w+#.]+", title.lower())
        for size in (3, 2, 1):
            for start in range(len(words) - size + 1):
                phrase = " ".join(words[start:start + size]).strip()
                if phrase:
                    title_canon.add(skills_taxonomy.canon(phrase))

    for term, pattern in compiled_vocab(path):
        canonical = skills_taxonomy.canon(term)
        if not canonical or canonical in found:
            continue

        for match in pattern.finditer(text):
            start, end = match.span()
            # Skip a hit sitting inside an already-matched longer term.
            if any(s <= start and end <= e for s, e in consumed):
                continue

            section = classify_position(start, marks)
            if section == "irrelevant":
                continue    # a skill named under 'Benefits' is not a requirement

            consumed.append((start, end))

            mentions = len(pattern.findall(text))
            weight = WEIGHT_REQUIRED if section == "required" else WEIGHT_OPTIONAL
            if canonical in title_canon:
                # Named in the title, so it is the job rather than a detail of
                # it. Overrides the section entirely.
                weight = WEIGHT_IN_TITLE
            weight += WEIGHT_REPEAT_BONUS * min(mentions - 1, MAX_REPEAT_BONUS)

            found[canonical] = ExtractedSkill(
                term=term, canonical=canonical,
                required=(section == "required"),
                evidence=text[max(0, start - 40):end + 40].strip(),
                weight=round(weight, 2), mentions=mentions,
            )
            break

        if len(found) >= max_skills:
            break

    # Heaviest first, so a truncated list keeps what the job is actually about.
    return sorted(found.values(), key=lambda s: (-s.weight, s.canonical))


def enrich(job: dict[str, Any], path: Path = VOCAB_PATH) -> dict[str, Any]:
    """Fill skills_raw/skills_key from the description, if the source gave none.

    Never overwrites structured skills. MCF's own tags are better evidence than
    anything read out of prose, so a source that publishes them keeps them.
    """
    if job.get("skills_raw"):
        return job

    text = job.get("jd_text") or ""
    skills = extract(text, path=path, title=job.get("title") or "")
    if not skills:
        return job

    job["skills_raw"] = [s.term for s in skills]
    job["skills_key"] = [s.term for s in skills if s.required]
    job["skills_weight"] = {s.term: s.weight for s in skills}
    provenance = job.setdefault("_provenance", {})
    provenance["skills"] = (f"extracted from description against a "
                            f"{len(load_vocab(path))}-term vocabulary; "
                            f"{len(job['skills_key'])} of {len(skills)} required")
    return job
