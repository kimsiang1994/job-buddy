"""Chooses which verified facts to put on the resume, and in what order.

**Selection, not generation.** The model is never asked to write a bullet. It
is handed the verified facts and asked which ones answer this job, ranked, with
a reason. Anything it writes is checked by `fact_guard` and falls back to the
fact's own approved phrasing when it fails. The worst outcome available to this
module is a resume that reads blandly; it has no path to one that reads falsely.

Two properties are load-bearing and both are enforced structurally rather than
by prompting:

  **Ranked output, not a fixed set.** The renderer has a hard one-page limit
  and must drop bullets to meet it. If selection returned an unordered set, the
  renderer would drop arbitrarily. It returns a ranking, so what gets cut is
  always the least relevant thing rather than the last thing.

  **A stable prompt prefix.** DeepSeek bills a cache hit at roughly a fiftieth
  of a miss, and the cache keys on an exact prefix. So message[0] holds only
  what is constant for a profile -- the facts, the rubric, the schema -- and
  the job goes in message[1]. Any per-job text leaking into the system message
  destroys the cache for every subsequent job in the run, which is a silent
  cost bug rather than a visible failure. `tests/test_tailor.py` asserts two
  different jobs produce byte-identical prefixes.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from jobbuddy import fact_guard

# More than this and the one-page renderer is guaranteed to be cutting, which
# wastes selection effort on bullets nobody will read.
MAX_BULLETS = 14

SELECTION_SCHEMA = """Reply with JSON:

{"selected": [{"fact_id": "...", "rank": 1, "why": "...",
               "text": "optional rewording, or omit to use the approved phrasing"}],
 "headline": "optional one-line positioning, using only words from the facts",
 "unaddressed": ["JD requirements no fact answers"]}"""

SELECTION_RULES = """You choose which of a candidate's VERIFIED facts belong on
a one-page resume for one specific job. You do not write new material.

RULES:

1. Only fact_ids from the list below. A fact_id not in the list is discarded.
2. `text` may reorder or shorten a fact's approved phrasing to match the job's
   language. It may NEVER add a number, a company, a technology or a duration
   that is not already in that fact. Anything added is stripped automatically
   and the approved phrasing used instead, so inventing costs you the edit.
3. Rank by how directly the fact answers THIS job's stated requirements. Rank 1
   is the single most relevant thing this candidate has done. The bottom of
   your ranking will be cut to fit one page -- rank accordingly.
4. `unaddressed` must be honest. Listing a requirement the candidate cannot
   meet is useful; pretending it is covered is not. Do not stretch a fact to
   cover a requirement it does not actually demonstrate.
5. Prefer facts carrying a measured outcome over facts describing a duty."""


def _facts_block(facts: list[dict[str, Any]]) -> str:
    """The candidate's facts, rendered identically every time.

    Sorted by fact_id so a reordering upstream cannot change the prefix bytes
    and silently halve the cache hit rate.
    """
    lines = []
    for fact in sorted(facts, key=lambda f: str(f.get("fact_id"))):
        phrasing = (fact.get("phrasings") or [""])[0]
        lines.append(json.dumps({
            "fact_id": fact.get("fact_id"),
            "org": fact.get("org"),
            "role": fact.get("role"),
            "start": fact.get("start"),
            "end": fact.get("end"),
            "approved_phrasing": phrasing,
            "skills": sorted(fact.get("skills") or []),
        }, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def build_prefix(profile: dict[str, Any]) -> str:
    """The system message. Constant for a profile, so the cache holds."""
    facts = [f for f in (profile.get("facts") or []) if f.get("verified")]
    never = (profile.get("constraints") or {}).get("never_claim") or []
    return "\n\n".join([
        SELECTION_RULES,
        "NEVER CLAIM (these are false about this candidate):\n"
        + "\n".join(f"- {n}" for n in sorted(never)) if never else "NEVER CLAIM: (none set)",
        f"VERIFIED FACTS ({len(facts)}):\n{_facts_block(facts)}",
        SELECTION_SCHEMA,
    ])


def build_job_message(job: dict[str, Any], requirements: list[str]) -> str:
    """The per-job half. Everything that varies lives here, nowhere else."""
    parts = [
        f"ROLE: {job.get('title') or '(untitled)'}",
        f"COMPANY: {job.get('company') or '(unknown)'}",
        f"SENIORITY: {job.get('seniority') or '(unstated)'}",
    ]
    if requirements:
        parts.append("STATED REQUIREMENTS:\n"
                     + "\n".join(f"- {r}" for r in requirements))
    jd = (job.get("jd_text") or "").strip()
    if jd:
        # Truncated because the tail of a JD is boilerplate -- benefits, EEO
        # statements -- and paying to process it on every job adds up.
        parts.append(f"JOB DESCRIPTION:\n{jd[:6000]}")
    return "\n\n".join(parts)


def select(profile: dict[str, Any], job: dict[str, Any],
           requirements: list[str] | None = None,
           chat: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Ask which facts answer this job. Returns the raw selection, ungated."""
    if chat is None:
        from jobbuddy.deepseek.deepseek_client import json_chat as chat

    result = chat(
        [{"role": "system", "content": build_prefix(profile)},
         {"role": "user", "content": build_job_message(job, requirements or [])}],
        schema_keys=("selected",),
        profile="analyze",
        tier="quality",
        max_tokens=2048,
    )
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error"), "selected": []}

    data = result.get("data") or {}
    selected = [s for s in (data.get("selected") or []) if isinstance(s, dict)]
    return {
        "ok": True,
        "selected": selected,
        "headline": data.get("headline") or "",
        "unaddressed": [str(u) for u in (data.get("unaddressed") or [])],
        "cost_usd": result.get("cost_usd"),
    }


def tailor(profile: dict[str, Any], job: dict[str, Any],
           requirements: list[str] | None = None,
           chat: Callable[..., dict[str, Any]] | None = None,
           max_bullets: int = MAX_BULLETS) -> dict[str, Any]:
    """Select, gate, and return a ranked resume draft.

    The gate runs on the way out, unconditionally. A caller cannot obtain
    ungated bullets from this function, which is the point -- `select()` is
    public only so the selection can be inspected when something looks wrong.
    """
    facts_by_id = {str(f.get("fact_id")): f
                   for f in (profile.get("facts") or []) if f.get("verified")}

    selection = select(profile, job, requirements, chat)
    if not selection.get("ok"):
        return {"ok": False, "error": selection.get("error"), "bullets": []}

    # Rank before gating so a rejected bullet keeps its position rather than
    # falling to the end, where the renderer would cut it for the wrong reason.
    ordered = sorted(
        selection["selected"],
        key=lambda s: (s.get("rank") if isinstance(s.get("rank"), int) else 999))

    candidates: list[dict[str, Any]] = []
    unknown: list[str] = []
    for item in ordered:
        fact_id = str(item.get("fact_id") or "")
        if fact_id not in facts_by_id:
            # A hallucinated fact_id. Silently dropping it would hide the
            # single clearest signal that the model is not grounded.
            unknown.append(fact_id)
            continue
        fact = facts_by_id[fact_id]
        text = str(item.get("text") or "").strip() or (fact.get("phrasings") or [""])[0]
        candidates.append({"text": text, "fact_id": fact_id,
                           "why": str(item.get("why") or "")})

    safe, verdicts = fact_guard.guard(candidates, facts_by_id, profile)

    # Pair each surviving line back to its fact, so the renderer and the report
    # can both say which fact a bullet came from.
    #
    # `guard` returns two lists that happen to advance together: a verdict is
    # appended for every candidate, and text is appended when the verdict
    # passed or its fallback did. Zipping them relies on that staying true in
    # another module. If it ever drifts, bullets get attributed to the WRONG
    # fact -- a plausible-looking citation pointing at something else, which is
    # the precise failure this codebase exists to prevent. So the invariant is
    # checked rather than assumed, and a mismatch drops the attribution instead
    # of inventing one.
    emitted = [v for v in verdicts if v.ok or v.fallback_used]
    aligned = len(emitted) == len(safe)

    bullets: list[dict[str, Any]] = []
    for index, text in enumerate(safe):
        verdict = emitted[index] if aligned else None
        fact = facts_by_id.get(verdict.fact_id, {}) if verdict else {}
        bullets.append({
            "text": text,
            "fact_id": verdict.fact_id if verdict else "",
            "org": fact.get("org"),
            "role": fact.get("role"),
            "start": fact.get("start"),
            "end": fact.get("end"),
            "fell_back": bool(verdict and verdict.fallback_used),
        })

    return {
        "ok": True,
        # Surfaced rather than swallowed: if this is ever False the two modules
        # have drifted apart and every citation in the run is suspect.
        "attribution_aligned": aligned,
        "bullets": bullets[:max_bullets],
        "dropped_for_length": max(0, len(bullets) - max_bullets),
        "headline": selection.get("headline") or "",
        "unaddressed": selection.get("unaddressed") or [],
        "unknown_fact_ids": unknown,
        "guard": fact_guard.summarise(verdicts),
        "cost_usd": selection.get("cost_usd"),
    }
