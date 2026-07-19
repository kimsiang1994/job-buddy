"""Grades a rendered resume against one job, by simulating the screen it faces.

This module is an EVALUATOR of tailored output, and nothing more. It has no
authority over factual truth -- that is `fact_guard`'s job and the user's. A
panel verdict can say a resume reads thin, that a requirement went unaddressed,
or that a recruiter would bin it in six seconds. It can never authorise a claim,
and nothing it returns is evidence that a bullet is true.

Three things make the difference between a useful judge and a number that only
looks authoritative:

  built from the job    Personas are derived from the job record -- rubric from
                        its extracted requirements, bar from its seniority,
                        expected scope from its salary band. Not static prompts.

  blind                 The grader sees the rendered resume and the job. Never
                        the fact list, the tailoring rationale, which bullets
                        were selected, or a prior panel score. Enforced by
                        `screen()` having no parameter to pass them through.

  calibrated            Scores are trusted for a job only after the panel is
                        shown to rank tailored >= baseline > control. An
                        uncalibrated judge is worse than no judge -- a number
                        that looks authoritative and is not will be trusted.

Every grader claim must quote the resume, and every quote is checked against the
resume text. A fabricated quote is the same failure mode `fact_guard` exists for,
and it is the cheapest available signal that the grader is confabulating.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence

from jobbuddy import job_schema
from jobbuddy import skills_taxonomy
from jobbuddy.deepseek import deepseek_client

# The three lenses, in the order a real pipeline applies them.
PERSONA_NAMES = ("recruiter", "hiring_manager", "skeptic")

DECISIONS = ("advance", "borderline", "reject")

# What `screen` demands back. Passed to json_chat as schema_keys, so a reply
# missing any of them gets one repair attempt before it counts as a failure.
VERDICT_KEYS = ("decision", "score", "reasons", "missing_requirements", "evidence")

MAX_REASONS = 3

# A recruiter's six seconds does not reach requirement forty.
RECRUITER_RUBRIC_SIZE = 6

# Monthly SGD band -> the scope of work that pay implies. Used to set what the
# persona expects a candidate at this price to have owned.
_SCOPE_BANDS = (
    (6000, "individual delivery: doing the work, under someone else's direction"),
    (12000, "owning a workstream end to end, including its stakeholders"),
    (20000, "owning a system or a team's roadmap, and the trade-offs in it"),
    (10 ** 9, "org-level scope: setting direction others deliver against"),
)

_BARS = {
    "intern": "can they learn fast; no track record expected",
    "junior": "supervised delivery, with evidence they finish what they start",
    "mid": "independent delivery on a defined problem",
    "senior": "independent delivery on an ambiguous problem, plus impact beyond their own tasks",
    "lead": "sets technical direction and is accountable for others' output",
    "principal": "solves problems the organisation could not previously solve",
    "manager": "accountable for a team's delivery, hiring and performance",
    "director": "accountable for a function and its budget",
    "executive": "accountable for a business outcome",
}

_DEFAULT_BAR = "unstated level: judge against what the description actually asks for"


def _norm_ws(text: Any) -> str:
    """Whitespace-normalised, case-folded text for substring comparison.

    Case is folded as well as whitespace collapsed. The check exists to catch a
    grader inventing a quote, not to catch it re-casing a real one -- dropping a
    genuine quote over a capital letter would make the counter meaningless.
    """
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _requirement_texts(requirements: Any) -> list[str]:
    """Requirements as plain strings, whatever shape the caller had them in.

    Accepts strings, `skill_extract.ExtractedSkill`, and the dict forms that
    come back from JSON. Unparseable entries are skipped rather than raising --
    a bad requirement list must degrade the rubric, not lose the screen.
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in requirements or []:
        text = ""
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            for key in ("text", "term", "requirement", "skill", "name"):
                if item.get(key):
                    text = str(item[key])
                    break
        else:
            text = str(getattr(item, "term", "") or "")
        text = job_schema.norm_text(text)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out


def _is_required(item: Any) -> bool:
    """True unless the requirement is explicitly flagged optional."""
    if isinstance(item, dict):
        return bool(item.get("required", True))
    return bool(getattr(item, "required", True))


def _required_first(requirements: Any) -> list[str]:
    """Requirement strings, mandatory ones first, order otherwise preserved."""
    items = list(requirements or [])
    required = _requirement_texts([i for i in items if _is_required(i)])
    optional = [t for t in _requirement_texts(items) if t not in required]
    return required + optional


def _scope_for(job: dict[str, Any]) -> str | None:
    """Expected scope implied by the stated salary band, or None if unstated."""
    if not job.get("salary_is_stated"):
        return None
    lo, hi = job.get("salary_min_sgd"), job.get("salary_max_sgd")
    if lo is None and hi is None:
        return None
    midpoint = ((lo or hi) + (hi or lo)) / 2.0
    for ceiling, description in _SCOPE_BANDS:
        if midpoint < ceiling:
            return f"pays about SGD {int(midpoint):,}/month, which buys {description}"
    return None


def build_personas(job: dict[str, Any],
                   requirements: Any) -> list[dict[str, Any]]:
    """Three graders, derived from this job record. Never raises.

    Deliberately diverse lenses. Three graders with the same rubric do not vote
    three times -- they vote once, three times over, and the agreement between
    them measures nothing but the shared prompt. Disagreement is the only part
    of a panel that carries information, so each persona is given a different
    question to answer and a different reason to say no.

    Everything persona-specific comes from the job: the rubric is its own
    extracted requirements, the bar is its seniority, the scope is its salary
    band. A static prompt would grade a graduate posting and a principal one
    against the same expectations.
    """
    job = job or {}
    ranked = _required_first(requirements)
    seniority = job.get("seniority")
    bar = _BARS.get(seniority or "", _DEFAULT_BAR)
    scope = _scope_for(job)
    title = job_schema.norm_text(job.get("title")) or "the advertised role"
    company = job_schema.norm_text(job.get("company")) or "the employer"

    common = {
        "job_key": job.get("job_key"),
        "title": title,
        "company": company,
        "seniority": seniority,
        "bar": bar,
        "expected_scope": scope,
    }

    recruiter = {
        **common,
        "name": "recruiter",
        "lens": "a six-second scan, screening a stack of eighty",
        "reads_deeply": False,
        "rubric": [
            f"does the resume title read as {title}, or as something else",
            "are the obvious keywords present where a scan would land: "
            + ", ".join(ranked[:RECRUITER_RUBRIC_SIZE]),
            f"is the most recent employer recognisable, or does {company} have to guess",
            "any instant disqualifier: wrong location, wrong seniority, wrong field",
        ],
        "instruction": (
            "You are a recruiter screening for this role. You spend six seconds "
            "per resume and you do not read the bullets in detail. Judge only on "
            "title alignment, whether the expected keywords are visible at a "
            "glance, whether the employers are recognisable, and obvious "
            "disqualifiers. Do not reward depth you had to read for -- you did "
            "not read it."
        ),
    }

    hiring_manager = {
        **common,
        "name": "hiring_manager",
        "lens": "reads for evidence: mechanism and scale, or nothing",
        "reads_deeply": True,
        "rubric": [f"is there evidence of {r}, with a mechanism and a scale" for r in ranked[:12]]
                  + [f"does the work described reach the bar: {bar}"]
                  + ([f"does the scope match what the pay implies -- {scope}"] if scope else []),
        "instruction": (
            "You are the hiring manager for this role and you would own this "
            "person. Read every line for evidence. A bullet that names a verb "
            "and an outcome but no mechanism and no scale is not evidence -- "
            "'improved data quality' tells you nothing, 'cut a 10-day cycle to "
            "5 by automating 4 ETL jobs' does. Say which claims are substantive "
            "and which are decoration."
        ),
    }

    skeptic = {
        **common,
        "name": "skeptic",
        "lens": "looks for the reason to say no",
        "reads_deeply": True,
        "rubric": [
            "unexplained gaps between roles, and short tenures",
            "claims that read inflated for the level or the tenure they sit in",
            "requirements the resume never addresses at all, silently: "
            + ", ".join(ranked[:12]),
            f"a candidate presenting above the bar for this role ({bar})",
        ],
        "instruction": (
            "You are the panel's skeptic. Your job is to find the reason to "
            "reject, not to be fair. Look for gaps, short tenures, claims that "
            "read inflated for the stated level, and requirements the resume "
            "quietly never addresses. If you cannot find a real reason to "
            "reject, say so plainly -- inventing one is as useless as missing one."
        ),
    }

    return [recruiter, hiring_manager, skeptic]


def _prompt(resume_text: str, job: dict[str, Any], requirements: Sequence[str],
            persona: dict[str, Any]) -> list[dict[str, str]]:
    """The messages for one screen. Resume and job only -- see `screen`."""
    rubric = "\n".join(f"  - {line}" for line in persona.get("rubric") or [])
    requirement_block = "\n".join(f"  - {r}" for r in requirements) or "  (none extracted)"
    scope = persona.get("expected_scope")

    system = (
        f"{persona.get('instruction', '')}\n\n"
        f"Your lens: {persona.get('lens', '')}\n"
        f"The bar for this role: {persona.get('bar', '')}\n"
        + (f"Scope implied by the posting: {scope}\n" if scope else "")
        + "\nYour rubric:\n" + rubric + "\n\n"
        "Reply with a JSON object only, with exactly these keys:\n"
        '  "decision": "advance" | "borderline" | "reject"\n'
        '  "score": integer 0-100\n'
        f'  "reasons": up to {MAX_REASONS} short strings\n'
        '  "missing_requirements": requirements this resume does not address, '
        "each quoted verbatim from the requirement list you were given\n"
        '  "evidence": objects {"claim": ..., "resume_span": ...} where '
        "resume_span is copied EXACTLY from the resume text, character for "
        "character. Do not paraphrase a span. A span that is not in the resume "
        "is discarded and counted against you."
    )

    user = (
        f"ROLE: {persona.get('title')}\n"
        f"COMPANY: {persona.get('company')}\n"
        f"LEVEL: {persona.get('seniority') or 'unstated'}\n\n"
        f"REQUIREMENTS FROM THE JOB DESCRIPTION:\n{requirement_block}\n\n"
        f"JOB DESCRIPTION:\n{job_schema.norm_jd_text(job.get('jd_text'))[:6000]}\n\n"
        f"RESUME:\n{resume_text}"
    )

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _coerce_score(value: Any) -> int | None:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def _string_list(value: Any, limit: int | None = None) -> list[str]:
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in value if isinstance(value, (list, tuple)) else []:
        text = job_schema.norm_text(item if not isinstance(item, dict)
                                    else item.get("text") or item.get("requirement") or "")
        if text and text not in out:
            out.append(text)
    return out[:limit] if limit else out


def _failed(persona_name: str, error: str) -> dict[str, Any]:
    """A screen that did not produce a usable verdict, recorded not raised."""
    return {
        "persona": persona_name,
        "ok": False,
        "decision": None,
        "score": None,
        "reasons": [],
        "missing_requirements": [],
        "evidence": [],
        "dropped_evidence": [],
        "fabricated_spans": 0,
        "error": error,
    }


def screen(resume_text: str, job: dict[str, Any], requirements: Any,
           persona: dict[str, Any],
           chat: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """One persona's verdict on one resume. Never raises.

    THE GRADER IS BLIND. It receives the rendered resume text and the job, and
    that is the complete list. It does not see the fact list, the tailoring
    rationale, which bullets were selected over which others, or any earlier
    persona's score.

    That is enforced structurally rather than by discipline: there is no
    `profile` or `facts` parameter on this function, so there is no way to leak
    them into the prompt without editing this signature -- which is a change a
    reviewer would see. The same model generates and grades here, and a model
    shown its own reasoning agrees with it; blinding is the only mitigation that
    survives the prompt being rewritten later.

    `chat` is injected so tests can run the whole path offline. It defaults to
    the real `json_chat`.
    """
    chat = chat or deepseek_client.json_chat
    name = str((persona or {}).get("name") or "unknown")
    texts = _requirement_texts(requirements)

    try:
        result = chat(_prompt(resume_text or "", job or {}, texts, persona or {}),
                      schema_keys=VERDICT_KEYS, profile="extract", tier="fast")
    except Exception as exc:            # a grader outage is not a resume defect
        return _failed(name, f"grader call raised: {exc}")

    if not isinstance(result, dict):
        return _failed(name, "grader returned a non-dict result")
    data = result.get("data")
    if not isinstance(data, dict):
        return _failed(name, result.get("error") or "grader returned no JSON object")

    decision = str(data.get("decision") or "").strip().lower()
    if decision not in DECISIONS:
        return _failed(name, f"decision {data.get('decision')!r} is not one of {DECISIONS}")

    score = _coerce_score(data.get("score"))
    if score is None:
        return _failed(name, f"score {data.get('score')!r} is not a number")

    # --- evidence spans ---------------------------------------------------
    # Every quote must actually be in the resume. A grader that invents a quote
    # has stopped reading and started composing, and its score is worthless --
    # so the drops are counted and surfaced, not silently swallowed.
    haystack = _norm_ws(resume_text)
    evidence: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for item in data.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        span = job_schema.norm_text(item.get("resume_span"))
        entry = {"claim": job_schema.norm_text(item.get("claim")), "resume_span": span}
        if span and _norm_ws(span) in haystack:
            evidence.append(entry)
        else:
            dropped.append(entry)

    return {
        "persona": name,
        "ok": True,
        "decision": decision,
        "score": score,
        "reasons": _string_list(data.get("reasons"), MAX_REASONS),
        "missing_requirements": _string_list(data.get("missing_requirements")),
        "evidence": evidence,
        "dropped_evidence": dropped,
        "fabricated_spans": len(dropped),
        "error": "",
    }


def consensus(verdicts: Iterable[dict[str, Any]]) -> str:
    """Panel decision: reject on 2+ rejects, advance only on 2+ advances.

    Asymmetric on purpose. Two graders wanting to reject is enough to stop,
    because the cost of one more tailoring pass is an hour and the cost of a
    wasted application is a role you cannot re-apply to for six months.
    """
    decisions = [v.get("decision") for v in verdicts if v.get("ok")]
    if not decisions:
        return "unknown"
    if decisions.count("reject") >= 2:
        return "reject"
    if decisions.count("advance") >= 2:
        return "advance"
    return "borderline"


def run_panel(resume_text: str, job: dict[str, Any], requirements: Any,
              chat: Callable[..., dict[str, Any]] | None = None,
              personas: Sequence[dict[str, Any]] | None = None,
              calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run all three personas over one resume. Never raises.

    `trustworthy` is carried in the result and is None until `calibrate` has
    run for this job. None means "not established", which is not the same as
    True and must not be reported as a score anyone acts on.
    """
    personas = personas or build_personas(job or {}, requirements)
    verdicts = [screen(resume_text, job, requirements, p, chat=chat) for p in personas]

    scored = [v["score"] for v in verdicts if v.get("ok")]
    missing: list[str] = []
    for verdict in verdicts:
        for requirement in verdict.get("missing_requirements") or []:
            if requirement not in missing:
                missing.append(requirement)

    if calibration is None:
        trustworthy, reason = None, "panel not calibrated for this job"
    else:
        trustworthy = bool(calibration.get("trustworthy"))
        reason = str(calibration.get("reason") or "")

    return {
        "consensus": consensus(verdicts),
        "score": round(sum(scored) / len(scored), 1) if scored else None,
        "personas": {v["persona"]: v for v in verdicts},
        "missing_requirements": missing,
        "fabricated_spans": sum(v.get("fabricated_spans", 0) for v in verdicts),
        "failures": [v["persona"] for v in verdicts if not v.get("ok")],
        "trustworthy": trustworthy,
        "reason": reason,
    }


def calibrate(tailored_text: str, baseline_text: str, control_text: str,
              job: dict[str, Any], requirements: Any,
              chat: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Is this panel able to tell good from bad on THIS job? Never raises.

    Runs the panel over three inputs whose true ordering is already known:

      tailored   the resume produced for this job
      baseline   the untailored master resume
      control    a deliberately mismatched resume -- a plumbing CV for an ML role

    The panel is trustworthy for this job only if it scores
    tailored >= baseline > control. If that ordering does not hold, the panel's
    scores for this job MUST be discarded rather than reported with a caveat.
    An uncalibrated judge is worse than no judge -- a number that looks
    authoritative and is not will be trusted.

    Returns {"trustworthy": bool, "ordering": [...], "reason": str}. Feed the
    whole dict back into `run_panel(..., calibration=...)` so the flag travels
    with the score it qualifies.
    """
    personas = build_personas(job or {}, requirements)
    runs = {
        label: run_panel(text, job, requirements, chat=chat, personas=personas)
        for label, text in (("tailored", tailored_text),
                            ("baseline", baseline_text),
                            ("control", control_text))
    }
    scores = {label: run["score"] for label, run in runs.items()}
    ordering = sorted(
        ({"label": label, "score": score} for label, score in scores.items()),
        key=lambda row: (row["score"] is not None, row["score"] or 0.0),
        reverse=True,
    )

    unscored = [label for label, score in scores.items() if score is None]
    if unscored:
        return {
            "trustworthy": False,
            "ordering": ordering,
            "reason": f"no usable panel score for: {', '.join(sorted(unscored))}",
            "runs": runs,
        }

    tailored, baseline, control = scores["tailored"], scores["baseline"], scores["control"]

    # `>=` between tailored and baseline, `>` above control. Tailoring selects
    # from the same true facts, so a tie there is a real outcome. Failing to
    # rank a plumbing CV below either is not a tie -- it is the judge being
    # unable to see the thing it exists to measure.
    if control >= baseline:
        reason = (f"control scored {control} against baseline {baseline}: the panel "
                  "cannot distinguish a mismatched resume from an untailored one")
        return {"trustworthy": False, "ordering": ordering, "reason": reason, "runs": runs}
    if tailored < baseline:
        reason = (f"tailored scored {tailored} below baseline {baseline}: the panel "
                  "does not reward tailoring on this job")
        return {"trustworthy": False, "ordering": ordering, "reason": reason, "runs": runs}

    return {
        "trustworthy": True,
        "ordering": ordering,
        "reason": f"ordering holds: tailored {tailored} >= baseline {baseline} > control {control}",
        "runs": runs,
    }


# --------------------------------------------------------------------------
# Gap attribution -- the only function here that touches facts, and it runs
# AFTER screening. Feeding facts in before a screen would un-blind the grader,
# which is the one property this module is built around.
# --------------------------------------------------------------------------

def _fact_terms(fact: dict[str, Any]) -> dict[str, float]:
    """A fact's claimable skills/entities, as skills_taxonomy's `owned` map.

    Only `skills` and `entities` -- not `org`, `role` or `team`. A requirement
    is not covered because the candidate's job title contained the same noun.
    """
    owned: dict[str, float] = {}
    for key in ("skills", "entities"):
        for item in fact.get(key) or []:
            term = skills_taxonomy.canon(item)
            if term:
                owned[term] = 1.0
    return owned


def attribute_gaps(missing_requirements: Iterable[str],
                   facts: Any) -> dict[str, Any]:
    """Split missing requirements into "we cut it" and "we do not have it".

    Deterministic set matching against the fact records, no LLM. Reuses
    `skills_taxonomy.match`, so the deliberate absence of a fuzzy token-overlap
    pass applies here too: a false cover would tell the user they already have
    experience they do not, which is the failure this codebase spends the most
    effort preventing.

    Returns:
      have_but_cut       [{"requirement", "fact_id", "matched", "how"}] -- a fact
                         covers it, so it was dropped for space. Actionable now:
                         put it back.
      genuinely_lacking  [str] -- nothing in the profile covers it. An honest
                         development plan, not a tailoring bug.
    """
    records = list(facts.values()) if isinstance(facts, dict) else list(facts or [])
    indexed = [(rec, _fact_terms(rec)) for rec in records if isinstance(rec, dict)]

    have_but_cut: list[dict[str, Any]] = []
    genuinely_lacking: list[str] = []

    for requirement in _requirement_texts(missing_requirements):
        best: tuple[float, dict[str, Any], str, str] | None = None
        for record, owned in indexed:
            if not owned:
                continue
            weight, via, how = skills_taxonomy.match(requirement, owned)
            if weight > 0 and (best is None or weight > best[0]):
                best = (weight, record, via, how)
        if best is None:
            genuinely_lacking.append(requirement)
        else:
            _, record, via, how = best
            have_but_cut.append({
                "requirement": requirement,
                "fact_id": str(record.get("fact_id") or ""),
                "matched": via,
                "how": how,
            })

    return {
        "have_but_cut": have_but_cut,
        "genuinely_lacking": genuinely_lacking,
        "explanation": (
            f"{len(have_but_cut)} of {len(have_but_cut) + len(genuinely_lacking)} "
            "unaddressed requirements are already covered by a recorded fact and "
            "were cut, not absent; the rest are real gaps."
        ),
    }
