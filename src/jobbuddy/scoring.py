"""Deterministic job scoring. No LLM, no network.

Every component returns a value in 0-100 *plus* the inputs it used and a
sentence explaining itself, so a ranking can be audited by eye in the workbook
without re-running anything. A score you cannot explain is a score you cannot
trust, and this one decides where you spend your evenings.

Two rules that keep it honest:

  - Hard filters run BEFORE scoring. A job that fails work authorisation or the
    salary floor is excluded with a reason, not ranked at 12.
  - A component with no data returns None and its weight is *removed* from the
    denominator. Imputing 50 would quietly claim knowledge we do not have.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jobbuddy import job_schema
from jobbuddy import skills_taxonomy

REPO_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_DIR / "run_config.json"

SKILL_TIER_WEIGHT = {"expert": 1.0, "working": 0.7, "familiar": 0.35}


@dataclass
class ScoreContext:
    """Whatever a component needs beyond the job and the profile.

    Exists so that every component can share one signature. Add a field here
    rather than adding a parameter to one function -- a component that cannot
    reach the registry ends up outside the loop, and outside the error handling
    and renormalisation the loop provides.
    """

    velocity: dict[str, dict[str, Any]] = field(default_factory=dict)
    market_wages: dict[str, Any] = field(default_factory=dict)
    home_postal_code: str | None = None

# A posting collects most of its applications early; after about three weeks
# the pile is deep and the recruiter is usually already interviewing.
FRESHNESS_HALFLIFE_DAYS = 21.0

# What an unmeasurable component is worth. Not 50-as-a-guess -- 50 is the point
# at which a missing signal neither helps nor hurts, which is the only honest
# thing to say about evidence you do not have.
NEUTRAL_PRIOR = 50.0

_warned: set[str] = set()
_config_cache: dict[str, Any] | None = None


def _warn(message: str) -> None:
    if message in _warned:
        return
    _warned.add(message)
    try:
        import sys

        print(f"scoring: {message}", file=sys.stderr)
    except Exception:
        pass


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read run_config.json. Never raises; falls back to safe defaults."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    fallback: dict[str, Any] = {
        "filters": {"singapore_only": True, "open_only": True},
        "profile": {"skills": {}, "target_seniority": None, "years_experience": None},
        "weights": {"skill_match": 30, "seniority_fit": 15, "comp_signal": 15,
                    "competition": 20, "company_signal": 10,
                    "application_friction": 5, "freshness": 5},
        "scopes": [],
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("run_config.json is not an object")
        _config_cache = data
    except (OSError, ValueError) as exc:
        _warn(f"could not read {path.name} ({exc}); using built-in defaults")
        _config_cache = fallback
    return _config_cache


def reload() -> None:
    global _config_cache
    _config_cache = None


def _flat_skills(profile: dict[str, Any]) -> dict[str, float]:
    """Canonical skill -> proficiency weight."""
    return skills_taxonomy.build_owned(profile.get("skills") or {}, SKILL_TIER_WEIGHT)


# --------------------------------------------------------------------------
# Components. Each returns (score_or_None, detail_dict).
# --------------------------------------------------------------------------

def score_skill_match(job: dict[str, Any], profile: dict[str, Any],
                     ctx: "ScoreContext | None" = None) -> tuple[float | None, dict]:
    """Share of the job's requested skills the profile actually covers.

    Key skills (MCF's `isKeySkill`) count double when the board bothers to set
    them; in practice it usually does not, so the fallback is all skills equal.
    """
    owned = _flat_skills(profile)
    if not owned:
        return None, {"reason": "profile lists no skills"}

    # Drop extraction noise BEFORE scoring. MCF's extractor emits things like
    # 'Ship Building' and 'scientific discipline'; counting those as skills the
    # candidate lacks is what made a matching role score 0/16.
    raw_terms = job.get("skills_raw") or []
    terms = skills_taxonomy.clean_job_skills(raw_terms)
    dropped = len(raw_terms) - len(terms)
    if not terms:
        return None, {"reason": "job lists no usable skills", "noise_dropped": dropped}

    key_terms = {skills_taxonomy.canon(s) for s in job.get("skills_key") or []}
    # Per-skill importance, set by the extractor from where the term appeared.
    # A job description is not a flat list: a skill named in the title is the
    # job, one under 'nice to have' is a wish. Scoring them equally made a
    # candidate who met every core requirement and none of the twenty
    # peripheral ones look like a 13% match.
    declared = job.get("skills_weight") or {}
    title_text = f" {(job.get('title') or '').lower()} "

    def importance_of(term: str) -> float:
        if term in declared:
            return float(declared[term])
        # MCF and any other source that flags key skills but gives no weights.
        if skills_taxonomy.canon(term) in key_terms:
            return 2.0
        if term.lower() in title_text:
            return 4.0
        return 1.0

    total = 0.0
    earned = 0.0
    core_total = 0.0
    core_earned = 0.0
    matched: list[str] = []
    missing: list[str] = []
    missing_core: list[str] = []

    for term in terms:
        importance = importance_of(term)
        total += importance
        is_core = importance >= 1.0        # required or better; excludes wishes
        if is_core:
            core_total += importance

        weight, via, how = skills_taxonomy.match(term, owned)
        if weight > 0:
            earned += importance * weight
            if is_core:
                core_earned += importance * weight
            matched.append(f"{term} ({how} -> {via})" if how != "exact" else term)
        else:
            missing.append(term)
            if is_core:
                missing_core.append(term)

    score = 100.0 * earned / total if total else None
    core_score = (100.0 * core_earned / core_total) if core_total else None

    detail = {
        "matched": matched[:20],
        "missing": missing[:20],
        "matched_count": len(matched),
        "total_count": len(terms),
        "noise_dropped": dropped,
        # Reported separately because it answers the question that actually
        # matters: not "did you tick every box", but "do you have what this job
        # is about". Nobody is a perfect match for a 38-item wishlist.
        "core_score": None if core_score is None else round(core_score, 1),
        "missing_core": missing_core[:10],
        "core_requirements": len([t for t in terms if importance_of(t) >= 1.0]),
    }
    return score, detail


def score_seniority_fit(job: dict[str, Any], profile: dict[str, Any],
                     ctx: "ScoreContext | None" = None) -> tuple[float | None, dict]:
    """How well the job's level matches what the candidate is aiming for.

    Scored against `target_seniority` -- the level they WANT -- not the one they
    currently hold. That distinction matters: defaulting the target to the
    current level made the scorer rank staying-put roles top and stretch roles
    35 points lower, which is the opposite of why anyone runs a job search.

    Asymmetric, but now tilted toward the stretch. Reaching one level above
    target is a live application; dropping one below is a step backwards that
    only pay can justify, and pay is scored separately.
    """
    target = profile.get("target_seniority")
    level = job.get("seniority")
    if not target or not level:
        return None, {"reason": "seniority unknown on job or profile"}

    gap = job_schema.seniority_distance(target, level)
    if gap is None:
        return None, {"reason": "level not on ladder"}

    if gap == 0:
        score = 100.0
    elif gap > 0:            # above target: a reach, and reaching is the point
        score = max(0.0, 100.0 - 25.0 * gap)
    else:                    # below target: safe, but it is why you would not move
        score = max(0.0, 100.0 - 32.0 * abs(gap))

    # An inference from a dropdown is weaker evidence than one from the title,
    # so pull a weak basis toward neutral rather than trusting it fully.
    basis = job.get("seniority_basis") or "unknown"
    if basis == "position_level":
        score = 50.0 + (score - 50.0) * 0.6

    years = job.get("min_years_exp")
    have = profile.get("years_experience")
    years_note = None
    if isinstance(years, int) and years > 0 and isinstance(have, (int, float)):
        if years > have:
            shortfall = years - have
            score = max(0.0, score - 8.0 * shortfall)
            years_note = f"asks {years}y, profile has {have}y"
        else:
            years_note = f"asks {years}y, profile has {have}y (met)"

    return score, {
        "job_level": level, "target_level": target, "gap": gap,
        "current_level": profile.get("current_seniority"),
        "basis": basis, "years": years_note,
        "direction": ("at target" if gap == 0
                      else f"{gap} level(s) above target"
                      if gap > 0 else f"{abs(gap)} level(s) below target"),
    }


def score_comp_signal(job: dict[str, Any], profile: dict[str, Any],
                     ctx: "ScoreContext | None" = None) -> tuple[float | None, dict]:
    """Pay, relative to what the profile is worth.

    Uses the stated range midpoint. MCF makes salary mandatory, so this is real
    for nearly every SG posting -- which is unusual and worth leaning on.
    """
    lo, hi = job.get("salary_min_sgd"), job.get("salary_max_sgd")
    if not job.get("salary_is_stated") or lo is None:
        return None, {"reason": "salary not stated"}

    midpoint = (lo + hi) / 2.0 if hi else float(lo)
    current = profile.get("current_salary_sgd_monthly")
    reference = current or 12000.0  # sane SG senior-AI anchor when unset

    ratio = midpoint / reference
    # 1.0x reference -> 60, 1.3x -> 100, 0.7x -> 20. Linear between.
    score = max(0.0, min(100.0, 60.0 + (ratio - 1.0) * 133.0))
    return score, {
        "midpoint_sgd": int(midpoint), "range": [lo, hi],
        "reference_sgd": int(reference),
        "reference_basis": "current salary" if current else "default anchor",
        "ratio": round(ratio, 2),
    }


def score_competition(job: dict[str, Any], profile: dict[str, Any],
                     ctx: "ScoreContext | None" = None) -> tuple[float | None, dict]:
    """How crowded this posting is. Higher score = less competition.

    This is the component the whole state layer exists for. On MCF it uses the
    REAL submitted-application count, which the portal publishes -- not a proxy,
    and not LinkedIn's click-through number.
    """
    applications = job.get("applications")
    views = job.get("views")
    age = job.get("age_days")
    vacancies = job.get("vacancies") or 1

    detail: dict[str, Any] = {
        "applications": applications, "views": views,
        "age_days": age, "vacancies": vacancies,
        "repost_count": job.get("repost_count"),
    }

    if not isinstance(applications, int):
        # No published count -- fall back to age and reposts alone.
        if age is None:
            return None, {**detail, "reason": "no application count and no posting date"}
        score = max(0.0, 100.0 * (0.5 ** (age / FRESHNESS_HALFLIFE_DAYS)))
        detail["basis"] = "age proxy (no published count)"
        return score, detail

    # Applications per vacancy is the number that actually matters: 20 applicants
    # for 5 openings is a very different race from 20 for 1.
    per_vacancy = applications / max(vacancies, 1)

    # Calibrated against live MCF data, where a typical SG tech posting sits
    # around 5-20 applications. 0 -> 100, 10 -> ~65, 30 -> ~30, 80+ -> near 0.
    score = 100.0 * (0.5 ** (per_vacancy / 18.0))

    # Arrival rate matters as much as the total. A job with 15 applications in
    # 30 days is calmer than one with 15 in 2 days.
    apps_per_day = job.get("apps_per_day")
    if isinstance(apps_per_day, (int, float)) and apps_per_day > 0:
        if apps_per_day > 2.0:
            score *= 0.75
        elif apps_per_day < 0.3:
            score = min(100.0, score * 1.15)
        detail["apps_per_day"] = apps_per_day

    # Repeated reposting means the role has failed to fill. Mildly good for you
    # (they are still looking, the bar may soften) but it also signals churn.
    reposts = job.get("repost_count") or 0
    if reposts >= 2:
        score *= 0.9
        detail["repost_penalty"] = True

    if isinstance(views, int) and views > 0:
        detail["apps_per_view"] = round(applications / views, 4)

    detail["applications_per_vacancy"] = round(per_vacancy, 2)
    detail["basis"] = "published application count"
    return max(0.0, min(100.0, score)), detail


def score_freshness(job: dict[str, Any], profile: dict[str, Any],
                     ctx: "ScoreContext | None" = None) -> tuple[float | None, dict]:
    """How recently posted. Separate from competition so each is legible."""
    age = job.get("age_days")
    if age is None:
        return None, {"reason": "no posting date"}
    score = max(0.0, 100.0 * (0.5 ** (age / FRESHNESS_HALFLIFE_DAYS)))
    detail = {"age_days": age, "posted_at": job.get("posted_at")}
    expires = job.get("expires_at")
    if expires:
        detail["expires_at"] = expires
        from datetime import date as _date
        try:
            detail["days_left"] = (_date.fromisoformat(expires) - _date.today()).days
        except ValueError:
            pass
    return score, detail


def score_application_friction(job: dict[str, Any], profile: dict[str, Any],
                     ctx: "ScoreContext | None" = None) -> tuple[float | None, dict]:
    """Directness of the application path. Higher = less friction.

    Agency postings mean a recruiter screen before anyone technical sees you,
    and the same requisition often appears under three agencies at once.
    """
    if job.get("is_agency"):
        return 30.0, {"path": "recruitment agency", "note": "extra screen, possible duplicate req"}
    if job.get("source") == "mcf":
        return 85.0, {"path": "direct employer via MCF"}
    return 100.0, {"path": "direct employer ATS"}


def score_company_signal(job: dict[str, Any], profile: dict[str, Any],
                         ctx: "ScoreContext | None" = None) -> tuple[float | None, dict]:
    """Hiring posture. None until enough history exists -- never imputed."""
    velocity = (ctx or ScoreContext()).velocity
    if not velocity:
        return None, {"reason": "no company history yet"}
    stats = velocity.get(job.get("company_norm") or "")
    if not stats:
        return None, {"reason": "company not seen before"}
    if not stats.get("sufficient"):
        return None, {
            "reason": f"insufficient history ({stats.get('history_days', 0)} days)",
            "open_reqs": stats.get("open_reqs"),
        }
    open_reqs = stats.get("open_reqs", 0)
    score = min(100.0, 40.0 + 12.0 * open_reqs)
    return score, {
        "open_reqs": open_reqs,
        "new_in_window": stats.get("new_in_window"),
        "history_days": stats.get("history_days"),
    }


# --------------------------------------------------------------------------
# Hard filters
# --------------------------------------------------------------------------

def check_filters(job: dict[str, Any], config: dict[str, Any]) -> str | None:
    """Return an exclusion reason, or None to keep the job."""
    filters = config.get("filters") or {}
    profile = config.get("profile") or {}

    if filters.get("singapore_only", True) and job.get("is_overseas"):
        return "not in Singapore"
    if filters.get("open_only", True) and not job.get("is_open", True):
        return f"not open (status={job.get('source_status')})"

    # Both sides go through norm_company. Comparing a raw string against an
    # already-normalised one fails silently on any name carrying a legal suffix:
    # 'bytedance' matched, but 'bytedance technology', 'TikTok Singapore' and
    # 'Grab Holdings' did not -- because norm_company strips technology, sg,
    # singapore, pte, ltd, holdings. You then get suggestions from the employer
    # you explicitly excluded, which is the one filter people actually check.
    company = job.get("company_norm") or ""
    for excluded in filters.get("exclude_companies") or []:
        needle = job_schema.norm_company(excluded)
        if needle and needle in company:
            return f"excluded company ({excluded})"

    title = (job.get("title") or "").lower()
    for pattern in filters.get("exclude_title_patterns") or []:
        try:
            if re.search(pattern, title, re.I):
                return f"title matched exclusion /{pattern}/"
        except re.error:
            _warn(f"bad regex in exclude_title_patterns: {pattern!r}")

    if filters.get("exclude_agencies") and job.get("is_agency"):
        return "recruitment agency posting"

    floor = filters.get("min_salary_sgd_monthly")
    if isinstance(floor, (int, float)):
        if job.get("salary_is_stated") and job.get("salary_max_sgd") is not None:
            # Compare against the TOP of the range: if the most they will pay is
            # under your floor, it is out regardless of where they start.
            if job["salary_max_sgd"] < floor:
                return f"max salary {job['salary_max_sgd']} below floor {int(floor)}"
        elif not filters.get("allow_unstated_salary", True):
            return "salary not stated"

    max_gap = filters.get("max_seniority_gap_below")
    if isinstance(max_gap, int):
        gap = job_schema.seniority_distance(profile.get("target_seniority"), job.get("seniority"))
        if gap is not None and gap < -max_gap:
            return f"{abs(gap)} levels below target seniority"

    return None


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------

# Every component has the same signature and lives in this tuple. company_signal
# used to sit outside it, because it needed one extra argument -- and the copy of
# the accumulate-and-renormalise block that came with it was missing the
# try/except the loop has, so the one component most likely to hit missing data
# was the one that could take down the whole job. A third context-needing
# component (commute distance, wage benchmark) would have meant a third copy.
_COMPONENTS = (
    ("skill_match", score_skill_match),
    ("seniority_fit", score_seniority_fit),
    ("comp_signal", score_comp_signal),
    ("competition", score_competition),
    ("company_signal", score_company_signal),
    ("application_friction", score_application_friction),
    ("freshness", score_freshness),
)


def score_job(
    job: dict[str, Any],
    config: dict[str, Any] | None = None,
    velocity: dict[str, dict[str, Any]] | None = None,
    ctx: ScoreContext | None = None,
) -> dict[str, Any]:
    """Score one job. Writes into job['scores'] and returns it.

    Components that return None are dropped and the remaining weights
    renormalise, so a missing signal never silently counts as average.

    `velocity` is accepted for compatibility with existing callers; prefer
    passing a ScoreContext, which is where any future extra context belongs.
    """
    config = config or load_config()
    profile = config.get("profile") or {}
    weights = config.get("weights") or {}
    if ctx is None:
        ctx = ScoreContext(velocity=velocity or {})

    components: dict[str, Any] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for name, func in _COMPONENTS:
        try:
            value, detail = func(job, profile, ctx)
        except Exception as exc:  # one bad component must not lose the job
            _warn(f"component {name} raised on {job.get('job_key')} ({exc})")
            value, detail = None, {"error": str(exc)}
        weight = float(weights.get(name, 0) or 0)
        components[name] = {"value": None if value is None else round(value, 1),
                            "weight": weight, "detail": detail}
        if value is not None and weight > 0:
            weighted_sum += value * weight
            weight_total += weight

    total = round(weighted_sum / weight_total, 1) if weight_total else 0.0

    # Confidence-adjusted rank.
    #
    # Renormalisation alone is right within one source and wrong across several.
    # MyCareersFuture publishes salary and a real application count; Workable,
    # the ATS boards and HN publish neither. So a sparse job is scored only on
    # the components where it happens to do well -- and freshness and low
    # application friction are exactly those. Measured on a live run: nine jobs
    # with no salary scored 90-95 while a Micron role with a stated
    # 10-20k range scored 82. Knowing less made a job look better.
    #
    # So the score stays honest about what was measurable, and the RANK shrinks
    # it toward neutral in proportion to how little was measurable. A job graded
    # on half the evidence moves half-way to 50. Nothing is imputed; the
    # uncertainty is priced instead of hidden.
    available = sum(float(w or 0) for k, w in weights.items() if not k.startswith("_"))
    confidence = (weight_total / available) if available else 0.0
    adjusted = round(total * confidence + NEUTRAL_PRIOR * (1.0 - confidence), 1)

    job["scores"] = {
        "total": total,
        "adjusted": adjusted,
        "confidence": round(confidence, 3),
        "components": components,
        "weight_used": weight_total,
        "weight_available": available,
        "explanation": explain(components, total),
    }
    return job["scores"]


def explain(components: dict[str, Any], total: float) -> str:
    """One human sentence: what lifted this job, and what held it back."""
    scored = [
        (name, c["value"], c["weight"])
        for name, c in components.items()
        if c.get("value") is not None and c.get("weight", 0) > 0
    ]
    if not scored:
        return "no components could be scored"

    # Rank by contribution to the weighted mean, not raw score -- a 100 on a
    # 5-weight component is not what moved this job up the list.
    ranked = sorted(scored, key=lambda x: x[1] * x[2], reverse=True)
    best = ranked[:2]
    worst = sorted(scored, key=lambda x: x[1])[:2]

    ups = ", ".join(f"{n.replace('_', ' ')} {v:.0f}" for n, v, _ in best)
    downs = ", ".join(f"{n.replace('_', ' ')} {v:.0f}" for n, v, _ in worst)
    missing = [n for n, c in components.items() if c.get("value") is None]
    tail = f"; not scored: {', '.join(missing)}" if missing else ""
    return f"{total:.0f} overall. Strongest: {ups}. Weakest: {downs}{tail}."
