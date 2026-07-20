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
    # The whole run_config, so a component can read a tunable threshold without
    # reaching for load_config() and silently ignoring the config a caller
    # passed to score_job. Populated by score_job; None when a component is
    # called directly, in which case that component must fall back to its own
    # documented defaults rather than raise.
    config: dict[str, Any] | None = None

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
        "weights": {"skill_match": 21, "reach": 15, "seniority_fit": 15,
                    "comp_signal": 13, "competition": 16, "company_signal": 10,
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

    def is_core(term: str) -> bool:
        """True only when the posting itself SAYS this is required.

        `importance_of` returns 1.0 for a term that is merely present, and the
        old test was `importance >= 1.0` -- so every extracted term counted as
        core and `core_score` actually measured "coverage of everything the ad
        mentions". The name says mandatory, and `score_reach` reads it believing
        that, so a stretch role and a comfortable one scored more alike than
        they should.

        Membership is now positive evidence only: an explicit weight at or
        above required, a source-flagged key skill, or the job title itself. A
        term the ad merely lists is not evidence of a requirement.

        For a posting that publishes no weighting at all this leaves nothing
        core, `core_score` is None, and the reach component is dropped and its
        weight renormalised. That is the honest outcome -- we do not know which
        of its requirements are mandatory -- and it is what two thirds of
        postings avoid, since they do publish weights.
        """
        if term in declared:
            return float(declared[term]) >= 1.0
        if skills_taxonomy.canon(term) in key_terms:
            return True
        return term.lower() in title_text

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
        core = is_core(term)
        if core:
            core_total += importance

        weight, via, how = skills_taxonomy.match(term, owned)
        if weight > 0:
            earned += importance * weight
            if core:
                core_earned += importance * weight
            matched.append(f"{term} ({how} -> {via})" if how != "exact" else term)
        else:
            missing.append(term)
            if core:
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
        "core_requirements": len([t for t in terms if is_core(t)]),
    }
    return score, detail


# How much of a job's MANDATORY requirements you have to cover before it stops
# counting as a stretch. Both are percentages of `core_score`.
#
# Defaults chosen to be legible rather than clever: cover 70% of what a job says
# it must have and it is a normal application, so reach stops penalising it at
# all. Cover 30% or less and you are applying to a job that is mostly about
# things you have not done. Between the two the penalty is linear.
#
# These sit in run_config.json under weights/_reach; the constants here are the
# fallback for a component called outside score_job.
REACH_COMFORTABLE_COVERAGE = 70.0
REACH_OUT_OF_DEPTH_COVERAGE = 30.0

# What a credential you do not hold costs when the posting only *prefers* it.
#
# Small on purpose. "PhD preferred" is not a gap in ability -- plenty of those
# roles hire strong MSc candidates -- but it is not free either: you are
# starting behind anyone who has one. So it marks the job down rather than
# hiding it, which is the whole point of having reach separate from knockouts.
#
# 0.12 was chosen so the deduction is visible in a ranking without ever
# reordering it against a genuine capability gap: 12% of reach at weight 15 is
# under two points of the final score, where the difference between a fit and a
# stretch is fifteen. A well-matched job with 'PhD preferred' must still
# comfortably outrank a role the candidate cannot do.
REACH_PREFERRED_CREDENTIAL_PENALTY = 0.12

# A hand-edited config must not be able to turn a modest tilt into the component.
MAX_CREDENTIAL_PENALTY = 0.5


def score_reach(job: dict[str, Any], profile: dict[str, Any],
                ctx: "ScoreContext | None" = None) -> tuple[float | None, dict]:
    """How far out of reach this job is, on its MANDATORY requirements alone.

    Distinct from skill_match, which averages over everything a posting names --
    including the twenty-item wishlist under 'nice to have'. A job can score
    respectably there while the candidate misses most of what it says it must
    have, and that job is the one this component exists to push down the list.

    So it reads only `core_score` from score_skill_match -- the coverage of
    terms weighted at required-or-better. It does NOT recompute that matching:
    a second copy would eventually disagree with the first, and the two numbers
    would then contradict each other in the same report.

    Deliberately flat at 100 above `comfortable_coverage`. A component that kept
    discriminating at the top would be scoring the same signal as skill_match
    twice; the job here is only to separate "you can do this" from "this is
    someone else's job", not to re-rank the jobs that already fit.

    On top of that coverage, a credential the posting WANTS and the profile
    lacks costs a small fixed percentage -- see
    REACH_PREFERRED_CREDENTIAL_PENALTY. "PhD preferred" is not a knockout and
    must not be one, but costing nothing at all was wrong too: it is precisely
    the role where you start behind the candidates who have one.

    Returns None -- never a neutral 50 -- when the posting states no mandatory
    requirements. Imputing there would claim the candidate half-fits a job
    nobody has measured, and the renormalisation in score_job already handles a
    missing component by removing its weight from the denominator.
    """
    _, detail = score_skill_match(job, profile, ctx)
    coverage = detail.get("core_score")
    if coverage is None:
        return None, {
            "reason": detail.get("reason") or "job states no mandatory requirements",
            "core_requirements": detail.get("core_requirements", 0),
        }

    reach_cfg = ((ctx.config if ctx else None) or {}).get("weights", {})
    comfortable = _as_float(reach_cfg.get("_reach_comfortable_coverage"),
                            REACH_COMFORTABLE_COVERAGE)
    out_of_depth = _as_float(reach_cfg.get("_reach_out_of_depth_coverage"),
                             REACH_OUT_OF_DEPTH_COVERAGE)
    # A misconfigured pair (out_of_depth above comfortable) would divide by a
    # negative span and invert the whole component, ranking the worst fits top.
    if out_of_depth >= comfortable:
        _warn(f"reach thresholds inverted ({out_of_depth} >= {comfortable}); "
              f"using built-in {REACH_OUT_OF_DEPTH_COVERAGE}/"
              f"{REACH_COMFORTABLE_COVERAGE}")
        comfortable, out_of_depth = REACH_COMFORTABLE_COVERAGE, REACH_OUT_OF_DEPTH_COVERAGE

    span = comfortable - out_of_depth
    score = 100.0 * (coverage - out_of_depth) / span
    score = max(0.0, min(100.0, score))
    before_credentials = score

    penalty = min(MAX_CREDENTIAL_PENALTY, max(0.0, _as_float(
        reach_cfg.get("_reach_preferred_credential_penalty"),
        REACH_PREFERRED_CREDENTIAL_PENALTY)))
    lacked = wanted_credentials_lacked(job, profile)
    if lacked and penalty:
        # Applied once however many are named. Two preferred credentials do not
        # make a job twice as hard to get, and stacking would quietly turn a
        # modest tilt into something that competes with a real capability gap.
        score *= (1.0 - penalty)

    missing = detail.get("missing_core") or []
    if before_credentials >= 100.0 and not lacked:
        verdict = "within reach"
    elif score <= 0.0:
        verdict = "out of reach"
    elif before_credentials >= 100.0:
        verdict = "within reach, but they would prefer a credential you lack"
    else:
        verdict = "a stretch"

    explanation = f"covers {coverage:.0f}% of what this job says it requires"
    if missing:
        explanation += f"; missing {', '.join(missing[:4])}"
    if lacked and penalty:
        explanation += (f"; they want {', '.join(lacked)} and you hold none"
                        f" (reach -{penalty * 100:.0f}%)")

    return score, {
        "core_coverage": coverage,
        "core_requirements": detail.get("core_requirements", 0),
        "missing_core": missing[:10],
        "comfortable_coverage": comfortable,
        "out_of_depth_coverage": out_of_depth,
        # Both sides of the deduction are reported so the report can show its
        # working. A silent adjustment is the thing this component exists to
        # stop happening.
        "coverage_score": round(before_credentials, 1),
        "credentials_wanted_not_held": lacked,
        "credential_penalty": penalty if lacked else 0.0,
        "verdict": verdict,
        "explanation": explanation,
    }


def _as_float(value: Any, fallback: float) -> float:
    """A config number, or the fallback. Never raises on a hand-edited file."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


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

# --------------------------------------------------------------------------
# Knockouts -- credential gaps that no amount of preparation closes
# --------------------------------------------------------------------------
#
# Separate from reach, and deliberately so. Reach says "you would be stretching"
# and lowers the rank; a knockout says "you cannot apply to this at all today"
# and removes the job. A doctorate is not a weekend of study, and ranking a
# PhD-required research post at 41 instead of excluding it just means reading
# the same rejection twice.
#
# The whole design of this section is asymmetric on purpose. Excluding a job the
# candidate could actually have got is silent and unrecoverable -- they never
# see it. Keeping one they cannot get costs thirty seconds of reading. So every
# rule below fires ONLY on explicitly required phrasing, an optional marker
# anywhere in the same sentence vetoes the knockout, and anything ambiguous
# keeps the job.

# Split a description into the units a requirement is actually written in:
# lines, bullets and sentences. Matching "PhD" and "required" anywhere in a
# 4,000-character description would knock out a job whose only 'required' is in
# "a valid work pass is required" three paragraphs away.
_SEGMENT_SPLIT_RE = re.compile(r"[\n\r•·;]+|(?<=[.!?])\s+")

# Checked FIRST and always wins. "PhD preferred" and "PhD or equivalent
# experience" are exactly the postings a strong candidate gets interviewed for,
# and hiding those would defeat the point of running the search.
_OPTIONAL_MARKERS = re.compile(
    r"\b(?:preferred|preferable|preferably|plus|bonus|nice[- ]to[- ]have|"
    r"nice to have|good to have|desirable|desired|advantage|advantageous|"
    r"ideally|ideal candidate|would be great|welcome|optional|not required|"
    r"not necessary|or equivalent|equivalent experience|equivalent practical|"
    r"or comparable|or similar|an asset|beneficial)\b", re.I)

_REQUIRED_MARKERS = re.compile(
    # 'requirement' alone would not match the commonest heading of all,
    # 'Requirements:', because \b fails on the trailing s.
    r"\b(?:require(?:s|d|ment|ments)?|must have|must hold|must possess|"
    r"must be|mandatory|essential|minimum|at least|you must|need to have|"
    r"is a must|non[- ]negotiable|only candidates)\b", re.I)

# The trailing s? matters: without it 'PhDs required' matched nothing, because
# \b fails between the D and the s. That silently exempted every posting using
# the plural, in both directions -- no knockout and no credential deduction.
_DOCTORATE_RE = re.compile(
    r"\b(?:ph\.?\s?d|doctorate|doctoral|d\.?\s?phil)s?\b", re.I)

# Licences and registrations, not certifications. A CISSP or an AWS cert is
# something a working engineer can go and get; bar admission is not.
_LICENCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("bar admission", r"\b(?:admitted to the bar|bar admission|called to the bar|"
                      r"practi[sc]ing certificate|qualified (?:solicitor|lawyer))\b"),
    ("medical registration", r"\b(?:medical registration|medical council|"
                             r"registered nurse|registered pharmacist|"
                             r"practi[sc]ing licence|practi[sc]ing license)\b"),
    ("professional engineer registration",
     r"\b(?:registered professional engineer|professional engineer \(pe\)|"
     r"\bp\.?e\.? licen[cs]e)\b"),
    ("CFA charter", r"\b(?:cfa charter|cfa charterholder|"
                    r"chartered financial analyst)\b"),
    ("accountancy charter", r"\b(?:certified public accountant|chartered accountant)\b"),
)
_LICENCE_RES = tuple((name, re.compile(pattern, re.I))
                     for name, pattern in _LICENCE_PATTERNS)

# Degrees the candidate may well already hold. Present ONLY so that a posting
# saying "Master's preferred" can be recognised as a credential and then
# matched against what the profile holds -- at which point it costs nothing.
# Without them, reach would have to guess, and the safe guess (never penalise)
# would make the whole soft-credential signal unreachable.
_MASTERS_RE = re.compile(
    r"\b(?:master'?s?(?:\s+degree)?|m\.?\s?sc|m\.?\s?eng|mba|"
    r"postgraduate degree)\b", re.I)
_BACHELORS_RE = re.compile(
    r"\b(?:bachelor'?s?(?:\s+degree)?|b\.?\s?sc|b\.?\s?eng|"
    r"undergraduate degree)\b", re.I)

# Ordered: the strongest credential named in a sentence is the one reported.
_SOFT_CREDENTIALS: tuple[tuple[str, Any], ...] = (
    (("doctorate", _DOCTORATE_RE),)
    + _LICENCE_RES
    + (("master's degree", _MASTERS_RE), ("bachelor's degree", _BACHELORS_RE))
)

# What the profile already has, so a posting asking for it costs nothing.
# Overridden by profile.credentials_held in run_config.json.
DEFAULT_CREDENTIALS_HELD = ("master's degree", "bachelor's degree")


def wanted_credentials_lacked(job: dict[str, Any],
                              profile: dict[str, Any]) -> list[str]:
    """Credentials this posting asks for that the profile does not hold.

    Reuses the same segment split and the same optional/required markers as the
    knockouts, because identifying "PhD preferred" is work already done there --
    it is how the knockout knows NOT to fire. This reads that signal rather than
    scanning the description a second time with slightly different rules, which
    is how two parts of one report start disagreeing about the same sentence.

    A credential is counted only when the posting states a verdict on it.
    "Several of our researchers hold a PhD" is describing the team, not asking
    for one, and must cost nothing -- exactly as it knocks nothing out.

    Never raises: a malformed profile returns an empty list, so the failure mode
    is "no deduction", never "an unexplained deduction".
    """
    held_raw = profile.get("credentials_held")
    if not isinstance(held_raw, (list, tuple)):
        held_raw = DEFAULT_CREDENTIALS_HELD
    held = [str(h).lower().strip() for h in held_raw if str(h or "").strip()]

    found: list[str] = []
    try:
        for segment in _segments(_knockout_text(job)):
            # Either verdict counts. 'preferred' is the case this exists for;
            # 'required' can only get this far when the matching knockout was
            # switched off, and a job kept on purpose should still rank below
            # one with no credential gap at all.
            if not (_OPTIONAL_MARKERS.search(segment)
                    or _REQUIRED_MARKERS.search(segment)):
                continue
            for name, pattern in _SOFT_CREDENTIALS:
                if name in found or not pattern.search(segment):
                    continue
                if any(h in name or name in h for h in held):
                    continue
                found.append(name)
    except (re.error, TypeError, AttributeError) as exc:
        _warn(f"credential scan failed on {job.get('job_key')} ({exc}); "
              f"scoring reach without a credential deduction")
        return []
    return found

# "Bachelor's degree in X", "MSc in X", "Degree in X". The field list is
# captured WHOLE rather than up to the first comma: a posting saying "Physics,
# Computer Science or a related field" must be read as the offer it is, not
# knocked out on its first item.
_DEGREE_FIELD_RE = re.compile(
    r"\b(?:bachelor'?s?|master'?s?|b\.?\s?sc|m\.?\s?sc|b\.?\s?eng|m\.?\s?eng|"
    r"b\.?\s?a\b|degree|diploma)\b[^.\n]{0,40}?\bin\s+([^.\n]{3,140})", re.I)

# Any of these in the captured field list means the employer has already said
# adjacent study counts, so there is nothing to knock out.
_DEGREE_ESCAPE_RE = re.compile(
    r"\b(?:related|relevant|similar|equivalent|comparable|or other|any)\b", re.I)

_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:to|-|–)?\s*\d{0,2}\s*\+?\s*years?\b", re.I)

# Defaults, used when run_config.json says nothing. Documented in the config.
KNOCKOUT_YEARS_MULTIPLE = 2.0
_DEFAULT_DEGREE_FIELDS_HELD = (
    "information technology", "information system", "computer", "computing",
    "software", "economic", "business", "data", "analytic", "statistic",
    "mathematic", "artificial intelligence", "machine learning",
    "engineering", "science", "technology",
)


def _segments(text: str) -> list[str]:
    """Description split into the units a single requirement is written in."""
    return [s.strip() for s in _SEGMENT_SPLIT_RE.split(text or "") if s and s.strip()]


def _demands(segment: str) -> bool:
    """True only when this segment states a hard requirement.

    Optional wording wins over required wording in the same breath, because
    "a PhD or equivalent experience is required" is a requirement the candidate
    already meets. Silence loses: a segment naming a credential with no verdict
    either way keeps the job.
    """
    if _OPTIONAL_MARKERS.search(segment):
        return False
    return bool(_REQUIRED_MARKERS.search(segment))


def _knockout_text(job: dict[str, Any]) -> str:
    return f"{job.get('title') or ''}\n{job.get('jd_text') or ''}"


def _years_demanded(job: dict[str, Any], segments: list[str]) -> int | None:
    """The smallest number of years this posting actually insists on.

    The SMALLEST, not the largest. A posting saying "5+ years in ML, 10+ years
    in a leadership role preferred" demands five; taking the largest figure on
    the page would exclude it on a line the employer marked optional.
    """
    candidates: list[int] = []
    stated = job.get("min_years_exp")
    if isinstance(stated, int) and stated > 0:
        candidates.append(stated)
    for segment in segments:
        if not _demands(segment):
            continue
        for found in _YEARS_RE.findall(segment):
            try:
                years = int(found)
            except ValueError:
                continue
            if 0 < years <= 40:
                candidates.append(years)
    return min(candidates) if candidates else None


def check_knockouts(job: dict[str, Any], config: dict[str, Any]) -> str | None:
    """Return an exclusion reason for a credential the profile cannot obtain.

    Every return value is a short, stable phrase rather than a quotation from
    the posting, because pipeline.exclusion_reasons() groups by this exact
    string -- a per-job snippet would turn the end-of-run summary into a list of
    one-job categories and the user would stop reading it.

    Never raises. A bad pattern or a hand-edited config returns None (keep the
    job) with a warning, since the failure mode of this function must always be
    "shows you too much", never "hid something and did not say so".
    """
    settings = ((config.get("filters") or {}).get("knockouts")) or {}
    if not settings.get("enabled", True):
        return None

    profile = config.get("profile") or {}
    text = _knockout_text(job)
    try:
        segments = _segments(text)
        demanding = [s for s in segments if _demands(s)]

        if settings.get("doctorate", True):
            for segment in demanding:
                if _DOCTORATE_RE.search(segment):
                    return "requires a doctorate"

        if settings.get("professional_licence", True):
            for segment in demanding:
                for name, pattern in _LICENCE_RES:
                    if pattern.search(segment):
                        return f"requires a professional licence ({name})"

        if settings.get("degree_field", True):
            held = [str(f).lower() for f in
                    (settings.get("degree_fields_held")
                     or _DEFAULT_DEGREE_FIELDS_HELD)]
            for segment in demanding:
                match = _DEGREE_FIELD_RE.search(segment)
                if not match:
                    continue
                fields = match.group(1).lower()
                if _DEGREE_ESCAPE_RE.search(fields):
                    continue
                if any(f and f in fields for f in held):
                    continue
                short = re.sub(r"\s+", " ", fields).strip()[:40]
                return f"requires a degree in a field you do not hold ({short})"

        multiple = settings.get("years_experience_multiple", KNOCKOUT_YEARS_MULTIPLE)
        have = profile.get("years_experience")
        if isinstance(multiple, (int, float)) and not isinstance(multiple, bool) \
                and isinstance(have, (int, float)) and have > 0:
            ceiling = float(have) * float(multiple)
            demanded = _years_demanded(job, segments)
            if demanded is not None and demanded > ceiling:
                return (f"demands {demanded}y experience, "
                        f"over the {ceiling:.0f}y ceiling for {have:.0f}y held")
    except (re.error, TypeError, ValueError, AttributeError) as exc:
        _warn(f"knockout check failed on {job.get('job_key')} ({exc}); "
              f"keeping the job rather than excluding it unexplained")
        return None

    return None


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

    # Last, because it is the most expensive check and the only one that reads
    # the whole description.
    return check_knockouts(job, config)


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
    ("reach", score_reach),
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
    # A component reading a threshold must see the config THIS call was given,
    # not whatever is on disk -- otherwise a test or an A/B arm that passes a
    # config would be scored against run_config.json without saying so.
    if ctx.config is None:
        ctx.config = config

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
