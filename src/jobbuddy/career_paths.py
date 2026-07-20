"""What to target next, computed from the postings we already hold.

**Fully deterministic. There is no LLM call anywhere in this module, and there
must never be one.** This is a design constraint, not a preference.

Everything below is arithmetic over data already collected: counts of postings,
medians of stated salaries, coverage of a posting's mandatory requirements by
the verified profile. A model narrating the same numbers would produce fluent,
confident prose about somebody's career with nothing gating it. The tailoring
path has `fact_guard` standing between the model and the resume, so a claim
about the candidate cannot ship unless a verified fact backs it. There is no
equivalent guard for a claim about the *market* -- no fact store to check
"MLOps roles pay 20% more" against -- so the only safe design is to never let a
model make one. The numbers here are the output; the caveats travel with them.

What this can see, and what it therefore cannot say
---------------------------------------------------
This reads **vacancies, not careers.** It knows what employers wrote down that
they wanted, in the runs this repo has recorded. It does not know what anyone
with this background actually did next, whether they got the job, what they
were paid once hired, or whether the transition was any good. Every number is a
statement about job adverts.

The other trap this analysis walks into by default is causation. "Roles
requiring Kubernetes pay more" is a true statement about the corpus and a false
statement about the world: those roles are usually *more senior*, and the
seniority is doing the work, not the Kubernetes. Both statements are carried in
`result["caveats"]` rather than in a docstring or a report footer, so a later
renderer edit cannot quietly drop them.

Honesty rules that are structural rather than stylistic
-------------------------------------------------------
  - **`n` travels with every figure.** A cluster of 4 postings and a cluster of
    40 are different kinds of claim, and the renderer is given what it needs to
    show that (`n`, `thin`).
  - **A median is never emitted from a sample below
    `min_stated_salaries` (default 5).** It reports
    `"not enough stated salaries (n=3)"` instead.
  - **The salary denominator is always reported.** A median over 3 of 40
    postings and one over 35 of 40 are not the same number wearing the same
    label, so `stated_n` and `of_n` sit next to every one.
  - **A trend is never imputed.** Too few runs, or runs spanning too few days,
    reports its own insufficiency and stops.

Skill matching is delegated to `skills_taxonomy` and `scoring`, unchanged.
There is deliberately no fuzzy or head-noun matching here either -- see the note
at the foot of `skills_taxonomy`. A false skill match inflates fit, and inflated
fit is what would later justify a false resume bullet.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from . import job_schema, scoring, skills_taxonomy

# --------------------------------------------------------------------------
# tunables -- overridable from run_config.json under a "career_paths" key
# --------------------------------------------------------------------------

# Below this many stated salaries a median is a rumour. Five is not a
# statistical threshold, it is the point at which one outlier stops being the
# answer; the figure is reported with its n either way so the reader can
# disagree.
MIN_STATED_SALARIES = 5

# At or below this many postings a cluster is marked `thin`. It is still
# reported -- a small cluster can be the interesting one -- but it must never
# render like a thick one.
THIN_CLUSTER_N = 8

# A title n-gram must appear in this many postings before it can name a
# cluster. At 1 every one-off title becomes its own "adjacent role", which is
# noise wearing the costume of a finding.
MIN_CLUSTER_N = 2

MAX_CLUSTERS = 12
MAX_SKILLS_GAP = 12

# A requirement must appear in at least this many of a cluster's postings to
# count as a gap. The brief for this section is "requirements RECURRING across
# the cluster", and recurring means more than one -- at 1 the list fills with
# whatever single adjectives one verbose advert happened to use, which reads as
# a development plan and is actually a transcription of one employer's mood.
MIN_GAP_POSTINGS = 2

# Movement compares the most recent `TREND_WINDOW_RUNS` runs against the
# `TREND_WINDOW_RUNS` immediately before them, so both halves need to exist.
TREND_WINDOW_RUNS = 5

# ...and the whole span needs to be long enough that a change could mean
# something. Twenty runs in an afternoon is twenty looks at one morning's job
# board, not a trend. This is the check that stops the analysis from dressing
# up sampling frequency as market movement.
MIN_TREND_SPAN_DAYS = 14

# Words that describe a level or a posting's packaging rather than the role.
# Stripped before clustering so "Senior Data Scientist" and "Data Scientist"
# land in one cluster -- the whole question being asked is which ROLE to target,
# and the level is answered by seniority elsewhere.
_LEVEL_TOKENS = frozenset({
    "senior", "snr", "sr", "junior", "jnr", "jr", "principal", "staff",
    "mid", "midlevel", "entry", "graduate", "trainee", "intern",
    "associate", "assistant", "deputy", "chief", "executive", "exec",
    "i", "ii", "iii", "iv", "v", "1", "2", "3",
})

_PACKAGING_TOKENS = frozenset({
    "singapore", "sg", "apac", "asia", "sea", "regional", "global",
    "remote", "hybrid", "onsite", "contract", "permanent", "perm",
    "fulltime", "full", "part", "time", "urgent", "hiring", "immediate",
    "new", "up", "to", "and", "or", "of", "the", "a", "an", "for", "with",
    "in", "on", "at", "job", "role", "position", "opening", "vacancy",
    "west", "east", "north", "south", "central",
})

_TITLE_STOPWORDS = _LEVEL_TOKENS | _PACKAGING_TOKENS

# The noun a role name ends on. A cluster label must end in one of these, which
# is what stops "generative ai" or "large language" from becoming a "role":
# those are subject matter, not jobs. Small and explicit on purpose -- this is a
# lexicon, not a model, and every entry is reviewable.
_ROLE_HEADS = frozenset({
    "engineer", "engineering", "scientist", "science", "architect",
    "developer", "analyst", "manager", "lead", "consultant", "researcher",
    "specialist", "administrator", "director", "officer", "designer",
    "strategist", "head", "practitioner", "programmer",
})

# Multi-word subjects CONTRACTED to one token, so that "AI Engineer" and
# "Artificial Intelligence Engineer" cluster together.
#
# Contracting rather than expanding, which is what this did first and got
# wrong. Expanding "ai" to "artificial intelligence" turns one concept into two
# tokens, and the 3-gram window then slices across it: "AI Product Manager"
# became the cluster "intelligence product manager" and "AI Stack Engineer"
# became "intelligence stack engineer". Both are artefacts of the tokeniser,
# not roles anyone advertises. One concept must occupy one token.
#
# The pairs are checked longest-first and the list is deliberately tiny -- it
# covers the abbreviations that genuinely split one role across two spellings
# in this corpus, and nothing else. It is a lexicon, not a model.
_PHRASE_CONTRACTIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("large", "language", "model"), "llm"),
    (("large", "language", "models"), "llm"),
    (("natural", "language", "processing"), "nlp"),
    (("artificial", "intelligence"), "ai"),
    (("machine", "learning"), "ml"),
    (("gen", "ai"), "genai"),
    (("generative", "ai"), "genai"),
    # Without this, "full" is dropped as packaging and the leftover "stack"
    # names a cluster called "stack ai engineer", which is not a job.
    (("full", "stack"), "fullstack"),
    (("back", "end"), "backend"),
    (("front", "end"), "frontend"),
)

# Level words that are only levels when they come FIRST. "Lead" opening a title
# is a rank ("Lead ML Engineer"); "lead" at the end is the role itself
# ("Engineering Lead"), so it cannot simply join the stopword set. Position is
# the whole distinction, and it is exact rather than guessed.
_LEADING_LEVEL_TOKENS = frozenset({"lead", "head", "chief", "director"})

# Single tokens that are already one concept and just need a canonical spelling.
_TOKEN_ALIASES = {
    "a.i": "ai", "artificial": "ai", "llms": "llm", "genai": "genai",
}

# --------------------------------------------------------------------------
# the caveats -- part of the return value, not documentation
# --------------------------------------------------------------------------

CAVEAT_CAUSATION = (
    "Correlation only. Where a cluster pays more than another, that is a fact "
    "about which adverts state which numbers -- it is NOT evidence that "
    "learning that cluster's skills would raise your pay. Higher-paying "
    "clusters are usually more senior, and the seniority is doing the work. "
    "Treat a skills gap as the price of entry to a role, never as a lever on "
    "salary."
)

CAVEAT_COVERAGE = (
    "This sees vacancies, not careers. It knows what employers asked for in "
    "the postings this tool collected. It does not know what anyone with your "
    "background actually did next, who was hired, or what they were paid once "
    "hired. Nothing here is a career outcome; all of it is job-advert text."
)

CAVEAT_THIN = (
    "Cluster sizes differ by an order of magnitude. Every figure carries its "
    f"n, and any cluster at or below {THIN_CLUSTER_N} postings is flagged "
    "thin. A thin cluster's median is one or two employers' opinions."
)

CAVEAT_SALARY_DENOMINATOR = (
    "Salary medians are computed only over postings that STATED a salary, and "
    "the count that did is reported next to every median as stated_n of "
    "of_n. Employers who state a salary are not a random sample of employers."
)

CAVEAT_COVERAGE_METRIC = (
    "Coverage is the share of a posting's required-or-better skill terms that "
    "the profile matches, taken from scoring.score_skill_match's core_score. "
    "Read it as a RELATIVE ranking between clusters, not as an absolute "
    "readiness percentage. The terms come from the job board's own skill "
    "extractor, which emits a long tail per posting, so nobody scores near "
    "100 and a 30% cluster does not mean you meet 30% of the job. Compare the "
    "clusters to each other; do not read the number on its own."
)

CAVEAT_SELECTION = (
    "The corpus is whatever the configured search scopes returned, so the "
    "clusters found are bounded by the queries in run_config.json. A role "
    "nobody searched for cannot appear here, however good a fit it would be."
)


def _caveats() -> dict[str, str]:
    return {
        "causation": CAVEAT_CAUSATION,
        "coverage": CAVEAT_COVERAGE,
        "thin_samples": CAVEAT_THIN,
        "salary_denominator": CAVEAT_SALARY_DENOMINATOR,
        "coverage_metric": CAVEAT_COVERAGE_METRIC,
        "selection": CAVEAT_SELECTION,
    }


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """Tunables, config over defaults. A bad value falls back rather than raises."""
    raw = ((config or {}).get("career_paths") or {})
    out = {
        "min_stated_salaries": MIN_STATED_SALARIES,
        "thin_cluster_n": THIN_CLUSTER_N,
        "min_cluster_n": MIN_CLUSTER_N,
        "max_clusters": MAX_CLUSTERS,
        "max_skills_gap": MAX_SKILLS_GAP,
        "min_gap_postings": MIN_GAP_POSTINGS,
        "trend_window_runs": TREND_WINDOW_RUNS,
        "min_trend_span_days": MIN_TREND_SPAN_DAYS,
    }
    for key, default in list(out.items()):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value >= 0:
            out[key] = int(value)
    return out


def _owned_skills(profile: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Canonical skill -> weight, plus a plain statement of where it came from.

    Two different objects are called "the profile" in this codebase: the tiered
    matching block in run_config.json (`skills`), and the verified fact store in
    profile/master_profile.json (`skill_groups`). Rather than guess, this reads
    whichever is present and SAYS which, because the two produce different
    coverage numbers and a reader comparing runs needs to know why.
    """
    profile = profile or {}
    tiers = profile.get("skills")
    if isinstance(tiers, dict) and any(tiers.values()):
        return (skills_taxonomy.build_owned(tiers, scoring.SKILL_TIER_WEIGHT),
                "run_config profile.skills (proficiency tiers as stated)")

    groups = profile.get("skill_groups")
    if isinstance(groups, (list, tuple)) and groups:
        items: list[str] = []
        for group in groups:
            if isinstance(group, dict):
                items.extend(str(i) for i in (group.get("items") or []))
        if items:
            # The verified profile records what the candidate has, not how well.
            # Everything therefore enters at the middle tier, and this string
            # says so -- an unstated proficiency silently promoted to expert is
            # exactly how coverage gets inflated.
            return (skills_taxonomy.build_owned(
                {"working": items}, scoring.SKILL_TIER_WEIGHT),
                "master_profile skill_groups (no proficiency stated; all "
                "treated as 'working')")

    return {}, "no skills found on the profile"


def _scoring_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """A profile in the shape `scoring.score_skill_match` expects.

    `score_skill_match` is reused rather than reimplemented -- a second copy of
    the matching would eventually disagree with the first, and the two numbers
    would then contradict each other in one report.
    """
    profile = profile or {}
    tiers = profile.get("skills")
    if isinstance(tiers, dict) and any(tiers.values()):
        return profile

    groups = profile.get("skill_groups")
    items: list[str] = []
    for group in groups if isinstance(groups, (list, tuple)) else []:
        if isinstance(group, dict):
            items.extend(str(i) for i in (group.get("items") or []))
    if not items:
        return profile
    derived = dict(profile)
    derived["skills"] = {"working": items}
    return derived


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _midpoint(job: dict[str, Any]) -> float | None:
    """The stated monthly SGD midpoint, or None when nothing was stated.

    `salary_is_stated` is trusted over the presence of the numbers: a board that
    fills a range in from a category average has stated a guess, and averaging
    guesses produces a confident number about nothing.
    """
    if not job.get("salary_is_stated"):
        return None
    low = job.get("salary_min_sgd")
    high = job.get("salary_max_sgd")
    values = [float(v) for v in (low, high)
              if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0]
    return sum(values) / len(values) if values else None


def _current_salary(config: dict[str, Any] | None) -> float | None:
    """The user's current pay, read from run_config -- never hardcoded here."""
    value = ((config or {}).get("profile") or {}).get("current_salary_sgd_monthly")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


# --------------------------------------------------------------------------
# 1. clustering -- deterministic, no ML
# --------------------------------------------------------------------------

def title_tokens(title: Any) -> list[str]:
    """The significant tokens of a title, in order.

    `job_schema.norm_title` first (it already strips the bracketed salary
    teasers and the after-dash decoration), then multi-word subjects contract to
    one token, then level and packaging words go.

    Order is preserved because clusters are named from CONTIGUOUS n-grams. That
    is an exact-substring test, not a fuzzy one: "ml engineer" matches a title
    containing those two tokens in that order and nothing else. A bag-of-words
    test would have merged "engineering manager" with "manager, engineering
    support".
    """
    raw = [t.strip(".") for t in job_schema.norm_title(title).split()]
    raw = [t for t in raw if t]

    # Contract first, longest phrase first, so "artificial intelligence" is one
    # token before anything counts tokens.
    contracted: list[str] = []
    index = 0
    while index < len(raw):
        for phrase, replacement in _PHRASE_CONTRACTIONS:
            size = len(phrase)
            if tuple(raw[index:index + size]) == phrase:
                contracted.append(replacement)
                index += size
                break
        else:
            contracted.append(_TOKEN_ALIASES.get(raw[index], raw[index]))
            index += 1

    significant = [t for t in contracted if t not in _TITLE_STOPWORDS]

    # Strip a leading rank, but never the last token -- "Lead" alone, or "Head"
    # alone, is all the role the title gives us and deleting it leaves nothing.
    while len(significant) > 1 and significant[0] in _LEADING_LEVEL_TOKENS:
        significant = significant[1:]
    return significant


def _ngrams(tokens: Sequence[str]) -> list[tuple[str, ...]]:
    """Contiguous 1-3 grams that END on a role head, longest first.

    Ending on a role head is what keeps a cluster a job rather than a topic:
    "generative ai" is a subject, "generative ai engineer" is a role.
    """
    found: list[tuple[str, ...]] = []
    for size in (3, 2, 1):
        for start in range(len(tokens) - size + 1):
            gram = tuple(tokens[start:start + size])
            if gram[-1] in _ROLE_HEADS:
                found.append(gram)
    return found


def cluster_titles(jobs: Sequence[dict[str, Any]],
                   min_cluster_n: int = MIN_CLUSTER_N,
                   max_clusters: int = MAX_CLUSTERS) -> list[dict[str, Any]]:
    """Group postings into role clusters. Same input always gives same output.

    Three deterministic passes:

      1. every posting yields its role-head n-grams;
      2. an n-gram seen in at least `min_cluster_n` postings becomes a seed;
      3. each posting joins the FIRST seed it contains, seeds ordered by length
         descending, then by how many postings hold them, then alphabetically.

    Length-first is what makes it stable and legible: "machine learning
    engineer" claims a posting before the bare "engineer" can, so the generic
    seed collects only what nothing more specific described. Every tie is
    broken by a total order, so there is no dependence on dict iteration or
    input order for the RESULT -- only the membership lists preserve input
    order, which is intentional.

    A posting whose title names no role head at all is not forced anywhere. It
    lands in `unclustered`, reported as a count, because inventing a home for it
    would put postings into a cluster that does not describe them.
    """
    tokenised = [(index, title_tokens(job.get("title") or job.get("title_norm")))
                 for index, job in enumerate(jobs or [])]

    counts: dict[tuple[str, ...], int] = {}
    for _, tokens in tokenised:
        for gram in set(_ngrams(tokens)):
            counts[gram] = counts.get(gram, 0) + 1

    seeds = sorted(
        (gram for gram, count in counts.items() if count >= max(1, min_cluster_n)),
        key=lambda gram: (-len(gram), -counts[gram], gram),
    )

    members: dict[tuple[str, ...], list[int]] = {seed: [] for seed in seeds}
    unclustered: list[int] = []
    for index, tokens in tokenised:
        grams = set(_ngrams(tokens))
        for seed in seeds:
            if seed in grams:
                members[seed].append(index)
                break
        else:
            unclustered.append(index)

    clusters = [
        {"label": " ".join(seed), "tokens": list(seed), "indices": indices}
        for seed, indices in members.items() if indices
    ]
    # Biggest first, then the same total order as the seeds, so two runs over
    # the same corpus emit the clusters in the same sequence.
    clusters.sort(key=lambda c: (-len(c["indices"]), tuple(c["tokens"])))
    if max_clusters > 0:
        dropped = clusters[max_clusters:]
        clusters = clusters[:max_clusters]
        for cluster in dropped:
            unclustered.extend(cluster["indices"])
    return clusters + ([{"label": "unclustered", "tokens": [],
                         "indices": sorted(unclustered)}] if unclustered else [])


# --------------------------------------------------------------------------
# 2. per-cluster measurement
# --------------------------------------------------------------------------

def _coverage(cluster_jobs: Sequence[dict[str, Any]],
              scoring_profile: dict[str, Any]) -> dict[str, Any]:
    """Median coverage of the cluster's MANDATORY requirements by the profile.

    Reuses `scoring.score_skill_match` and reads its `core_score` -- the same
    number `scoring.score_reach` reads, so a cluster's coverage and a job's
    reach cannot disagree. A posting that states no mandatory requirements is
    excluded from the median rather than counted as zero; `measured_n` reports
    how many actually contributed.
    """
    values: list[float] = []
    for job in cluster_jobs:
        _, detail = scoring.score_skill_match(job, scoring_profile)
        core = detail.get("core_score")
        if isinstance(core, (int, float)) and not isinstance(core, bool):
            values.append(float(core))

    median = _median(sorted(values))
    return {
        "median_pct": None if median is None else round(median, 1),
        "measured_n": len(values),
        "of_n": len(cluster_jobs),
        "reason": None if values else
                  "no posting in this cluster states a mandatory requirement",
    }


def _pay(cluster_jobs: Sequence[dict[str, Any]], current: float | None,
         min_stated: int) -> dict[str, Any]:
    """Median stated pay against the user's current, with its denominator.

    The denominator is the point. A median over 3 of 40 postings and one over
    35 of 40 read identically once the fraction is dropped, and only one of them
    is worth anything -- so `stated_n`, `of_n` and `stated_fraction` sit beside
    every figure, and below `min_stated` there is no figure at all.
    """
    stated = sorted(v for v in (_midpoint(j) for j in cluster_jobs) if v is not None)
    of_n = len(cluster_jobs)
    fraction = round(len(stated) / of_n, 3) if of_n else 0.0

    base = {
        "stated_n": len(stated),
        "of_n": of_n,
        "stated_fraction": fraction,
        "median_sgd": None,
        "delta_vs_current_sgd": None,
        "delta_pct": None,
        "reason": None,
    }

    if len(stated) < max(1, min_stated):
        base["reason"] = f"not enough stated salaries (n={len(stated)})"
        return base

    median = _median(stated)
    base["median_sgd"] = int(round(median))
    base["range_sgd"] = [int(round(stated[0])), int(round(stated[-1]))]
    if current:
        base["delta_vs_current_sgd"] = int(round(median - current))
        base["delta_pct"] = round(100.0 * (median - current) / current, 1)
    else:
        base["reason"] = ("no current salary in run_config "
                          "(profile.current_salary_sgd_monthly); "
                          "median reported without a delta")
    return base


def _skills_gap(cluster_jobs: Sequence[dict[str, Any]],
                owned: dict[str, float], limit: int,
                min_postings: int = MIN_GAP_POSTINGS) -> list[dict[str, Any]]:
    """Requirements recurring across the cluster that the profile does not cover.

    Counted by POSTINGS naming the skill, not by mentions, so one verbose advert
    cannot invent a trend. Noise is dropped by `skills_taxonomy.clean_job_skills`
    before counting, which is what keeps "communication", "teamwork" and
    "scientific discipline" out of a development plan.

    Coverage is decided by `skills_taxonomy.match` alone -- exact, containment
    and reverse-containment. No fuzzy pass is added here. A skill wrongly called
    covered is silently dropped from the plan; a skill wrongly called missing
    sends the user to learn something they already know. Both are bad, and the
    taxonomy's rules are the reviewed ones.
    """
    counts: dict[str, int] = {}
    surface: dict[str, str] = {}
    for job in cluster_jobs:
        seen: set[str] = set()
        for term in skills_taxonomy.clean_job_skills(job.get("skills_raw") or []):
            key = skills_taxonomy.canon(term)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            surface.setdefault(key, term)

    of_n = len(cluster_jobs)
    gaps = []
    for key, count in counts.items():
        if count < max(1, min_postings):
            continue
        weight, _, _ = skills_taxonomy.match(key, owned)
        if weight > 0:
            continue
        gaps.append({
            "skill": surface.get(key, key),
            "canonical": key,
            "postings": count,
            "of_n": of_n,
            "share": round(count / of_n, 3) if of_n else 0.0,
        })
    gaps.sort(key=lambda g: (-g["postings"], g["canonical"]))
    return gaps[:limit] if limit > 0 else gaps


# --------------------------------------------------------------------------
# 3. movement over time
# --------------------------------------------------------------------------

def _sightings_of(history: Any) -> list[dict[str, Any]]:
    """The raw sightings behind a JobHistory, or a list passed directly.

    `JobHistory` exposes run counts and per-job lookups but not the whole log,
    and adding a public accessor is a change to `job_store`, which this module
    does not own. So the private attribute is read defensively and a plain list
    of sighting dicts is accepted too -- which is also what the tests use, so
    the trend logic is exercised without a JobHistory at all.
    """
    if isinstance(history, (list, tuple)):
        return [s for s in history if isinstance(s, dict)]
    raw = getattr(history, "_sightings", None)
    return [s for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []


def _run_order(sightings: Sequence[dict[str, Any]]) -> list[str]:
    """Run ids oldest first, ordered by their first timestamp then by id."""
    first: dict[str, str] = {}
    for sighting in sightings:
        run_id = str(sighting.get("run_id") or "")
        ts = str(sighting.get("ts") or "")
        if not run_id:
            continue
        if run_id not in first or ts < first[run_id]:
            first[run_id] = ts
    return sorted(first, key=lambda r: (first[r], r))


def _span_days(sightings: Sequence[dict[str, Any]]) -> float | None:
    stamps = sorted(str(s.get("ts") or "") for s in sightings if s.get("ts"))
    if len(stamps) < 2:
        return None
    try:
        start = datetime.fromisoformat(stamps[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(stamps[-1].replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return round((end - start).total_seconds() / 86400.0, 2)


def _movement(history: Any, clusters: Sequence[dict[str, Any]],
              settings: dict[str, Any]) -> dict[str, Any]:
    """Which clusters appear in more postings now than a window of runs ago.

    Counts DISTINCT job_keys per half, never sightings: twenty-three looks at
    the same advert is one vacancy, and counting rows would make polling
    frequency look like hiring.

    Refuses to answer twice over. Too few runs for two windows, or a span too
    short for a change to mean anything, both return an `insufficient` status
    with the actual figures -- never an imputed trend. Twenty runs in two days
    is twenty looks at one morning's job board.
    """
    window = max(1, int(settings["trend_window_runs"]))
    min_span = max(0, int(settings["min_trend_span_days"]))
    sightings = _sightings_of(history)

    if not sightings:
        return {"status": "insufficient history (0 runs)", "runs": 0,
                "runs_needed": window * 2, "span_days": None}

    runs = _run_order(sightings)
    span = _span_days(sightings)

    if len(runs) < window * 2:
        return {"status": f"insufficient history ({len(runs)} runs)",
                "runs": len(runs), "runs_needed": window * 2, "span_days": span}

    recent = set(runs[-window:])
    prior = set(runs[-window * 2:-window])

    if span is not None and span < min_span:
        return {
            "status": f"insufficient history ({len(runs)} runs spanning "
                      f"{span:g} days)",
            "runs": len(runs),
            "runs_needed": window * 2,
            "span_days": span,
            "span_days_needed": min_span,
            "note": ("there are enough runs, but they are packed into too "
                     "short a period. Repeated runs over a couple of days "
                     "re-observe one snapshot of the market; a change between "
                     "them measures this tool's polling, not hiring."),
        }

    # Assign each sighting's title to a cluster using the SAME rules as the run
    # itself, so a movement figure and a cluster figure describe one thing.
    seeds = [(tuple(c["tokens"]), c["label"]) for c in clusters if c["tokens"]]
    recent_keys: dict[str, set[str]] = {label: set() for _, label in seeds}
    prior_keys: dict[str, set[str]] = {label: set() for _, label in seeds}

    for sighting in sightings:
        run_id = str(sighting.get("run_id") or "")
        bucket = recent_keys if run_id in recent else (
            prior_keys if run_id in prior else None)
        if bucket is None:
            continue
        grams = set(_ngrams(title_tokens(sighting.get("title_norm"))))
        for tokens, label in seeds:
            if tokens in grams:
                bucket[label].add(str(sighting.get("job_key") or ""))
                break

    moves = []
    for _, label in seeds:
        now, before = len(recent_keys[label]), len(prior_keys[label])
        moves.append({
            "label": label,
            "postings_recent": now,
            "postings_prior": before,
            "change": now - before,
            "thin": min(now, before) <= settings["thin_cluster_n"],
        })
    moves.sort(key=lambda m: (-m["change"], m["label"]))

    return {
        "status": "measured",
        "runs": len(runs),
        "window_runs": window,
        "span_days": span,
        "basis": (f"distinct job_keys in the last {window} runs vs the "
                  f"{window} runs before them"),
        "clusters": moves,
        "skills": {
            "status": "unavailable",
            "reason": ("the sightings log records title, salary and seniority "
                       "but not skills, so no skill can be compared against "
                       "its own past. Reading today's skills onto past "
                       "sightings would measure only which jobs are still "
                       "open, which is survivorship, not demand."),
        },
    }


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def analyse(jobs: Sequence[dict[str, Any]] | None,
            profile: dict[str, Any] | None,
            history: Any = None,
            config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Adjacent roles, pay deltas, skills gaps and movement -- all arithmetic.

    `jobs` is a run's ranked list. `profile` is either the run_config matching
    block or the verified master profile; whichever it is, the answer says so.
    `history` is a `job_store.JobHistory` (or a raw sightings list). `config` is
    run_config -- the current salary is read from it and never hardcoded.

    An empty `jobs` returns an empty, well-formed result. The caveats are
    present in every return value, including that one, because a report with no
    clusters is exactly where a reader is most tempted to fill the gap.
    """
    jobs = [j for j in (jobs or []) if isinstance(j, dict)]
    settings = _settings(config)
    owned, basis = _owned_skills(profile or {})
    scoring_profile = _scoring_profile(profile or {})
    current = _current_salary(config)

    raw_clusters = cluster_titles(jobs, settings["min_cluster_n"],
                                  settings["max_clusters"])

    reported: list[dict[str, Any]] = []
    unclustered_n = 0
    for cluster in raw_clusters:
        indices = cluster["indices"]
        if cluster["label"] == "unclustered":
            unclustered_n = len(indices)
            continue
        members = [jobs[i] for i in indices]
        coverage = _coverage(members, scoring_profile)
        n = len(members)

        # Rank = coverage x thickness, read as "postings in this cluster whose
        # stated requirements you already meet". Deliberately not a normalised
        # score: a 90%-coverage cluster with 3 postings is a worse target than a
        # 60%-coverage cluster with 30, and any formula that hid that would be
        # ranking on a number the market cannot supply.
        median_cov = coverage["median_pct"]
        rank = None if median_cov is None else round(median_cov / 100.0 * n, 2)

        reported.append({
            "label": cluster["label"],
            "n": n,
            "thin": n <= settings["thin_cluster_n"],
            "sample_titles": sorted({str(j.get("title") or j.get("title_norm") or "")
                                     for j in members})[:6],
            "seniority_mix": _mix(members, "seniority"),
            "coverage": coverage,
            "rank_score": rank,
            "rank_basis": "median mandatory-requirement coverage x postings",
            "pay": _pay(members, current, settings["min_stated_salaries"]),
            "skills_gap": _skills_gap(members, owned, settings["max_skills_gap"],
                                      settings["min_gap_postings"]),
        })

    # None ranks last -- an unmeasurable cluster must never lead the list.
    reported.sort(key=lambda c: (c["rank_score"] is None,
                                 -(c["rank_score"] or 0.0), c["label"]))

    return {
        "n_postings": len(jobs),
        "n_clusters": len(reported),
        "unclustered_postings": unclustered_n,
        "profile_basis": basis,
        "current_salary_sgd_monthly": None if current is None else int(current),
        "clusters": reported,
        "movement": _movement(history, raw_clusters, settings),
        "caveats": _caveats(),
        "settings": settings,
        "deterministic": True,
        "method": ("title n-gram clustering, median stated salary, "
                   "skills_taxonomy coverage -- no model involved"),
    }


def _mix(jobs: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    """Counts of one field's values, so a cluster's level is visible not assumed.

    Included because the causation caveat is otherwise unfalsifiable by the
    reader: if a better-paying cluster is 80% 'lead' and the current one is
    'senior', the seniority explanation is right there rather than asserted.
    """
    counts: dict[str, int] = {}
    for job in jobs:
        value = job.get(field)
        key = str(value) if value else "unstated"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
