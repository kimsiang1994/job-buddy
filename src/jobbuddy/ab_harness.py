"""Decides whether two resume variants actually differ, or only look like they do.

This module exists because of one measurement. Someone ran a SINGLE unchanged
resume through HackerRank's open-source LLM resume screener 100 times at
temperature 0.1 -- same resume, same job, same prompt, near-zero sampling
temperature -- and the total scores spanned 66 to 99. At a typical 85-point
cutoff, that one identical resume is rejected 65% of the time. Nothing about
the candidate changed across those 100 runs; the only variable was the grader
talking to itself.

The consequence is the whole reason this file exists:

  **A single-run score difference between two resume variants is mostly noise.**

"Variant A scored 82, variant B scored 79, therefore A is better" is confident
nonsense -- 3 points sits deep inside a 33-point null distribution. That
comparison is easy to make by accident, it produces a number that looks like a
result, and acting on it means rewriting a resume in a direction chosen by a
random seed. This module's entire job is to make that mistake impossible to
commit accidentally:

  noise floor       measured FIRST, on the same input twice over, so the range
                    attributable to nothing at all is known before anything is
                    compared against it. `compare_variants` takes the floor as a
                    REQUIRED argument -- an optional guard is omitted exactly
                    when it matters.

  forced choice     `compare_pair` shows one grader BOTH resumes and asks which
                    is stronger, instead of scoring each alone. Far lower
                    variance, because the grader is not calibrating against an
                    imagined 0-100 scale it re-invents on every call.

  swap the order    LLM graders favour whichever option they read first. Every
                    pair is run (a,b) AND (b,a), and a preference counts only if
                    it survives the swap. Orderings that disagree are
                    `no_preference` -- not a coin flip, not the average.

  honest statistics bootstrap CI on every win rate, `significant` only when the
                    interval excludes 0.5 AND the effect clears the noise floor,
                    and `underpowered` when the trial count could not have
                    detected an effect this size anyway.

**The honest default is "no detectable difference."** Most resume tweaks do not
produce a detectable effect, and reporting that plainly is the feature. There is
deliberately no "best guess winner" field: someone would read it as a result.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Callable, Sequence

from jobbuddy import hr_panel
from jobbuddy import job_schema
from jobbuddy.deepseek import deepseek_client

# What `compare_pair` demands back from the grader. Passed as schema_keys so a
# reply missing one gets a repair attempt before it counts as a failure.
CHOICE_KEYS = ("stronger", "reason")

# The two orderings every pair is run in. Position bias is the failure mode.
ORDERINGS = (("a", "b"), ("b", "a"))
CALLS_PER_COMPARISON = len(ORDERINGS)

BOOTSTRAP_ITERATIONS = 2000
CI_ALPHA = 0.05

# z for a two-sided 95% interval, used only for the power estimate.
_Z95 = 1.96

# A win rate of exactly 0.5 is "no signal". Significance is measured as distance
# from here, in either direction.
NULL_RATE = 0.5

# Outcome weights. A no_preference is genuinely half a win: the grader looked
# twice and could not tell them apart.
_WIN, _DRAW, _LOSS = 1.0, 0.5, 0.0


def _requirement_texts(requirements: Any) -> list[str]:
    """Requirement strings, mandatory first. Reuses `hr_panel`'s own parser.

    Deliberately not reimplemented: the shapes a requirement list arrives in
    (strings, ExtractedSkill, JSON dicts) are already handled there, and two
    parsers drifting apart would mean the comparison prompt and the scoring
    prompt quietly describe different jobs.
    """
    return hr_panel._required_first(requirements)


def _pair_prompt(first_text: str, second_text: str, job: dict[str, Any],
                 requirements: Sequence[str],
                 persona: dict[str, Any]) -> list[dict[str, str]]:
    """Messages for one forced choice between two resumes.

    Neither resume is identified as the original or the edit, and the grader is
    told nothing about what changed. A grader that knows which one is "the new
    version" is being asked to approve a change, not to judge two candidates.
    """
    rubric = "\n".join(f"  - {line}" for line in persona.get("rubric") or [])
    requirement_block = "\n".join(f"  - {r}" for r in requirements) or "  (none extracted)"
    scope = persona.get("expected_scope")

    system = (
        f"{persona.get('instruction', '')}\n\n"
        f"Your lens: {persona.get('lens', '')}\n"
        f"The bar for this role: {persona.get('bar', '')}\n"
        + (f"Scope implied by the posting: {scope}\n" if scope else "")
        + "\nYour rubric:\n" + rubric + "\n\n"
        "You are shown TWO resumes for the same role. Do not score them. Choose "
        "the one you would advance if you could only advance one. If they are "
        "genuinely indistinguishable for this role, say so rather than picking "
        "one to be decisive.\n\n"
        "Reply with a JSON object only, with exactly these keys:\n"
        '  "stronger": "1" | "2" | "tie"\n'
        '  "reason": one short string saying what decided it'
    )

    user = (
        f"ROLE: {persona.get('title')}\n"
        f"COMPANY: {persona.get('company')}\n"
        f"LEVEL: {persona.get('seniority') or 'unstated'}\n\n"
        f"REQUIREMENTS FROM THE JOB DESCRIPTION:\n{requirement_block}\n\n"
        f"JOB DESCRIPTION:\n{job_schema.norm_jd_text(job.get('jd_text'))[:6000]}\n\n"
        f"RESUME 1:\n{first_text}\n\n"
        f"RESUME 2:\n{second_text}"
    )

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _read_choice(value: Any) -> str | None:
    """'1' | '2' | 'tie' from whatever the grader actually wrote, or None."""
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return None
    if text.startswith(("tie", "equal", "neither", "same", "no ")):
        return "tie"
    if "1" in text and "2" not in text:
        return "1"
    if "2" in text and "1" not in text:
        return "2"
    return None


def _one_ordering(first_text: str, second_text: str, labels: tuple[str, str],
                  job: dict[str, Any], requirements: Sequence[str],
                  persona: dict[str, Any],
                  chat: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """One grader call in one ordering. Never raises."""
    run = {"order": list(labels), "choice": None, "ok": False,
           "reason": "", "error": ""}

    try:
        result = chat(_pair_prompt(first_text, second_text, job, requirements, persona),
                      schema_keys=CHOICE_KEYS, profile="extract", tier="fast")
    except Exception as exc:            # a grader outage is not a resume defect
        run["error"] = f"grader call raised: {exc}"
        return run

    if not isinstance(result, dict):
        run["error"] = "grader returned a non-dict result"
        return run
    data = result.get("data")
    if not isinstance(data, dict):
        run["error"] = str(result.get("error") or "grader returned no JSON object")
        return run

    choice = _read_choice(data.get("stronger"))
    if choice is None:
        run["error"] = f"choice {data.get('stronger')!r} is not one of '1', '2', 'tie'"
        return run

    run["ok"] = True
    run["reason"] = job_schema.norm_text(data.get("reason"))
    run["choice"] = "tie" if choice == "tie" else labels[0 if choice == "1" else 1]
    return run


def compare_pair(variant_a: str, variant_b: str, job: dict[str, Any],
                 requirements: Any, persona: dict[str, Any],
                 chat: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """THE PRIMARY COMPARISON METHOD. Forced choice, run both ways. Never raises.

    One grader sees both resumes and picks the stronger for this job. Forced
    choice has far lower variance than absolute scoring, because the grader is
    comparing two concrete documents rather than calibrating each against an
    imagined scale it reconstructs from scratch on every call -- which is what
    produced the 66-99 spread this module exists to defend against.

    POSITION BIAS IS THE FAILURE MODE HERE. LLM graders favour whichever option
    they read first, strongly enough to manufacture a clean-looking result out
    of nothing. So the pair is run twice, (a,b) and (b,a), and a preference is
    counted only when it survives the swap. If the two orderings disagree the
    answer is `"no_preference"` -- not a coin flip, not "the first one", not an
    average. A grader that just picks whatever it saw first therefore reports a
    preference exactly never, which is the correct reading of a grader that has
    told you nothing.

    Returns {"winner": "a"|"b"|"no_preference", "consistent": bool,
             "runs": [...], "reason": str}.
    """
    chat = chat or deepseek_client.json_chat
    texts = _requirement_texts(requirements)
    persona = persona or {}
    by_label = {"a": variant_a or "", "b": variant_b or ""}

    runs = [
        _one_ordering(by_label[first], by_label[second], (first, second),
                      job or {}, texts, persona, chat)
        for first, second in ORDERINGS
    ]

    failed = [r for r in runs if not r["ok"]]
    if failed:
        return {
            "winner": "no_preference",
            "consistent": False,
            "runs": runs,
            "reason": f"grader failed in {len(failed)} of {len(runs)} orderings: "
                      + "; ".join(r["error"] for r in failed),
        }

    choices = [r["choice"] for r in runs]
    if choices[0] == choices[1] and choices[0] in ("a", "b"):
        return {
            "winner": choices[0],
            "consistent": True,
            "runs": runs,
            "reason": f"{choices[0]} preferred in both orderings",
        }

    if all(c == "tie" for c in choices):
        reason = "grader called it a tie in both orderings"
    elif "tie" in choices:
        reason = "one ordering picked a winner and the other called it a tie"
    else:
        reason = (f"orderings disagree ({choices[0]} first-position, "
                  f"{choices[1]} second): consistent with position bias, not a preference")

    return {"winner": "no_preference", "consistent": False, "runs": runs, "reason": reason}


def noise_floor(resume_text: str, job: dict[str, Any], requirements: Any,
                trials: int = 10,
                chat: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """The null distribution: score spread produced by nothing at all. Never raises.

    Runs the full panel `trials` times on the SAME resume and the SAME job. Any
    variation in the result is variation the grader invented, because the input
    did not move. Measure this before comparing anything, and treat every
    smaller difference as what it is.

    Returns {"mean", "sd", "min", "max", "trials", "scores", "spread"}. `scores`
    holds only the runs that produced a usable panel score; `trials` counts the
    runs attempted, so a floor built on fewer scores than trials is visibly
    thin rather than silently narrow.
    """
    attempts = max(0, int(trials or 0))
    scores: list[float] = []
    failures: list[str] = []

    for _ in range(attempts):
        panel = hr_panel.run_panel(resume_text, job, requirements, chat=chat)
        if panel.get("score") is None:
            failures.append(panel.get("reason") or "panel produced no score")
        else:
            scores.append(float(panel["score"]))

    if not scores:
        return {"mean": None, "sd": 0.0, "min": None, "max": None,
                "trials": attempts, "scores": [], "spread": 0.0,
                "failures": failures,
                "reason": "no run produced a usable panel score"}

    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return {
        "mean": round(statistics.fmean(scores), 2),
        "sd": round(sd, 2),
        "min": min(scores),
        "max": max(scores),
        "trials": attempts,
        "scores": scores,
        "spread": round(max(scores) - min(scores), 2),
        "failures": failures,
        "reason": (f"{len(scores)} of {attempts} runs scored; identical input "
                   f"spanned {min(scores)}-{max(scores)}"),
    }


def estimate_cost(variants: Any, trials: int,
                  personas: Any) -> int:
    """Grader calls the plan would make, computed before any of them are made.

    n variants x trials x 2 orderings x personas grows quadratically in the
    number of variants, and a runaway comparison is a real bill. `personas` may
    be a count or the persona list itself.
    """
    names = list(variants.keys()) if isinstance(variants, dict) else list(variants or [])
    pairs = len(names) * (len(names) - 1) // 2
    count = personas if isinstance(personas, int) else len(list(personas or []))
    return pairs * max(0, int(trials or 0)) * CALLS_PER_COMPARISON * max(0, int(count))


def _floor_threshold(floor: dict[str, Any]) -> float:
    """The noise floor's sd, expressed as a win-rate advantage.

    The floor is measured in score points on a 0-100 scale; win rates live on
    0-1. The conversion is deliberately crude -- one sd of score noise maps to
    sd/100 of win rate -- and deliberately errs toward demanding MORE evidence:
    an effect that cannot clear the grader's own self-disagreement is not an
    effect, whatever the interval says.

    Uncapped on purpose. A floor wide enough to put every reachable effect out
    of range (sd above 50) does not mean the threshold is set wrong -- it means
    that grader disagrees with itself too much to compare anything with, and no
    win rate it produces should be called a result.
    """
    try:
        sd = float(floor.get("sd") or 0.0)
    except (TypeError, ValueError):
        sd = 0.0
    return max(0.0, sd / 100.0)


def _bootstrap_ci(outcomes: Sequence[float], seed_key: str,
                  iterations: int = BOOTSTRAP_ITERATIONS) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean outcome. Pure stdlib `random`.

    Seeded from `seed_key` so a reported interval can be reproduced exactly.
    An unreproducible confidence interval is a decoration.
    """
    if not outcomes:
        return (0.0, 0.0)
    rng = random.Random(seed_key)
    n = len(outcomes)
    pool = list(outcomes)
    means = sorted(sum(rng.choice(pool) for _ in range(n)) / n
                   for _ in range(max(1, iterations)))
    lo = means[int((CI_ALPHA / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - CI_ALPHA / 2) * len(means)))]
    return (round(lo, 4), round(hi, 4))


def _trials_needed(effect: float, threshold: float) -> int:
    """Comparisons needed to resolve an effect this size, at 95%.

    Uses the conservative p=0.5 variance. When the observed effect is smaller
    than the noise floor, the floor is used instead -- the question then is not
    "how many runs to confirm this difference" but "how many to have seen one
    at all".
    """
    target = max(abs(effect), threshold, 1e-6)
    return int(math.ceil((_Z95 ** 2) * 0.25 / (target ** 2)))


def compare_variants(variants: dict[str, str], job: dict[str, Any],
                     requirements: Any, floor: dict[str, Any],
                     trials: int = 10,
                     chat: Callable[..., dict[str, Any]] | None = None,
                     personas: Sequence[dict[str, Any]] | None = None,
                     seed: int | str = 0,
                     max_calls: int = 500) -> dict[str, Any]:
    """Repeated paired comparisons across all variants, with honest statistics.

    `floor` is REQUIRED and comes from `noise_floor`. It is not optional and has
    no default: a comparison that has not measured its own null distribution
    cannot say whether it found anything, and an optional guard is skipped
    exactly on the run where it would have mattered.

    A variant's win rate is `significant` only when ALL THREE hold:
      - its bootstrap 95% CI excludes 0.5,
      - the observed effect exceeds the noise floor's sd, and
      - it ran enough comparisons to resolve an effect that size.

    Otherwise the verdict is "no detectable difference", which is the expected
    outcome for most resume edits and is a result, not a failure. There is no
    best-guess winner field; per-variant win rates and their intervals are the
    whole answer.

    Refuses to start -- returning an error dict, never raising -- when the plan
    exceeds `max_calls`.
    """
    names = [str(n) for n in (variants or {})]
    personas = list(personas) if personas is not None else hr_panel.build_personas(
        job or {}, requirements)
    trials = max(0, int(trials or 0))

    planned = estimate_cost(names, trials, personas)
    base = {
        "verdict": "no detectable difference",
        "variants": {},
        "pairs": [],
        "winners": [],
        "planned_calls": planned,
        "max_calls": max_calls,
        "calls": 0,
        "trials": trials,
        "seed": seed,
        "floor": floor,
        "underpowered": False,
        "failures": [],
        "error": "",
    }

    if not isinstance(floor, dict) or floor.get("sd") is None:
        return {**base, "verdict": "not run", "reason": "no noise floor supplied",
                "error": "floor must be the dict returned by noise_floor()"}

    if len(names) < 2:
        return {**base, "verdict": "not run",
                "reason": "fewer than two variants to compare",
                "error": "at least two variants are required"}

    if planned > max_calls:
        return {
            **base,
            "verdict": "not run",
            "reason": (f"plan needs {planned} grader calls, over the {max_calls} "
                       f"limit: {len(names)} variants x {trials} trials x "
                       f"{CALLS_PER_COMPARISON} orderings x {len(personas)} personas"),
            "error": f"plan of {planned} grader calls exceeds max_calls={max_calls}",
        }

    threshold = _floor_threshold(floor)
    outcomes: dict[str, list[float]] = {name: [] for name in names}
    tally = {name: {"wins": 0, "losses": 0, "no_preference": 0} for name in names}
    pairs: list[dict[str, Any]] = []
    failures: list[str] = []
    calls = 0

    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pair = {"a": left, "b": right, "runs": [],
                    "a_wins": 0, "b_wins": 0, "no_preference": 0}
            for _ in range(trials):
                for persona in personas:
                    result = compare_pair(variants[left], variants[right], job,
                                          requirements, persona, chat=chat)
                    calls += len(result["runs"])
                    result["persona"] = str(persona.get("name") or "unknown")
                    pair["runs"].append(result)

                    # A grader failure is recorded and the comparison is counted
                    # as no_preference. The other trials are not lost -- a
                    # dropped run would quietly bias whichever variant happened
                    # to be losing the failed ones.
                    for run in result["runs"]:
                        if not run["ok"]:
                            failures.append(f"{left} vs {right} "
                                            f"({result['persona']}): {run['error']}")

                    if result["winner"] == "a":
                        pair["a_wins"] += 1
                        outcomes[left].append(_WIN)
                        outcomes[right].append(_LOSS)
                        tally[left]["wins"] += 1
                        tally[right]["losses"] += 1
                    elif result["winner"] == "b":
                        pair["b_wins"] += 1
                        outcomes[right].append(_WIN)
                        outcomes[left].append(_LOSS)
                        tally[right]["wins"] += 1
                        tally[left]["losses"] += 1
                    else:
                        pair["no_preference"] += 1
                        outcomes[left].append(_DRAW)
                        outcomes[right].append(_DRAW)
                        tally[left]["no_preference"] += 1
                        tally[right]["no_preference"] += 1
            pairs.append(pair)

    rows: dict[str, Any] = {}
    winners: list[str] = []
    for name in names:
        series = outcomes[name]
        rate = statistics.fmean(series) if series else NULL_RATE
        low, high = _bootstrap_ci(series, f"{seed}:{name}")
        effect = abs(rate - NULL_RATE)
        needed = _trials_needed(effect, threshold)
        excludes_null = low > NULL_RATE or high < NULL_RATE
        clears_floor = effect > threshold
        powered = len(series) >= needed

        # Power is a precondition for significance, not a footnote on a negative
        # result. Without this line a coin-flipping grader is called significant
        # on about 6% of seeds at 10 trials: a handful of lucky comparisons
        # produce a tight bootstrap interval that excludes 0.5, and the harness
        # reports the exact manufactured result it exists to prevent. A sample
        # too small to resolve the effect it just measured has not measured it.
        significant = bool(series) and excludes_null and clears_floor and powered
        if significant and rate > NULL_RATE:
            winners.append(name)
        rows[name] = {
            "win_rate": round(rate, 4),
            "ci_low": low,
            "ci_high": high,
            "comparisons": len(series),
            "effect": round(effect, 4),
            "floor_threshold": round(threshold, 4),
            "excludes_null": excludes_null,
            "clears_floor": clears_floor,
            "significant": significant,
            "trials_needed": needed,
            "powered": powered,
            "underpowered": not powered,
            **tally[name],
        }

    verdict = "significant" if winners else "no detectable difference"
    underpowered = verdict != "significant" and any(r["underpowered"] for r in rows.values())

    if verdict == "significant":
        reason = (f"{', '.join(winners)} wins above chance with a 95% CI excluding "
                  f"{NULL_RATE} and an effect above the noise floor "
                  f"(sd {floor.get('sd')} -> {threshold:.2f} win-rate)")
    elif underpowered:
        most = max(rows.values(), key=lambda r: r["trials_needed"])
        reason = (f"no detectable difference, and underpowered: "
                  f"{most['comparisons']} comparisons per variant cannot resolve an "
                  f"effect this size -- about {most['trials_needed']} would be needed")
    else:
        reason = ("no detectable difference: no variant's win rate clears both its "
                  f"bootstrap interval and the noise floor (sd {floor.get('sd')})")

    return {
        **base,
        "verdict": verdict,
        "variants": rows,
        "pairs": pairs,
        "winners": winners,
        "calls": calls,
        "underpowered": underpowered,
        "failures": failures,
        "reason": reason,
    }


def summarise(comparison: dict[str, Any]) -> dict[str, Any]:
    """What happened, for the run report. Never raises."""
    rows = comparison.get("variants") or {}
    return {
        "verdict": comparison.get("verdict"),
        "reason": comparison.get("reason", ""),
        "underpowered": bool(comparison.get("underpowered")),
        "calls": comparison.get("calls", 0),
        "failures": len(comparison.get("failures") or []),
        "rates": {name: [row["win_rate"], row["ci_low"], row["ci_high"]]
                  for name, row in rows.items()},
    }
