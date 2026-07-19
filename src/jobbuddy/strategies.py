"""Named tailoring tactics, so they can be measured instead of argued about.

Resume advice is mostly folklore repeated confidently. The only way to know
whether a tactic helps is to run it as one arm of a comparison against a
grader, with a noise floor underneath -- which is what `ab_harness` is for.
This module is the other half: it turns each tactic into something swappable,
so an experiment changes exactly one thing.

Each strategy is a small transform over the selection prompt, the selected
bullets, or both. They carry the evidence grade that justified building them,
because a tactic with no evidence behind it should have to say so at the point
where someone is deciding whether to switch it on:

  A  measured -- a study, an experiment, or a platform publishing its own data
  B  practitioner consensus -- multiple independent people who actually screen
  C  folklore -- widely repeated, no traceable source

Nothing graded C is enabled by default. Two are implemented anyway, because
"widely believed and never tested" is the most interesting thing to test, and
the harness can now answer it.

**What this module deliberately does not contain.** Keyword-density
maximisation, ATS-score threshold optimisation, invisible text, and any
detector-evasion tactic. The first two optimise against a threat that turns
out to be folklore -- the "75% of resumes are auto-rejected" figure traces to
a vendor that shut down in 2013 having never published a study, and Greenhouse's
own documentation confirms a resume that fails to parse still creates a
candidate record. The last two are deceptive toward an employer, and this tool
does not do that even where it would work. It does not work either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

# Flagged as inflated writing, NOT as an AI tell. An earlier reading of the
# research claimed this list doubled as an LLM-authorship signal; that was
# withdrawn -- "spearheaded" is folklore as an AI marker. Do not re-justify
# this list on authorship grounds.
INFLATED_VERBS = frozenset({
    "spearheaded", "orchestrated", "leveraged", "crafted", "pioneered",
    "enhanced", "revolutionized", "revolutionised", "transformed",
    "utilized", "utilised",
})

# Only these four have evidence behind them. The lists circulating in blog
# posts run to dozens of words and are folklore; a longer list here would
# strip ordinary English and make bullets worse for no measured gain.
LLM_TELLS = frozenset({"delve", "underscore", "meticulous", "crucial"})


@dataclass
class Strategy:
    """One tactic, with the evidence that justifies it and what it changes."""

    name: str
    grade: str                      # "A", "B" or "C"
    rationale: str
    prompt_suffix: str = ""
    post: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None
    default_on: bool = False
    conflicts_with: frozenset[str] = field(default_factory=frozenset)


def _cap_metric_density(bullets: list[dict[str, Any]],
                        cap: float = 0.4) -> list[dict[str, Any]]:
    """Demote surplus quantified bullets rather than deleting them.

    Screeners report that near-total quantification reads as padding at
    mid-level. The advice industry pushes the other way, toward quantifying
    everything, which is why this is a cap and not a target.

    Demotion rather than deletion because the bullet is true and may still be
    the best thing available once the renderer starts cutting for space.
    """
    if not bullets:
        return bullets
    has_number = [bool(re.search(r"\d", b.get("text", ""))) for b in bullets]
    allowed = max(1, int(len(bullets) * cap))
    if sum(has_number) <= allowed:
        return bullets

    kept: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    seen = 0
    for bullet, numeric in zip(bullets, has_number):
        if numeric:
            seen += 1
            if seen > allowed:
                demoted.append(bullet)
                continue
        kept.append(bullet)
    return kept + demoted


def _strip_inflated_language(bullets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag inflated verbs and LLM tells. Flags rather than rewrites.

    Rewriting here would put generated words back into a bullet that has
    already passed `fact_guard`, downstream of the gate -- which is exactly the
    hole the gate exists to close. So this annotates, and selection is asked to
    avoid the vocabulary in the first place via the prompt suffix.
    """
    out = []
    for bullet in bullets:
        words = {w.lower().strip(".,;:") for w in (bullet.get("text") or "").split()}
        found = sorted((words & INFLATED_VERBS) | (words & LLM_TELLS))
        out.append({**bullet, "language_flags": found} if found else bullet)
    return out


REGISTRY: dict[str, Strategy] = {
    "baseline": Strategy(
        name="baseline",
        grade="A",
        rationale="No tactic applied. The control arm every comparison needs.",
        default_on=True,
    ),

    "jd_relevance_order": Strategy(
        name="jd_relevance_order",
        grade="B",
        rationale=(
            "Rank bullets by how directly they answer the JD. Lossless -- it "
            "reorders existing verified content and invents nothing. Matters "
            "more than it sounds because screeners report reading the top of "
            "page one and deciding only whether to keep reading."),
        prompt_suffix=(
            "\n\nOrder strictly by how directly each fact answers THIS job's "
            "stated requirements. The first bullet is what a screener reads "
            "before deciding whether to continue."),
        default_on=True,
    ),

    "hard_to_fake_signals": Strategy(
        name="hard_to_fake_signals",
        grade="A",
        rationale=(
            "Prefer facts carrying signals that are costly to fabricate: a "
            "shipped system, named production impact, a public artifact. Cui "
            "et al. (5.5M cover letters) measured tailoring's correlation with "
            "callbacks falling 51% as automation spread, while verifiable "
            "history signals rose. Employers did not detect AI writing -- they "
            "repriced it. As everyone automates prose, prose stops carrying "
            "information and only the hard-to-fake residue does."),
        prompt_suffix=(
            "\n\nPrefer facts describing something SHIPPED and still running, "
            "with named systems and real production impact, over facts "
            "describing responsibilities or duties. A reader can check the "
            "former and cannot check the latter."),
        default_on=True,
    ),

    "metric_density_cap": Strategy(
        name="metric_density_cap",
        grade="B",
        rationale=(
            "Cap quantified bullets at ~40% rather than maximising them. "
            "Screeners report that at mid-level, everything-quantified reads "
            "as padding. Deliberately opposes the advice industry's push "
            "toward total quantification."),
        post=_cap_metric_density,
    ),

    "plain_language": Strategy(
        name="plain_language",
        grade="B",
        rationale=(
            "Avoid inflated verbs and the four LLM-tell words with evidence "
            "behind them. Cheap insurance against a weakly-evidenced risk."),
        prompt_suffix=(
            "\n\nAvoid these words: spearheaded, orchestrated, leveraged, "
            "crafted, pioneered, enhanced, revolutionised, transformed, "
            "utilised, delve, underscore, meticulous, crucial. Prefer the "
            "plainest verb that is accurate."),
        post=_strip_inflated_language,
        default_on=True,
    ),

    "xyz_formula": Strategy(
        name="xyz_formula",
        grade="C",
        rationale=(
            "'Accomplished X as measured by Y by doing Z'. Bock titled it 'My "
            "Personal Formula', cited no study, and supported it with "
            "hand-picked examples -- so the claim in circulation is untested. "
            "Off by default and worth running as an arm precisely because it "
            "is universally repeated and never measured."),
        prompt_suffix=(
            "\n\nWhere a fact supports it, shape the bullet as: accomplished X, "
            "as measured by Y, by doing Z. Never invent Y to fit the shape -- "
            "if the fact carries no measurement, leave it unmeasured."),
        conflicts_with=frozenset({"technical_decision"}),
    ),

    "technical_decision": Strategy(
        name="technical_decision",
        grade="C",
        rationale=(
            "For ML roles, surface the technical decision and its tradeoff "
            "rather than the headline metric -- 'less what, more how'. "
            "Attributed to a practitioner who hires for ML roles, but sourced "
            "only from search snippets, never a primary read. Directly "
            "contradicts xyz_formula's metrics-first shape, which is what "
            "makes the pair worth running against each other."),
        prompt_suffix=(
            "\n\nWhere a fact supports it, surface the technical decision and "
            "what it traded off, not only the outcome. What was hard, and what "
            "was chosen. Never invent a rationale the fact does not contain."),
        conflicts_with=frozenset({"xyz_formula"}),
    ),
}


def defaults() -> list[str]:
    return [name for name, s in REGISTRY.items() if s.default_on]


def resolve(names: list[str] | None = None) -> tuple[list[Strategy], list[str]]:
    """Turn names into strategies. Returns (strategies, problems).

    Unknown names and conflicting pairs are reported rather than raised -- an
    experiment misconfigured at 2am should say so and run the control, not die.
    """
    chosen = defaults() if names is None else list(names)
    problems: list[str] = []

    resolved: list[Strategy] = []
    for name in chosen:
        strategy = REGISTRY.get(name)
        if strategy is None:
            problems.append(f"unknown strategy {name!r}")
            continue
        resolved.append(strategy)

    active = {s.name for s in resolved}
    for strategy in resolved:
        clash = strategy.conflicts_with & active
        if clash:
            problems.append(
                f"{strategy.name!r} conflicts with {', '.join(sorted(clash))} -- "
                "running both makes the comparison uninterpretable")

    return resolved, problems


def apply_prompt(base_prompt: str, strategies: list[Strategy]) -> str:
    """Append each strategy's instruction, in a stable order.

    Sorted by name so two runs with the same set produce byte-identical
    prompts. Unstable ordering would halve the provider's prompt-cache hit
    rate and, worse, make an A/B comparison compare two different prompts.
    """
    suffixes = [s.prompt_suffix for s in sorted(strategies, key=lambda s: s.name)
                if s.prompt_suffix]
    return base_prompt + "".join(suffixes)


def apply_post(bullets: list[dict[str, Any]],
               strategies: list[Strategy]) -> list[dict[str, Any]]:
    """Run post-selection transforms, in the same stable order."""
    for strategy in sorted(strategies, key=lambda s: s.name):
        if strategy.post is not None:
            bullets = strategy.post(bullets)
    return bullets


def describe(names: list[str] | None = None) -> list[dict[str, Any]]:
    """What is switched on and what evidence justifies it, for the run report."""
    strategies, problems = resolve(names)
    return [{"name": s.name, "grade": s.grade, "rationale": s.rationale,
             "problems": problems} for s in strategies]
