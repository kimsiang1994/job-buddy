"""Mutation testing: prove the tests would actually notice the bug coming back.

    py tests/mutate.py                  # run every mutation
    py tests/mutate.py --list           # show them, change nothing
    py tests/mutate.py --only fact_     # run the ones whose id contains this

NOT part of the normal suite, by name and on purpose. It is slow, and it EDITS
SOURCE FILES -- `unittest discover` looks for `test*.py`, so `mutate.py` is
never collected.

Why this exists. Both audits mutation-checked by hand -- break the fix, see if
anything goes red -- and both times it caught a test that proved nothing: it
asserted on a return value that the bug never changed, so it passed against the
broken code just as happily. A test that cannot fail is worse than no test,
because it is counted as coverage.

Doing that by hand is a ritual nobody repeats. This makes it a command.

Each mutation is a file plus an EXACT old string and its replacement, so it is
reviewable in the diff of this file rather than being generated. Every one
reintroduces a defect that really shipped, and names the test that must catch
it. A mutation that SURVIVES -- the suite still passes with the bug back in --
is the finding.

Safety: the original bytes are held in memory and restored in a `finally`, and
again via `atexit` if the process is interrupted. If a restore ever fails the
script says so loudly rather than exiting quietly on a mutated tree.
"""

from __future__ import annotations

import argparse
import atexit
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "jobbuddy"


@dataclass
class Mutation:
    """One defect, put back exactly as it originally shipped."""

    id: str
    path: Path
    old: str
    new: str
    test: str                 # the unittest target that must go red
    bug: str                  # what this broke when it was real
    tests_the_test: str = ""  # what a SURVIVING mutation would tell us


M = Mutation

MUTATIONS: list[Mutation] = [

    # ----------------------------------------------------------------------
    # The lazy-cache inversion. Failed roughly 75% of tailoring jobs.
    # ----------------------------------------------------------------------
    M(
        id="token_budget.flag_before_data",
        path=SRC / "deepseek" / "token_budget.py",
        old='        try:\n'
            '            with open(PROFILES_PATH, "r", encoding="utf-8-sig") as fh:',
        new='        _profiles_cache["loaded"] = True\n'
            '        try:\n'
            '            with open(PROFILES_PATH, "r", encoding="utf-8-sig") as fh:',
        test="tests.test_token_budget",
        bug="`loaded` set before the file read, so a second thread saw the flag "
            "true and got `data` = None. Presented as an intermittent "
            "AttributeError on NoneType that never reproduced on a single job.",
        tests_the_test="the thread test does not actually widen the window, or "
                       "asserts on something the race does not change",
    ),
    M(
        id="model_config.flag_before_data",
        path=SRC / "deepseek" / "model_config.py",
        old="        config = None\n        try:\n",
        new='        config = None\n        _cache["loaded"] = True\n        try:\n',
        test="tests.test_audit_regressions.LazyCacheThreadSafetyTests",
        bug="Same inversion in models.json loading. A racing thread got None, "
            "silently took the hardcoded-fallback model and warned 'config "
            "unavailable' about a file that read perfectly.",
    ),
    M(
        id="token_budget.publish_outside_the_lock",
        path=SRC / "deepseek" / "token_budget.py",
        old="    with _profiles_lock:",
        new="    if True:",
        test="tests.test_invariants.LazyCachesArePublishedSafely",
        bug="Ordering alone narrows the race; the lock is what closes it. "
            "Removing it leaves code that looks correct and still races.",
        tests_the_test="the invariant checks ordering but never checks the lock",
    ),

    # ----------------------------------------------------------------------
    # fact_guard: the anti-fabrication gate failing OPEN
    # ----------------------------------------------------------------------
    M(
        id="fact_guard.date_fails_open",
        path=SRC / "fact_guard.py",
        old="        parsed_end = _year_month(end)\n"
            "        if parsed_end is None:\n"
            "            # Unknown, not \"today\". See the docstring.\n"
            "            return None\n"
            "        end_year, end_month = parsed_end",
        new="        parsed_end = _year_month(end)\n"
            "        if parsed_end is None:\n"
            "            today = date.today()\n"
            "            parsed_end = (today.year, today.month)\n"
            "        end_year, end_month = parsed_end",
        test="tests.test_audit_regressions",
        bug="An unparseable `end` was treated as 'still employed', so an end of "
            "2019 mistyped as '20I9' inflated the supported span by years and a "
            "bullet claiming '8 years of Kubernetes' passed against a two-year "
            "fact. The guard failed open on the one input it exists to catch.",
    ),
    M(
        id="fact_guard.month_range_unchecked",
        path=SRC / "fact_guard.py",
        old="    if not (1900 <= year <= 2200) or not (1 <= month <= 12):\n"
            "        return None",
        new="    if False:\n"
            "        return None",
        test="tests.test_fuzz.FactGuardNeverPassesAnInventedNumber",
        bug="`int('99')` is a fine integer, so an end of '2019-99' parsed as "
            "month ninety-nine and added eight years to the span the guard "
            "measures against. Found by fuzzing, not by example.",
        tests_the_test="the fuzz corpus never generates a numerically-valid but "
                       "semantically-impossible date",
    ),
    M(
        id="fact_guard.numbers_unchecked",
        path=SRC / "fact_guard.py",
        old="        if not (forms & allowed_numbers):",
        new="        if False:",
        test="tests.test_fuzz.FactGuardNeverPassesAnInventedNumber",
        bug="The core anti-fabrication check. Without it any invented figure "
            "reaches the rendered resume.",
        tests_the_test="the fuzz test never generates a bullet with a number "
                       "absent from its fact, or never accepts one at all",
    ),

    # ----------------------------------------------------------------------
    # resume_rules: a clean report from a check that read nothing
    # ----------------------------------------------------------------------
    M(
        id="resume_rules.unreadable_input_passes",
        path=SRC / "resume_rules.py",
        old='    if not isinstance(model, dict):\n        return Report([Violation(',
        new='    if not isinstance(model, dict):\n        return Report()  # was: Report([Violation(\n    if False:\n        return Report([Violation(',
        test="tests.test_resume_rules",
        bug="A model this module could not read returned a CLEAN report. "
            "`pipeline` blocks a render on `report.errors`, so that was an "
            "affirmative 'no personal data found' from a check that inspected "
            "nothing -- and the render proceeded.",
    ),

    # ----------------------------------------------------------------------
    # company_registry: the silent empty read that overwrote the real file
    # ----------------------------------------------------------------------
    M(
        id="company_registry.silent_empty_read",
        path=SRC / "company_registry.py",
        old='        net._warn(f"registry: could not read {path.name} ({exc}); "\n'
            '                  f"starting from an EMPTY registry -- the next write will "\n'
            '                  f"replace the file, so move it aside if you want it back")\n'
            "        return {}",
        new="        return {}",
        test="tests.test_silent_failures",
        bug="One bad parse returned {} without a word, and `observe()` then "
            "called `save()` on it -- silently replacing weeks of accumulated "
            "company-to-board discovery with an empty file.",
    ),

    # ----------------------------------------------------------------------
    # Structural invariants: prove the reflective rules are not decorative
    # ----------------------------------------------------------------------
    M(
        id="invariants.bom_read_regressed",
        path=SRC / "verify_profile.py",
        old='        return json.loads(path.read_text(encoding="utf-8-sig"))',
        new='        return json.loads(path.read_text(encoding="utf-8"))',
        test="tests.test_invariants.ConfigReadsTolerateABOM",
        bug="A draft profile edited in Notepad comes back with a BOM. Read as "
            "plain utf-8 it fails to parse, lands in the except handler, and "
            "reports as 'no draft' -- so hand-done verification work vanishes.",
        tests_the_test="the utf-8-sig invariant does not actually scan read_text",
    ),
    M(
        id="invariants.bare_except_introduced",
        path=SRC / "deepseek" / "deepseek_common.py",
        old="            except Exception:\n                pass",
        new="            except:\n                pass",
        test="tests.test_invariants.NoBareExcept",
        bug="No bare `except:` exists today. This proves the rule would reject "
            "the first one on the way in, which is the only time it is cheap.",
    ),
    M(
        id="invariants.loader_falls_back_silently",
        path=SRC / "deepseek" / "calibrate_budgets.py",
        old="            except json.JSONDecodeError:\n                damaged += 1",
        new="            except json.JSONDecodeError:\n                continue",
        test="tests.test_invariants.LoadersReportTheirFallback",
        bug="Every number this module produces is a percentile over these rows, "
            "and percentiles do not announce that their sample shrank. A log "
            "half of which fails to parse tunes budgets on the other half and "
            "looks exactly like a clean run.",
    ),

    # ----------------------------------------------------------------------
    # The other silent-success shapes the fuzzing covers
    # ----------------------------------------------------------------------
    M(
        id="job_schema.implausible_salary_reported",
        path=SRC / "job_schema.py",
        old="        return None\n    return int(round(monthly))",
        new="        pass\n    return int(round(monthly))",
        test="tests.test_fuzz.SalaryConversionIsPlausibleOrDisclosed",
        bug="A figure still above the ceiling after being read as annual was "
            "returned anyway, so it maxed the pay score and took the top of the "
            "ranking -- the ceiling's own failure mode, one twelfth as large.",
    ),
    M(
        id="job_schema.non_finite_salary",
        path=SRC / "job_schema.py",
        old="    if not math.isfinite(value) or value <= 0:",
        new="    if value <= 0:",
        test="tests.test_fuzz.ParsersNeverRaise",
        bug="float('1e309') is inf, which survives the conversion and then "
            "raises OverflowError out of a pure normaliser documented as "
            "returning None when a figure cannot be trusted.",
    ),
    M(
        id="html_text.non_string_input",
        path=SRC / "html_text.py",
        old="    if not isinstance(html, str):\n"
            '        html = "" if html is None else str(html)',
        new="    pass",
        test="tests.test_fuzz.ParsersNeverRaise",
        bug="A non-string description raised TypeError inside the handler that "
            "exists to prevent exactly that -- the function documented as never "
            "raising raised, out of its own recovery path.",
    ),
    M(
        id="paths.digest_of_truncation_not_original",
        path=SRC / "paths.py",
        old="    return f\"{prefix}-{_digest(original)}\"",
        new="    return f\"{prefix}-{_digest(prefix)}\"",
        test="tests.test_fuzz.PathsAreAlwaysUsableOnWindows",
        bug="Two long titles sharing a 40-character prefix would hash the SAME "
            "truncated prefix, collide on one directory, and have the second "
            "job overwrite the first job's resume with nothing reported.",
        tests_the_test="the fuzz test does not generate long titles sharing a "
                       "prefix, so truncation collisions are never exercised",
    ),

    # ----------------------------------------------------------------------
    # career_paths -- every one of these is a way to publish a confident
    # statement about someone's career that the postings do not support. There
    # is no fact_guard between this module and the reader, so the tests ARE the
    # guard, and a survivor here means a false claim could ship unnoticed.
    # ----------------------------------------------------------------------
    M(
        id="career_paths.median_from_a_tiny_sample",
        path=SRC / "career_paths.py",
        old="    if len(stated) < max(1, min_stated):\n"
            '        base["reason"] = f"not enough stated salaries (n={len(stated)})"\n'
            "        return base",
        new="    if False:\n"
            '        base["reason"] = f"not enough stated salaries (n={len(stated)})"\n'
            "        return base",
        test="tests.test_career_paths.MediansNeverEscapeATinySample",
        bug="A median over three stated salaries was published looking exactly "
            "like a median over thirty-five, so two employers' asking price "
            "became 'what this role pays' and could talk someone into a move.",
        tests_the_test="the suppression test asserts on a reason string the "
                       "suppression never sets, rather than on median_sgd "
                       "actually being None",
    ),
    M(
        id="career_paths.thin_cluster_looks_thick",
        path=SRC / "career_paths.py",
        old='            "thin": n <= settings["thin_cluster_n"],\n'
            '            "sample_titles"',
        new='            "thin": False,\n'
            '            "sample_titles"',
        test="tests.test_career_paths.ThinClustersAreVisiblyThin",
        bug="A four-posting cluster carried no thin flag, so the renderer drew "
            "it solid beside a thirty-four-posting one and four adverts read "
            "as a market.",
    ),
    M(
        id="career_paths.trend_from_a_single_afternoon",
        path=SRC / "career_paths.py",
        old="    if span is not None and span < min_span:",
        new="    if False:",
        test="tests.test_career_paths.HistoryIsNeverImputed",
        bug="Twenty-six runs spanning 1.17 days passed the run-count check and "
            "produced a 'trend'. It was measuring how often this tool polled "
            "the job board, presented as hiring demand.",
        tests_the_test="the history tests only ever supply too FEW runs, so the "
                       "span guard is never the thing under test",
    ),
    M(
        id="career_paths.gap_names_a_skill_you_already_have",
        path=SRC / "career_paths.py",
        old="        weight, _, _ = skills_taxonomy.match(key, owned)\n"
            "        if weight > 0:\n"
            "            continue",
        new="        weight, _, _ = skills_taxonomy.match(key, owned)\n"
            "        if False:\n"
            "            continue",
        test="tests.test_career_paths.TheSkillsGapNamesOnlyRealGaps",
        bug="The development plan listed Python and SQL for a candidate who is "
            "expert in both -- the whole output reduced to a frequency count of "
            "the job board's extractor.",
    ),
    M(
        id="career_paths.one_advert_becomes_a_development_plan",
        path=SRC / "career_paths.py",
        old="        if count < max(1, min_postings):\n            continue",
        new="        if False:\n            continue",
        test="tests.test_career_paths.TheSkillsGapNamesOnlyRealGaps",
        bug="A requirement named by exactly one posting ranked alongside one "
            "named by nineteen, so a single verbose advert's vocabulary was "
            "presented as a recurring market requirement.",
    ),
    M(
        id="career_paths.causation_caveat_dropped",
        path=SRC / "career_paths.py",
        old='        "causation": CAVEAT_CAUSATION,',
        new='        "coverage_only": CAVEAT_CAUSATION,',
        test="tests.test_career_paths.CaveatsCannotBeDropped",
        bug="The caveat lived only in the report template, so a renderer edit "
            "removed it and the pay deltas went out unqualified -- 'these roles "
            "pay 59% more' with nothing saying they are all manager-level.",
        tests_the_test="the caveat test checks the renderer's output instead of "
                       "the returned structure, which is the thing that "
                       "survives a renderer rewrite",
    ),
    M(
        id="career_paths.clusters_depend_on_input_order",
        path=SRC / "career_paths.py",
        old="    seeds = sorted(\n"
            "        (gram for gram, count in counts.items() if count >= max(1, min_cluster_n)),\n"
            "        key=lambda gram: (-len(gram), -counts[gram], gram),\n"
            "    )",
        new="    seeds = [gram for gram, count in counts.items()\n"
            "             if count >= max(1, min_cluster_n)]",
        test="tests.test_career_paths.ClusteringIsDeterministic",
        bug="Seed order followed dict insertion, so the same corpus in a "
            "different order produced differently named and differently sized "
            "clusters -- 'engineer' claiming postings that 'ml engineer' should "
            "have had.",
        tests_the_test="determinism is checked by running the SAME ordering "
                       "twice, which a dict-order dependency passes happily",
    ),
    M(
        id="career_paths.movement_counts_polling_not_vacancies",
        path=SRC / "career_paths.py",
        old='                bucket[label].add(str(sighting.get("job_key") or ""))',
        new='                bucket[label].add(str(sighting.get("ts") or ""))',
        test="tests.test_career_paths.HistoryIsNeverImputed",
        bug="Counting sightings rather than distinct vacancies meant that "
            "running the tool more often in recent weeks showed as rising "
            "demand for every role at once.",
    ),
    M(
        id="render_career.thin_clusters_drawn_solid",
        path=SRC / "render_career.py",
        old="    ceiling = max(1.0, float(thin_n) * _CONFIDENCE_DIVISOR)\n"
            "    return min(1.0, middle / ceiling)",
        new="    return 1.0",
        test="tests.test_career_paths.TheRendererDegradesAndNeverRaises",
        bug="Thin clusters rendered at full confidence: a bar built on three "
            "postings was drawn as solidly as one built on thirty-four, which "
            "is the impute-the-missing-value bug back at the picture level.",
    ),
    M(
        id="render_career.suppressed_median_drawn_as_a_low_bar",
        path=SRC / "render_career.py",
        old='            "value": None if suppressed else round(',
        new='            "value": 0.0 if suppressed else round(',
        test="tests.test_career_paths.TheRendererDegradesAndNeverRaises",
        bug="A cluster whose median was suppressed for too few stated salaries "
            "drew a bar at zero, which reads as 'this pays nothing' rather than "
            "'we do not know'.",
    ),

    # ----------------------------------------------------------------------
    # skills_taxonomy -- compound advert-phrase filtering
    #
    # These guard BOTH directions, because the two failures look nothing alike.
    # Too loose and the development plan fills with 'Design' and 'Liaising with
    # cross functional teams'. Too tight and a real requirement vanishes from
    # the denominator, coverage reads higher than it is, and the gap silently
    # disappears -- which is the one nobody can see.
    # ----------------------------------------------------------------------
    M(
        id="skills_taxonomy.advert_phrases_pass_the_filter",
        path=SRC / "skills_taxonomy.py",
        old="    return is_advert_phrase(text)",
        new="    return False",
        test="tests.test_skill_noise.AdvertPhrasesAreDropped",
        bug="Noise patterns were anchored against single generic words, so "
            "every compound advert phrase walked past them. On 331 real "
            "postings the top missing skills were 'Design', 'Development', "
            "'Liaising with cross functional teams', 'Scalability' and 'Data "
            "Solutions' -- at the same frequency as AI Evaluation and CI/CD.",
    ),
    M(
        id="skills_taxonomy.known_terms_lose_their_protection",
        path=SRC / "skills_taxonomy.py",
        old="    if canonical in ALIASES:\n        return False",
        new="    if False:\n        return False",
        test="tests.test_skill_noise.KnownTaxonomyTermsAreProtected",
        bug="An earlier frequency-based filter deleted Python (73% document "
            "frequency), AWS and API as 'too common'. Once a real skill leaves "
            "the vocabulary, a job asking for it stops counting as a match the "
            "candidate actually has.",
        tests_the_test="the protection test asserts on a handful of hand-picked "
                       "terms rather than sweeping every alias, so it passes "
                       "while some other alias is being eaten",
    ),
    M(
        id="skills_taxonomy.abstract_override_applied_to_the_canonical_form",
        path=SRC / "skills_taxonomy.py",
        old="    if surface(term) in _ABSTRACT_TAXONOMY_TERMS:",
        new="    if canonical in _ABSTRACT_TAXONOMY_TERMS:",
        test="tests.test_skill_noise.KnownTaxonomyTermsAreProtected",
        bug="Dropping bare 'Scalability' by its CANONICAL form also deleted "
            "'High Availability', 'Distributed Systems' and 'Scalable "
            "Systems', which all alias onto it and are all real skills -- a "
            "false drop, which hides a genuine gap instead of showing clutter.",
    ),
    M(
        id="skills_taxonomy.anchor_accepts_any_shared_word",
        path=SRC / "skills_taxonomy.py",
        old='            if " ".join(words[start:start + size]) in ALIASES:\n'
            "                return True",
        new="            if any(w in key.split() for key in ALIASES\n"
            "                   for w in words[start:start + size]):\n"
            "                return True",
        test="tests.test_skill_noise.TheAnchorRuleIsNotGenerous",
        bug="Treating 'this word appears somewhere in the taxonomy' as a "
            "technical anchor let 'Data Solutions' anchor on 'data' (from "
            "'data engineering'), which exempted almost every advert phrase "
            "and quietly restored the original bug.",
        tests_the_test="the anchor rule is only checked through its "
                       "consequences on a few phrases, so a generous version "
                       "that still happens to drop those phrases survives",
    ),
    M(
        id="skills_taxonomy.length_rule_eats_long_product_names",
        path=SRC / "skills_taxonomy.py",
        old="_TOO_MANY_WORDS = 6",
        new="_TOO_MANY_WORDS = 4",
        test="tests.test_skill_noise.RealSkillsSurvive",
        bug="A tighter word-count backstop dropped 'Software as a Service "
            "(SaaS)' and 'BIRT (Business Intelligence and Reporting Tools)' -- "
            "real, named things that are simply long.",
    ),
    M(
        id="skills_taxonomy.soft_vocabulary_over_reaches",
        path=SRC / "skills_taxonomy.py",
        old="    r\"ambiguity)\\b\", re.I)",
        new="    r\"ambiguity|integrity|autonomy)\\b\", re.I)",
        test="tests.test_skill_noise.RealSkillsSurvive",
        bug="Adding plausible-looking soft words to the vocabulary cost 'Data "
            "Integrity' and 'Autonomy iManage', both real. Every widening of "
            "this list has to be read against the corpus, not guessed.",
    ),
    M(
        id="skills_taxonomy.bias_flips_toward_dropping",
        path=SRC / "skills_taxonomy.py",
        old='    "registration", "diversity", "lifestyle", "autonomy", "accountability",',
        new='    "registration", "diversity", "lifestyle", "autonomy", "accountability",\n'
            '    "monitoring", "debugging", "orchestration", "testing", "research",',
        test="tests.test_skill_noise.TheBiasIsTowardKeeping",
        bug="The bare-noun list is the easiest place to over-reach: every one "
            "of these reads generic and every one is a real, nameable skill. "
            "Dropping them removes real requirements from the denominator and "
            "inflates coverage.",
        tests_the_test="nothing asserts the KEEP side of the bias, so a later "
                       "tightening lands with a green suite",
    ),
]


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

@dataclass
class Result:
    mutation: Mutation
    status: str               # "caught", "SURVIVED", "not applied"
    detail: str = ""
    failing: list[str] = field(default_factory=list)


_originals: dict[Path, str] = {}


def _restore_everything() -> None:
    """Put every touched file back. Runs on exit, including on Ctrl-C."""
    for path, text in list(_originals.items()):
        try:
            if path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8", newline="")
        except OSError as exc:
            print(f"!! COULD NOT RESTORE {path}: {exc}", file=sys.stderr)
        else:
            _originals.pop(path, None)


atexit.register(_restore_everything)


def run_tests(target: str) -> tuple[bool, list[str]]:
    """(passed, failing test ids) for one unittest target."""
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", target, "-v"],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    failing = [line.split(" ")[0] for line in proc.stderr.splitlines()
               if line.startswith(("FAIL: ", "ERROR: "))]
    failing = [line.split(": ", 1)[1].split(" ")[0]
               for line in proc.stderr.splitlines()
               if line.startswith(("FAIL: ", "ERROR: "))]
    return proc.returncode == 0, failing


def apply_mutation(mutation: Mutation) -> Result | None:
    """Write the mutated source, or return a Result explaining why not."""
    if not mutation.path.is_file():
        return Result(mutation, "not applied", f"{mutation.path.name} is missing")
    original = mutation.path.read_text(encoding="utf-8")
    count = original.count(mutation.old)
    if count != 1:
        return Result(
            mutation, "not applied",
            f"anchor matched {count} times, expected exactly 1 -- the source "
            f"moved and this mutation needs re-anchoring")
    _originals[mutation.path] = original
    mutation.path.write_text(original.replace(mutation.old, mutation.new),
                             encoding="utf-8", newline="")
    return None


def check(mutation: Mutation) -> Result:
    problem = apply_mutation(mutation)
    if problem is not None:
        return problem
    try:
        passed, failing = run_tests(mutation.test)
    finally:
        _restore_everything()

    if passed:
        return Result(mutation, "SURVIVED",
                      f"{mutation.test} still passes with the bug reintroduced")
    return Result(mutation, "caught", f"{len(failing)} test(s) went red", failing)


def print_table(results: list[Result]) -> None:
    width = max(len(r.mutation.id) for r in results) + 2
    print()
    print("=" * (width + 62))
    print(f"{'MUTATION'.ljust(width)}{'VERDICT'.ljust(12)}DETAIL")
    print("=" * (width + 62))
    for result in results:
        mark = {"caught": "PASS", "SURVIVED": "FAIL", "not applied": "SKIP"}[result.status]
        print(f"{result.mutation.id.ljust(width)}{mark.ljust(12)}{result.detail}")
        if result.status == "SURVIVED":
            print(f"{''.ljust(width)}{''.ljust(12)}"
                  f"-> likely: {result.mutation.tests_the_test or 'the test asserts on something the bug does not change'}")
    print("=" * (width + 62))

    caught = sum(1 for r in results if r.status == "caught")
    survived = [r for r in results if r.status == "SURVIVED"]
    skipped = [r for r in results if r.status == "not applied"]
    print(f"\n{caught}/{len(results)} mutations caught.")

    if survived:
        print(f"\n{len(survived)} SURVIVED -- a bug was put back and the suite "
              f"stayed green. That is a hole in the tests, not in the code:\n")
        for result in survived:
            print(f"  {result.mutation.id}")
            print(f"    file:  {result.mutation.path.relative_to(REPO)}")
            print(f"    test:  {result.mutation.test}")
            print(f"    bug:   {result.mutation.bug}")
            print()
    if skipped:
        print(f"\n{len(skipped)} could not be applied (source moved under them):\n")
        for result in skipped:
            print(f"  {result.mutation.id}: {result.detail}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true",
                        help="show the mutations and exit, changing nothing")
    parser.add_argument("--only", default="",
                        help="run only mutations whose id contains this string")
    args = parser.parse_args(argv)

    selected = [m for m in MUTATIONS if args.only in m.id]
    if not selected:
        print(f"no mutation id contains {args.only!r}", file=sys.stderr)
        return 2

    if args.list:
        for mutation in selected:
            print(f"{mutation.id}\n    {mutation.path.relative_to(REPO)}"
                  f"\n    -> {mutation.test}\n    {mutation.bug}\n")
        return 0

    print(f"Running {len(selected)} mutation(s). Each one edits a source file, "
          f"runs its test\ntarget, and restores the file. A mutation that does "
          f"NOT make its test fail is\nthe finding.\n")

    results = []
    for index, mutation in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {mutation.id} -> {mutation.test} ... ",
              end="", flush=True)
        result = check(mutation)
        print({"caught": "caught", "SURVIVED": "SURVIVED",
               "not applied": "skipped"}[result.status])
        results.append(result)

    print_table(results)

    if any(r.status != "caught" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _restore_everything()
