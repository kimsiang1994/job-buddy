"""Seeded random input against the parsers, asserting properties not absence of crashes.

    py -m unittest tests.test_fuzz

The modules here all take input written by somebody else: a job title typed by
an employer, a JD scraped as HTML, a salary field somebody filled in wrong, a
resume span produced by a model. "Does not crash" is the floor, not the
property -- every silent-failure bug in this repo returned cleanly. So each
test below asserts the guarantee the module actually sells:

    fact_guard      a passing bullet contains no number its fact does not have
    job_schema      a returned salary is plausible, or a predicate discloses why
    paths           the output is a path Windows will accept, and truncation
                    does not merge two different jobs into one directory
    verify_profile  a fact is only auto-verified when its span is really in the
    html_text       resume, and flattening never invents text
    skills_taxonomy a match is exact or a containment, never a coincidence

SEED is fixed and printed on failure, so a red run reproduces exactly. Raise
FUZZ_ITERATIONS locally to hunt; the committed value keeps the suite quick.

Offline: pure functions only, no network and no API key.
"""

from __future__ import annotations

import random
import re
import string
import unittest
from pathlib import Path

from jobbuddy import fact_guard, html_text, job_schema, paths, skills_taxonomy, verify_profile

# Fixed so a failure reproduces. Printed in every failure message.
SEED = 20260718
FUZZ_ITERATIONS = 300


# --------------------------------------------------------------------------
# hostile input corpus
# --------------------------------------------------------------------------

# The shapes that have historically broken something, kept as a fixed list so
# every fuzz test covers all of them regardless of what the RNG picks.
PATHOLOGICAL = [
    None, "", " ", "\t\n", 0, -1, 1.5, True, False,
    [], {}, [None], {"a": None}, object(),
    "\x00", "\x00\x01\x02\x1f\x7f",                 # control characters
    "﻿leading BOM", "trailing BOM﻿",
    "a" * 10000, "中" * 5000,                   # huge, and huge CJK
    "😀 emoji", "é combining", "‮RTL override",
    "NUL", "CON.txt", "com1", "aux ", "LPT9.pdf",   # Windows reserved
    "C:\\Windows\\System32", "../../etc/passwd", "a/b\\c:d*e?f\"g<h>i|j",
    ".", "..", "...", "   .   ", "-", "--",
    "#{}$@\\", "^$.*+?()[]{}|",                     # Typst and regex metacharacters
    "2024-13-45", "31/02/2024", "not a date", "0000-00-00", "9999-99-99",
    "1e309", "NaN", "inf", "-inf", "0x10", "1_000",
    "S$12,000 - S$18,000", "up to 20k", "negotiable", "--", "1e5",
    "<script>alert(1)</script>", "<p>unclosed", "<<<>>>", "<!--", "&#x41;",
    "&nbsp;" * 500, "<div" + " a=1" * 500 + ">",
]

_ALPHABET = string.printable + "中文éü﻿​"


def random_text(rng: random.Random, max_len: int = 80) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randrange(max_len)))


def random_value(rng: random.Random):
    """Any shape a caller might realistically (or unrealistically) pass."""
    pick = rng.randrange(10)
    if pick == 0:
        return rng.choice(PATHOLOGICAL)
    if pick == 1:
        return rng.randrange(-10**9, 10**9)
    if pick == 2:
        return rng.uniform(-10**6, 10**6)
    if pick == 3:
        return [random_value(rng) for _ in range(rng.randrange(4))]
    if pick == 4:
        return {random_text(rng, 8): random_value(rng) for _ in range(rng.randrange(3))}
    return random_text(rng)


class FuzzCase(unittest.TestCase):
    """Every failure carries the seed and the exact input that produced it."""

    def setUp(self):
        self.rng = random.Random(SEED)

    def context(self, value, extra: str = "") -> str:
        return (f"\n  SEED={SEED} (fixed; set it in tests/test_fuzz.py to reproduce)"
                f"\n  input={value!r:.300}"
                + (f"\n  {extra}" if extra else ""))

    def assertNeverRaises(self, fn, value, *args):
        try:
            return fn(value, *args)
        except Exception as exc:                     # noqa: BLE001 - that is the point
            self.fail(f"{fn.__module__}.{fn.__name__} raised "
                      f"{type(exc).__name__}: {exc}{self.context(value)}")


# --------------------------------------------------------------------------
# the read-path convention: these must not raise, on anything
# --------------------------------------------------------------------------

class ParsersNeverRaise(FuzzCase):
    """The repo convention: a malformed input degrades, it does not kill a run.

    This is the floor rather than the point, but it is the floor every other
    property in this file stands on -- a function that raises has no behaviour
    left to assert anything about.
    """

    def test_job_schema_normalisers(self):
        for value in PATHOLOGICAL + [random_value(self.rng) for _ in range(FUZZ_ITERATIONS)]:
            for fn in (job_schema.norm_text, job_schema.norm_jd_text,
                       job_schema.norm_title, job_schema.norm_company,
                       job_schema.parse_date, job_schema.days_between,
                       job_schema.normalise_seniority, job_schema.seniority_from_years,
                       job_schema.looks_like_agency):
                self.assertNeverRaises(fn, value)

    def test_job_schema_salary(self):
        for amount in PATHOLOGICAL + [random_value(self.rng) for _ in range(100)]:
            for period in ("monthly", "annually", "hourly", "", None, "per fortnight", 7):
                for fn in (job_schema.to_monthly_sgd, job_schema.salary_was_adjusted,
                           job_schema.salary_period_was_guessed):
                    try:
                        fn(amount, period)
                    except Exception as exc:         # noqa: BLE001
                        self.fail(f"{fn.__name__} raised {type(exc).__name__}: {exc}"
                                  f"{self.context(amount, f'period={period!r}')}")

    def test_html_text_flatten(self):
        for value in PATHOLOGICAL + [random_text(self.rng, 400) for _ in range(FUZZ_ITERATIONS)]:
            for preserve in (True, False):
                try:
                    html_text.flatten_html(value, preserve)
                except Exception as exc:             # noqa: BLE001
                    self.fail(f"flatten_html raised {type(exc).__name__}: {exc}"
                              f"{self.context(value, f'preserve_blocks={preserve}')}")

    def test_skills_taxonomy(self):
        for value in PATHOLOGICAL + [random_value(self.rng) for _ in range(FUZZ_ITERATIONS)]:
            self.assertNeverRaises(skills_taxonomy.canon, value)
            self.assertNeverRaises(skills_taxonomy.is_noise, value)
            self.assertNeverRaises(skills_taxonomy.clean_job_skills, [value])
            try:
                skills_taxonomy.match(str(value), {"python": 1.0, "machine learning": 0.8})
            except Exception as exc:                 # noqa: BLE001
                self.fail(f"match raised {type(exc).__name__}: {exc}{self.context(value)}")

    def test_paths_sanitisation(self):
        for value in PATHOLOGICAL + [random_text(self.rng, 300) for _ in range(FUZZ_ITERATIONS)]:
            self.assertNeverRaises(paths.sanitise_component, value)
            self.assertNeverRaises(paths.job_label, {"title": value, "company": value})

    def test_verify_profile_normalise_and_check(self):
        for value in PATHOLOGICAL + [random_value(self.rng) for _ in range(FUZZ_ITERATIONS)]:
            self.assertNeverRaises(verify_profile.normalise, value)
            try:
                verify_profile.check_fact({"source_span": value, "numbers": [value],
                                           "entities": [value], "org": value,
                                           "start": value, "end": value},
                                          verify_profile.normalise("some resume text"))
            except Exception as exc:                 # noqa: BLE001
                self.fail(f"check_fact raised {type(exc).__name__}: {exc}"
                          f"{self.context(value)}")

    def test_fact_guard_survives_any_bullet(self):
        """The BULLET is the untrusted input -- it is what a model produced."""
        well_formed = {"fact_id": "f1", "numbers": ["12"], "entities": ["Python"],
                       "skills": [], "phrasings": [], "start": "2019-01",
                       "end": "2023-01"}
        for value in PATHOLOGICAL + [random_value(self.rng) for _ in range(FUZZ_ITERATIONS)]:
            for fact in (None, {}, {"fact_id": "f1"}, well_formed):
                try:
                    fact_guard.check_bullet(str(value), fact, {}, None)
                except Exception as exc:             # noqa: BLE001
                    self.fail(f"check_bullet raised {type(exc).__name__}: {exc}"
                              f"{self.context(value, f'fact={fact!r:.120}')}")

    def test_a_malformed_fact_never_yields_a_pass(self):
        """A malformed FACT is a different contract, and a weaker one on purpose.

        `check_bullet` is a gate, not a read path, so it is not held to "never
        raises": a fact whose `numbers` is an int instead of a list is a broken
        profile, and raising there stops the run rather than shipping a
        document. What must never happen is the third outcome -- returning
        cleanly with `ok=True`, which is a gate reporting that it approved
        something it could not evaluate.

        So this asserts fail-CLOSED: raise or reject, never accept. (It does
        currently raise on some of these; that is recorded rather than hidden.)
        """
        for value in PATHOLOGICAL + [random_value(self.rng) for _ in range(100)]:
            fact = {"fact_id": "f1", "numbers": value, "entities": value,
                    "start": value, "end": value}
            try:
                verdict = fact_guard.check_bullet("Delivered 4,200,000 records", fact,
                                                  {}, None)
            except Exception:                        # noqa: BLE001 - fail-closed
                continue
            self.assertFalse(
                verdict.ok,
                f"a bullet asserting 4,200,000 was ACCEPTED against a fact whose "
                f"fields are malformed -- the gate approved what it could not read"
                + self.context(value))


# --------------------------------------------------------------------------
# fact_guard: the anti-fabrication guarantee, fuzzed
# --------------------------------------------------------------------------

# Deliberately NOT fact_guard's own NUMBER_RE. An oracle built from the code
# under test moves with it: mutate `_fact_numbers` to accept everything and a
# self-referential check would still agree with itself. This is a plain,
# independent reading of "a digit run in the text".
_INDEPENDENT_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _independent_allowed_numbers(fact: dict) -> set[str]:
    """Every number an honest bullet could take from this fact.

    Written from the fact directly rather than by calling into fact_guard:
    digits appearing in its `numbers` entries, plus the years in its dates.
    """
    allowed: set[str] = set()
    for raw in fact.get("numbers") or []:
        for match in _INDEPENDENT_NUMBER.finditer(str(raw)):
            digits = match.group(0).replace(",", "").rstrip(".")
            allowed.add(digits)
            try:
                value = float(digits)
            except ValueError:
                continue
            if value.is_integer():
                allowed.add(str(int(value)))
                allowed.add(f"{int(value):,}")
    for key in ("start", "end"):
        if fact.get(key):
            allowed.add(str(fact[key])[:4])
    return allowed


class FactGuardNeverPassesAnInventedNumber(FuzzCase):
    """The one guarantee the whole tailoring design rests on.

    Tailoring is a selection problem: the moment a model is asked to write a
    bullet, it is authorised to invent the numbers in it. Prompting cannot
    close that, so the deterministic check does -- every digit in an accepted
    bullet must trace to the cited fact.

    Fuzzed rather than exampled because the failure is not "the guard rejects
    everything", it is "the guard accepts one shape nobody thought to try".
    That shape is what random generation is for.

    The only numbers exempt are duration claims ("3 years"), which the duration
    rule checks against the fact's dates instead of its number list. Those are
    excluded here and asserted separately below.
    """

    def _random_fact(self, rng):
        numbers = [rng.choice(["12", "4,200,000", "97%", "3.5", "18", "250k", ""])
                   for _ in range(rng.randrange(4))]
        return {
            "fact_id": f"f{rng.randrange(100)}",
            "numbers": numbers,
            "entities": [rng.choice(["Python", "AWS", "Kafka", "PySpark"])
                         for _ in range(rng.randrange(3))],
            "skills": [], "org": "Acme", "role": "Engineer",
            "phrasings": ["Built the thing at Acme"],
            "source_span": "Built the thing at Acme",
            "start": rng.choice(["2019-01", "2021-06", None, "bad"]),
            "end": rng.choice(["2023-04", None, "20I9", "not-a-date"]),
        }

    def _random_bullet(self, rng):
        parts = [rng.choice([
            "Built", "Delivered", "Reduced", "Scaled", "Owned",
            str(rng.randrange(0, 5000)),
            f"{rng.randrange(1, 99)}%",
            f"{rng.randrange(1, 20)} years",
            "4,200,000", "12", "97%", "3.5", "250k",
            "Python", "AWS", "Kafka", "Acme", "Databricks", "Snowflake",
            "the pipeline", "for the team",
        ]) for _ in range(rng.randrange(2, 9))]
        return " ".join(parts)

    def test_an_accepted_bullet_invents_no_number(self):
        rng = self.rng
        checked = 0
        accepted = 0
        for _ in range(FUZZ_ITERATIONS * 4):
            fact = self._random_fact(rng)
            bullet = self._random_bullet(rng)
            verdict = fact_guard.check_bullet(bullet, fact, {}, None)
            checked += 1
            if not verdict.ok:
                continue
            accepted += 1

            allowed = _independent_allowed_numbers(fact)
            # Spans the duration rule owns; checked separately, not here.
            duration_spans = [m.span() for m in
                              re.finditer(r"\d+(?:\.\d+)?\+?\s*(?:years?|yrs?)\b",
                                          bullet, re.I)]
            for match in _INDEPENDENT_NUMBER.finditer(bullet):
                if any(s <= match.start() < e for s, e in duration_spans):
                    continue
                token = match.group(0).replace(",", "").rstrip(".")
                self.assertIn(
                    token, allowed,
                    f"fact_guard ACCEPTED a bullet asserting {token!r}, which is "
                    f"not in fact {fact['fact_id']}."
                    + self.context(bullet, f"fact numbers={fact['numbers']!r} "
                                           f"dates={fact['start']!r}..{fact['end']!r} "
                                           f"independently-allowed={sorted(allowed)}"))

        # A fuzz test where nothing was ever accepted proves nothing: it would
        # pass identically against a guard that rejects everything.
        self.assertGreater(accepted, 0,
                           f"no bullet was accepted in {checked} attempts -- the "
                           f"property was never actually exercised")

    def test_an_unparseable_date_never_supports_a_duration_claim(self):
        """The fail-open that shipped: a mistyped end date meant 'still there'.

        `_years_between` used to fall back to today when `end` failed to parse,
        so an end of '20I9' inflated a two-year fact into a six-year one and a
        bullet claiming '8 years' passed.
        """
        rng = self.rng
        for _ in range(FUZZ_ITERATIONS):
            fact = {"fact_id": "f1", "numbers": [], "entities": [],
                    "start": "2019-01",
                    "end": rng.choice(["20I9", "not-a-date", "", "2019-99",
                                       "中文", "0000-00-00"])}
            years = rng.randrange(6, 40)
            bullet = f"Delivered {years} years of work"
            verdict = fact_guard.check_bullet(bullet, fact, {}, None)
            if fact["end"] == "":
                continue     # empty end genuinely means "current role"
            self.assertFalse(
                verdict.ok,
                f"a {years}-year claim passed against an unparseable end date "
                f"{fact['end']!r} -- the guard failed OPEN"
                + self.context(bullet))

    def test_an_uncited_bullet_is_always_rejected(self):
        for value in PATHOLOGICAL[:40]:
            verdict = fact_guard.check_bullet(str(value), None, {}, None)
            self.assertFalse(verdict.ok,
                             f"an uncited bullet was accepted{self.context(value)}")

    def test_guard_never_emits_a_bullet_it_rejected(self):
        """`guard()` may only return text that passed, generated or fallback."""
        rng = self.rng
        for _ in range(FUZZ_ITERATIONS):
            fact = self._random_fact(rng)
            facts = {fact["fact_id"]: fact}
            items = [{"text": self._random_bullet(rng),
                      "fact_id": rng.choice([fact["fact_id"], "missing", ""])}
                     for _ in range(rng.randrange(1, 5))]
            safe, verdicts = fact_guard.guard(items, facts, {})
            for text in safe:
                cited = facts.get(fact["fact_id"])
                self.assertTrue(
                    fact_guard.check_bullet(text, cited, {}, None).ok,
                    f"guard() emitted a bullet that does not pass its own check"
                    + self.context(text))


# --------------------------------------------------------------------------
# job_schema: a salary is plausible, or something says why not
# --------------------------------------------------------------------------

class SalaryConversionIsPlausibleOrDisclosed(FuzzCase):
    """A returned salary feeds the pay score, so a wrong one reorders the ranking.

    Seen live: a posting listing 200000-300000 "Monthly". Left alone it maxes
    the pay component and takes the top of the results. The module divides such
    a figure down, and the trade-off is deliberate -- but a silent adjustment is
    a number the reader cannot distinguish from a stated one, so the
    corresponding predicate must admit it happened.
    """

    PERIODS = ["monthly", "annually", "annual", "yearly", "weekly", "daily",
               "hourly", "MONTHLY", " Monthly ", "", None, "per fortnight",
               "bi-weekly", "中文", 7, [], {"a": 1}, "1e5"]

    def test_returned_salary_is_never_implausible(self):
        rng = self.rng
        amounts = ([rng.uniform(0, 10**7) for _ in range(FUZZ_ITERATIONS)]
                   + [rng.randrange(0, 10**8) for _ in range(FUZZ_ITERATIONS)]
                   + PATHOLOGICAL)
        for amount in amounts:
            for period in self.PERIODS:
                monthly = job_schema.to_monthly_sgd(amount, period)
                if monthly is None:
                    continue
                self.assertLessEqual(
                    monthly, job_schema.MONTHLY_PLAUSIBILITY_CEILING_SGD,
                    f"to_monthly_sgd returned {monthly}, above the plausibility "
                    f"ceiling it exists to enforce"
                    + self.context(amount, f"period={period!r}"))

    def test_an_unrecognised_period_is_always_disclosed(self):
        rng = self.rng
        recognised = set(job_schema._SALARY_PERIOD_TO_MONTHLY)
        for _ in range(FUZZ_ITERATIONS):
            amount = rng.choice([rng.uniform(1, 10**6), rng.randrange(1, 10**7)])
            period = rng.choice(self.PERIODS)
            monthly = job_schema.to_monthly_sgd(amount, period)
            if monthly is None:
                continue
            if job_schema.norm_text(period).lower() in recognised:
                continue
            self.assertTrue(
                job_schema.salary_period_was_guessed(amount, period),
                f"the period was unrecognised and the magnitude guess was used, "
                f"but salary_period_was_guessed() says otherwise -- so the "
                f"number reads as stated"
                + self.context(amount, f"period={period!r} -> {monthly}"))

    def test_a_divided_down_salary_is_always_disclosed(self):
        """In the recoverable band, the override happens AND is admitted."""
        rng = self.rng
        ceiling = job_schema.MONTHLY_PLAUSIBILITY_CEILING_SGD
        for _ in range(FUZZ_ITERATIONS):
            # Above the ceiling but readable as an annual figure. Beyond 12x
            # there is no period that makes it a salary; that band is asserted
            # separately below.
            amount = rng.uniform(ceiling + 1, ceiling * 12)
            monthly = job_schema.to_monthly_sgd(amount, "monthly")
            self.assertIsNotNone(
                monthly, f"{amount} is readable as an annual figure"
                         + self.context(amount))
            self.assertTrue(
                job_schema.salary_was_adjusted(amount, "monthly"),
                f"{amount} monthly was silently divided down to {monthly} with "
                f"no predicate admitting the override"
                + self.context(amount))

    def test_a_figure_beyond_any_period_is_refused_not_reported(self):
        """Past 12x the ceiling there is no reading that makes it a salary.

        Returning the divided-down number anyway kept it above the ceiling, so
        it still maxed the pay score and still took the top of the ranking --
        the ceiling's own failure mode, one twelfth as large.
        """
        rng = self.rng
        ceiling = job_schema.MONTHLY_PLAUSIBILITY_CEILING_SGD
        for _ in range(FUZZ_ITERATIONS):
            amount = rng.uniform(ceiling * 12 + 1, 10**9)
            self.assertIsNone(
                job_schema.to_monthly_sgd(amount, "monthly"),
                f"an impossible figure was reported as a usable salary"
                + self.context(amount))

    def test_validate_job_never_passes_an_inverted_salary_range(self):
        rng = self.rng
        for _ in range(FUZZ_ITERATIONS):
            job = job_schema.new_job("fuzz", rng.randrange(10**6))
            job["title"], job["company"], job["url"] = "T", "C", "u"
            lo, hi = rng.randrange(0, 10**5), rng.randrange(0, 10**5)
            job["salary_min_sgd"], job["salary_max_sgd"] = lo, hi
            job = job_schema.finalise(job)
            problems = job_schema.validate_job(job)
            if lo > hi:
                self.assertTrue(
                    any("exceeds" in p for p in problems),
                    f"min {lo} > max {hi} passed validation{self.context(job['job_key'])}")


# --------------------------------------------------------------------------
# paths: always a usable Windows path, and distinct where it must be
# --------------------------------------------------------------------------

_ILLEGAL_IN_NAME = re.compile(r'[<>:"/\\|?*]|[\x00-\x1f\x7f]')


class PathsAreAlwaysUsableOnWindows(FuzzCase):
    """Every component of the output tree is attacker-shaped input.

    A job title is written by whoever posted it, and it lands in a directory
    name on a filesystem with opinions: reserved device names, illegal
    characters, a trailing dot the shell strips after `mkdir` accepted it, and
    a 260-character ceiling whose failure surfaces as FileNotFoundError on a
    directory that plainly exists.
    """

    def _labels(self, rng, count):
        return ([random_text(rng, 300) for _ in range(count)]
                + [str(v) for v in PATHOLOGICAL])

    def test_component_is_a_name_windows_accepts(self):
        rng = self.rng
        parent = Path(r"C:\Users\someone\Desktop\job buddy\potential applications"
                      r"\machine learning engineer\2026-07-18_120000")
        for label in self._labels(rng, FUZZ_ITERATIONS):
            name = paths.job_component(label, parent)

            self.assertTrue(name, f"empty component{self.context(label)}")
            self.assertIsNone(
                _ILLEGAL_IN_NAME.search(name),
                f"component {name!r} contains a character Windows rejects"
                + self.context(label))
            self.assertNotIn(name[-1], " .",
                             f"component {name!r} ends in a dot or space, which "
                             f"the shell strips after mkdir accepted it"
                             + self.context(label))
            stem = name.split(".", 1)[0].strip().upper()
            self.assertNotIn(stem, paths.RESERVED_NAMES,
                             f"component {name!r} is a reserved device name"
                             + self.context(label))

    def test_full_path_fits_max_path_unless_the_root_alone_ate_it(self):
        rng = self.rng
        parent = Path(r"C:\p\scope\2026-07-18_120000")
        for label in self._labels(rng, FUZZ_ITERATIONS):
            name = paths.job_component(label, parent)
            total = len(str(parent)) + 1 + len(name) + paths.FILENAME_HEADROOM
            if total > paths.MAX_PATH:
                # The documented escape: below MIN_COMPONENT a name carries no
                # information, so the budget stops and the OS is left to
                # complain rather than the deliverable being silently moved.
                self.assertLessEqual(
                    len(name), paths.MIN_COMPONENT,
                    f"path is {total} chars (over MAX_PATH {paths.MAX_PATH}) and "
                    f"the component was not at the MIN_COMPONENT floor"
                    + self.context(label, f"component={name!r}"))

    def test_truncation_keeps_different_jobs_apart(self):
        """Two long titles sharing a prefix must not share a directory.

        Truncation alone would have the second job overwrite the first's
        resume, and nothing would report it. The digest is of the FULL original
        for exactly this reason.
        """
        rng = self.rng
        # The parent must be LONG enough that the budget actually truncates.
        # It was not, in the first version of this test: labels came in under
        # the budget, nothing was ever shortened, and the test passed happily
        # against a `_fit` that hashed the truncated prefix instead of the
        # original -- the exact collision it claims to rule out. Caught by
        # tests/mutate.py (paths.digest_of_truncation_not_original), which is
        # what that script is for.
        parent = Path(r"C:\Users\someone\Desktop\job buddy\potential applications"
                      r"\senior machine learning engineer singapore"
                      r"\2026-07-18_120000")
        budget = paths.MAX_PATH - len(str(parent)) - 1 - paths.FILENAME_HEADROOM
        self.assertGreater(budget, paths.MIN_COMPONENT)

        prefix = "Senior Machine Learning Engineer, " * 3
        self.assertGreater(len(prefix), budget,
                           "the shared prefix must exceed the budget or nothing "
                           "is truncated and this test proves nothing")

        seen: dict[str, str] = {}
        truncated = 0
        for _ in range(FUZZ_ITERATIONS * 2):
            label = prefix + random_text(rng, 120)
            name = paths.job_component(label, parent)
            if len(name) < len(paths.sanitise_component(label)):
                truncated += 1
            if name in seen and seen[name] != label:
                self.fail(
                    f"two different titles produced the same directory {name!r}"
                    f"{self.context(label, f'other={seen[name]!r:.200}')}")
            seen[name] = label

        # Not every label: a random tail of control characters can sanitise
        # away to under the budget. The floor is here so the test cannot go
        # quietly vacuous the way its first version did.
        self.assertGreater(truncated, FUZZ_ITERATIONS,
                           f"only {truncated} of {FUZZ_ITERATIONS * 2} labels "
                           f"were truncated, so the collision this test exists "
                           f"to rule out was barely exercised")

    def test_taken_set_guarantees_uniqueness_even_for_identical_titles(self):
        rng = self.rng
        parent = Path(r"C:\p\scope\2026-07-18_120000")
        taken: set[str] = set()
        produced = []
        labels = ["Data Engineer"] * 20 + self._labels(rng, 100)
        for label in labels:
            produced.append(paths.job_component(label, parent, taken))
        lowered = [p.casefold() for p in produced]
        self.assertEqual(len(lowered), len(set(lowered)),
                         "job_component produced a duplicate despite `taken` -- "
                         "one job's deliverables would overwrite another's"
                         + self.context(SEED))

    def test_job_dir_is_stable_and_pure(self):
        rng = self.rng
        for label in self._labels(rng, 100):
            a = paths.job_dir("scope", "2026-07-18_120000", label, Path(r"C:\p"))
            b = paths.job_dir("scope", "2026-07-18_120000", label, Path(r"C:\p"))
            self.assertEqual(a, b, f"job_dir is not deterministic{self.context(label)}")


# --------------------------------------------------------------------------
# html_text and skills_taxonomy
# --------------------------------------------------------------------------

class FlatteningNeverInventsText(FuzzCase):
    """Flattening may lose markup. It may not add words that were not there."""

    def test_no_word_appears_that_was_not_in_the_input(self):
        rng = self.rng
        words = ["Kubernetes", "Python", "Nice", "have", "required", "中文"]
        for _ in range(FUZZ_ITERATIONS):
            chosen = [rng.choice(words) for _ in range(rng.randrange(1, 6))]
            tags = ["<p>", "<div>", "<li>", "<h3>", "<br/>", "<script>", "</script>"]
            html = "".join(rng.choice(tags) + w for w in chosen)
            out = html_text.flatten_html(html)
            for token in re.findall(r"[A-Za-z一-鿿]{3,}", out):
                self.assertTrue(
                    any(token in w or w in token for w in words + ["script"]),
                    f"flatten_html produced the token {token!r}, absent from the "
                    f"input{self.context(html)}")

    def test_script_contents_never_survive(self):
        rng = self.rng
        for _ in range(FUZZ_ITERATIONS):
            secret = "SECRET" + str(rng.randrange(10**6))
            html = f"<p>visible</p><script>var x = '{secret}';</script>"
            out = html_text.flatten_html(html)
            self.assertNotIn(secret, out,
                             f"script contents leaked into the flattened text"
                             + self.context(html))


class SkillMatchesAreNeverCoincidental(FuzzCase):
    """A false skill match is strictly worse than a missed one.

    It inflates the fit score, and downstream it could justify a resume bullet
    asserting a skill the candidate does not have -- the exact failure the
    whole design exists to prevent. The fuzzed property is that any non-zero
    match is an exact hit or a genuine token containment, never an overlap.
    """

    def test_a_match_is_exact_or_a_containment(self):
        rng = self.rng
        owned = {"python": 1.0, "machine learning": 0.9, "generative ai": 0.8,
                 "stakeholder management": 0.6, "data engineering": 0.7}
        candidates = (["Model Deployment", "Agentic Memory Management",
                       "Data Science", "Generative AI Application Development",
                       "Python3", "large language model"]
                      + [random_text(rng, 40) for _ in range(FUZZ_ITERATIONS)])
        for term in candidates:
            weight, matched, how = skills_taxonomy.match(term, owned)
            if weight == 0.0:
                continue
            self.assertIn(how, ("exact", "contains", "narrower"),
                          f"unknown match kind {how!r}{self.context(term)}")
            job_tokens = skills_taxonomy.tokens(term)
            owned_tokens = skills_taxonomy.tokens(matched)
            if how == "exact":
                self.assertEqual(skills_taxonomy.canon(term), matched)
            elif how == "contains":
                self.assertTrue(owned_tokens <= job_tokens,
                                f"'contains' match without containment"
                                + self.context(term, f"matched={matched!r}"))
            else:
                self.assertTrue(job_tokens <= owned_tokens,
                                f"'narrower' match without containment"
                                + self.context(term, f"matched={matched!r}"))

    def test_noise_is_never_matched_as_a_skill(self):
        rng = self.rng
        owned = {"python": 1.0}
        for term in [random_text(rng, 60) for _ in range(FUZZ_ITERATIONS)]:
            if not skills_taxonomy.is_noise(term):
                continue
            cleaned = skills_taxonomy.clean_job_skills([term])
            self.assertEqual(cleaned, [],
                             f"a term flagged as noise survived cleaning"
                             + self.context(term))


# --------------------------------------------------------------------------
# verify_profile
# --------------------------------------------------------------------------

class VerificationRequiresARealSpan(FuzzCase):
    """A fact is auto-verified only when its span is literally in the resume.

    The claim `auto_verify` makes is narrow and precise: "copied from the
    resume, not invented by the model". Anything that gets marked verified
    without the span actually being present has turned that claim into a lie,
    and the whole downstream guard trusts it.
    """

    def test_a_fact_is_never_verified_without_its_span_in_the_resume(self):
        rng = self.rng
        resume = ("Built an ETL pipeline at Acme processing 4,200,000 records. "
                  "Led a team of 12 engineers from 2019 to 2023.")
        verified_any = 0
        for _ in range(FUZZ_ITERATIONS):
            span = rng.choice([
                "Built an ETL pipeline at Acme processing 4,200,000 records.",
                "Led a team of 12 engineers",
                random_text(rng, 60),
                "Built an ETL pipeline at Acme processing 9,999,999 records.",
                "", " ",
                # JSON-serialisable only: `auto_verify` deep-copies through
                # json, and a draft always arrives parsed FROM json, so an
                # arbitrary object is not an input this function can receive.
                rng.choice([v for v in PATHOLOGICAL[:24]
                            if isinstance(v, (str, int, float, bool, type(None)))]),
            ])
            draft = {"facts": [{"fact_id": "f1", "source_span": span,
                                "numbers": [], "entities": []}]}
            out = verify_profile.auto_verify(draft, resume)
            fact = out["facts"][0]
            if not fact.get("verified"):
                continue
            verified_any += 1
            self.assertIn(
                verify_profile.normalise(span), verify_profile.normalise(resume),
                f"a fact was auto-verified whose span is not in the resume"
                + self.context(span))
        self.assertGreater(verified_any, 0,
                           "nothing was ever verified -- the property was never "
                           "exercised")

    def test_a_short_span_is_never_distinctive_enough(self):
        resume = "Built an ETL pipeline at Acme"
        for span in ["Built", "at Acme", "ETL", "a", "an ETL"]:
            draft = {"facts": [{"fact_id": "f1", "source_span": span,
                                "numbers": [], "entities": []}]}
            out = verify_profile.auto_verify(draft, resume)
            self.assertFalse(
                out["facts"][0].get("verified"),
                f"the short span {span!r} was accepted as distinctive -- short "
                f"spans match by accident{self.context(span)}")

    def test_auto_verify_does_not_mutate_its_input(self):
        draft = {"facts": [{"fact_id": "f1", "source_span": "x" * 40}]}
        before = repr(draft)
        verify_profile.auto_verify(draft, "some resume")
        self.assertEqual(repr(draft), before,
                         "auto_verify mutated the draft it was handed, so a "
                         "caller cannot diff before against after")


if __name__ == "__main__":
    print(f"fuzz seed: {SEED}")
    unittest.main()
