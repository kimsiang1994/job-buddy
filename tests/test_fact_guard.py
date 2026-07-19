"""Adversarial tests for the anti-fabrication gate.

    py -m unittest tests.test_fact_guard

These are the actual guarantee. The model cannot be tested to certainty --
`fact_guard` can, so it gets the most fixtures and the most hostile ones. Every
case here is a way a plausible-sounding resume line could be false.

The principle throughout: do not test that the LLM said the right thing; test
that the deterministic code around it rejects the wrong thing.
"""

from __future__ import annotations

import unittest

from jobbuddy import fact_guard

FACT = {
    "fact_id": "citibank.etl",
    "org": "Citibank Singapore",
    "role": "Manager, Data Engineering",
    "start": "2023-08", "end": "2024-11",
    "numbers": ["10", "5", "4"],
    "entities": ["Citibank", "ETL", "PySpark", "SAS"],
    "skills": ["etl", "pyspark", "data-governance"],
    "phrasings": [
        "Cut data generation from 10 to 5 working days by automating 4 ETL processes",
    ],
    "verified": True,
}

PROFILE = {
    "years_experience": 4.9,
    "constraints": {"never_claim": ["managed a team", "PhD", "founder"]},
    "entity_allowlist": ["Singapore"],
}

FACTS = {FACT["fact_id"]: FACT}


class NumbersMustBeReal(unittest.TestCase):
    def test_true_bullet_passes(self):
        verdict = fact_guard.check_bullet(
            "Cut data generation from 10 to 5 working days by automating 4 ETL processes",
            FACT, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_invented_percentage_is_rejected(self):
        """The classic hallucination: a plausible metric nobody measured."""
        verdict = fact_guard.check_bullet(
            "Cut data generation by 60% by automating 4 ETL processes", FACT, PROFILE)
        self.assertFalse(verdict.ok)
        self.assertTrue(any("invented number" in r for r in verdict.reasons))

    def test_inflated_number_is_rejected(self):
        verdict = fact_guard.check_bullet(
            "Automated 40 ETL processes", FACT, PROFILE)
        self.assertFalse(verdict.ok)

    def test_comma_formatting_does_not_cause_a_false_rejection(self):
        """A guard that rejects true statements gets switched off."""
        fact = dict(FACT, numbers=["4200000"])
        verdict = fact_guard.check_bullet("Processed 4,200,000 records", fact, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_year_from_the_facts_dates_is_allowed(self):
        verdict = fact_guard.check_bullet("Led the 2023 migration", FACT, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)


class EntitiesMustBeReal(unittest.TestCase):
    def test_company_never_worked_at_is_rejected(self):
        verdict = fact_guard.check_bullet(
            "Cut data generation from 10 to 5 days at Goldman Sachs", FACT, PROFILE)
        self.assertFalse(verdict.ok)
        self.assertTrue(any("Goldman" in r for r in verdict.reasons))

    def test_technology_not_in_the_fact_is_rejected(self):
        verdict = fact_guard.check_bullet(
            "Automated 4 ETL processes using Kubernetes", FACT, PROFILE)
        self.assertFalse(verdict.ok)

    def test_named_technology_in_the_fact_is_allowed(self):
        verdict = fact_guard.check_bullet(
            "Automated 4 ETL processes in PySpark", FACT, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_ordinary_capitalised_words_do_not_trip_it(self):
        """Sentence openers and role nouns are grammar, not claims."""
        verdict = fact_guard.check_bullet(
            "Led the ETL work and delivered it in 5 days", FACT, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_allowlisted_entity_passes(self):
        verdict = fact_guard.check_bullet("Delivered ETL work in Singapore", FACT, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)


class TheFactsOwnWordsAreAlwaysAllowed(unittest.TestCase):
    """Found on the first live run: the guard rejected true bullets AND the
    approved phrasings they fell back to, so the bullets vanished entirely.

    A word standing in the resume line a fact came from cannot be a
    fabrication. Rejecting it is not strictness, it is a broken guard -- and
    one that deletes verified content while reporting success.
    """

    def test_a_plural_of_a_listed_entity_passes(self):
        """'APIs' against an entity list saying 'API'."""
        fact = dict(FACT, entities=["API"],
                    phrasings=["Built and deployed REST APIs in containers"])
        verdict = fact_guard.check_bullet(
            "Built and deployed REST APIs in containers", fact, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_a_generic_term_in_the_approved_phrasing_passes(self):
        """'ROI' -- no extractor lists it as an entity, but it is in the resume."""
        fact = dict(FACT, entities=["Bayesian"],
                    phrasings=["Built marketing-mix models quantifying ROI"])
        verdict = fact_guard.check_bullet(
            "Built marketing-mix models quantifying ROI", fact, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_every_approved_phrasing_passes_its_own_guard(self):
        """The invariant that was violated. If a fact's own phrasing fails,
        the fallback has nothing to fall back to."""
        for phrasing in FACT["phrasings"]:
            with self.subTest(phrasing=phrasing[:50]):
                verdict = fact_guard.check_bullet(phrasing, FACT, PROFILE)
                self.assertTrue(verdict.ok, verdict.reasons)

    def test_the_source_span_passes_its_own_guard(self):
        fact = dict(FACT, source_span="Automated 4 ETL processes using SAS Viya",
                    phrasings=[])
        verdict = fact_guard.check_bullet(
            "Automated 4 ETL processes using SAS Viya", fact, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)

    def test_this_does_not_weaken_the_guard(self):
        """The loosening must not admit anything outside the fact's own text."""
        fact = dict(FACT, phrasings=["Automated 4 ETL processes in PySpark"])
        for hostile in ("Automated 4 ETL processes at Goldman Sachs",
                        "Automated 4 ETL processes using Kubernetes",
                        "Automated 40 ETL processes in PySpark"):
            with self.subTest(hostile=hostile):
                self.assertFalse(
                    fact_guard.check_bullet(hostile, fact, PROFILE).ok)

    def test_a_plural_does_not_admit_an_unrelated_word(self):
        """Depluralisation must not turn 'AWS' into a licence for 'AW'."""
        fact = dict(FACT, entities=["ETL"], phrasings=["Automated 4 ETL processes"])
        self.assertFalse(
            fact_guard.check_bullet("Automated 4 ETL processes on AWS",
                                    fact, PROFILE).ok)


class DurationsMustBeSupported(unittest.TestCase):
    def test_overstated_tenure_is_rejected(self):
        """The fact spans ~1.3 years; claiming 5 is a lie the dates disprove."""
        verdict = fact_guard.check_bullet(
            "5 years of ETL leadership at Citibank", FACT, PROFILE)
        self.assertFalse(verdict.ok)
        self.assertTrue(any("duration" in r for r in verdict.reasons))

    def test_supported_tenure_passes(self):
        verdict = fact_guard.check_bullet("1 year of ETL work at Citibank", FACT, PROFILE)
        self.assertTrue(verdict.ok, verdict.reasons)


class DenylistAndCitation(unittest.TestCase):
    def test_denylisted_claim_is_rejected(self):
        verdict = fact_guard.check_bullet(
            "Managed a team of 4 ETL engineers", FACT, PROFILE)
        self.assertFalse(verdict.ok)
        self.assertTrue(any("denylist" in r for r in verdict.reasons))

    def test_uncited_bullet_is_rejected_unconditionally(self):
        """No citation, no bullet -- that is the shape a hallucination arrives in."""
        verdict = fact_guard.check_bullet("Cut data generation from 10 to 5 days",
                                          None, PROFILE)
        self.assertFalse(verdict.ok)
        self.assertTrue(any("uncited" in r for r in verdict.reasons))

    def test_empty_bullet_is_rejected(self):
        self.assertFalse(fact_guard.check_bullet("   ", FACT, PROFILE).ok)


class GuardFallsBackRatherThanEmittingFalsehood(unittest.TestCase):
    def test_rejected_bullet_falls_back_to_approved_phrasing(self):
        """A blander resume is acceptable. A false one is not."""
        safe, verdicts = fact_guard.guard(
            [{"text": "Cut data generation by 90% at Goldman Sachs",
              "fact_id": "citibank.etl"}],
            FACTS, PROFILE)
        self.assertEqual(len(safe), 1)
        self.assertEqual(safe[0], FACT["phrasings"][0])
        self.assertTrue(verdicts[0].fallback_used)

    def test_uncited_bullet_is_dropped_entirely(self):
        safe, verdicts = fact_guard.guard(
            [{"text": "Grew revenue 3x", "fact_id": ""}], FACTS, PROFILE)
        self.assertEqual(safe, [])
        self.assertFalse(verdicts[0].ok)

    def test_bullet_citing_an_unknown_fact_is_dropped(self):
        safe, _ = fact_guard.guard(
            [{"text": "Did something impressive", "fact_id": "no.such.fact"}],
            FACTS, PROFILE)
        self.assertEqual(safe, [])

    def test_true_bullets_pass_through_untouched(self):
        safe, verdicts = fact_guard.guard(
            [{"text": FACT["phrasings"][0], "fact_id": "citibank.etl"}],
            FACTS, PROFILE)
        self.assertEqual(safe, [FACT["phrasings"][0]])
        self.assertFalse(verdicts[0].fallback_used)

    def test_summary_reports_what_was_caught(self):
        _, verdicts = fact_guard.guard(
            [{"text": "Cut costs by 80% at Meta", "fact_id": "citibank.etl"},
             {"text": FACT["phrasings"][0], "fact_id": "citibank.etl"}],
            FACTS, PROFILE)
        summary = fact_guard.summarise(verdicts)
        self.assertEqual(summary["bullets"], 2)
        self.assertGreaterEqual(summary["rejected"], 1)
        self.assertIn("examples", summary)


class NothingUnverifiedEverEscapes(unittest.TestCase):
    """The property that matters, stated as one test.

    Whatever a model produces, every emitted line must survive the guard.
    """

    HOSTILE = [
        "Increased revenue by 250% at Google",
        "10 years of experience leading ETL teams",
        "Managed a team of 12 engineers",
        "PhD in Computer Science",
        "Founder of a data startup",
        "Reduced latency 99% using Kafka and Snowflake",
        "",
        "Cut data generation from 10 to 5 working days by automating 4 ETL processes",
    ]

    def test_every_emitted_bullet_passes_the_guard(self):
        safe, _ = fact_guard.guard(
            [{"text": t, "fact_id": "citibank.etl"} for t in self.HOSTILE],
            FACTS, PROFILE)
        for bullet in safe:
            with self.subTest(bullet=bullet[:60]):
                verdict = fact_guard.check_bullet(bullet, FACT, PROFILE)
                self.assertTrue(verdict.ok,
                                f"emitted an unverified bullet: {verdict.reasons}")

    def test_the_one_true_bullet_survives(self):
        safe, _ = fact_guard.guard(
            [{"text": t, "fact_id": "citibank.etl"} for t in self.HOSTILE],
            FACTS, PROFILE)
        self.assertIn(FACT["phrasings"][0], safe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
