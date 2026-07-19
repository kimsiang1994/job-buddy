"""Tests for the tailoring tactics.

    py -m unittest tests.test_strategies

Offline, no API key, no network, no cost.

The properties that matter: a strategy set must produce a byte-identical
prompt every time (otherwise an A/B comparison is comparing two different
prompts and the result means nothing), and no post-transform may put generated
words back into a bullet that has already passed fact_guard.
"""

from __future__ import annotations

import unittest

from jobbuddy import strategies


def bullet(text, fact_id="f1"):
    return {"text": text, "fact_id": fact_id}


class ResolutionIsForgivingButHonest(unittest.TestCase):
    def test_defaults_resolve_cleanly(self):
        resolved, problems = strategies.resolve()
        self.assertTrue(resolved)
        self.assertEqual(problems, [])

    def test_an_unknown_name_is_reported_not_raised(self):
        resolved, problems = strategies.resolve(["baseline", "nonsense"])
        self.assertEqual(len(resolved), 1)
        self.assertTrue(any("nonsense" in p for p in problems))

    def test_conflicting_tactics_are_reported(self):
        """Running both makes the comparison uninterpretable."""
        _, problems = strategies.resolve(["xyz_formula", "technical_decision"])
        self.assertTrue(any("conflicts" in p for p in problems))

    def test_nothing_graded_C_is_on_by_default(self):
        """An untested tactic must be opted into, never inherited."""
        for name in strategies.defaults():
            with self.subTest(name=name):
                self.assertNotEqual(strategies.REGISTRY[name].grade, "C")

    def test_every_strategy_carries_a_rationale(self):
        for name, strategy in strategies.REGISTRY.items():
            with self.subTest(name=name):
                self.assertTrue(strategy.rationale.strip())
                self.assertIn(strategy.grade, ("A", "B", "C"))


class PromptsAreDeterministic(unittest.TestCase):
    def test_the_same_set_produces_identical_prompts(self):
        a, _ = strategies.resolve(["plain_language", "hard_to_fake_signals"])
        b, _ = strategies.resolve(["hard_to_fake_signals", "plain_language"])
        self.assertEqual(strategies.apply_prompt("BASE", a),
                         strategies.apply_prompt("BASE", b))

    def test_different_sets_produce_different_prompts(self):
        """Guards against making prompts stable by ignoring the strategies."""
        a, _ = strategies.resolve(["baseline"])
        b, _ = strategies.resolve(["baseline", "xyz_formula"])
        self.assertNotEqual(strategies.apply_prompt("BASE", a),
                            strategies.apply_prompt("BASE", b))

    def test_the_base_prompt_is_preserved(self):
        resolved, _ = strategies.resolve()
        self.assertTrue(strategies.apply_prompt("BASE", resolved).startswith("BASE"))


class MetricDensityIsCappedNotDeleted(unittest.TestCase):
    def test_surplus_quantified_bullets_are_demoted_not_dropped(self):
        bullets = [bullet(f"Did thing {n} with 5 units") for n in range(5)]
        out = strategies._cap_metric_density(bullets, cap=0.4)
        self.assertEqual(len(out), len(bullets))

    def test_a_sparse_resume_is_left_alone(self):
        """This is a cap, not a target. There is deliberately no rule that
        tells a resume to add numbers."""
        bullets = [bullet("Led the migration"), bullet("Cut latency by 40%")]
        self.assertEqual(strategies._cap_metric_density(bullets, cap=0.4), bullets)

    def test_an_empty_list_is_handled(self):
        self.assertEqual(strategies._cap_metric_density([]), [])


class LanguageIsFlaggedNotRewritten(unittest.TestCase):
    """Rewriting here would inject generated words downstream of fact_guard --
    the precise hole the gate exists to close."""

    def test_an_inflated_verb_is_flagged_and_the_text_is_untouched(self):
        original = "Spearheaded the migration"
        out = strategies._strip_inflated_language([bullet(original)])
        self.assertEqual(out[0]["text"], original)
        self.assertIn("spearheaded", out[0]["language_flags"])

    def test_an_llm_tell_is_flagged(self):
        out = strategies._strip_inflated_language([bullet("Delve into the data")])
        self.assertIn("delve", out[0]["language_flags"])

    def test_clean_text_gets_no_flag_key(self):
        out = strategies._strip_inflated_language([bullet("Cut latency by 40%")])
        self.assertNotIn("language_flags", out[0])

    def test_the_llm_tell_list_stays_short(self):
        """Blog lists run to dozens of words and are folklore. A longer list
        would strip ordinary English for no measured gain."""
        self.assertEqual(strategies.LLM_TELLS,
                         {"delve", "underscore", "meticulous", "crucial"})


class PostTransformsCompose(unittest.TestCase):
    def test_transforms_run_in_a_stable_order(self):
        resolved, _ = strategies.resolve(["metric_density_cap", "plain_language"])
        bullets = [bullet("Spearheaded 5 things"), bullet("Led the work")]
        first = strategies.apply_post(bullets, resolved)
        second = strategies.apply_post(bullets, list(reversed(resolved)))
        self.assertEqual([b["text"] for b in first], [b["text"] for b in second])

    def test_a_strategy_with_no_post_transform_is_a_no_op(self):
        resolved, _ = strategies.resolve(["baseline"])
        bullets = [bullet("Led the work")]
        self.assertEqual(strategies.apply_post(bullets, resolved), bullets)


class ForbiddenTacticsAreAbsent(unittest.TestCase):
    """These optimise against folklore or deceive an employer. Their absence
    is a design decision and should fail loudly if someone adds one."""

    def test_no_keyword_stuffing_or_evasion_strategy_exists(self):
        for name in strategies.REGISTRY:
            with self.subTest(name=name):
                self.assertNotRegex(
                    name, r"keyword_density|stuff|invisible|hidden|evade|detector")


if __name__ == "__main__":
    unittest.main(verbosity=2)
