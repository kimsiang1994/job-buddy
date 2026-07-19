"""Tests for the A/B harness.

    py -m unittest tests.test_ab_harness

Offline: no API key, no network, no cost. The grader is injected as `chat`
everywhere -- no monkeypatching -- so every path runs against a stub that
behaves the way a real LLM grader misbehaves: random on identical input, biased
toward whatever it read first, and occasionally dead mid-run.

The single most important test in this file is
`NoiseProducesNoResult.test_a_random_grader_yields_no_detectable_difference`.
If that ever fails, the harness has started manufacturing results out of noise,
which is precisely the thing it was written to prevent.
"""

from __future__ import annotations

import random
import unittest

from jobbuddy import ab_harness
from tests.test_hr_panel import JOB, REQUIREMENTS, RESUME, StubGrader, verdict

# Two variants that differ only by a marker, so a stub can tell them apart
# without anything in the harness knowing which is "the edit".
VARIANT_A = RESUME + "\nA-MARKER\n"
VARIANT_B = RESUME + "\nB-MARKER\n"

# A plausible floor: the HackerRank observation was a 33-point spread on
# identical input, which is roughly an sd of 8 on a 0-100 scale.
FLOOR = {"mean": 82.0, "sd": 8.0, "min": 66, "max": 99, "trials": 10,
         "scores": [82.0] * 10, "spread": 33.0}

# No noise at all. Used only where the floor must not be what blocks a verdict.
FLAT_FLOOR = {**FLOOR, "sd": 0.0, "spread": 0.0}

ONE_PERSONA = [{"name": "hiring_manager", "title": "Senior Data Engineer",
                "company": "Example Bank Pte Ltd", "seniority": "senior",
                "bar": "independent delivery", "lens": "reads for evidence",
                "rubric": ["evidence of ETL"], "instruction": "You are the hiring manager."}]


def _resumes_of(user_prompt):
    """(first, second) resume text, in the order the grader was shown them."""
    first = user_prompt.split("RESUME 1:\n", 1)[-1]
    first, second = first.split("\n\nRESUME 2:\n", 1)
    return first, second


class ChoiceGrader:
    """Stands in for `deepseek_client.json_chat` on a forced-choice prompt.

    Same shape as `test_hr_panel.StubGrader` -- a callable recording its calls
    and returning a json_chat-shaped dict -- but it answers the "which of these
    two" question instead of scoring one resume.

    `decide(first, second)` returns "1", "2", "tie", or "malformed".
    """

    def __init__(self, decide, raises_on=()):
        self.decide = decide
        self.raises_on = set(raises_on)
        self.calls = []

    def __call__(self, messages, schema_keys=(), **kwargs):
        self.calls.append(messages)
        if len(self.calls) in self.raises_on:
            raise RuntimeError("connection reset")
        first, second = _resumes_of(messages[1]["content"])
        choice = self.decide(first, second)
        if choice == "malformed":
            return {"ok": False, "data": None, "error": "reply is not a JSON object"}
        return {"ok": True, "data": {"stronger": choice, "reason": "stub"},
                "error": "", "repaired": False}


def prefers_marker(marker):
    """A grader with a real, position-independent preference."""
    return ChoiceGrader(lambda first, second: "1" if marker in first else "2")


def always_first():
    """A grader with no opinion at all, only a position bias."""
    return ChoiceGrader(lambda first, second: "1")


def noisy(seed):
    """A grader that answers identical questions differently. Seeded."""
    rng = random.Random(seed)
    return ChoiceGrader(lambda first, second: rng.choice(["1", "2"]))


class ScoreGrader:
    """`json_chat` for the panel's scoring prompt, with a score per call.

    Used only by the noise floor tests, where the whole point is that the same
    input scores differently on each run.
    """

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def __call__(self, messages, schema_keys=(), **kwargs):
        self.calls.append(messages)
        score = self.scores[(len(self.calls) - 1) % len(self.scores)]
        if score is None:
            return {"ok": False, "data": None, "error": "reply is not a JSON object"}
        return {"ok": True, "data": verdict("advance", score), "error": "",
                "repaired": False}


class TheNoiseFloorIsMeasuredFirst(unittest.TestCase):
    def test_it_reports_the_spread_of_identical_input(self):
        # Three personas per run, so nine calls covers three trials.
        grader = ScoreGrader([66, 66, 66, 99, 99, 99, 80, 80, 80])
        floor = ab_harness.noise_floor(RESUME, JOB, REQUIREMENTS, trials=3, chat=grader)
        self.assertEqual(floor["trials"], 3)
        self.assertEqual(floor["scores"], [66.0, 99.0, 80.0])
        self.assertEqual(floor["min"], 66.0)
        self.assertEqual(floor["max"], 99.0)
        self.assertEqual(floor["spread"], 33.0)
        self.assertGreater(floor["sd"], 10)

    def test_an_unchanging_grader_reports_a_zero_floor(self):
        floor = ab_harness.noise_floor(RESUME, JOB, REQUIREMENTS, trials=4,
                                       chat=StubGrader(default=verdict("advance", 70)))
        self.assertEqual(floor["sd"], 0.0)
        self.assertEqual(floor["spread"], 0.0)
        self.assertEqual(floor["mean"], 70.0)

    def test_a_grader_that_never_scores_degrades_rather_than_raising(self):
        floor = ab_harness.noise_floor(RESUME, JOB, REQUIREMENTS, trials=3,
                                       chat=StubGrader(default="malformed"))
        self.assertIsNone(floor["mean"])
        self.assertEqual(floor["scores"], [])
        self.assertEqual(floor["trials"], 3)


class ForcedChoiceSurvivesTheSwap(unittest.TestCase):
    def test_a_real_preference_is_reported_when_it_survives_both_orderings(self):
        result = ab_harness.compare_pair(VARIANT_A, VARIANT_B, JOB, REQUIREMENTS,
                                         ONE_PERSONA[0], chat=prefers_marker("A-MARKER"))
        self.assertEqual(result["winner"], "a")
        self.assertTrue(result["consistent"])
        self.assertEqual(len(result["runs"]), 2)

    def test_the_preference_is_not_an_artefact_of_which_argument_came_first(self):
        result = ab_harness.compare_pair(VARIANT_A, VARIANT_B, JOB, REQUIREMENTS,
                                         ONE_PERSONA[0], chat=prefers_marker("B-MARKER"))
        self.assertEqual(result["winner"], "b")
        self.assertTrue(result["consistent"])

    def test_both_orderings_are_actually_run(self):
        grader = prefers_marker("A-MARKER")
        ab_harness.compare_pair(VARIANT_A, VARIANT_B, JOB, REQUIREMENTS,
                                ONE_PERSONA[0], chat=grader)
        firsts = [_resumes_of(call[1]["content"])[0] for call in grader.calls]
        self.assertIn("A-MARKER", firsts[0])
        self.assertIn("B-MARKER", firsts[1])

    def test_a_position_biased_grader_yields_no_preference(self):
        """Always picking what it read first is not a preference. It is a bias."""
        result = ab_harness.compare_pair(VARIANT_A, VARIANT_B, JOB, REQUIREMENTS,
                                         ONE_PERSONA[0], chat=always_first())
        self.assertEqual(result["winner"], "no_preference")
        self.assertFalse(result["consistent"])
        self.assertIn("position bias", result["reason"])

    def test_a_tie_in_both_orderings_is_no_preference(self):
        grader = ChoiceGrader(lambda first, second: "tie")
        result = ab_harness.compare_pair(VARIANT_A, VARIANT_B, JOB, REQUIREMENTS,
                                         ONE_PERSONA[0], chat=grader)
        self.assertEqual(result["winner"], "no_preference")

    def test_an_unreadable_choice_is_recorded_not_raised(self):
        grader = ChoiceGrader(lambda first, second: "malformed")
        result = ab_harness.compare_pair(VARIANT_A, VARIANT_B, JOB, REQUIREMENTS,
                                         ONE_PERSONA[0], chat=grader)
        self.assertEqual(result["winner"], "no_preference")
        self.assertIn("JSON", result["reason"])

    def test_a_transport_exception_is_caught(self):
        grader = ChoiceGrader(lambda first, second: "1", raises_on=(1, 2))
        result = ab_harness.compare_pair(VARIANT_A, VARIANT_B, JOB, REQUIREMENTS,
                                         ONE_PERSONA[0], chat=grader)
        self.assertEqual(result["winner"], "no_preference")
        self.assertIn("raised", result["reason"])

    def test_the_grader_is_never_told_which_variant_is_the_edit(self):
        grader = prefers_marker("A-MARKER")
        ab_harness.compare_pair(VARIANT_A, VARIANT_B, JOB, REQUIREMENTS,
                                ONE_PERSONA[0], chat=grader)
        blob = "\n".join(m["content"] for call in grader.calls for m in call)
        for word in ("variant_a", "variant_b", "original", "tailored", "baseline"):
            self.assertNotIn(word, blob.lower())


class NoiseProducesNoResult(unittest.TestCase):
    """The tests this module exists for."""

    def test_a_random_grader_yields_no_detectable_difference(self):
        """THE test. A coin-flipping grader must never produce a winner.

        If this fails, the harness is manufacturing results out of noise and
        every comparison it has ever reported is suspect.
        """
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=10, chat=noisy(20260718), personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["verdict"], "no detectable difference")
        self.assertEqual(result["winners"], [])
        self.assertNotIn("best_guess", result)

    def test_a_random_grader_produces_a_winner_far_below_the_nominal_rate(self):
        """One lucky seed proves nothing, so sweep a hundred of them.

        A 95% interval is wrong 5% of the time BY CONSTRUCTION -- asserting a
        coin flip never yields a winner would be asserting something false, and
        the test would be tuned to a seed rather than to the harness. So this
        bounds the rate instead. Measured: 1 of 100 seeds.

        The bound is what makes it bite. Before significance required adequate
        power, this stood at 6% -- a handful of lucky comparisons producing a
        tight interval that excluded 0.5 -- and that regression fails here.
        """
        winners = [seed for seed in range(100)
                   if ab_harness.compare_variants(
                       {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
                       trials=10, chat=noisy(seed), personas=ONE_PERSONA,
                       seed=seed)["verdict"] != "no detectable difference"]
        self.assertLessEqual(len(winners), 3, f"noise produced winners on {winners}")

    def test_a_position_biased_grader_never_produces_a_winner(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B, "c": RESUME + "\nC-MARKER\n"},
            JOB, REQUIREMENTS, FLOOR, trials=5, chat=always_first(),
            personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["verdict"], "no detectable difference")
        self.assertEqual(result["winners"], [])
        for pair in result["pairs"]:
            with self.subTest(pair=(pair["a"], pair["b"])):
                self.assertEqual(pair["a_wins"], 0)
                self.assertEqual(pair["b_wins"], 0)
                self.assertEqual(pair["no_preference"], 5)

    def test_an_effect_smaller_than_the_noise_floor_is_not_significant(self):
        """A real but small edge, under a wider floor, is still not a result.

        The interval alone would call this one. The floor is what stops it.
        """
        # A genuine preference on the first three comparisons, a tie after:
        # win rate 0.65, an effect of 0.15 under a floor threshold of 0.20.
        grader = ChoiceGrader(lambda first, second:
                              ("1" if "A-MARKER" in first else "2")
                              if len(grader.calls) <= 6 else "tie")
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS,
            {**FLOOR, "sd": 20.0}, trials=10, chat=grader,
            personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["variants"]["a"]["win_rate"], 0.65)
        self.assertTrue(result["variants"]["a"]["excludes_null"])
        self.assertFalse(result["variants"]["a"]["clears_floor"])
        self.assertEqual(result["verdict"], "no detectable difference")

    def test_a_grader_too_noisy_to_use_can_produce_no_result_at_all(self):
        """An sd above 50 puts every reachable effect out of range. Correctly."""
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS,
            {**FLOOR, "sd": 60.0}, trials=10, chat=prefers_marker("A-MARKER"),
            personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["variants"]["a"]["win_rate"], 1.0)
        self.assertEqual(result["verdict"], "no detectable difference")


class ARealEffectIsDetected(unittest.TestCase):
    def test_a_large_consistent_effect_is_significant(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=10, chat=prefers_marker("A-MARKER"), personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["verdict"], "significant")
        self.assertEqual(result["winners"], ["a"])
        self.assertEqual(result["variants"]["a"]["win_rate"], 1.0)
        self.assertEqual(result["variants"]["b"]["win_rate"], 0.0)
        self.assertGreater(result["variants"]["a"]["ci_low"], 0.5)
        self.assertFalse(result["underpowered"])

    def test_the_losing_variant_is_not_reported_as_a_winner(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=10, chat=prefers_marker("B-MARKER"), personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["winners"], ["b"])


class BootstrapIsReproducible(unittest.TestCase):
    def test_the_same_seed_gives_the_same_interval(self):
        def run(seed):
            return ab_harness.compare_variants(
                {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
                trials=6, chat=noisy(7), personas=ONE_PERSONA, seed=seed)

        first, second = run(99), run(99)
        self.assertEqual(first["variants"]["a"]["ci_low"],
                         second["variants"]["a"]["ci_low"])
        self.assertEqual(first["variants"]["a"]["ci_high"],
                         second["variants"]["a"]["ci_high"])

    def test_the_interval_brackets_the_observed_rate(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=8, chat=noisy(3), personas=ONE_PERSONA, seed=5)
        row = result["variants"]["a"]
        self.assertLessEqual(row["ci_low"], row["win_rate"])
        self.assertLessEqual(row["win_rate"], row["ci_high"])


class TooFewTrialsIsSaidOutLoud(unittest.TestCase):
    def test_underpowered_is_reported_with_the_trials_that_would_be_needed(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=1, chat=always_first(), personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["verdict"], "no detectable difference")
        self.assertTrue(result["underpowered"])
        self.assertGreater(result["variants"]["a"]["trials_needed"],
                           result["variants"]["a"]["comparisons"])
        self.assertIn("would be needed", result["reason"])

    def test_too_few_comparisons_cannot_be_significant_however_clean_they_look(self):
        """Two unanimous comparisons are not evidence, they are two comparisons.

        Power is a precondition for significance, not a caveat printed under
        one. Without that rule a tiny sample of lucky agreements produces a
        tight interval and a confident verdict.
        """
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=2, chat=prefers_marker("A-MARKER"), personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["variants"]["a"]["win_rate"], 1.0)
        self.assertTrue(result["variants"]["a"]["excludes_null"])
        self.assertFalse(result["variants"]["a"]["powered"])
        self.assertEqual(result["verdict"], "no detectable difference")
        self.assertTrue(result["underpowered"])

    def test_a_detected_effect_is_not_reported_as_underpowered(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=10, chat=prefers_marker("A-MARKER"), personas=ONE_PERSONA, seed=1)
        self.assertFalse(result["underpowered"])


class TheCostGuardRunsBeforeTheCalls(unittest.TestCase):
    def test_estimate_counts_pairs_times_trials_times_orderings_times_personas(self):
        # 3 variants -> 3 pairs; 3 pairs x 10 trials x 2 orderings x 3 personas.
        self.assertEqual(ab_harness.estimate_cost({"a": "", "b": "", "c": ""}, 10, 3), 180)
        self.assertEqual(ab_harness.estimate_cost({"a": "", "b": ""}, 10, ONE_PERSONA), 20)
        self.assertEqual(ab_harness.estimate_cost({"a": ""}, 10, 3), 0)

    def test_a_plan_over_the_limit_returns_an_error_and_makes_no_calls(self):
        grader = prefers_marker("A-MARKER")
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B, "c": RESUME}, JOB, REQUIREMENTS,
            FLOOR, trials=50, chat=grader, personas=ONE_PERSONA, max_calls=10)
        self.assertTrue(result["error"])
        self.assertEqual(result["verdict"], "not run")
        self.assertEqual(result["planned_calls"], 300)
        self.assertEqual(grader.calls, [])

    def test_it_returns_the_error_rather_than_raising(self):
        try:
            result = ab_harness.compare_variants(
                {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
                trials=1000, chat=prefers_marker("A-MARKER"),
                personas=ONE_PERSONA, max_calls=5)
        except Exception as exc:                    # pragma: no cover - the bug
            self.fail(f"cost guard raised instead of returning: {exc}")
        self.assertIn("max_calls", result["error"])

    def test_a_plan_within_the_limit_runs(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=2, chat=prefers_marker("A-MARKER"), personas=ONE_PERSONA,
            max_calls=4)
        self.assertEqual(result["error"], "")
        self.assertEqual(result["calls"], 4)


class TheGuardCannotBeOmitted(unittest.TestCase):
    def test_the_noise_floor_is_a_required_argument(self):
        import inspect
        floor_param = inspect.signature(ab_harness.compare_variants).parameters["floor"]
        self.assertIs(floor_param.default, inspect.Parameter.empty)

    def test_a_missing_floor_is_refused_rather_than_assumed_to_be_zero(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, {},
            trials=2, chat=prefers_marker("A-MARKER"), personas=ONE_PERSONA)
        self.assertEqual(result["verdict"], "not run")
        self.assertIn("noise_floor", result["error"])

    def test_fewer_than_two_variants_is_refused(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A}, JOB, REQUIREMENTS, FLOOR, trials=2,
            chat=prefers_marker("A-MARKER"), personas=ONE_PERSONA)
        self.assertEqual(result["verdict"], "not run")


class AFailedGraderLosesOneTrialNotTheRun(unittest.TestCase):
    def test_a_mid_run_failure_is_recorded_and_the_other_trials_survive(self):
        # 5 trials x 2 orderings = 10 calls; kill the third comparison only.
        grader = ChoiceGrader(lambda first, second: "1" if "A-MARKER" in first else "2",
                              raises_on=(5, 6))
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLAT_FLOOR,
            trials=5, chat=grader, personas=ONE_PERSONA, seed=1)
        self.assertEqual(len(grader.calls), 10)
        self.assertEqual(result["pairs"][0]["a_wins"], 4)
        self.assertEqual(result["pairs"][0]["no_preference"], 1)
        self.assertEqual(len(result["failures"]), 2)
        self.assertIn("raised", result["failures"][0])
        # The surviving four trials still carry the comparison.
        self.assertEqual(result["variants"]["a"]["win_rate"], 0.9)
        self.assertEqual(result["variants"]["a"]["comparisons"], 5)

    def test_a_grader_that_fails_every_call_reports_no_difference(self):
        grader = ChoiceGrader(lambda first, second: "malformed")
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=4, chat=grader, personas=ONE_PERSONA, seed=1)
        self.assertEqual(result["verdict"], "no detectable difference")
        self.assertEqual(len(result["failures"]), 8)


class Summary(unittest.TestCase):
    def test_it_reports_the_verdict_and_the_intervals(self):
        result = ab_harness.compare_variants(
            {"a": VARIANT_A, "b": VARIANT_B}, JOB, REQUIREMENTS, FLOOR,
            trials=4, chat=prefers_marker("A-MARKER"), personas=ONE_PERSONA, seed=1)
        summary = ab_harness.summarise(result)
        self.assertEqual(summary["verdict"], "significant")
        self.assertEqual(summary["calls"], 8)
        self.assertEqual(len(summary["rates"]["a"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
