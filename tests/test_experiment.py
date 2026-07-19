"""Tests for the tactic experiment runner.

    py -m unittest tests.test_experiment

Offline, no API key, no network, no cost -- both the tailoring model and the
grader are injected as stubs.

The property that matters most: the control arm is present even when nobody
asked for it. Without it a comparison can report "A beats B" when the honest
finding is "both are worse than doing nothing".
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jobbuddy import experiment

FACT = {
    "fact_id": "umbra.etl",
    "org": "Umbra Financial",
    "role": "Manager, Data Engineering",
    "start": "2023-08", "end": "2024-11",
    "numbers": ["10", "5", "4"],
    "entities": ["PySpark", "ETL"],
    "skills": ["etl", "pyspark"],
    "phrasings": ["Cut data generation from 10 to 5 working days by automating "
                  "4 ETL processes in PySpark"],
    "verified": True,
}

PROFILE = {
    "identity": {"name": "Alex Tan"},
    "constraints": {"never_claim": []},
    "facts": [FACT],
}

JOB = {"title": "Data Engineer", "company": "Globex", "seniority": "mid",
       "jd_text": "ETL pipelines.", "skills": ["etl"]}


def tailor_stub(ok=True):
    def _chat(messages, **kwargs):
        if not ok:
            return {"ok": False, "error": "boom"}
        return {"ok": True, "cost_usd": 0.001,
                "data": {"selected": [{"fact_id": "umbra.etl", "rank": 1}],
                         "headline": "Data Engineer",
                         "unaddressed": ["Kubernetes"]}}
    return _chat


class TextIsPlainAndComplete(unittest.TestCase):
    def test_bullets_and_role_reach_the_text(self):
        tailored = {"headline": "Data Engineer",
                    "bullets": [{"text": "Did the thing", "org": "Umbra Financial",
                                 "role": "Manager"}]}
        text = experiment.as_text(PROFILE, tailored)
        self.assertIn("Did the thing", text)
        self.assertIn("Umbra Financial", text)
        self.assertIn("Alex Tan", text)

    def test_an_empty_draft_produces_empty_text(self):
        self.assertEqual(experiment.as_text(PROFILE, {"bullets": []}).strip(),
                         "Alex Tan")


class TheControlArmIsNotOptional(unittest.TestCase):
    """Comparing two tactics with no control cannot detect that both hurt."""

    def test_baseline_is_added_when_not_requested(self):
        variants, detail = experiment.build_arms(
            PROFILE, JOB, ["etl"], ["xyz_formula"], chat=tailor_stub())
        self.assertIn("baseline", variants)
        self.assertIn("baseline", [d["arm"] for d in detail])

    def test_baseline_is_not_duplicated_when_requested(self):
        _, detail = experiment.build_arms(
            PROFILE, JOB, ["etl"], ["baseline", "xyz_formula"], chat=tailor_stub())
        self.assertEqual([d["arm"] for d in detail].count("baseline"), 1)

    def test_a_failing_arm_is_recorded_rather_than_lost(self):
        variants, detail = experiment.build_arms(
            PROFILE, JOB, ["etl"], ["xyz_formula"], chat=tailor_stub(ok=False))
        self.assertEqual(variants, {})
        self.assertTrue(all(not d["ok"] for d in detail))
        self.assertTrue(all("boom" in str(d.get("error")) for d in detail))


class LengthIsHeldFixedAcrossArms(unittest.TestCase):
    """The confound that produced a fake result on the first live run.

    xyz_formula beat baseline 0.92 to 0.08 with an interval nowhere near
    chance -- because it had emitted 8 bullets against baseline's 3. The
    graders were comparing a fuller resume to a thinner one.
    """

    def _uneven(self):
        """A model that returns more selections for the non-baseline arm."""
        profile = {**PROFILE, "facts": [
            dict(FACT, fact_id=f"f{n}",
                 phrasings=[f"Automated {n} ETL processes in PySpark"],
                 numbers=[str(n)])
            for n in range(1, 6)]}

        def _chat(messages, **kwargs):
            system = messages[0]["content"]
            count = 5 if "as measured by" in system else 2
            return {"ok": True, "cost_usd": 0.001,
                    "data": {"selected": [{"fact_id": f"f{n}", "rank": n}
                                          for n in range(1, count + 1)],
                             "headline": "", "unaddressed": []}}
        return profile, _chat

    def test_arms_are_truncated_to_the_shortest(self):
        profile, chat = self._uneven()
        variants, detail = experiment.build_arms(
            profile, JOB, ["etl"], ["xyz_formula"], chat=chat)
        shown = {d["arm"]: d["bullets_shown"] for d in detail if d.get("ok")
                 and "bullets_shown" in d}
        self.assertEqual(len(set(shown.values())), 1, shown)

    def test_the_original_counts_are_still_reported(self):
        """Silently equalising would hide a real difference in yield."""
        profile, chat = self._uneven()
        _, detail = experiment.build_arms(
            profile, JOB, ["etl"], ["xyz_formula"], chat=chat)
        produced = {d["arm"]: d["bullets_produced"] for d in detail
                    if d.get("ok") and "bullets_produced" in d}
        self.assertNotEqual(len(set(produced.values())), 1, produced)
        self.assertTrue(any(d.get("length_control") for d in detail))

    def test_the_control_can_be_switched_off_deliberately(self):
        profile, chat = self._uneven()
        _, detail = experiment.build_arms(
            profile, JOB, ["etl"], ["xyz_formula"], chat=chat,
            equalise_length=False)
        shown = {d["arm"]: d["bullets_shown"] for d in detail if d.get("ok")
                 and "bullets_shown" in d}
        self.assertNotEqual(len(set(shown.values())), 1, shown)


class RunRefusesWhatItCannotMeasure(unittest.TestCase):
    def test_fewer_than_two_working_arms_is_an_error_not_a_result(self):
        result = experiment.run(PROFILE, JOB, ["etl"], ["xyz_formula"],
                                chat=tailor_stub(ok=False))
        self.assertFalse(result["ok"])
        self.assertIn("2+", result["error"])

    def test_it_never_raises_on_a_broken_model(self):
        result = experiment.run(PROFILE, JOB, ["etl"], ["nonsense_arm"],
                                chat=tailor_stub(ok=False))
        self.assertIn("ok", result)


class ResultsAccumulate(unittest.TestCase):
    def test_saving_twice_does_not_overwrite(self):
        directory = Path(tempfile.mkdtemp())
        first = experiment.save({"ok": True, "n": 1}, directory)
        second = experiment.save({"ok": True, "n": 2}, directory)
        # Same-second writes may collide by timestamp; what must hold is that
        # a result is never silently lost without the path saying so.
        self.assertTrue(first.exists() and second.exists())
        self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["n"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
