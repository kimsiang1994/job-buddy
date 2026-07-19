"""Tests for the simulated HR panel.

    py -m unittest tests.test_hr_panel

Offline: no API key, no network, no cost. The grader is injected as `chat`, so
every path through the module runs against a stub that returns exactly the
malformed, fabricated or contradictory replies a real model produces on a bad
day. Nothing here tests what the LLM says -- it tests that the deterministic
code around it drops a fabricated quote, counts a panel's votes correctly,
refuses to report an uncalibrated score, and never raises.
"""

from __future__ import annotations

import inspect
import unittest

from jobbuddy import hr_panel
from jobbuddy import job_schema

RESUME = """Alex Tan
Manager, Data Engineering -- Umbra Financial, Aug 2023 to Nov 2024
- Cut data generation from 10 to 5 working days by automating 4 ETL processes
- Built PySpark pipelines replacing legacy SAS reporting
"""

REQUIREMENTS = [
    {"term": "PySpark", "required": True},
    {"term": "ETL", "required": True},
    {"term": "Kubernetes", "required": True},
    {"term": "Airflow", "required": False},
]

FACTS = {
    "umbra.etl": {
        "fact_id": "umbra.etl",
        "org": "Umbra Financial",
        "role": "Manager, Data Engineering",
        "skills": ["etl", "pyspark", "data governance"],
        "entities": ["Umbra", "SAS"],
        "verified": True,
    },
}


def make_job(job_id="1", title="Senior Data Engineer", seniority="senior",
             jd_text="Build and own data pipelines.", salary=(9000, 12000)):
    job = job_schema.new_job("mcf", job_id)
    job["title"] = title
    job["company"] = "Example Bank Pte Ltd"
    job["url"] = f"https://example.test/{job_id}"
    job["jd_text"] = jd_text
    job["seniority"] = seniority
    if salary:
        job["salary_min_sgd"], job["salary_max_sgd"] = salary
        job["salary_is_stated"] = True
    return job_schema.finalise(job)


JOB = make_job()


def verdict(decision="advance", score=70, reasons=(), missing=(), evidence=()):
    """One well-formed grader reply."""
    return {
        "decision": decision,
        "score": score,
        "reasons": list(reasons),
        "missing_requirements": list(missing),
        "evidence": [{"claim": c, "resume_span": s} for c, s in evidence],
    }


def _persona_of(system_prompt):
    if "You are a recruiter" in system_prompt:
        return "recruiter"
    if "You are the hiring manager" in system_prompt:
        return "hiring_manager"
    return "skeptic"


def _resume_of(user_prompt):
    return user_prompt.split("RESUME:\n", 1)[-1]


class StubGrader:
    """Stands in for `deepseek_client.json_chat`. Never touches the network.

    Replies can be keyed by persona or by a marker in the resume text, which is
    what the calibration tests need -- the same three personas must score three
    different resumes differently.
    """

    def __init__(self, default=None, by_persona=None, by_resume=None, raises=False):
        self.default = default if default is not None else verdict()
        self.by_persona = by_persona or {}
        self.by_resume = by_resume or {}
        self.raises = raises
        self.calls = []

    def __call__(self, messages, schema_keys=(), **kwargs):
        self.calls.append(messages)
        if self.raises:
            raise RuntimeError("connection reset")
        system, user = messages[0]["content"], messages[1]["content"]
        payload = self.by_persona.get(_persona_of(system))
        if payload is None:
            resume = _resume_of(user)
            for marker, value in self.by_resume.items():
                if marker in resume:
                    payload = value
                    break
        if payload is None:
            payload = self.default
        if payload == "malformed":
            return {"ok": False, "data": None, "error": "reply is not a JSON object"}
        return {"ok": True, "data": payload, "error": "", "repaired": False}


class EvidenceMustQuoteTheResume(unittest.TestCase):
    def test_a_real_quote_is_kept(self):
        grader = StubGrader(default=verdict(evidence=[
            ("automated ETL", "automating 4 ETL processes")]))
        result = hr_panel.screen(RESUME, JOB, REQUIREMENTS,
                                 hr_panel.build_personas(JOB, REQUIREMENTS)[0],
                                 chat=grader)
        self.assertTrue(result["ok"], result["error"])
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["fabricated_spans"], 0)

    def test_a_fabricated_quote_is_dropped_and_counted(self):
        """A grader inventing a quote has stopped reading and started composing."""
        grader = StubGrader(default=verdict(evidence=[
            ("automated ETL", "automating 4 ETL processes"),
            ("led a team of 12", "Led a team of 12 engineers at Google")]))
        result = hr_panel.screen(RESUME, JOB, REQUIREMENTS,
                                 hr_panel.build_personas(JOB, REQUIREMENTS)[0],
                                 chat=grader)
        self.assertEqual([e["claim"] for e in result["evidence"]], ["automated ETL"])
        self.assertEqual(result["fabricated_spans"], 1)
        self.assertEqual(result["dropped_evidence"][0]["claim"], "led a team of 12")

    def test_whitespace_differences_do_not_drop_a_true_quote(self):
        """Wrapping a real line is not fabrication; dropping it would be noise."""
        grader = StubGrader(default=verdict(evidence=[
            ("etl", "automating   4 ETL\n  processes")]))
        result = hr_panel.screen(RESUME, JOB, REQUIREMENTS,
                                 hr_panel.build_personas(JOB, REQUIREMENTS)[0],
                                 chat=grader)
        self.assertEqual(result["fabricated_spans"], 0)

    def test_fabricated_spans_are_totalled_across_the_panel(self):
        grader = StubGrader(default=verdict(evidence=[("x", "not in this resume")]))
        panel = hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)
        self.assertEqual(panel["fabricated_spans"], 3)


class TheGraderIsBlind(unittest.TestCase):
    def test_screen_has_no_parameter_that_could_carry_facts(self):
        """Structural, not advisory: there is nowhere to put the leak."""
        params = set(inspect.signature(hr_panel.screen).parameters)
        self.assertEqual(params & {"profile", "facts", "rationale", "selected"}, set())

    def test_the_prompt_contains_only_the_resume_and_the_job(self):
        grader = StubGrader()
        hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)
        blob = "\n".join(m["content"] for call in grader.calls for m in call)
        self.assertNotIn("fact_id", blob)
        self.assertNotIn("umbra.etl", blob)

    def test_no_persona_sees_another_personas_verdict(self):
        """Each grader gets the same inputs; nothing accumulates between them."""
        grader = StubGrader(by_persona={
            "recruiter": verdict("reject", 20, reasons=["title mismatch"]),
            "hiring_manager": verdict("advance", 90),
            "skeptic": verdict("advance", 88),
        })
        hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)
        users = {call[1]["content"] for call in grader.calls}
        self.assertEqual(len(users), 1, "the user message differed between personas")
        systems = "\n".join(call[0]["content"] for call in grader.calls)
        self.assertNotIn("title mismatch", systems)


class Consensus(unittest.TestCase):
    def test_two_rejects_outvote_one_advance(self):
        grader = StubGrader(by_persona={
            "recruiter": verdict("advance", 80),
            "hiring_manager": verdict("reject", 40),
            "skeptic": verdict("reject", 35),
        })
        panel = hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)
        self.assertEqual(panel["consensus"], "reject")

    def test_advance_requires_two_advances(self):
        grader = StubGrader(by_persona={
            "recruiter": verdict("advance", 80),
            "hiring_manager": verdict("borderline", 60),
            "skeptic": verdict("reject", 40),
        })
        panel = hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)
        self.assertEqual(panel["consensus"], "borderline")

    def test_two_advances_carry_it(self):
        grader = StubGrader(by_persona={
            "recruiter": verdict("advance", 80),
            "hiring_manager": verdict("advance", 75),
            "skeptic": verdict("reject", 40),
        })
        self.assertEqual(
            hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)["consensus"],
            "advance")

    def test_missing_requirements_are_unioned(self):
        grader = StubGrader(by_persona={
            "recruiter": verdict(missing=["Kubernetes"]),
            "hiring_manager": verdict(missing=["Kubernetes", "Airflow"]),
            "skeptic": verdict(missing=["Airflow"]),
        })
        panel = hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)
        self.assertEqual(sorted(panel["missing_requirements"]), ["Airflow", "Kubernetes"])


class Calibration(unittest.TestCase):
    """The panel is only worth reading once it has proved it can see."""

    TAILORED = RESUME + "\nTAILORED-MARKER\n"
    BASELINE = RESUME + "\nBASELINE-MARKER\n"
    CONTROL = "Licensed plumber. Installed 300 domestic boilers.\nCONTROL-MARKER\n"

    def _grader(self, tailored, baseline, control):
        return StubGrader(by_resume={
            "TAILORED-MARKER": verdict("advance", tailored),
            "BASELINE-MARKER": verdict("borderline", baseline),
            "CONTROL-MARKER": verdict("reject", control),
        })

    def test_correct_ordering_is_trustworthy(self):
        result = hr_panel.calibrate(
            self.TAILORED, self.BASELINE, self.CONTROL, JOB, REQUIREMENTS,
            chat=self._grader(82, 64, 18))
        self.assertTrue(result["trustworthy"], result["reason"])
        self.assertEqual([row["label"] for row in result["ordering"]],
                         ["tailored", "baseline", "control"])

    def test_a_tie_between_tailored_and_baseline_still_calibrates(self):
        result = hr_panel.calibrate(
            self.TAILORED, self.BASELINE, self.CONTROL, JOB, REQUIREMENTS,
            chat=self._grader(64, 64, 18))
        self.assertTrue(result["trustworthy"], result["reason"])

    def test_control_scoring_above_the_tailored_resume_fails_calibration(self):
        """A judge that likes the plumbing CV cannot grade the ML resume."""
        result = hr_panel.calibrate(
            self.TAILORED, self.BASELINE, self.CONTROL, JOB, REQUIREMENTS,
            chat=self._grader(60, 55, 88))
        self.assertFalse(result["trustworthy"])
        self.assertIn("control", result["reason"])

    def test_tailored_below_baseline_fails_calibration(self):
        result = hr_panel.calibrate(
            self.TAILORED, self.BASELINE, self.CONTROL, JOB, REQUIREMENTS,
            chat=self._grader(50, 70, 18))
        self.assertFalse(result["trustworthy"])
        self.assertIn("tailored", result["reason"])

    def test_a_panel_that_could_not_score_is_not_trustworthy(self):
        grader = StubGrader(by_resume={
            "TAILORED-MARKER": "malformed",
            "BASELINE-MARKER": verdict("borderline", 64),
            "CONTROL-MARKER": verdict("reject", 18),
        })
        result = hr_panel.calibrate(self.TAILORED, self.BASELINE, self.CONTROL,
                                    JOB, REQUIREMENTS, chat=grader)
        self.assertFalse(result["trustworthy"])
        self.assertIn("tailored", result["reason"])

    def test_an_uncalibrated_run_reports_trustworthy_as_unknown(self):
        """None is not True. An unqualified score must not read as a good one."""
        panel = hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=StubGrader())
        self.assertIsNone(panel["trustworthy"])
        self.assertIn("not calibrated", panel["reason"])

    def test_a_failed_calibration_travels_with_the_score(self):
        calibration = hr_panel.calibrate(
            self.TAILORED, self.BASELINE, self.CONTROL, JOB, REQUIREMENTS,
            chat=self._grader(60, 55, 88))
        panel = hr_panel.run_panel(self.TAILORED, JOB, REQUIREMENTS,
                                   chat=StubGrader(), calibration=calibration)
        self.assertFalse(panel["trustworthy"])
        self.assertIn("control", panel["reason"])


class PersonasComeFromTheJob(unittest.TestCase):
    def test_three_personas_with_distinct_lenses(self):
        personas = hr_panel.build_personas(JOB, REQUIREMENTS)
        self.assertEqual([p["name"] for p in personas], list(hr_panel.PERSONA_NAMES))
        self.assertEqual(len({p["lens"] for p in personas}), 3)
        self.assertFalse(personas[0]["reads_deeply"])

    def test_two_different_jobs_produce_different_rubrics(self):
        """A static prompt would grade a graduate and a principal role the same."""
        other = make_job("2", title="Graduate Analyst", seniority="junior",
                         jd_text="Support the reporting team.", salary=(3500, 4500))
        mine = hr_panel.build_personas(JOB, REQUIREMENTS)
        theirs = hr_panel.build_personas(other, [{"term": "Excel", "required": True}])
        for a, b in zip(mine, theirs):
            with self.subTest(persona=a["name"]):
                self.assertNotEqual(a["rubric"], b["rubric"])
        self.assertNotEqual(mine[1]["bar"], theirs[1]["bar"])

    def test_the_rubric_quotes_the_jobs_own_requirements(self):
        rubric = " ".join(hr_panel.build_personas(JOB, REQUIREMENTS)[0]["rubric"])
        self.assertIn("PySpark", rubric)
        self.assertIn("Kubernetes", rubric)

    def test_expected_scope_comes_from_the_salary_band_when_stated(self):
        with_pay = hr_panel.build_personas(JOB, REQUIREMENTS)[1]
        self.assertIsNotNone(with_pay["expected_scope"])
        without = hr_panel.build_personas(make_job("3", salary=None), REQUIREMENTS)[1]
        self.assertIsNone(without["expected_scope"])

    def test_a_job_with_no_requirements_still_builds_a_panel(self):
        personas = hr_panel.build_personas(make_job("4"), [])
        self.assertEqual(len(personas), 3)


class BadGraderRepliesDegradeRatherThanRaise(unittest.TestCase):
    """The read path cannot raise. A broken grader is a recorded failure."""

    def _screen(self, grader):
        return hr_panel.screen(RESUME, JOB, REQUIREMENTS,
                               hr_panel.build_personas(JOB, REQUIREMENTS)[0],
                               chat=grader)

    def test_unparseable_reply_is_recorded_not_raised(self):
        result = self._screen(StubGrader(default="malformed"))
        self.assertFalse(result["ok"])
        self.assertIn("JSON", result["error"])
        self.assertIsNone(result["score"])

    def test_an_invalid_decision_is_a_failure_not_a_guess(self):
        result = self._screen(StubGrader(default=verdict("maybe", 70)))
        self.assertFalse(result["ok"])
        self.assertIn("maybe", result["error"])

    def test_a_non_numeric_score_is_a_failure(self):
        result = self._screen(StubGrader(default=verdict("advance", "very good")))
        self.assertFalse(result["ok"])

    def test_a_transport_exception_is_caught(self):
        result = self._screen(StubGrader(raises=True))
        self.assertFalse(result["ok"])
        self.assertIn("raised", result["error"])

    def test_a_score_outside_the_range_is_clamped(self):
        result = self._screen(StubGrader(default=verdict("advance", 140)))
        self.assertEqual(result["score"], 100)

    def test_reasons_are_capped_at_three(self):
        result = self._screen(StubGrader(default=verdict(reasons=list("abcdef"))))
        self.assertEqual(len(result["reasons"]), hr_panel.MAX_REASONS)

    def test_a_whole_panel_of_failures_has_no_consensus_and_no_score(self):
        panel = hr_panel.run_panel(RESUME, JOB, REQUIREMENTS,
                                   chat=StubGrader(default="malformed"))
        self.assertEqual(panel["consensus"], "unknown")
        self.assertIsNone(panel["score"])
        self.assertEqual(len(panel["failures"]), 3)

    def test_one_failure_does_not_lose_the_other_two_votes(self):
        grader = StubGrader(by_persona={
            "recruiter": "malformed",
            "hiring_manager": verdict("reject", 40),
            "skeptic": verdict("reject", 35),
        })
        panel = hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)
        self.assertEqual(panel["consensus"], "reject")
        self.assertEqual(panel["failures"], ["recruiter"])
        self.assertEqual(panel["score"], 37.5)


class GapAttribution(unittest.TestCase):
    def test_a_requirement_covered_by_a_facts_skills_was_cut_not_missing(self):
        result = hr_panel.attribute_gaps(["PySpark", "Kubernetes"], FACTS)
        self.assertEqual([g["requirement"] for g in result["have_but_cut"]], ["PySpark"])
        self.assertEqual(result["have_but_cut"][0]["fact_id"], "umbra.etl")
        self.assertEqual(result["genuinely_lacking"], ["Kubernetes"])

    def test_an_entity_also_counts_as_coverage(self):
        result = hr_panel.attribute_gaps(["SAS"], FACTS)
        self.assertEqual(len(result["have_but_cut"]), 1)

    def test_a_job_title_noun_does_not_count_as_coverage(self):
        """'Manager' in the role field is not evidence of a management skill."""
        result = hr_panel.attribute_gaps(["Manager"], FACTS)
        self.assertEqual(result["genuinely_lacking"], ["Manager"])

    def test_it_accepts_a_list_of_facts_as_well_as_a_dict(self):
        result = hr_panel.attribute_gaps(["ETL"], list(FACTS.values()))
        self.assertEqual(len(result["have_but_cut"]), 1)

    def test_no_facts_means_everything_is_a_real_gap(self):
        result = hr_panel.attribute_gaps(["PySpark", "Kubernetes"], {})
        self.assertEqual(result["genuinely_lacking"], ["PySpark", "Kubernetes"])
        self.assertEqual(result["have_but_cut"], [])

    def test_it_runs_on_a_panels_own_missing_requirements(self):
        grader = StubGrader(default=verdict(missing=["Kubernetes", "PySpark"]))
        panel = hr_panel.run_panel(RESUME, JOB, REQUIREMENTS, chat=grader)
        result = hr_panel.attribute_gaps(panel["missing_requirements"], FACTS)
        self.assertEqual(result["genuinely_lacking"], ["Kubernetes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
