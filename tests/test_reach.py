"""Knockouts and the reach component.

Two mechanisms with deliberately different consequences, tested separately:

  knockouts  hard filters. They EXCLUDE a job, and the failure mode that
             matters is a false positive -- a job wrongly hidden is one the
             candidate never learns about. So every knockout is tested in both
             directions, and the 'preferred' direction is the important one.
  reach      a soft score. It lowers a rank and hides nothing.
"""

from __future__ import annotations

import unittest

from jobbuddy import job_schema
from jobbuddy import scoring


PROFILE = {
    "target_seniority": "senior",
    "years_experience": 5,
    "current_salary_sgd_monthly": 8000,
    "skills": {
        "expert": ["python", "sql", "llm", "rag", "machine learning", "nlp"],
        "working": ["aws", "docker", "kubernetes", "langchain", "deep learning"],
        "familiar": ["pytorch", "tensorflow", "gcp"],
    },
}

CONFIG = {
    "filters": {"singapore_only": True, "open_only": True},
    "profile": PROFILE,
    "weights": {
        "skill_match": 21, "reach": 15, "seniority_fit": 15, "comp_signal": 13,
        "competition": 16, "company_signal": 10, "application_friction": 5,
        "freshness": 5,
    },
}


def job(jd_text: str = "", **overrides):
    record = job_schema.new_job("mcf", "test-1")
    record.update({
        "title": "Machine Learning Engineer",
        "company": "Acme Analytics",
        "url": "https://example.com/1",
        "jd_text": jd_text,
        "seniority": "senior",
        "seniority_basis": "title",
    })
    record.update(overrides)
    return job_schema.finalise(record)


def config_with(**knockouts):
    merged = {**CONFIG, "filters": {**CONFIG["filters"], "knockouts": knockouts}}
    return merged


class DoctorateKnockout(unittest.TestCase):
    """The candidate holds an MSc. A required PhD is not closeable."""

    def test_required_phd_excludes(self):
        reason = scoring.check_filters(
            job("Requirements: PhD in Machine Learning is required."), CONFIG)
        self.assertEqual(reason, "requires a doctorate")

    def test_doctorate_spelled_out_excludes(self):
        reason = scoring.check_filters(
            job("You must hold a doctorate in a quantitative field."), CONFIG)
        self.assertEqual(reason, "requires a doctorate")

    def test_preferred_phd_does_not_exclude(self):
        self.assertIsNone(scoring.check_filters(
            job("PhD preferred. Strong Python and ML background."), CONFIG))

    def test_phd_a_plus_does_not_exclude(self):
        self.assertIsNone(scoring.check_filters(
            job("A PhD is a plus but not required."), CONFIG))

    def test_phd_or_equivalent_experience_does_not_exclude(self):
        self.assertIsNone(scoring.check_filters(
            job("PhD or equivalent industry experience is required."), CONFIG))

    def test_phd_mentioned_with_no_verdict_does_not_exclude(self):
        """Silence keeps the job. Ambiguity must never hide anything."""
        self.assertIsNone(scoring.check_filters(
            job("Our team includes PhDs from NUS and NTU."), CONFIG))

    def test_required_elsewhere_in_the_document_does_not_exclude(self):
        """'required' three paragraphs away is not this sentence's verdict."""
        self.assertIsNone(scoring.check_filters(job(
            "Several of our researchers hold a PhD.\n"
            "A valid Singapore work pass is required.\n"
            "Requirements: 4 years of Python."), CONFIG))

    def test_the_knockout_can_be_switched_off(self):
        self.assertIsNone(scoring.check_filters(
            job("PhD required."), config_with(doctorate=False)))

    def test_disabling_all_knockouts_switches_off_the_section(self):
        self.assertIsNone(scoring.check_filters(
            job("PhD required."), config_with(enabled=False)))


class LicenceKnockout(unittest.TestCase):

    def test_required_bar_admission_excludes(self):
        reason = scoring.check_filters(
            job("Candidates must be admitted to the bar in Singapore."), CONFIG)
        self.assertIn("professional licence", reason or "")
        self.assertIn("bar admission", reason or "")

    def test_required_cfa_charter_excludes(self):
        reason = scoring.check_filters(
            job("A CFA charter is required for this role."), CONFIG)
        self.assertIn("CFA charter", reason or "")

    def test_preferred_cfa_does_not_exclude(self):
        self.assertIsNone(scoring.check_filters(
            job("CFA charter preferred."), CONFIG))

    def test_a_closeable_certification_is_not_a_knockout(self):
        """AWS and CISSP are things a working engineer can go and sit."""
        self.assertIsNone(scoring.check_filters(
            job("AWS Solutions Architect certification is required."), CONFIG))


class DegreeFieldKnockout(unittest.TestCase):

    def test_required_unrelated_degree_excludes(self):
        reason = scoring.check_filters(
            job("Requirements: Bachelor's degree in Chemistry."), CONFIG)
        self.assertIn("degree in a field you do not hold", reason or "")

    def test_a_held_field_does_not_exclude(self):
        self.assertIsNone(scoring.check_filters(
            job("Requirements: Bachelor's degree in Computer Science."), CONFIG))

    def test_or_related_field_never_excludes(self):
        self.assertIsNone(scoring.check_filters(
            job("Required: Master's degree in Physics or a related field."), CONFIG))

    def test_a_held_field_later_in_the_list_does_not_exclude(self):
        """The first item in a list is not the whole requirement."""
        self.assertIsNone(scoring.check_filters(
            job("Required: degree in Physics, Computer Science or Statistics."),
            CONFIG))

    def test_preferred_unrelated_degree_does_not_exclude(self):
        self.assertIsNone(scoring.check_filters(
            job("A degree in Chemistry is preferred."), CONFIG))


class YearsKnockout(unittest.TestCase):
    """Default ceiling is 2.0 x 5 years, so more than 10 is out."""

    def test_a_demand_far_above_the_profile_excludes(self):
        reason = scoring.check_filters(
            job("Requirements: minimum 15 years of experience in ML."), CONFIG)
        self.assertIn("demands 15y", reason or "")

    def test_the_boards_stated_minimum_is_read(self):
        reason = scoring.check_filters(job("", min_years_exp=12), CONFIG)
        self.assertIn("demands 12y", reason or "")

    def test_a_demand_at_the_profile_level_does_not_exclude(self):
        self.assertIsNone(scoring.check_filters(
            job("Requirements: at least 5 years of experience."), CONFIG))

    def test_a_demand_just_under_the_ceiling_does_not_exclude(self):
        self.assertIsNone(scoring.check_filters(
            job("Requirements: minimum 9 years of experience."), CONFIG))

    def test_the_smallest_demanded_figure_wins(self):
        """A preferred 12y line must not knock out a job asking for 5y."""
        self.assertIsNone(scoring.check_filters(job(
            "Requirements: minimum 5 years of Python.\n"
            "12+ years of leadership experience preferred."), CONFIG))

    def test_two_figures_in_one_required_sentence_take_the_smaller(self):
        """The entry bar is the smallest thing they insist on, not the largest.

        Guards the min/max choice directly: 'at least 5 years in ML and 12
        years overall' is one required sentence, so the _demands filter cannot
        do this job -- only taking the minimum can.
        """
        self.assertIsNone(scoring.check_filters(job(
            "Requirements: at least 5 years in ML and 12 years of overall "
            "industry experience."), CONFIG))

    def test_a_low_jd_figure_beats_an_inflated_board_minimum(self):
        """MCF's min_years_exp field is frequently wrong. It must not be the
        only voice: a description asking for 4 years keeps the job."""
        self.assertIsNone(scoring.check_filters(
            job("Requirements: at least 4 years of Python.", min_years_exp=15),
            CONFIG))

    def test_the_multiple_is_configurable(self):
        tight = config_with(years_experience_multiple=1.5)
        self.assertIsNotNone(scoring.check_filters(
            job("Requirements: minimum 9 years of experience."), tight))

    def test_null_multiple_switches_the_years_knockout_off(self):
        loose = config_with(years_experience_multiple=None)
        self.assertIsNone(scoring.check_filters(
            job("Requirements: minimum 20 years of experience."), loose))


class KnockoutsAreVisible(unittest.TestCase):
    """Rule one: never silently hide a job."""

    def test_every_knockout_produces_a_non_empty_reason(self):
        for text in ("PhD required.",
                     "Must be admitted to the bar.",
                     "Requirements: Bachelor's degree in Chemistry.",
                     "Requirements: minimum 15 years of experience."):
            with self.subTest(text=text):
                reason = scoring.check_filters(job(text), CONFIG)
                self.assertTrue(reason and reason.strip())

    def test_reasons_are_stable_phrases_so_the_summary_groups_them(self):
        first = scoring.check_filters(
            job("PhD in Physics required."), CONFIG)
        second = scoring.check_filters(
            job("A PhD is required for this position.", source_job_id="x"), CONFIG)
        self.assertEqual(first, second)

    def test_the_reason_reaches_the_pipeline_exclusion_summary(self):
        from jobbuddy import pipeline
        result = pipeline.RunResult(run_id="r", jobs=[])
        bad = job("PhD required.")
        result.excluded.append({
            "job_key": bad["job_key"], "title": bad["title"],
            "company": bad["company"],
            "reason": scoring.check_filters(bad, CONFIG),
        })
        self.assertEqual(result.exclusion_reasons(), {"requires a doctorate": 1})

    def test_a_broken_config_keeps_the_job_rather_than_hiding_it(self):
        broken = {**CONFIG,
                  "filters": {"knockouts": {"degree_fields_held": "not a list"}}}
        self.assertIsNone(scoring.check_knockouts(job("PhD is nice."), broken))


class Reach(unittest.TestCase):

    def _reach(self, record, profile=PROFILE):
        ctx = scoring.ScoreContext(config=CONFIG)
        return scoring.score_reach(record, profile, ctx)

    def test_a_job_the_candidate_fits_scores_high(self):
        fit = job(skills_raw=["Python", "Machine Learning", "SQL", "NLP"],
                  skills_weight={"Python": 1.0, "Machine Learning": 1.0,
                                 "SQL": 1.0, "NLP": 1.0})
        value, detail = self._reach(fit)
        self.assertIsNotNone(value)
        self.assertGreaterEqual(value, 99.0)
        self.assertEqual(detail["verdict"], "within reach")

    def test_a_job_the_candidate_mostly_misses_scores_low(self):
        stretch = job(skills_raw=["Rust", "Verilog", "FPGA", "CUDA kernels",
                                  "Python"],
                      skills_weight={"Rust": 1.0, "Verilog": 1.0, "FPGA": 1.0,
                                     "CUDA kernels": 1.0, "Python": 1.0})
        value, detail = self._reach(stretch)
        self.assertIsNotNone(value)
        self.assertLess(value, 20.0)
        self.assertTrue(detail["missing_core"])

    def test_reach_is_none_when_the_job_states_no_requirements(self):
        value, detail = self._reach(job())
        self.assertIsNone(value)
        self.assertTrue(detail["reason"])

    def test_reach_is_none_when_every_requirement_is_optional(self):
        wishes = job(skills_raw=["Rust", "Verilog"],
                     skills_weight={"Rust": 0.25, "Verilog": 0.25})
        value, _ = self._reach(wishes)
        self.assertIsNone(value)

    def test_an_unmeasurable_reach_does_not_change_the_total(self):
        """The documented bug this guards: imputing 50 for a missing component.

        A job with no stated requirements must be scored on the other seven
        weights alone, at the same total as a config that never had reach.
        """
        blank = job(applications=4, age_days=3, salary_min_sgd=10000,
                    salary_max_sgd=14000, salary_is_stated=True)
        with_reach = scoring.score_job(dict(blank), CONFIG)
        without = {k: v for k, v in CONFIG["weights"].items() if k != "reach"}
        no_reach = scoring.score_job(dict(blank), {**CONFIG, "weights": without})

        self.assertIsNone(with_reach["components"]["reach"]["value"])
        self.assertEqual(with_reach["total"], no_reach["total"])
        self.assertEqual(with_reach["weight_used"], no_reach["weight_used"])

    def test_reach_reuses_skill_match_rather_than_recomputing_coverage(self):
        record = job(skills_raw=["Python", "Rust"],
                     skills_weight={"Python": 1.0, "Rust": 1.0})
        _, match_detail = scoring.score_skill_match(record, PROFILE)
        _, reach_detail = self._reach(record)
        self.assertEqual(reach_detail["core_coverage"], match_detail["core_score"])

    def test_thresholds_come_from_config(self):
        record = job(skills_raw=["Python", "Rust", "Verilog", "FPGA"],
                     skills_weight={"Python": 1.0, "Rust": 1.0,
                                    "Verilog": 1.0, "FPGA": 1.0})
        strict = {**CONFIG, "weights": {**CONFIG["weights"],
                                        "_reach_out_of_depth_coverage": 40}}
        lenient = {**CONFIG, "weights": {**CONFIG["weights"],
                                         "_reach_out_of_depth_coverage": 0,
                                         "_reach_comfortable_coverage": 90}}
        strict_value, _ = scoring.score_reach(
            record, PROFILE, scoring.ScoreContext(config=strict))
        lenient_value, _ = scoring.score_reach(
            record, PROFILE, scoring.ScoreContext(config=lenient))
        self.assertLess(strict_value, lenient_value)

    def test_inverted_thresholds_fall_back_instead_of_inverting_the_ranking(self):
        record = job(skills_raw=["Python", "Rust"],
                     skills_weight={"Python": 1.0, "Rust": 1.0})
        silly = {**CONFIG, "weights": {**CONFIG["weights"],
                                       "_reach_out_of_depth_coverage": 90,
                                       "_reach_comfortable_coverage": 10}}
        value, detail = scoring.score_reach(
            record, PROFILE, scoring.ScoreContext(config=silly))
        self.assertEqual(detail["comfortable_coverage"],
                         scoring.REACH_COMFORTABLE_COVERAGE)
        self.assertEqual(detail["out_of_depth_coverage"],
                         scoring.REACH_OUT_OF_DEPTH_COVERAGE)
        self.assertIsNotNone(value)

    def test_reach_never_raises_on_a_profile_with_no_skills(self):
        value, detail = self._reach(job(skills_raw=["Python"]), profile={})
        self.assertIsNone(value)
        self.assertTrue(detail["reason"])


class PreferredCredentials(unittest.TestCase):
    """A credential the posting wants and the profile lacks costs a little.

    Not a knockout -- those roles do hire strong MSc candidates -- but not free
    either. The direction that matters most is the second class of test: a
    credential the candidate HOLDS must cost exactly nothing, because a
    deduction there would be telling them they are short of something they have.
    """

    FIT = dict(
        skills_raw=["Python", "Machine Learning", "SQL", "NLP"],
        skills_weight={"Python": 1.0, "Machine Learning": 1.0,
                       "SQL": 1.0, "NLP": 1.0})

    def _reach(self, jd_text, profile=PROFILE, config=CONFIG):
        record = job(jd_text, **self.FIT)
        return scoring.score_reach(
            record, profile, scoring.ScoreContext(config=config))

    def test_a_preferred_doctorate_reduces_reach(self):
        plain, _ = self._reach("Requirements: Python and ML.")
        gated, detail = self._reach("Requirements: Python and ML. PhD preferred.")
        self.assertLess(gated, plain)
        self.assertEqual(detail["credentials_wanted_not_held"], ["doctorate"])

    def test_the_deduction_is_the_configured_size(self):
        plain, _ = self._reach("Requirements: Python and ML.")
        gated, detail = self._reach("Requirements: Python and ML. PhD preferred.")
        self.assertAlmostEqual(gated, plain * (1 - 0.12), places=6)
        self.assertEqual(detail["credential_penalty"], 0.12)

    def test_a_preferred_credential_the_profile_holds_costs_nothing(self):
        plain, _ = self._reach("Requirements: Python and ML.")
        for text in ("Master's degree preferred.", "An MSc is a plus.",
                     "Bachelor's degree in Computer Science preferred."):
            with self.subTest(text=text):
                value, detail = self._reach(f"Requirements: Python and ML. {text}")
                self.assertEqual(value, plain)
                self.assertEqual(detail["credentials_wanted_not_held"], [])
                self.assertEqual(detail["credential_penalty"], 0.0)

    def test_a_credential_merely_mentioned_costs_nothing(self):
        """Describing the team is not asking for a qualification.

        Both forms, because 'PhDs' once slipped past the detector entirely on
        the trailing s -- which made this test pass for the wrong reason.
        """
        plain, _ = self._reach("Requirements: Python and ML.")
        for text in ("Our team includes PhDs from NUS.",
                     "Your manager holds a PhD in statistics.",
                     "We are a research group with several doctorates."):
            with self.subTest(text=text):
                value, detail = self._reach(f"Requirements: Python and ML.\n{text}")
                self.assertEqual(value, plain)
                self.assertEqual(detail["credentials_wanted_not_held"], [])

    def test_the_plural_form_is_detected(self):
        """Guards the regex directly: 'PhDs preferred' must cost the same as
        'PhD preferred', not slip through unpenalised."""
        singular, _ = self._reach("Requirements: Python and ML. PhD preferred.")
        plural, detail = self._reach(
            "Requirements: Python and ML. PhDs are preferred.")
        self.assertEqual(plural, singular)
        self.assertEqual(detail["credentials_wanted_not_held"], ["doctorate"])

    def test_a_required_plural_doctorate_still_knocks_out(self):
        self.assertEqual(
            scoring.check_filters(job("Requirements: PhDs required."), CONFIG),
            "requires a doctorate")

    def test_a_preferred_licence_is_counted(self):
        _, detail = self._reach(
            "Requirements: Python. A CFA charter is preferred.")
        self.assertEqual(detail["credentials_wanted_not_held"], ["CFA charter"])

    def test_two_credentials_are_deducted_once_not_twice(self):
        one, _ = self._reach("Requirements: Python and ML. PhD preferred.")
        two, detail = self._reach(
            "Requirements: Python and ML. PhD preferred. CFA charter a plus.")
        self.assertEqual(one, two)
        self.assertEqual(len(detail["credentials_wanted_not_held"]), 2)

    def test_the_penalty_is_surfaced_in_the_explanation(self):
        _, detail = self._reach("Requirements: Python and ML. PhD preferred.")
        self.assertIn("doctorate", detail["explanation"])
        self.assertIn("12%", detail["explanation"])
        self.assertIn("credential", detail["verdict"])

    def test_the_pre_deduction_score_is_reported_too(self):
        _, detail = self._reach("Requirements: Python and ML. PhD preferred.")
        self.assertEqual(detail["coverage_score"], 100.0)

    def test_the_penalty_size_is_configurable(self):
        heavy = {**CONFIG, "weights": {
            **CONFIG["weights"], "_reach_preferred_credential_penalty": 0.4}}
        off = {**CONFIG, "weights": {
            **CONFIG["weights"], "_reach_preferred_credential_penalty": 0}}
        text = "Requirements: Python and ML. PhD preferred."
        plain, _ = self._reach("Requirements: Python and ML.")
        self.assertAlmostEqual(self._reach(text, config=heavy)[0],
                               plain * 0.6, places=6)
        self.assertEqual(self._reach(text, config=off)[0], plain)

    def test_an_absurd_configured_penalty_is_clamped(self):
        silly = {**CONFIG, "weights": {
            **CONFIG["weights"], "_reach_preferred_credential_penalty": 5.0}}
        plain, _ = self._reach("Requirements: Python and ML.")
        value, _ = self._reach(
            "Requirements: Python and ML. PhD preferred.", config=silly)
        self.assertAlmostEqual(
            value, plain * (1 - scoring.MAX_CREDENTIAL_PENALTY), places=6)

    def test_credentials_held_comes_from_the_profile(self):
        doctor = {**PROFILE, "credentials_held": ["master's degree",
                                                  "bachelor's degree",
                                                  "doctorate"]}
        plain, _ = self._reach("Requirements: Python and ML.")
        value, detail = self._reach(
            "Requirements: Python and ML. PhD preferred.", profile=doctor)
        self.assertEqual(value, plain)
        self.assertEqual(detail["credentials_wanted_not_held"], [])

    def test_a_required_credential_still_counts_when_its_knockout_is_off(self):
        """Choosing to see PhD-required roles is not choosing to rank them
        alongside roles with no credential gap at all."""
        lenient = config_with(doctorate=False)
        record = job("Requirements: Python and ML. PhD required.", **self.FIT)
        self.assertIsNone(scoring.check_filters(record, lenient))
        value, detail = scoring.score_reach(
            record, PROFILE, scoring.ScoreContext(config=lenient))
        self.assertEqual(detail["credentials_wanted_not_held"], ["doctorate"])
        self.assertLess(value, 100.0)

    def test_a_malformed_credentials_held_does_not_deduct_wrongly(self):
        broken = {**PROFILE, "credentials_held": "master's degree"}
        value, detail = self._reach(
            "Requirements: Python and ML. Master's preferred.", profile=broken)
        self.assertEqual(detail["credentials_wanted_not_held"], [])
        self.assertEqual(value, 100.0)


class Ranking(unittest.TestCase):
    """The weights still order jobs the way the user asked them to."""

    @staticmethod
    def _posting(skills, weights, **extra):
        return job(
            skills_raw=skills, skills_weight=weights,
            applications=8, age_days=5, vacancies=1,
            salary_min_sgd=10000, salary_max_sgd=13000, salary_is_stated=True,
            **extra)

    def test_a_well_matched_job_outranks_a_stretch_job(self):
        fit = self._posting(
            ["Python", "Machine Learning", "SQL", "NLP", "AWS"],
            {"Python": 4.0, "Machine Learning": 4.0, "SQL": 1.0,
             "NLP": 1.0, "AWS": 1.0})
        stretch = self._posting(
            ["Rust", "Verilog", "FPGA", "CUDA kernels", "Python"],
            {"Rust": 4.0, "Verilog": 1.0, "FPGA": 1.0,
             "CUDA kernels": 1.0, "Python": 1.0},
            source_job_id="test-2")

        fit_scores = scoring.score_job(fit, CONFIG)
        stretch_scores = scoring.score_job(stretch, CONFIG)
        self.assertGreater(fit_scores["total"], stretch_scores["total"])
        self.assertGreater(fit_scores["adjusted"], stretch_scores["adjusted"])

    def test_reach_widens_the_gap_rather_than_merely_repeating_skill_match(self):
        """Two jobs with the same overall skill coverage, differing only in
        whether the misses are mandatory or wishes. skill_match cannot tell
        them apart; reach is the component that can."""
        mandatory_misses = self._posting(
            ["Rust", "Verilog", "Python", "SQL"],
            {"Rust": 1.0, "Verilog": 1.0, "Python": 1.0, "SQL": 1.0})
        optional_misses = self._posting(
            ["Rust", "Verilog", "Python", "SQL"],
            {"Rust": 0.25, "Verilog": 0.25, "Python": 1.0, "SQL": 1.0},
            source_job_id="test-3")

        hard = scoring.score_job(mandatory_misses, CONFIG)
        soft = scoring.score_job(optional_misses, CONFIG)
        self.assertLess(hard["components"]["reach"]["value"],
                        soft["components"]["reach"]["value"])
        self.assertLess(hard["total"], soft["total"])

    def test_a_credential_penalty_cannot_flip_a_fit_below_a_stretch(self):
        """The bound that makes the deduction safe to ship.

        A job the candidate fits which merely prefers a doctorate must still
        rank clearly above one they cannot do -- otherwise a soft signal is
        overruling a capability gap, which is the wrong way round.
        """
        gated_fit = self._posting(
            ["Python", "Machine Learning", "SQL", "NLP", "AWS"],
            {"Python": 4.0, "Machine Learning": 4.0, "SQL": 1.0,
             "NLP": 1.0, "AWS": 1.0},
            jd_text="Requirements: 5 years of ML. PhD preferred.")
        stretch = self._posting(
            ["Rust", "Verilog", "FPGA", "CUDA kernels", "Python"],
            {"Rust": 4.0, "Verilog": 1.0, "FPGA": 1.0,
             "CUDA kernels": 1.0, "Python": 1.0},
            source_job_id="test-4")

        gated = scoring.score_job(gated_fit, CONFIG)
        weak = scoring.score_job(stretch, CONFIG)
        self.assertGreater(gated["total"], weak["total"])
        self.assertGreater(gated["components"]["reach"]["value"],
                           weak["components"]["reach"]["value"])

    def test_the_credential_deduction_is_worth_under_two_points_of_total(self):
        """States the size claim the config comment makes, so a later reweight
        that quietly makes this a major signal fails here."""
        skills = (["Python", "Machine Learning", "SQL", "NLP", "AWS"],
                  {"Python": 4.0, "Machine Learning": 4.0, "SQL": 1.0,
                   "NLP": 1.0, "AWS": 1.0})
        plain = scoring.score_job(
            self._posting(*skills, jd_text="Requirements: 5 years of ML."),
            CONFIG)
        gated = scoring.score_job(
            self._posting(*skills, source_job_id="test-5",
                          jd_text="Requirements: 5 years of ML. PhD preferred."),
            CONFIG)
        drop = plain["total"] - gated["total"]
        self.assertGreater(drop, 0.0)
        # 12% of reach is 1.8 points at the full weight of 100, and up to ~2.0
        # once an unmeasurable component (here company_signal) renormalises
        # reach's share upward. Both are far below the fit-vs-stretch gap.
        self.assertLessEqual(drop, 2.5)

    def test_reach_appears_in_the_explanation_when_it_is_the_weak_point(self):
        stretch = self._posting(
            ["Rust", "Verilog", "FPGA", "CUDA kernels"],
            {"Rust": 1.0, "Verilog": 1.0, "FPGA": 1.0, "CUDA kernels": 1.0})
        scores = scoring.score_job(stretch, CONFIG)
        self.assertIn("reach", scores["explanation"])


if __name__ == "__main__":
    unittest.main()


class CoreMeansMandatoryNotMerelyMentioned(unittest.TestCase):
    """`core_score` is read by `score_reach` as mandatory-requirement coverage.

    It was computed as `importance >= 1.0`, and `importance_of` returns 1.0 for
    any term merely present -- so every extracted term counted as core and the
    metric actually measured "coverage of everything the ad mentions". A
    stretch role and a comfortable one scored more alike than they should.
    """

    PROFILE = {"skills": {"expert": ["Python"], "working": [], "familiar": []}}

    def _job(self, **extra):
        return {"title": "Engineer", "skills_raw": ["Python", "Kubernetes"], **extra}

    def test_a_merely_listed_term_is_not_core(self):
        """No weights, no key flags, nothing in the title -- nothing is
        mandatory, so coverage is unknowable rather than 50%."""
        _, detail = scoring.score_skill_match(self._job(), self.PROFILE)
        self.assertIsNone(detail["core_score"])

    def test_a_source_flagged_key_skill_is_core(self):
        _, detail = scoring.score_skill_match(
            self._job(skills_key=["Kubernetes"]), self.PROFILE)
        self.assertIsNotNone(detail["core_score"])
        self.assertIn("Kubernetes", detail["missing_core"])

    def test_an_explicit_required_weight_is_core(self):
        _, detail = scoring.score_skill_match(
            self._job(skills_weight={"Kubernetes": 2.0}), self.PROFILE)
        self.assertIn("Kubernetes", detail["missing_core"])

    def test_an_explicit_wish_weight_is_not_core(self):
        """'Nice to have' is exactly what must not count as mandatory."""
        _, detail = scoring.score_skill_match(
            self._job(skills_weight={"Kubernetes": 0.25, "Python": 0.25}),
            self.PROFILE)
        self.assertIsNone(detail["core_score"])

    def test_a_skill_in_the_title_is_core(self):
        _, detail = scoring.score_skill_match(
            {"title": "Kubernetes Engineer", "skills_raw": ["Kubernetes"]},
            self.PROFILE)
        self.assertIn("Kubernetes", detail["missing_core"])

    def test_covering_the_mandatory_ones_scores_full_marks(self):
        """The guard against a definition so tight nothing ever qualifies."""
        _, detail = scoring.score_skill_match(
            self._job(skills_key=["Python"]), self.PROFILE)
        self.assertEqual(detail["core_score"], 100.0)
