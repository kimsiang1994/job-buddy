"""Offline regression tests for the job pipeline.

    py test_pipeline.py

No network, no API key, no cost -- same contract as test_scrapers.py.

Every fixture in this file is either real data captured from a live API or the
exact shape that broke something once. Following the convention already set in
test_scrapers.py: **add to these fixtures, do not rewrite them.** The comments
say which bug each one is guarding, because a test whose purpose is forgotten
gets deleted the first time it becomes inconvenient.
"""

from __future__ import annotations

import os
import tempfile
import unittest

# Redirected here as well as in tests/__init__.py, because `unittest discover
# -s tests` (without `-t .`) treats this directory as the top level and never
# imports the package __init__. Under that invocation the suite wrote its
# fixtures into the real config/companies.json -- three duplicated lines are a
# cheap price for a fix that does not depend on how the suite is launched.
os.environ.setdefault(
    "JB_COMPANY_REGISTRY",
    os.path.join(tempfile.mkdtemp(prefix="jobbuddy-tests-"), "companies.json"))

import json
from jobbuddy import job_schema
from jobbuddy import job_store
from jobbuddy import scoring
from jobbuddy import skills_taxonomy
from jobbuddy import source_mcf
from jobbuddy import user_input

# Trimmed from a real api.mycareersfuture.gov.sg/v2/jobs response, 2026-07-19.
# Kept structurally faithful -- the nesting is what parsers trip over.
MCF_RECORD = {
    "uuid": "59501ac0a69b6c5cc91d7b8e9a23d276",
    "title": "Machine Learning Engineer",
    "description": "<p>Role: <b>ML Engineer</b></p><ul><li>10+ years experience</li></ul>",
    "status": {"id": 102, "jobStatus": "Open"},
    "salary": {"maximum": 8000, "minimum": 6500, "type": {"id": 4, "salaryType": "Monthly"}},
    "postedCompany": {"uen": "200613609M", "name": "GECO ASIA PTE. LTD."},
    "hiringCompany": None,
    "address": {"isOverseas": False, "overseasCountry": None, "postalCode": None,
                "districts": [{"id": 998, "location": "Islandwide", "region": "Islandwide"}]},
    "positionLevels": [{"id": 7, "position": "Professional"}],
    "employmentTypes": [{"id": 3, "employmentType": "Contract"}],
    "minimumYearsExperience": 10,
    "numberOfVacancies": 1,
    "ssocCode": "25190",
    "skills": [{"skill": "Machine Learning", "isKeySkill": False},
               {"skill": "PySpark", "isKeySkill": False}],
    "categories": [{"id": 21, "category": "Information Technology"}],
    "metadata": {
        "jobPostId": "MCF-2026-1078704",
        "deletedAt": None,
        "expiryDate": "2099-07-24",
        "newPostingDate": "2026-06-24",
        "originalPostingDate": "2026-06-24",
        "repostCount": 0,
        "editCount": 0,
        "totalNumberOfView": 155,
        "totalNumberJobApplication": 11,
        "isHideSalary": False,
        "isPostedOnBehalf": False,
        "jobDetailsUrl": "https://www.mycareersfuture.gov.sg/job/x",
    },
}


def _mcf(**overrides):
    """A copy of the fixture with a deep-ish override applied."""
    import copy

    record = copy.deepcopy(MCF_RECORD)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(record.get(key), dict):
            record[key].update(value)
        else:
            record[key] = value
    return record


# --- shared factories -------------------------------------------------------
# These are module-level on purpose. `_job` used to be a method on the Scoring
# class, and three other classes reached through it as
# `Scoring._job(Scoring(), ...)` -- an unbound call with a throwaway TestCase
# built only to satisfy `self`. That construct also fails to import on Python
# 3.10, an undeclared version dependency introduced by a workaround.

def make_job(**overrides):
    """A finalised canonical Job, ready to score."""
    job = job_schema.new_job("mcf", overrides.pop("source_job_id", "test"))
    job.update({
        "title": "Machine Learning Engineer", "company": "Acme Pte Ltd",
        "url": "https://example.com/j", "jd_text": "Build models.",
        "seniority": "senior", "seniority_basis": "title",
        "salary_min_sgd": 10000, "salary_max_sgd": 14000, "salary_is_stated": True,
        "skills_raw": ["Machine Learning", "Python", "AWS"],
        "applications": 5, "views": 100, "posted_at": "2026-07-15",
        "vacancies": 1, "is_open": True,
    })
    job.update(overrides)
    return job_schema.finalise(job)


def make_config(**profile_overrides):
    """A scoring config. Deep-copied so no test can corrupt another's fixture."""
    import copy

    config = copy.deepcopy(BASE_CONFIG)
    config["profile"].update(profile_overrides)
    return config


BASE_CONFIG = {
    "filters": {"singapore_only": True, "open_only": True,
                "min_salary_sgd_monthly": 8000, "allow_unstated_salary": True,
                "exclude_companies": ["tiktok"],
                "exclude_title_patterns": [r"\bintern(ship)?\b"]},
    "profile": {"target_seniority": "senior", "years_experience": 5,
                "skills": {"expert": ["python", "machine learning"],
                           "working": ["aws"]}},
    "weights": {"skill_match": 30, "seniority_fit": 15, "comp_signal": 15,
                "competition": 20, "company_signal": 10,
                "application_friction": 5, "freshness": 5},
}


def with_resume(text, **kwargs):
    """Build an IntakeProfile from resume text written to a temp file."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cv.txt"
        path.write_text(text, encoding="utf-8")
        return user_input.build_profile(resume_path=path, **kwargs)


class SeniorityResolution(unittest.TestCase):
    """MCF's positionLevels dropdown is unreliable and must not outrank the title.

    Live data showed 'Professional' on a role wanting 10 years, and 'Manager' on
    one wanting 0. Trusting it produced 'mid' for a lead role and 'manager' for
    a junior one. Title wins, then years, then the dropdown.
    """

    def test_title_beats_position_level(self):
        level, basis = job_schema.resolve_seniority(
            title="Principal Machine Learning Engineer",
            years=5,
            position_level="Manager",
        )
        self.assertEqual(level, "principal")
        self.assertEqual(basis, "title")

    def test_years_used_when_title_is_bare(self):
        level, basis = job_schema.resolve_seniority(
            title="Machine Learning Engineer", years=10, position_level="Professional"
        )
        self.assertEqual(level, "lead")
        self.assertEqual(basis, "years")

    def test_zero_years_is_unknown_not_junior(self):
        # MCF stores 0 when the employer left the field alone. Reading that as
        # "no experience needed" mislabels senior roles as junior.
        self.assertIsNone(job_schema.seniority_from_years(0))
        self.assertIsNone(job_schema.seniority_from_years(None))

    def test_falls_back_to_position_level_and_says_so(self):
        level, basis = job_schema.resolve_seniority(
            title="Machine Learning Engineer", years=0, position_level="Manager"
        )
        self.assertEqual(level, "manager")
        self.assertEqual(basis, "position_level")

    def test_senior_manager_resolves_to_manager_not_senior(self):
        level, _ = job_schema.resolve_seniority(title="Senior Engineering Manager")
        self.assertEqual(level, "manager")

    def test_graduate_is_junior(self):
        level, basis = job_schema.resolve_seniority(
            title="Machine Learning Engineer Graduate", years=0,
            position_level="Professional",
        )
        self.assertEqual(level, "junior")
        self.assertEqual(basis, "title")


class SkillMatching(unittest.TestCase):
    """MCF's skill extractor emits odd casings and outright noise.

    An AI engineering role scored 0/16 against an AI engineer's profile before
    this module existed: 'Ai' was dropped by a length guard, 'Fine Tuning' did
    not equal 'fine-tuning', and 'Ship Building' counted as a missing skill.
    """

    OWNED = skills_taxonomy.build_owned(
        {"expert": ["artificial intelligence", "llm", "fine tuning", "python"],
         "working": ["docker", "aws"]},
        {"expert": 1.0, "working": 0.7},
    )

    def test_two_letter_ai_still_matches(self):
        weight, via, _ = skills_taxonomy.match("Ai", self.OWNED)
        self.assertGreater(weight, 0.0)
        self.assertEqual(via, "artificial intelligence")

    def test_hyphen_and_space_are_the_same_skill(self):
        self.assertEqual(skills_taxonomy.canon("Fine-Tuning"),
                         skills_taxonomy.canon("fine tuning"))

    def test_plural_and_expansion_alias(self):
        self.assertEqual(skills_taxonomy.canon("LLMs"),
                         skills_taxonomy.canon("Large Language Models"))

    def test_containment_matches_longer_board_phrase(self):
        weight, _, how = skills_taxonomy.match("Docker Container", self.OWNED)
        self.assertGreater(weight, 0.0)
        self.assertIn(how, ("exact", "contains"))

    def test_noise_terms_are_dropped(self):
        for noise in ("Ship Building", "scientific discipline",
                      "developed software systems", "technological knowledge"):
            self.assertTrue(skills_taxonomy.is_noise(noise), noise)

    def test_real_skills_are_not_dropped_as_noise(self):
        for real in ("Machine Learning", "PyTorch", "AWS", "Data Warehousing"):
            self.assertFalse(skills_taxonomy.is_noise(real), real)

    def test_generic_head_noun_does_not_create_a_match(self):
        """The removed overlap pass matched anything sharing one common word.

        'Model Deployment' matched 'large language model', 'Agentic Memory
        Management' matched 'stakeholder management'. A false skill match is
        worse than a missed one -- it inflates fit and could later justify a
        resume bullet claiming a skill the candidate lacks.
        """
        owned = skills_taxonomy.build_owned(
            {"expert": ["stakeholder management", "large language model"]},
            {"expert": 1.0},
        )
        for bogus in ("Agentic Memory Management", "Supply chain management software"):
            weight, via, how = skills_taxonomy.match(bogus, owned)
            self.assertEqual(weight, 0.0, f"{bogus} wrongly matched {via} via {how}")

    def test_unheld_skill_reports_missing(self):
        weight, _, _ = skills_taxonomy.match("Reinforcement Learning", self.OWNED)
        self.assertEqual(weight, 0.0)


class SalaryNormalisation(unittest.TestCase):
    def test_monthly_passes_through(self):
        self.assertEqual(job_schema.to_monthly_sgd(8000, "Monthly"), 8000)

    def test_annual_is_divided(self):
        self.assertEqual(job_schema.to_monthly_sgd(120000, "Annually"), 10000)

    def test_unknown_period_guesses_from_magnitude(self):
        # A 12x error in either direction ruins the comp score and every
        # ranking that depends on it, so an unknown period must not pass
        # through unchanged.
        self.assertEqual(job_schema.to_monthly_sgd(180000, "per annum-ish"), 15000)
        self.assertEqual(job_schema.to_monthly_sgd(9000, "per annum-ish"), 9000)

    def test_zero_and_junk_are_none(self):
        for bad in (0, -5, None, "", "negotiable"):
            self.assertIsNone(job_schema.to_monthly_sgd(bad, "Monthly"))

    def test_implausible_monthly_is_treated_as_annual(self):
        """Live bug: a Coupang posting listed 200000-300000 as 'Monthly'.

        Every MCF posting reports period 'Monthly', so the period field cannot
        discriminate. Left alone this maxed out the pay score and took the top
        of the ranking with SGD 2.4M/year for an IC role.
        """
        self.assertEqual(job_schema.to_monthly_sgd(200000, "Monthly"), 16667)
        self.assertTrue(job_schema.salary_was_adjusted(200000, "Monthly"))

    def test_plausible_high_salary_is_left_alone(self):
        self.assertEqual(job_schema.to_monthly_sgd(35000, "Monthly"), 35000)
        self.assertFalse(job_schema.salary_was_adjusted(35000, "Monthly"))


class CompanyAndTitleNormalisation(unittest.TestCase):
    def test_legal_suffixes_collapse(self):
        self.assertEqual(job_schema.norm_company("TIKTOK PTE. LTD."),
                         job_schema.norm_company("TikTok Singapore"))

    def test_title_decoration_stripped(self):
        self.assertEqual(
            job_schema.norm_title("Senior ML Engineer (AI Platform) - Up to $12k!"),
            job_schema.norm_title("Senior ML Engineer"),
        )

    def test_agency_detected_without_word_boundary(self):
        # \btalent\b matched neither TALENTSIS nor Talent Pulse, so both were
        # scored as direct employers.
        self.assertTrue(job_schema.looks_like_agency("TALENTSIS PTE. LTD."))
        self.assertTrue(job_schema.looks_like_agency("TALENT PULSE PTE. LTD."))
        self.assertTrue(job_schema.looks_like_agency("MICHAEL PAGE INTERNATIONAL"))

    def test_direct_employer_not_flagged_as_agency(self):
        for direct in ("GRABTAXI HOLDINGS PTE. LTD.", "MICRON SEMICONDUCTOR",
                       "BYTEDANCE PTE. LTD."):
            self.assertFalse(job_schema.looks_like_agency(direct), direct)

    def test_posted_on_behalf_forces_agency(self):
        self.assertTrue(job_schema.looks_like_agency("ACME CORP", posted_on_behalf=True))


class ContentKey(unittest.TestCase):
    """content_key identifies a ROLE; job_key identifies a POSTING.

    Collapsing them would make repost detection impossible.
    """

    def test_same_role_reposted_shares_content_key(self):
        a = job_schema.content_key("ML Engineer", "Acme Pte Ltd", "Build models.")
        b = job_schema.content_key("ML Engineer (Urgent!)", "ACME PTE. LTD.", "Build models.")
        self.assertEqual(a, b)

    def test_different_role_differs(self):
        a = job_schema.content_key("ML Engineer", "Acme", "Build models.")
        b = job_schema.content_key("Data Analyst", "Acme", "Write reports.")
        self.assertNotEqual(a, b)

    def test_trailing_boilerplate_ignored(self):
        base = "Build models. " + "x" * 4100
        a = job_schema.content_key("ML Engineer", "Acme", base + "EEO statement A")
        b = job_schema.content_key("ML Engineer", "Acme", base + "EEO statement B")
        self.assertEqual(a, b)


class McfMapping(unittest.TestCase):
    def test_maps_a_real_record(self):
        job = source_mcf.to_job(MCF_RECORD)
        self.assertIsNotNone(job)
        self.assertEqual(job["job_key"], "mcf:59501ac0a69b6c5cc91d7b8e9a23d276")
        self.assertEqual(job["company_uen"], "200613609M")
        self.assertEqual(job["ssoc_code"], "25190")
        self.assertEqual(job["salary_min_sgd"], 6500)
        self.assertTrue(job["salary_is_stated"])
        self.assertEqual(job["applications"], 11)
        self.assertEqual(job["views"], 155)
        self.assertEqual(job["posted_at"], "2026-06-24")
        self.assertTrue(job["is_open"])
        self.assertTrue(job["is_agency"])          # GECO is on the agency list
        self.assertEqual(job["seniority"], "lead")  # 10 years, not 'Professional'
        self.assertEqual(job_schema.validate_job(job), [])

    def test_html_description_is_flattened(self):
        job = source_mcf.to_job(MCF_RECORD)
        self.assertNotIn("<p>", job["jd_text"])
        self.assertIn("ML Engineer", job["jd_text"])

    def test_apps_per_view_computed(self):
        job = source_mcf.to_job(MCF_RECORD)
        self.assertAlmostEqual(job["apps_per_view"], 11 / 155, places=4)

    def test_hidden_salary_is_not_stated(self):
        job = source_mcf.to_job(_mcf(metadata={"isHideSalary": True}))
        self.assertFalse(job["salary_is_stated"])

    def test_closed_status_is_not_open(self):
        job = source_mcf.to_job(_mcf(status={"id": 104, "jobStatus": "Closed"}))
        self.assertFalse(job["is_open"])
        self.assertEqual(job["liveness"], "LIKELY_DEAD")

    def test_reopen_counts_as_open(self):
        job = source_mcf.to_job(_mcf(status={"id": 103, "jobStatus": "Re-open"}))
        self.assertTrue(job["is_open"])

    def test_expired_posting_is_not_open(self):
        job = source_mcf.to_job(_mcf(metadata={"expiryDate": "2020-01-01"}))
        self.assertFalse(job["is_open"])

    def test_soft_deleted_is_not_open(self):
        job = source_mcf.to_job(_mcf(metadata={"deletedAt": "2026-07-01T00:00:00.000Z"}))
        self.assertFalse(job["is_open"])

    def test_overseas_is_flagged(self):
        job = source_mcf.to_job(_mcf(address={"isOverseas": True, "overseasCountry": "Malaysia"}))
        self.assertTrue(job["is_overseas"])

    def test_hiring_company_preferred_over_posting_agency(self):
        job = source_mcf.to_job(_mcf(
            hiringCompany={"name": "REAL EMPLOYER PTE LTD", "uen": "999999999X"}
        ))
        self.assertEqual(job["company"], "REAL EMPLOYER PTE LTD")
        self.assertTrue(job["is_agency"])  # still an agency posting

    def test_missing_uuid_is_rejected_not_crashed(self):
        self.assertIsNone(source_mcf.to_job(_mcf(uuid="")))

    def test_original_posting_date_preferred_over_new(self):
        # newPostingDate resets on a repost; using it makes a stale role look
        # fresh, which inverts both the freshness and competition scores.
        job = source_mcf.to_job(_mcf(metadata={
            "originalPostingDate": "2026-01-01", "newPostingDate": "2026-07-18",
        }))
        self.assertEqual(job["posted_at"], "2026-01-01")


class StoreFold(unittest.TestCase):
    def _sightings(self):
        return [
            {"ts": "2026-07-01T00:00:00Z", "run_id": "r1", "job_key": "mcf:a",
             "content_key": "AAA", "company_norm": "acme", "applications": 2, "is_open": True},
            {"ts": "2026-07-02T00:00:00Z", "run_id": "r2", "job_key": "mcf:a",
             "content_key": "AAA", "company_norm": "acme", "applications": 5, "is_open": True},
            {"ts": "2026-07-02T00:00:00Z", "run_id": "r2", "job_key": "mcf:b",
             "content_key": "BBB", "company_norm": "acme", "applications": 0, "is_open": True},
        ]

    def test_first_seen_is_the_earliest_and_immutable(self):
        history = job_store.fold(self._sightings())
        self.assertEqual(history["mcf:a"]["first_seen_at"], "2026-07-01T00:00:00Z")
        self.assertEqual(history["mcf:a"]["last_seen_at"], "2026-07-02T00:00:00Z")
        self.assertEqual(history["mcf:a"]["seen_count"], 2)

    def test_out_of_order_rows_do_not_move_first_seen(self):
        rows = list(reversed(self._sightings()))
        history = job_store.fold(rows)
        self.assertEqual(history["mcf:a"]["first_seen_at"], "2026-07-01T00:00:00Z")

    def test_application_delta_is_tracked(self):
        history = job_store.fold(self._sightings())
        self.assertEqual(history["mcf:a"]["first_applications"], 2)
        self.assertEqual(history["mcf:a"]["last_applications"], 5)

    def test_repost_detected_across_changed_job_key(self):
        rows = self._sightings() + [
            {"ts": "2026-07-20T00:00:00Z", "run_id": "r9", "job_key": "mcf:c",
             "content_key": "AAA", "company_norm": "acme", "is_open": True},
        ]
        history = job_store.fold(rows)
        job = job_schema.new_job("mcf", "c")
        job["content_key"] = "AAA"
        job_store.apply_history([job], history, "r9", ["r1", "r2", "r9"])
        self.assertTrue(job["reposted"])
        self.assertIn("mcf:a", job["repost_of"])

    def test_feed_absence_never_sets_disappeared_at(self):
        """Absence from a search feed is not evidence of closure.

        Search relevance shifts and pagination truncates. Concluding 'closed'
        from a missing row would make time-to-fill fiction. Absent jobs are
        only ever flagged for an explicit re-check.
        """
        history = job_store.fold(self._sightings())
        suspects = job_store.mark_absent(history, {"mcf:b"}, ["r1", "r2", "r3", "r4"])
        keys = {s["job_key"] for s in suspects}
        self.assertIn("mcf:a", keys)
        for suspect in suspects:
            self.assertTrue(suspect["needs_recheck"])
            self.assertNotIn("disappeared_at", suspect)

    def test_no_absence_claimed_without_enough_runs(self):
        history = job_store.fold(self._sightings())
        self.assertEqual(job_store.mark_absent(history, set(), ["r1"]), [])

    def test_damaged_line_does_not_raise(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            path.write_text(
                '{"ts":"2026-07-01T00:00:00Z","run_id":"r1","job_key":"mcf:a"}\n'
                '{"ts":"broken\n',
                encoding="utf-8",
            )
            rows = job_store.read_sightings(path)
        self.assertEqual(len(rows), 1)


class CompanyVelocity(unittest.TestCase):
    def test_insufficient_history_is_reported_not_imputed(self):
        """A number invented from three days of data is worse than no number."""
        history = job_store.fold([
            {"ts": "2026-07-18T00:00:00Z", "run_id": "r1", "job_key": "mcf:a",
             "content_key": "A", "company_norm": "acme"},
        ])
        stats = job_store.company_velocity(history, window_days=30)
        self.assertFalse(stats["acme"]["sufficient"])

        job = job_schema.new_job("mcf", "a")
        job["company_norm"] = "acme"
        value, detail = scoring.score_company_signal(
            job, {}, scoring.ScoreContext(velocity=stats))
        self.assertIsNone(value)
        self.assertIn("insufficient history", detail["reason"])


class Scoring(unittest.TestCase):
    CONFIG = {
        "filters": {"singapore_only": True, "open_only": True,
                    "min_salary_sgd_monthly": 8000, "allow_unstated_salary": True,
                    "exclude_companies": ["tiktok"],
                    "exclude_title_patterns": [r"\bintern(ship)?\b"]},
        "profile": {"target_seniority": "senior", "years_experience": 5,
                    "skills": {"expert": ["python", "machine learning"],
                               "working": ["aws"]}},
        "weights": {"skill_match": 30, "seniority_fit": 15, "comp_signal": 15,
                    "competition": 20, "company_signal": 10,
                    "application_friction": 5, "freshness": 5},
    }

    def _job(self, **overrides):
        job = job_schema.new_job("mcf", "test")
        job.update({
            "title": "Machine Learning Engineer", "company": "Acme Pte Ltd",
            "url": "https://example.com/j", "jd_text": "Build models.",
            "seniority": "senior", "seniority_basis": "title",
            "salary_min_sgd": 10000, "salary_max_sgd": 14000, "salary_is_stated": True,
            "skills_raw": ["Machine Learning", "Python", "AWS"],
            "applications": 5, "views": 100, "posted_at": "2026-07-15",
            "vacancies": 1, "is_open": True,
        })
        job.update(overrides)
        return job_schema.finalise(job)

    def test_missing_component_renormalises_rather_than_imputing(self):
        """A null component must be removed from the denominator.

        Imputing 50 would claim knowledge we do not have and quietly drag every
        score toward the middle.
        """
        job = self._job(salary_is_stated=False, salary_min_sgd=None, salary_max_sgd=None)
        scores = scoring.score_job(job, self.CONFIG, velocity=None)
        self.assertIsNone(scores["components"]["comp_signal"]["value"])
        # company_signal (10) and comp_signal (15) both absent from 100.
        self.assertEqual(scores["weight_used"], 75.0)
        self.assertGreater(scores["total"], 0)

    def test_more_applications_scores_worse(self):
        quiet = scoring.score_job(self._job(applications=1), self.CONFIG)
        busy = scoring.score_job(self._job(applications=60), self.CONFIG)
        self.assertGreater(quiet["components"]["competition"]["value"],
                           busy["components"]["competition"]["value"])

    def test_vacancies_dilute_competition(self):
        one = scoring.score_job(self._job(applications=20, vacancies=1), self.CONFIG)
        many = scoring.score_job(self._job(applications=20, vacancies=10), self.CONFIG)
        self.assertGreater(many["components"]["competition"]["value"],
                           one["components"]["competition"]["value"])

    def test_competition_falls_back_to_age_without_counts(self):
        job = self._job(applications=None, views=None)
        scores = scoring.score_job(job, self.CONFIG)
        detail = scores["components"]["competition"]["detail"]
        self.assertIsNotNone(scores["components"]["competition"]["value"])
        self.assertIn("age proxy", detail["basis"])

    def test_weak_seniority_basis_is_discounted(self):
        strong = scoring.score_job(self._job(seniority="lead", seniority_basis="title"),
                                   self.CONFIG)
        weak = scoring.score_job(self._job(seniority="lead", seniority_basis="position_level"),
                                 self.CONFIG)
        self.assertNotEqual(strong["components"]["seniority_fit"]["value"],
                            weak["components"]["seniority_fit"]["value"])

    def test_explanation_names_real_components(self):
        scores = scoring.score_job(self._job(), self.CONFIG)
        self.assertIn("overall", scores["explanation"])
        self.assertIn("Strongest", scores["explanation"])


class HardFilters(unittest.TestCase):
    # A copy, not an alias. Scoring.CONFIG used to be the SAME object across
    # four classes, so one test mutating a nested dict would corrupt the others
    # order-dependently.
    CONFIG = make_config()

    def _job(self, **overrides):
        return make_job(**overrides)

    def test_current_employer_excluded(self):
        job = self._job(company="TikTok Pte Ltd")
        self.assertIn("excluded company", scoring.check_filters(job, self.CONFIG) or "")

    def test_salary_floor_uses_top_of_range(self):
        # If the most they will pay is under the floor, the job is out --
        # comparing against the bottom of the range would keep it.
        low = self._job(salary_min_sgd=4000, salary_max_sgd=6000)
        self.assertIsNotNone(scoring.check_filters(low, self.CONFIG))
        straddle = self._job(salary_min_sgd=6000, salary_max_sgd=12000)
        self.assertIsNone(scoring.check_filters(straddle, self.CONFIG))

    def test_unstated_salary_allowed_when_configured(self):
        job = self._job(salary_is_stated=False, salary_min_sgd=None, salary_max_sgd=None)
        self.assertIsNone(scoring.check_filters(job, self.CONFIG))

    def test_title_pattern_excluded(self):
        job = self._job(title="Machine Learning Intern")
        self.assertIsNotNone(scoring.check_filters(job, self.CONFIG))

    def test_overseas_excluded(self):
        job = self._job(is_overseas=True)
        self.assertEqual(scoring.check_filters(job, self.CONFIG), "not in Singapore")

    def test_closed_excluded(self):
        job = self._job(is_open=False, source_status="Closed")
        self.assertIn("not open", scoring.check_filters(job, self.CONFIG) or "")

    def test_bad_regex_does_not_crash_the_run(self):
        config = dict(self.CONFIG)
        config["filters"] = dict(config["filters"], exclude_title_patterns=["[unclosed"])
        self.assertIsNone(scoring.check_filters(self._job(), config))


class ResumeDerivation(unittest.TestCase):
    """Derivation from resume text, without an LLM.

    Both bugs pinned here were found by running against a real resume, not by
    reading the code.
    """

    # Synthetic, but shaped exactly like the real resume that exposed both bugs:
    # an EDUCATION block with date ranges above EXPERIENCE, and a bullet naming
    # someone else's job title. No real contact details -- this repo is public.
    RESUME = """ALEX EXAMPLE
person@example.com | +65 8000 1234 | linkedin.com/in/alex-example-00000000

EDUCATION
Example University
Master of IT in Business, AI Track (2021-2024); BSc Economics (2015-2020)

EXPERIENCE
Northwind, AI Engineer, Global Marketing Science  Feb 2026 - Present
- Built an end-to-end RAG pipeline with LLM integration and fine-tuning.
Contoso, Senior Data Analyst, AI & Optimisation  Nov 2024 - Jan 2026
- AI product owner for a budget optimiser.
Fabrikam, Senior Developer, Data & AI  Sep 2022 - Jul 2023
Direct report to CEO, risk and commercial teams
- Defined the company's AI strategy and roadmap.

TECHNICAL SKILLS
AI / ML: LLMs, RAG pipelines, LangChain, prompt engineering, fine-tuning
Stack: Python, PySpark, SQL, AWS, Docker, Kubernetes
"""

    def test_education_dates_excluded_from_experience(self):
        """Live bug: counting degree dates gave 11.3 years for a ~5 year career.

        That pushed every seniority comparison two levels too senior.
        """
        years, evidence = user_input.derive_years_experience(self.RESUME)
        self.assertTrue(evidence["read_experience_section_only"])
        self.assertLess(years, 8.0, f"education dates leaked in: {evidence}")
        self.assertGreater(years, 2.0)
        joined = " ".join(evidence["spans_found"])
        self.assertNotIn("2015", joined)

    def test_warns_when_no_experience_heading_found(self):
        years, evidence = user_input.derive_years_experience(
            "Some Company 2019 - 2023\nAnother 2015 - 2018"
        )
        self.assertFalse(evidence["read_experience_section_only"])
        self.assertIn("education dates may be included", evidence["note"])

    def test_concurrent_roles_merged_not_summed(self):
        text = "EXPERIENCE\nA, Eng  Jan 2020 - Jan 2024\nB, Advisor  Jan 2021 - Jan 2023\n"
        years, evidence = user_input.derive_years_experience(text)
        self.assertAlmostEqual(years, 4.0, delta=0.2)
        self.assertEqual(evidence["merged_ranges"], 1)

    def test_seniority_ignores_prose_about_other_people(self):
        """Live bug: 'Direct report to CEO' made the candidate an 'executive'.

        A title matcher fed a blob of resume prose matches anyone's job title,
        not the candidate's.
        """
        level, basis = user_input.derive_seniority(self.RESUME, years=4.9)
        self.assertNotEqual(level, "executive", f"matched prose, basis={basis}")

    def test_bullet_lines_are_not_role_lines(self):
        self.assertFalse(user_input._looks_like_role_line(
            "- Defined the company's AI strategy, roadmap, and architecture."))
        self.assertTrue(user_input._looks_like_role_line(
            "Shopee, Senior Data Analyst, AI & Optimisation  Nov 2024 - Jan 2026"))

    def test_contact_details_extracted(self):
        contact = user_input.derive_contact(self.RESUME)
        self.assertEqual(contact["email"], "person@example.com")
        self.assertIn("8000", contact["phone"])
        self.assertIn("linkedin.com/in/", contact["linkedin"])

    def test_skills_derived_only_from_what_is_present(self):
        skills = user_input.derive_skills(self.RESUME)
        self.assertIn("python", skills)
        self.assertIn("large language model", skills)   # via the 'LLMs' alias
        # Never invent a skill the resume does not mention.
        self.assertNotIn("reinforcement learning", skills)
        self.assertNotIn("terraform", skills)

    def test_validation_requires_the_compulsory_three(self):
        problems = user_input.validate(user_input.IntakeProfile())
        joined = " ".join(problems)
        self.assertIn("full_name", joined)
        self.assertIn("resume_path", joined)
        self.assertIn("target_roles", joined)

    def test_typed_values_beat_derived_values(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv.txt"
            path.write_text(self.RESUME, encoding="utf-8")
            profile = user_input.build_profile(
                full_name="Explicit Name", resume_path=path,
                target_roles="AI Engineer", years_experience=9.0,
            )
        self.assertEqual(profile.full_name, "Explicit Name")
        self.assertEqual(profile.years_experience, 9.0)
        self.assertNotIn("years_experience", profile.derived)

    def test_desired_below_current_pay_is_flagged(self):
        profile = user_input.IntakeProfile(
            full_name="A", resume_path=__file__, resume_text="x",
            target_roles=["Eng"],
            current_salary_sgd_monthly=12000, desired_salary_sgd_monthly=9000,
        )
        self.assertTrue(any("below current pay" in p for p in user_input.validate(profile)))

    def test_unreadable_resume_reports_reason_not_crash(self):
        text, how = user_input.read_resume_text("does/not/exist.pdf")
        self.assertEqual(text, "")
        self.assertIn("not found", how)

    def test_unsupported_type_is_explained(self):
        text, how = user_input.read_resume_text("resume.pages")
        self.assertEqual(text, "")
        self.assertIn("unsupported", how)


class SeniorityAmbition(unittest.TestCase):
    """A job search aims up. Scoring must reflect that.

    Defaulting target_seniority to the current level made staying-put roles
    score 100 and the stretch roles the user actually wants score 65 -- exactly
    backwards.
    """

    def test_target_defaults_one_level_above_current(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv.txt"
            path.write_text(ResumeDerivation.RESUME, encoding="utf-8")
            profile = user_input.build_profile(
                full_name="A", resume_path=path, target_roles="AI Engineer",
            )
        self.assertIsNotNone(profile.target_seniority)
        gap = job_schema.seniority_distance(
            profile.current_seniority, profile.target_seniority
        )
        self.assertEqual(gap, 1, "target should sit one level above current")

    def test_ambition_zero_targets_current_level(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv.txt"
            path.write_text(ResumeDerivation.RESUME, encoding="utf-8")
            profile = user_input.build_profile(
                full_name="A", resume_path=path, target_roles="AI Engineer",
                seniority_ambition=0,
            )
        self.assertEqual(profile.target_seniority, profile.current_seniority)

    def test_step_up_clamps_at_top_of_ladder(self):
        self.assertEqual(job_schema.step_up("executive", 3), "executive")
        self.assertIsNone(job_schema.step_up(None))

    def test_stretch_role_outranks_staying_put(self):
        """A mid-level candidate should see senior roles above mid roles."""
        profile = make_config(current_seniority="mid",
                              target_seniority="senior")["profile"]

        def at(level):
            return scoring.score_seniority_fit(
                make_job(seniority=level, seniority_basis="title"), profile)[0]

        self.assertGreater(at("senior"), at("mid"))
        self.assertGreater(at("lead"), at("junior"))

    def test_reaching_up_beats_dropping_down_by_the_same_distance(self):
        """One level above target must outrank one level below it.

        The penalties were first written the wrong way round -- the comment
        said 'tilted toward the stretch' while the numbers scored the
        staying-put level higher. Job searches are for better pay or better
        rank; a safe sideways move is the thing being moved away from.
        """
        profile = make_config(current_seniority="mid",
                              target_seniority="senior")["profile"]

        def at(level):
            return scoring.score_seniority_fit(
                make_job(seniority=level, seniority_basis="title"), profile)[0]

        self.assertGreater(at("lead"), at("mid"),
                           "one level above target should beat one level below")


class Confidentiality(unittest.TestCase):
    """Salary must never cross the process boundary.

    Prompts are retained by providers, so a leak here is permanent.
    """

    PROFILE = user_input.IntakeProfile(
        full_name="Alex Example",
        resume_path="cv.pdf",
        target_roles=["AI Engineer"],
        email="person@example.com",
        phone="+65 8000 1234",
        linkedin="https://www.linkedin.com/in/alex-example-00000000",
        current_salary_sgd_monthly=12000,
        desired_salary_sgd_monthly=16000,
        min_salary_sgd_monthly=10800,
        notes="leaving because of my manager",
        skills=["python", "llm"],
        years_experience=4.9,
    )

    def test_no_salary_field_survives_redaction(self):
        safe = user_input.redact_for_llm(self.PROFILE)
        for field_name in ("current_salary_sgd_monthly", "desired_salary_sgd_monthly",
                           "min_salary_sgd_monthly"):
            self.assertNotIn(field_name, safe)

    def test_no_salary_VALUE_appears_anywhere_in_the_payload(self):
        """Field names are not enough -- check the serialised values too."""
        blob = json.dumps(user_input.redact_for_llm(self.PROFILE))
        for secret in ("12000", "16000", "10800"):
            self.assertNotIn(secret, blob, f"salary {secret} leaked into payload")

    def test_direct_identifiers_are_stripped(self):
        blob = json.dumps(user_input.redact_for_llm(self.PROFILE))
        for secret in ("8000 1234", "person@example.com", "linkedin.com",
                       "because of my manager"):
            self.assertNotIn(secret, blob, f"{secret!r} leaked into payload")

    def test_matching_inputs_are_kept(self):
        safe = user_input.redact_for_llm(self.PROFILE)
        self.assertEqual(safe["target_seniority"], self.PROFILE.target_seniority)
        self.assertIn("python", safe["skills"])
        self.assertEqual(safe["years_experience"], 4.9)

    def test_redaction_is_an_allowlist_not_a_blocklist(self):
        """A newly added confidential field must be excluded by default.

        With a blocklist, forgetting to update it leaks silently and forever.
        """
        payload = dict(self.PROFILE.to_dict())
        payload["bank_account_number"] = "123-456-789"
        safe = user_input.redact_for_llm(payload)
        self.assertNotIn("bank_account_number", safe)

    def test_every_confidential_field_is_outside_the_allowlist(self):
        overlap = user_input.CONFIDENTIAL_FIELDS & user_input.LLM_SAFE_FIELDS
        self.assertEqual(overlap, set(), f"contradictory classification: {overlap}")


class Validation(unittest.TestCase):
    def test_inverted_salary_range_is_caught(self):
        job = job_schema.new_job("mcf", "x")
        job.update({"title": "T", "company": "C", "url": "u",
                    "salary_min_sgd": 9000, "salary_max_sgd": 5000})
        job_schema.finalise(job)
        self.assertTrue(any("exceeds max" in p for p in job_schema.validate_job(job)))

    def test_missing_fields_are_listed(self):
        job = job_schema.new_job("mcf", "x")
        problems = job_schema.validate_job(job)
        self.assertTrue(any("title" in p for p in problems))
        self.assertTrue(any("url" in p for p in problems))


if __name__ == "__main__":
    unittest.main(verbosity=2)
