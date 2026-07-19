"""Tests for the per-job analysis PDF.

    py -m unittest tests.test_render_report

Offline, no network, no API key, no cost. Every person, employer, salary and
job below is invented -- this repo is public and must never carry real CV or
salary data.

The property worth testing is not that the PDF compiles. It is that the report
never states a number the pipeline did not measure. So the assertions are on
the model and on the `.typ` source, where a fabricated figure would appear as
text, rather than on rendered output. A job with no salary and no published
application count is the fixture that matters: it must produce the words "not
measured" and "not published by this source", never a zero.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jobbuddy import render_report
from jobbuddy import render_resume
from jobbuddy import scoring

CONFIG = {
    "filters": {},
    "profile": {
        "skills": {"expert": ["Python", "PySpark"], "working": ["Airflow"]},
        "target_seniority": "manager",
        "years_experience": 8,
        "current_salary_sgd_monthly": 11000,
    },
    "weights": {"skill_match": 30, "seniority_fit": 15, "comp_signal": 15,
                "competition": 20, "company_signal": 10,
                "application_friction": 5, "freshness": 5},
}

FACTS = [
    {"fact_id": "umbra.etl", "verified": True, "org": "Umbra Financial",
     "role": "Manager, Data Engineering",
     "skills": ["Airflow", "PySpark"], "entities": [],
     "phrasings": ["Automated 4 ETL processes in PySpark"]},
    {"fact_id": "umbra.contracts", "verified": True, "org": "Umbra Financial",
     "role": "Manager, Data Engineering",
     "skills": ["Kubernetes"], "entities": [],
     "phrasings": ["Introduced schema contracts across 6 upstream feeds"]},
]

PROFILE = {"identity": {"name": "Ada Nakamura"}, "facts": FACTS}

TAILORED = {
    "ok": True,
    "bullets": [{"text": "Automated 4 ETL processes in PySpark",
                 "fact_id": "umbra.etl", "org": "Umbra Financial",
                 "role": "Manager, Data Engineering",
                 "start": "2023-08", "end": "2024-11"}],
    "headline": "Data engineer building measured ETL systems.",
    # One requirement a recorded fact covers, one nothing covers.
    "unaddressed": ["Kubernetes", "COBOL"],
    "dropped_for_length": 0,
    "unknown_fact_ids": [],
    "guard": {"bullets": 3, "passed": 2, "rejected": 1, "fell_back": 1,
              "by_kind": {"unsupported_number": 1},
              "examples": [{"bullet": "Cut costs by 90% across 12 teams",
                            "reasons": ["unsupported_number: 90 is not in the "
                                        "fact"]}]},
}

RESUME_RENDER = {
    "dropped": [{"text": "Introduced schema contracts across 6 upstream feeds",
                 "fact_id": "umbra.contracts",
                 "reason": "lowest-ranked (rank 9); did not fit 1 page(s)"}],
}


def rich_job() -> dict:
    """An MCF posting: salary stated, real application count published."""
    return {
        "job_key": "mcf:1", "source": "mcf",
        "url": "https://www.mycareersfuture.gov.sg/job/example-1",
        "title": "Manager, Data Platform", "company": "Umbra Financial",
        "company_norm": "umbra", "location": "Singapore",
        "seniority": "manager", "seniority_basis": "title",
        "employment_types": ["Full Time"],
        "salary_min_sgd": 10000, "salary_max_sgd": 14000,
        "salary_is_stated": True,
        "skills_raw": ["Python", "PySpark", "Airflow", "Kubernetes"],
        "skills_key": ["Python"],
        "applications": 18, "views": 900, "vacancies": 2, "apps_per_day": 1.5,
        "age_days": 12, "posted_at": "2026-07-06", "expires_at": "2026-08-06",
        "repost_count": 0, "seen_count": 3, "min_years_exp": 6,
        "is_agency": False,
    }


def sparse_job() -> dict:
    """An ATS board posting: no salary, no published application count.

    This is the fixture the module exists for. Both blanks are common on every
    source except MyCareersFuture, and both are where an analysis PDF is most
    tempted to quietly fill in a number.
    """
    return {
        "job_key": "workable:9", "source": "workable",
        "url": "https://example-co.workable.com/j/ABC123",
        "title": "Staff Data Engineer", "company": "Northwind Labs",
        "company_norm": "northwind", "location": "Singapore",
        "seniority": None, "employment_types": [],
        "salary_min_sgd": None, "salary_max_sgd": None,
        "salary_is_stated": False,
        "skills_raw": [], "skills_key": [],
        "applications": None, "views": None, "vacancies": None,
        "age_days": None, "posted_at": None, "expires_at": None,
        "is_agency": False,
    }


def scored(job: dict) -> dict:
    """Score with an explicit fictional config, never the repo's run_config."""
    scoring.score_job(job, CONFIG, velocity={})
    return job


def model_for(job: dict, **kwargs) -> dict:
    return render_report.build_model(job, **kwargs)


class RoleAndApplyLink(unittest.TestCase):

    def test_role_details_come_through_to_the_source(self):
        source = render_report.build_typst_source(model_for(scored(rich_job())))
        self.assertIn("Manager, Data Platform", source)
        self.assertIn("Umbra Financial", source)
        self.assertIn("Singapore", source)

    def test_an_absolute_http_url_is_reported_as_shape_validated(self):
        model = model_for(scored(rich_job()))
        self.assertIn("mycareersfuture", model["apply"]["url"])
        self.assertIn("validated: shape", model["apply"]["status"])

    def test_a_relative_url_is_flagged_rather_than_printed_as_clickable(self):
        job = scored(rich_job())
        job["url"] = "/job/example-1"
        self.assertIn("not a usable absolute URL",
                      model_for(job)["apply"]["status"])

    def test_a_missing_url_is_not_measured(self):
        job = scored(rich_job())
        job["url"] = ""
        self.assertEqual(model_for(job)["apply"]["url"], "not measured")


class Competition(unittest.TestCase):
    """MCF publishes a real count. Nothing else does, and that is not a zero."""

    def test_a_published_application_count_is_used_as_published(self):
        model = model_for(scored(rich_job()))
        self.assertTrue(model["competition"]["published"])
        self.assertEqual(model["competition"]["applications"], "18")
        self.assertIn("9.0 per vacancy", model["competition"]["per_vacancy"])

    def test_an_unpublished_count_says_so_and_never_shows_a_number(self):
        model = model_for(scored(sparse_job()))
        self.assertFalse(model["competition"]["published"])
        self.assertIn("not published by this source",
                      model["competition"]["applications"])
        self.assertIn("workable", model["competition"]["applications"])

    def test_an_unpublished_count_is_never_rendered_as_zero(self):
        source = render_report.build_typst_source(model_for(scored(sparse_job())))
        applications = [line for line in source.splitlines()
                        if line.startswith("*Applications:*")]
        self.assertEqual(len(applications), 1)
        self.assertNotIn("0", applications[0])
        self.assertIn("not published by this source", applications[0])


class NeverImputes(unittest.TestCase):
    """A job with no salary and no application count says so, and says nothing else."""

    def test_no_stated_salary_reports_not_measured_rather_than_zero(self):
        model = model_for(scored(sparse_job()))
        self.assertIn("not measured", model["salary"])
        self.assertIn("salary not stated", model["salary"])
        self.assertNotIn("SGD", model["salary"])

    def test_a_stated_salary_reports_the_range_and_its_own_midpoint(self):
        model = model_for(scored(rich_job()))
        self.assertIn("10,000", model["salary"])
        self.assertIn("14,000", model["salary"])
        self.assertIn("12,000", model["salary"])

    def test_a_posting_with_no_date_reports_time_on_market_as_not_measured(self):
        market = model_for(scored(sparse_job()))["market_time"]
        self.assertEqual(market["posted_at"], "not measured")
        self.assertEqual(market["age_days"], "not measured")
        self.assertEqual(market["days_left"], "not measured")

    def test_unmeasured_company_and_career_signals_carry_their_reason(self):
        model = model_for(scored(sparse_job()))
        self.assertEqual(model["company_signals"]["status"], "not measured")
        self.assertTrue(model["company_signals"]["reason"])
        self.assertEqual(model["career"]["direction"], "not measured")
        self.assertTrue(model["career"]["reason"])

    def test_a_fully_sparse_job_still_produces_a_report_source(self):
        source = render_report.build_typst_source(model_for(scored(sparse_job())))
        self.assertIn("not measured", source)
        self.assertIn("Staff Data Engineer", source)


class Gaps(unittest.TestCase):
    """The section worth the page: cut-for-space vs genuinely lacking."""

    def test_unaddressed_splits_into_cut_and_lacking_using_the_facts(self):
        gaps = model_for(scored(rich_job()), tailored=TAILORED,
                         profile=PROFILE)["gaps"]
        cut = [g["requirement"] for g in gaps["have_but_cut"]]
        self.assertIn("Kubernetes", cut)
        self.assertIn("COBOL", gaps["genuinely_lacking"])
        self.assertNotIn("Kubernetes", gaps["genuinely_lacking"])

    def test_a_cut_requirement_names_the_fact_that_covers_it(self):
        gaps = model_for(scored(rich_job()), tailored=TAILORED,
                         profile=PROFILE)["gaps"]
        covered = {g["requirement"]: g["fact_id"] for g in gaps["have_but_cut"]}
        self.assertEqual(covered["Kubernetes"], "umbra.contracts")

    def test_with_no_facts_nothing_is_declared_a_real_gap(self):
        """Defaulting to "you lack this" is as much an invention as the opposite."""
        gaps = model_for(scored(rich_job()), tailored=TAILORED,
                         profile={"facts": []})["gaps"]
        self.assertEqual(gaps["genuinely_lacking"], [])
        self.assertEqual(gaps["have_but_cut"], [])
        self.assertEqual(gaps["unclassified"], ["Kubernetes", "COBOL"])
        self.assertIn("not measured", gaps["explanation"])

    def test_both_gap_kinds_appear_in_the_rendered_source(self):
        source = render_report.build_typst_source(
            model_for(scored(rich_job()), tailored=TAILORED, profile=PROFILE))
        self.assertIn("it was cut, not missing", source)
        self.assertIn("You genuinely lack this", source)
        self.assertIn("COBOL", source)


class SilentLosses(unittest.TestCase):
    """What the guard rejected and what the page fit cut are both surfaced."""

    def test_guard_rejections_are_counted_and_exemplified(self):
        losses = model_for(scored(rich_job()), tailored=TAILORED)["losses"]
        self.assertEqual(losses["guard_rejected"], 1)
        self.assertEqual(losses["guard_fell_back"], 1)
        self.assertEqual(losses["guard_by_kind"], {"unsupported_number": 1})

    def test_bullets_cut_to_fit_the_page_are_named_in_the_source(self):
        source = render_report.build_typst_source(model_for(
            scored(rich_job()), tailored=TAILORED,
            resume_render=RESUME_RENDER))
        self.assertIn("schema contracts across 6 upstream feeds", source)
        self.assertIn("lowest-ranked", source)

    def test_no_tailoring_data_still_reports_zero_losses_without_crashing(self):
        losses = model_for(scored(rich_job()))["losses"]
        self.assertEqual(losses["guard_rejected"], 0)
        self.assertEqual(losses["cut_to_fit"], [])


class Charts(unittest.TestCase):

    def test_the_model_carries_one_svg_per_chart(self):
        charts = model_for(scored(rich_job()),
                           population=[70.0, 40.0])["charts"]
        self.assertEqual(set(charts),
                         {"components", "fit", "timeline", "distribution"})
        for svg in charts.values():
            self.assertTrue(svg.startswith("<svg"))


class Degradation(unittest.TestCase):
    """No typst wheel emits the .typ source and says so. It never raises."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="jobbuddy-report-"))
        self.addCleanup(self._cleanup)
        render_resume.reset_capability_cache()
        self.addCleanup(render_resume.reset_capability_cache)

    def _cleanup(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def _without_typst(self):
        original = render_resume._load_typst
        render_resume._load_typst = lambda: None
        render_resume.reset_capability_cache()
        self.addCleanup(setattr, render_resume, "_load_typst", original)

    def test_without_typst_it_writes_typ_source_and_names_the_degradation(self):
        self._without_typst()
        model = model_for(scored(rich_job()), tailored=TAILORED,
                          profile=PROFILE)
        with self.assertWarns(RuntimeWarning):
            result = render_report.render(model, self.dir / "report.pdf")
        self.assertTrue(result["ok"])
        self.assertEqual(result["degraded"], "typ-source")
        self.assertEqual(result["path"].suffix, ".typ")
        self.assertTrue(result["path"].exists())
        self.assertIn("typst compile", result["note"])
        self.assertIn("Manager, Data Platform",
                      result["path"].read_text(encoding="utf-8"))

    def test_charts_are_written_beside_the_document_either_way(self):
        self._without_typst()
        model = model_for(scored(rich_job()))
        with self.assertWarns(RuntimeWarning):
            result = render_report.render(model, self.dir / "report.pdf")
        self.assertEqual(len(result["charts"]), 4)
        for path in result["charts"]:
            self.assertTrue(path.exists())
            self.assertTrue(path.read_text(encoding="utf-8").startswith("<svg"))

    @unittest.skipUnless(render_resume._load_typst() is not None,
                         "typst wheel not installed")
    def test_with_typst_it_compiles_a_real_pdf(self):
        model = model_for(scored(rich_job()), tailored=TAILORED,
                          profile=PROFILE, resume_render=RESUME_RENDER)
        result = render_report.render(model, self.dir / "report.pdf")
        self.assertIsNone(result["degraded"])
        self.assertEqual(result["path"].suffix, ".pdf")
        self.assertTrue(result["path"].read_bytes().startswith(b"%PDF"))

    def test_a_sparse_job_renders_without_raising(self):
        self._without_typst()
        model = model_for(scored(sparse_job()))
        with self.assertWarns(RuntimeWarning):
            result = render_report.render(model, self.dir / "sparse.pdf")
        self.assertTrue(result["ok"])


class NeverCrashes(unittest.TestCase):
    """None and missing values must not take the report down."""

    def test_an_unscored_job_still_builds_a_model(self):
        model = render_report.build_model({"title": "Untitled"})
        self.assertEqual(model["company"], "not measured")
        self.assertEqual(model["explanation"], "not measured")
        render_report.build_typst_source(model)

    def test_an_empty_job_still_builds_a_model_and_a_source(self):
        source = render_report.build_typst_source(render_report.build_model({}))
        self.assertIn("not measured", source)

    def test_typst_special_characters_in_a_company_name_are_escaped(self):
        job = scored(rich_job())
        job["company"] = "R&D #1 [Asia] Pte Ltd"
        source = render_report.build_typst_source(model_for(job))
        self.assertIn("\\#1", source)
        self.assertIn("\\[Asia\\]", source)


if __name__ == "__main__":
    unittest.main()
