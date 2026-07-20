"""Tests for the tailoring stage -- search output in, deliverables out.

    py -m unittest tests.test_tailor_pipeline

Offline, no network, no API key, no cost: the model is injected as a stub, and
every write goes into a temporary directory. Nothing here may touch the real
`potential applications/` tree, because these tests run on the machine that
holds the user's actual applications.

The fixtures are the fictional Alex Tan, at Umbra Financial and Northwind Labs,
reused from `test_tailor.py` -- this repo is public and must never carry real CV
data.

What is asserted, and why each one rather than the code looking right:

  the stage runs end to end          a PDF, a DOCX and an analysis land on disk
  a blocked resume is NOT rendered   a personal-data leak that reaches a
                                     rendered document cannot be withdrawn
  one job failing loses one job      and it still reaches the workbook, because
                                     a job that vanishes silently is worse than
                                     a job that reports a failure
  the budget stops the run           and says how many jobs it did not do
  the stage is off by default        every existing `run_scopes` caller sees
                                     exactly what it saw before
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault(
    "JB_COMPANY_REGISTRY",
    os.path.join(tempfile.mkdtemp(prefix="jobbuddy-tests-"), "companies.json"))

import shutil
from unittest import mock

from jobbuddy import job_store
from jobbuddy import pipeline
from jobbuddy import render_resume
from jobbuddy import scoring
from tests.test_render_excel import read_sheets
from tests.test_seams import SCOPE, _fake_fetch, _job

# --------------------------------------------------------------------------
# Fixtures. Alex Tan is fictional and so is everything he has ever done.
# --------------------------------------------------------------------------

FACT_ETL = {
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

FACT_RETRIEVAL = {
    "fact_id": "northwind.retrieval",
    "org": "Northwind Labs",
    "role": "AI Engineer",
    "start": "2024-08", "end": None,
    "numbers": ["12", "340", "90"],
    "entities": ["PyTorch", "Triton"],
    "skills": ["pytorch", "retrieval"],
    "phrasings": ["Built a retrieval pipeline serving 12 million daily requests "
                  "in PyTorch, cutting p99 latency from 340ms to 90ms"],
    "verified": True,
}

# The approved phrasing of this one carries a personal particular. TAFEP and
# the PDPC list marital status for removal, so `resume_rules` calls it an error
# and the stage must refuse to render the document rather than warn about it.
FACT_LEAKS_A_PARTICULAR = {
    "fact_id": "umbra.hr",
    "org": "Umbra Financial",
    "role": "Manager, Data Engineering",
    "start": "2023-08", "end": "2024-11",
    "numbers": [], "entities": [], "skills": ["hr"],
    "phrasings": ["Standardised marital status reporting across regional "
                  "HR systems"],
    "verified": True,
}

PROFILE = {
    "identity": {
        "name": "Alex Tan",
        "email": "alex.tan@example.test",
        "phone": "+65 8000 0000",
        "location": "Singapore",
    },
    "constraints": {"never_claim": ["managed a team", "PhD"]},
    "facts": [FACT_ETL, FACT_RETRIEVAL, FACT_LEAKS_A_PARTICULAR],
    "skills_declared": {"expert": ["Python", "PySpark"],
                        "working": ["PyTorch", "Retrieval"]},
    "education": [{"qualification": "BSc Computer Science",
                   "institution": "Fictional University",
                   "year": "2019"}],
}

CLEAN_SELECTION = [{"fact_id": "northwind.retrieval", "rank": 1},
                   {"fact_id": "umbra.etl", "rank": 2}]
LEAKING_SELECTION = [{"fact_id": "umbra.hr", "rank": 1}]

COST_PER_CALL = 0.01


def stub_chat(selection_for=None, raise_for=None, cost=COST_PER_CALL):
    """A `json_chat` replacement keyed on the job title in message[1].

    Keyed on the job rather than on call order because the compute phase runs
    concurrently, so call order is not something a test may rely on.
    """
    selection_for = selection_for or {}
    raise_for = raise_for or set()

    def _chat(messages, **kwargs):
        job_message = messages[1]["content"]
        for title in raise_for:
            if title in job_message:
                raise RuntimeError(f"upstream exploded on {title}")
        selected = CLEAN_SELECTION
        for title, choice in selection_for.items():
            if title in job_message:
                selected = choice
                break
        return {"ok": True, "cost_usd": cost,
                "data": {"selected": list(selected),
                         "headline": "AI engineer, retrieval systems",
                         "unaddressed": ["Kubernetes at scale"]}}

    return _chat


def scored_job(key: str, title: str, adjusted: float = 80.0,
               company: str = "Umbra Financial") -> dict:
    """A ranked job in the shape the stage receives it from `run_scopes`."""
    return {
        "job_key": key, "title": title, "company": company,
        "url": "https://example.test/job", "source": "mcf",
        "seniority": "senior", "location": "Singapore",
        "jd_text": "Build retrieval systems.",
        "skills": ["pytorch", "retrieval", "etl"],
        "salary_is_stated": False, "applications": 12, "vacancies": 1,
        "scores": {"adjusted": adjusted, "total": adjusted,
                   "confidence": 0.8, "explanation": "strong skill match",
                   "components": {}},
    }


class StageCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jobbuddy-tailor-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.stamp = "2026-07-19_101500"

    def run_stage(self, jobs, chat, **kwargs):
        kwargs.setdefault("wave", 1)
        return pipeline.tailor_jobs(
            jobs, PROFILE, "ai-engineer-sg", output_dir=self.tmp,
            stamp=self.stamp, chat=chat, **kwargs)


class TheStageProducesTheDeliverables(StageCase):
    def test_a_tailored_job_gets_a_resume_an_analysis_and_a_directory(self):
        run = self.run_stage([scored_job("a", "Senior AI Engineer")],
                             stub_chat(), top=1)
        self.assertEqual(len(run.rendered), 1)
        outcome = run.outcomes[0]
        names = sorted(p.name for p in outcome.written)
        self.assertIn("resume.pdf", names)
        self.assertIn("resume.docx", names)
        self.assertIn("report.pdf", names)
        for path in outcome.written:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())

    def test_everything_lands_under_scope_then_stamp_then_job(self):
        run = self.run_stage([scored_job("a", "Senior AI Engineer")],
                             stub_chat(), top=1)
        directory = run.outcomes[0].directory
        self.assertEqual(directory.parent.name, self.stamp)
        self.assertEqual(directory.parent.parent.name, "ai-engineer-sg")
        self.assertTrue(directory.is_relative_to(self.tmp))

    def test_nothing_is_written_outside_the_temporary_tree(self):
        """This suite runs on the machine holding the real applications."""
        run = self.run_stage([scored_job("a", "Senior AI Engineer")],
                             stub_chat(), top=1)
        for path in run.written:
            with self.subTest(path=path):
                self.assertTrue(Path(path).is_relative_to(self.tmp))

    def test_the_page_one_check_runs_against_the_rendered_pdf(self):
        run = self.run_stage([scored_job("a", "Senior AI Engineer")],
                             stub_chat(), top=1)
        page_one = run.outcomes[0].page_one
        self.assertIsNotNone(page_one)
        self.assertIn("top_bullets", page_one)
        self.assertTrue(page_one["top_bullets"])

    def test_only_the_top_n_are_tailored(self):
        jobs = [scored_job(f"j{i}", f"AI Engineer {i}", adjusted=90 - i)
                for i in range(5)]
        run = self.run_stage(jobs, stub_chat(), top=2)
        self.assertEqual(len(run.outcomes), 2)
        self.assertEqual([o.job_key for o in run.outcomes], ["j0", "j1"])

    def test_one_workbook_covers_the_whole_run_not_just_the_tailored_jobs(self):
        jobs = [scored_job(f"j{i}", f"AI Engineer {i}", adjusted=90 - i)
                for i in range(4)]
        run = self.run_stage(jobs, stub_chat(), top=1)
        rows = read_sheets(run.workbook)["ai-engineer-sg"]
        self.assertEqual(len(rows) - 1, 4)


class ABlockedResumeIsNeverRendered(StageCase):
    """A personal-data leak that reaches a rendered document cannot be undone."""

    def _run(self):
        return self.run_stage(
            [scored_job("leak", "HR Data Lead")],
            stub_chat(selection_for={"HR Data Lead": LEAKING_SELECTION}),
            top=1)

    def test_the_job_is_marked_blocked(self):
        run = self._run()
        self.assertEqual(run.outcomes[0].status, "blocked")
        self.assertEqual(run.rendered, [])

    def test_no_document_is_written_for_it(self):
        run = self._run()
        outcome = run.outcomes[0]
        self.assertEqual(outcome.written, [])
        self.assertFalse(outcome.directory.exists())

    def test_the_reason_names_the_rule_that_blocked_it(self):
        run = self._run()
        self.assertIn("personal_data.marital_status", run.outcomes[0].reason)

    def test_the_reason_does_not_quote_the_offending_text(self):
        """A report echoing what it found has recreated the leak."""
        run = self._run()
        self.assertNotIn("Standardised marital status reporting",
                         run.outcomes[0].note())

    def test_a_blocked_job_still_reaches_the_workbook(self):
        run = self._run()
        rows = read_sheets(run.workbook)["ai-engineer-sg"]
        column = rows[0].index("tailoring")
        self.assertIn("blocked", rows[1][column])


class OneJobFailingDoesNotStopTheRun(StageCase):
    def _run(self):
        jobs = [scored_job("a", "Senior AI Engineer", adjusted=90),
                scored_job("b", "Broken Role", adjusted=80),
                scored_job("c", "Data Engineer", adjusted=70)]
        return jobs, self.run_stage(
            jobs, stub_chat(raise_for={"Broken Role"}), top=3)

    def test_the_other_jobs_are_still_tailored(self):
        _, run = self._run()
        self.assertEqual(sorted(o.job_key for o in run.rendered), ["a", "c"])

    def test_the_failure_is_recorded_with_the_stage_it_failed_at(self):
        _, run = self._run()
        failed = run.failed
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].job_key, "b")
        self.assertEqual(failed[0].status, "FAILED_AT_tailor")
        self.assertIn("upstream exploded", failed[0].reason)

    def test_the_failed_job_still_appears_in_the_workbook(self):
        """Vanishing silently is worse than reporting a failure."""
        jobs, run = self._run()
        rows = read_sheets(run.workbook)["ai-engineer-sg"]
        headers = rows[0]
        key_column = headers.index("job key")
        tailoring_column = headers.index("tailoring")
        by_key = {row[key_column]: row[tailoring_column] for row in rows[1:]}
        self.assertIn("b", by_key)
        self.assertIn("FAILED_AT_tailor", by_key["b"])

    def test_a_failure_after_selection_names_the_stage_it_failed_at(self):
        """Reporting every failure as FAILED_AT_tailor sends the user to the
        wrong module. The stage name is the actionable half of the record."""
        jobs = [scored_job("a", "Senior AI Engineer"),
                scored_job("b", "Data Engineer")]
        with mock.patch("jobbuddy.resume_rules.check",
                        side_effect=RuntimeError("rules blew up")):
            run = self.run_stage(jobs, stub_chat(), top=2)
        self.assertEqual({o.status for o in run.outcomes}, {"FAILED_AT_rules"})
        self.assertIn("rules blew up", run.outcomes[0].reason)

    def test_a_render_failure_loses_one_job_not_the_run(self):
        """The render is the serial half; a crash there must not stop the loop."""
        jobs = [scored_job("a", "Senior AI Engineer", adjusted=90),
                scored_job("b", "Data Engineer", adjusted=80)]
        calls = {"n": 0}
        real_render = render_resume.render

        def flaky(model, out_dir, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("typst fell over")
            return real_render(model, out_dir, **kwargs)

        with mock.patch("jobbuddy.render_resume.render", side_effect=flaky):
            run = self.run_stage(jobs, stub_chat(), top=2)
        self.assertEqual(run.outcomes[0].status, "FAILED_AT_render_resume")
        self.assertEqual(run.outcomes[1].status, "ok")

    def test_a_selection_error_is_reported_rather_than_raised(self):
        def refusing_chat(messages, **kwargs):
            return {"ok": False, "error": "429 rate limited"}

        run = self.run_stage([scored_job("a", "Senior AI Engineer")],
                             refusing_chat, top=1)
        self.assertEqual(run.outcomes[0].status, "FAILED_AT_tailor")
        self.assertIn("429", run.outcomes[0].reason)


class TheBudgetCeilingStopsTheRunHonestly(StageCase):
    def _run(self, max_cost_usd):
        jobs = [scored_job(f"j{i}", f"AI Engineer {i}", adjusted=90 - i)
                for i in range(5)]
        return self.run_stage(jobs, stub_chat(), top=5,
                              max_cost_usd=max_cost_usd)

    def test_the_run_stops_once_the_ceiling_is_reached(self):
        run = self._run(0.025)
        self.assertTrue(run.budget_exceeded)
        self.assertLess(len(run.rendered), 5)

    def test_the_jobs_not_done_are_named_rather_than_dropped(self):
        run = self._run(0.025)
        self.assertEqual(len(run.outcomes), 5)
        self.assertEqual(len(run.rendered) + len(run.skipped), 5)
        self.assertTrue(run.skipped)

    def test_the_summary_says_the_run_is_incomplete(self):
        summary = self._run(0.025).summary()
        self.assertIn("INCOMPLETE", summary)
        self.assertIn("skipped (budget)", summary)

    def test_the_skip_reason_states_what_was_already_spent(self):
        run = self._run(0.025)
        self.assertIn("ceiling", run.skipped[0].reason)
        self.assertIn("0.0", run.skipped[0].reason)

    def test_a_generous_ceiling_completes_the_whole_run(self):
        """Guards against a ceiling that stops early for the wrong reason."""
        run = self._run(10.0)
        self.assertFalse(run.budget_exceeded)
        self.assertEqual(len(run.rendered), 5)

    def test_cost_is_summed_from_the_tailor_results(self):
        run = self._run(10.0)
        self.assertAlmostEqual(run.cost_usd, 5 * COST_PER_CALL, places=6)

    def test_skipped_jobs_still_reach_the_workbook(self):
        run = self._run(0.025)
        rows = read_sheets(run.workbook)["ai-engineer-sg"]
        column = rows[0].index("tailoring")
        notes = [row[column] for row in rows[1:] if row[column]]
        self.assertTrue(any("skipped_budget" in n for n in notes))


class TwoJobsWithTheSameTitleDoNotOverwriteEachOther(StageCase):
    def test_each_job_gets_its_own_directory(self):
        jobs = [scored_job("a", "AI Engineer", adjusted=90,
                           company="Umbra Financial"),
                scored_job("b", "AI Engineer", adjusted=80,
                           company="Umbra Financial")]
        run = self.run_stage(jobs, stub_chat(), top=2)
        directories = [o.directory for o in run.outcomes]
        self.assertNotEqual(directories[0], directories[1])
        self.assertEqual(len({str(d) for d in directories}), 2)


class TheStageIsOffUnlessAskedFor(unittest.TestCase):
    """Every existing `run_scopes` caller must see exactly what it saw before."""

    def _run_scopes(self, tmp, **kwargs):
        history = job_store.JobHistory.load(
            path=Path(tmp) / "s.jsonl", snapshot_path=Path(tmp) / "st.json")
        return pipeline.run_scopes(
            [SCOPE], scoring.load_config(), history=history,
            fetch_jobs=_fake_fetch([_job("a"), _job("b", "B")]),
            output_dir=Path(tmp) / "out", **kwargs)

    def test_without_tailor_options_there_is_no_tailoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_scopes(tmp)
            self.assertIsNone(result.tailoring)

    def test_without_tailor_options_only_the_search_artefacts_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_scopes(tmp)
            self.assertEqual(sorted(p.name for p in result.written),
                             ["ranked.csv", "ranked.json"])

    def test_the_ranked_result_is_unchanged_by_the_new_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._run_scopes(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            explicit = self._run_scopes(tmp, tailor_options=None)
        self.assertEqual([j["job_key"] for j in plain.jobs],
                         [j["job_key"] for j in explicit.jobs])
        self.assertEqual(plain.counters, explicit.counters)

    def test_run_scopes_can_carry_the_stage_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_scopes(tmp, tailor_options={
                "profile": PROFILE, "top": 1, "max_cost_usd": 1.0,
                "chat": stub_chat(), "wave": 1})
            self.assertIsNotNone(result.tailoring)
            self.assertEqual(len(result.tailoring.rendered), 1)

    def test_a_dry_run_never_tailors(self):
        """Tailoring writes documents, which is what --dry-run promises not to."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_scopes(tmp, dry_run=True, tailor_options={
                "profile": PROFILE, "top": 1, "chat": stub_chat()})
            self.assertIsNone(result.tailoring)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class OneRunWritesOneDirectory(unittest.TestCase):
    """A single run wrote ranked.csv into 20260719T133510Z and every resume,
    report and workbook into 2026-07-19_213859 -- two directories, hours apart
    by name, for one run. `write_outputs` uses the UTC run_id and `tailor_jobs`
    minted its own local-time stamp.
    """

    def test_tailoring_reuses_the_runs_own_id(self):
        import inspect
        source = inspect.getsource(pipeline.run_scopes)
        self.assertIn('setdefault("stamp", result.run_id)', source,
                      "run_scopes must pass its run_id to tailor_jobs, or the "
                      "run's artefacts split across two directories")

    def test_an_explicit_stamp_lands_in_the_path(self):
        run = pipeline.tailor_jobs(
            [], {}, "scope-x", stamp="20260719T133510Z",
            output_dir=Path(tempfile.mkdtemp()))
        self.assertIn("20260719T133510Z", str(run.root))


class EveryArtefactOfOneRunSharesOneRoot(unittest.TestCase):
    """The outcome test the mechanism test missed.

    `OneRunWritesOneDirectory` asserts run_scopes passes its run_id to
    tailor_jobs -- the mechanism. It stayed green while a candidate segment was
    added to `paths.run_root` and to the tailoring stage but NOT to
    `write_outputs`, so ranked.csv landed in one tree and fifteen tailored
    resumes in another. Same bug as the earlier local-time/UTC split, second
    cause, and a mechanism test cannot see either.

    This asserts what actually matters: everything a run writes shares a
    parent, whoever the candidate is.
    """

    def _run(self, profile):
        tmp = tempfile.mkdtemp()
        out = Path(tmp) / "out"
        history = job_store.JobHistory.load(
            path=Path(tmp) / "s.jsonl", snapshot_path=Path(tmp) / "st.json")
        result = pipeline.run_scopes(
            [SCOPE], scoring.load_config(), history=history,
            fetch_jobs=_fake_fetch([_job("a"), _job("b", "B")]),
            output_dir=out,
            tailor_options={"profile": profile, "top": 1,
                            "chat": stub_chat(CLEAN_SELECTION),
                            "max_cost_usd": 10.0})
        return out, result

    def test_ranked_output_and_tailored_output_share_a_directory(self):
        out, result = self._run(dict(PROFILE, identity={"name": "Alex Tan"}))
        ranked = list(out.rglob("ranked.*"))
        self.assertTrue(ranked, "no ranked artefact written")
        for path in ranked:
            with self.subTest(artefact=path.name):
                self.assertEqual(
                    path.parent, result.tailoring.root,
                    f"{path.name} landed in {path.parent}, tailored output in "
                    f"{result.tailoring.root} -- one run, two directories")

    def test_the_named_candidate_appears_in_that_shared_root(self):
        out, result = self._run(dict(PROFILE, identity={"name": "Alex Tan"}))
        self.assertIn("Alex Tan", str(result.tailoring.root))
        for path in out.rglob("ranked.*"):
            self.assertIn("Alex Tan", str(path))

    def test_an_anonymous_profile_still_shares_one_root(self):
        """The no-candidate path must not diverge either."""
        out, result = self._run(dict(PROFILE, identity={}))
        ranked = list(out.rglob("ranked.*"))
        self.assertTrue(ranked)
        for path in ranked:
            self.assertEqual(path.parent, result.tailoring.root)
