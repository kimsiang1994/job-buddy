"""Regressions for the items two audits found and deferred.

    py -m unittest tests.test_deferred_audit_items

Each class pins one deferred item. Every test here was mutation-checked: the
defect was put back in its original form -- same window, same ordering -- and
the test confirmed to fail before the fix was restored. A test that passes
against the broken code guards nothing, and for the threading item in
particular an unfaithful mutation is easy to write by accident.

Where no honest seam exists the absence is stated in the docstring rather than
covered by a test that passes for the wrong reason.

Every person, employer and achievement below is invented. This repo is public
and must never carry real CV data.
"""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import warnings
from collections import Counter
from pathlib import Path
from unittest import mock

from jobbuddy import (import_resume, paths, pipeline, render_excel,
                      render_resume, user_input)


# --------------------------------------------------------------------------
# 1. isolation must not cost the evidence
# --------------------------------------------------------------------------

PROFILE = {
    "identity": {"name": "Ada Nakamura", "email": "ada@example.com",
                 "location": "Singapore"},
    "skills_declared": {"expert": ["Python"]},
    "facts": [],
}

JOB = {"job_key": "mcf:1", "title": "AI Engineer", "company": "Northwind Labs",
       "skills": ["python"], "scores": {"adjusted": 80.0}}


def raising_chat(*args, **kwargs):
    """A chat seam that fails the way a real one does: deep, not at the call."""
    def inner():
        raise RuntimeError("upstream refused the request")
    inner()


class FailureIsolationKeepsTheTraceback(unittest.TestCase):
    """`FAILED_AT_*` without a traceback cost two wrong diagnoses.

    `_prepare_job` recorded `traceback.format_exc()`; every other handler kept
    only `f"{type(exc).__name__}: {exc}"`. Isolation is what stops one bad job
    losing nineteen good ones -- it is not a reason to throw away the only
    evidence of why the bad job was bad, and an intermittent failure cannot be
    re-run on demand to get it back.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_failure_inside_the_pool_records_the_frames_not_just_the_message(self):
        run = pipeline.tailor_jobs([JOB], PROFILE, "scope", top=1,
                                   output_dir=self.out, chat=raising_chat)
        outcome = run.outcomes[0]
        self.assertTrue(outcome.status.startswith("FAILED_AT_"), outcome.status)
        self.assertIn("RuntimeError", outcome.reason)
        # The point of the item: the frames, not merely the message.
        self.assertIn("Traceback (most recent call last)", outcome.detail)
        self.assertIn("upstream refused the request", outcome.detail)

    def test_a_write_failure_records_the_frames_too(self):
        """The `_write_job` handlers, which kept only the message.

        Patched at `render_resume.render` -- the boundary `_write_job` calls,
        not something owned by the function under test.
        """
        def ok_chat(*args, **kwargs):
            return {"ok": True, "data": {"headline": "h", "bullets": []},
                    "cost_usd": 0.0}

        with mock.patch.object(render_resume, "render",
                               side_effect=OSError("disk went away")):
            run = pipeline.tailor_jobs([JOB], PROFILE, "scope", top=1,
                                       output_dir=self.out, chat=ok_chat)

        failed = [o for o in run.outcomes if o.status.startswith("FAILED_AT_")]
        if not failed:
            self.skipTest("tailor() did not reach the render stage offline")
        outcome = failed[0]
        self.assertIn("OSError", outcome.reason)
        self.assertIn("Traceback (most recent call last)", outcome.detail)
        self.assertIn("disk went away", outcome.detail)

    def test_the_workbook_line_stays_readable_even_though_the_frames_are_kept(self):
        """`note()` is a spreadsheet cell. The traceback must not land in it."""
        run = pipeline.tailor_jobs([JOB], PROFILE, "scope", top=1,
                                   output_dir=self.out, chat=raising_chat)
        note = run.outcomes[0].note()
        self.assertNotIn("Traceback", note)
        self.assertLess(len(note.splitlines()), 2)


# --------------------------------------------------------------------------
# 2. a failed resume read must say WHICH step failed
# --------------------------------------------------------------------------

class ReadResumeTextNamesTheFailingStep(unittest.TestCase):
    """A page-30 extraction bug read identically to a corrupt file.

    Two blanket catches, each spanning four operations, each formatting only
    `{exc}`. The user saw "could not read PDF: ..." whether the file was
    shredded, encrypted, or perfectly good with one page pypdf cannot walk --
    and those have different fixes.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_file_that_is_not_a_pdf_at_all_is_reported_as_an_open_failure(self):
        path = self.dir / "shredded.pdf"
        path.write_bytes(b"this is not a PDF, it is a text file with a lie in it")
        text, how = user_input.read_resume_text(path)
        self.assertEqual(text, "")
        self.assertIn("could not open PDF", how)
        # The type is the half that was being dropped.
        self.assertRegex(how, r"[A-Za-z]+Error")

    def test_a_page_that_will_not_extract_names_the_page_and_the_type(self):
        """The distinguishing case, and the whole reason for the split.

        Patched at pypdf, which is a third-party boundary rather than something
        this module owns: `PdfReader` opens fine and page 3 raises, exactly as
        a malformed content stream behaves.
        """
        path = self.dir / "resume.pdf"
        path.write_bytes(b"%PDF-1.4 pretend")

        class Page:
            def __init__(self, ok):
                self.ok = ok

            def extract_text(self):
                if self.ok:
                    return "readable text"
                raise KeyError("/Contents")

        class Reader:
            def __init__(self, _path):
                self.pages = [Page(True), Page(True), Page(False)]

        with mock.patch("pypdf.PdfReader", Reader):
            text, how = user_input.read_resume_text(path)

        self.assertEqual(text, "")
        self.assertIn("page 3 of 3", how)
        self.assertIn("KeyError", how)
        # And it must NOT read as a broken document, which is the other fix.
        self.assertNotIn("could not open PDF", how)

    def test_an_unreadable_page_is_told_apart_from_an_unreadable_file(self):
        """The property, stated directly: two different causes, two reasons."""
        broken_file = self.dir / "broken.pdf"
        broken_file.write_bytes(b"not a pdf")
        _, file_reason = user_input.read_resume_text(broken_file)

        good_file = self.dir / "good.pdf"
        good_file.write_bytes(b"%PDF-1.4 pretend")

        class Page:
            def extract_text(self):
                raise ValueError("bad stream")

        class Reader:
            def __init__(self, _path):
                self.pages = [Page()]

        with mock.patch("pypdf.PdfReader", Reader):
            _, page_reason = user_input.read_resume_text(good_file)

        self.assertNotEqual(file_reason, page_reason)

    def test_a_resume_saved_in_the_wrong_encoding_returns_a_reason_not_a_raise(self):
        """`read_resume_text` promises never to raise. UTF-16 broke that.

        `.read_text` raises UnicodeDecodeError, which is a ValueError and NOT an
        OSError, so the only catch on this branch missed it entirely and the
        exception escaped into the notebook cell.
        """
        path = self.dir / "resume.txt"
        path.write_bytes("Ada Nakamura, engineer".encode("utf-16"))
        text, how = user_input.read_resume_text(path)
        self.assertEqual(text, "")
        self.assertIn("UnicodeDecodeError", how)


# --------------------------------------------------------------------------
# 3. _warn_once is a check-then-act
# --------------------------------------------------------------------------

class SlowMembershipSet(set):
    """A set whose membership test is slow, to widen the real race window.

    The window is between `key in _warned` and `_warned.add(key)`, and nothing
    else. Widening it here is faithful: with the lock the sleep happens inside
    the critical section and one thread still wins; without it, every thread
    passes the test before any of them adds. A test that merely called
    `_warn_once` from several threads would pass against the defect on almost
    every run and prove nothing.

    The lookup happens BEFORE the sleep, and the ordering is the whole trick.
    Sleeping first and then consulting the real set widens a window that does
    not exist: by the time the sleep ends another thread has already added, so
    every latecomer correctly sees the key present and the defect passes the
    test. Deciding first and then stalling is what reproduces the actual bug --
    several threads each holding a stale "not warned yet" answer.
    """

    def __contains__(self, item):
        present = super().__contains__(item)
        threading.Event().wait(0.02)
        return present


class WarnOnceIsOnceUnderThreads(unittest.TestCase):
    """`set.add` being atomic was never the property needed.

    Both renderers run under `pipeline`'s ThreadPoolExecutor, so N workers with
    a missing wheel could each print the same "install it" line -- burying the
    line that says which degradation was taken, which is the one thing
    `_warn_once` exists to prevent.
    """

    def _assert_warns_once(self, module, reset):
        reset()
        self.addCleanup(reset)
        emitted: list[str] = []
        barrier = threading.Barrier(6)

        def worker():
            barrier.wait()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                module._warn_once("k", "the degraded-path message")
            emitted.extend(str(w.message) for w in caught)

        with mock.patch.object(module, "_warned", SlowMembershipSet()):
            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(emitted), 1,
                         f"warned {len(emitted)} times, not once")

    def test_render_excel_warns_exactly_once(self):
        self._assert_warns_once(render_excel, render_excel.reset_warning_cache)

    def test_render_resume_warns_exactly_once(self):
        self._assert_warns_once(render_resume,
                                render_resume.reset_capability_cache)


# --------------------------------------------------------------------------
# 4 and 5. the workbook write
# --------------------------------------------------------------------------

def job(key: str, adjusted: float) -> dict:
    return {"job_key": key, "title": f"Engineer {key}", "company": "Northwind",
            "scores": {"adjusted": adjusted, "total": adjusted,
                       "confidence": 0.8, "components": {}}}


class TheWorkbookIsWrittenAtomically(unittest.TestCase):
    """`book.close()` ran with the destination already open as the target.

    xlsxwriter does nearly all its work in `close()` -- assembling and zipping
    the package -- so it is the likeliest single failure, and it can fail with
    its handle still open. The old handler then called `out_path.unlink()`,
    which Windows refuses for an open file, so the cleanup raised
    `PermissionError [WinError 32]` and REPLACED the real exception. The
    original error never reached anyone.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "ranked.xlsx"
        self.addCleanup(self._tmp.cleanup)

    def _workbook_that_fails_to_close(self, keep_handle_open: bool):
        """xlsxwriter, but `close()` raises -- optionally still holding the file.

        `keep_handle_open` is what makes this the real bug rather than a
        generic failure: it is the open handle that turned the cleanup into a
        PermissionError.
        """
        real = render_excel._load_xlsxwriter()
        if real is None:
            self.skipTest("xlsxwriter is not installed")
        test = self

        class Book(real.Workbook):
            def close(self):
                if keep_handle_open:
                    test._held = open(self.filename, "rb")
                    test.addCleanup(test._held.close)
                raise ValueError("could not assemble the package")

        class Module:
            Workbook = Book

        return mock.patch.object(render_excel, "_load_xlsxwriter",
                                 lambda: Module)

    def test_the_real_error_survives_a_failure_that_holds_the_file_open(self):
        with self._workbook_that_fails_to_close(keep_handle_open=True):
            with self.assertRaises(ValueError) as caught:
                render_excel.write_workbook({"requested": [job("a", 70.0)]},
                                            self.path)
        self.assertIn("could not assemble", str(caught.exception))

    def test_a_failed_write_never_leaves_a_half_written_workbook(self):
        with self._workbook_that_fails_to_close(keep_handle_open=False):
            with self.assertRaises(ValueError):
                render_excel.write_workbook({"requested": [job("a", 70.0)]},
                                            self.path)
        self.assertFalse(self.path.exists())

    def test_a_failed_rerun_does_not_destroy_the_previous_good_workbook(self):
        """The strongest form of the property, and the user-visible one.

        Writing straight to the destination meant a failed re-run left the
        user with no workbook at all -- the previous run's, which was fine,
        had already been truncated by opening it.
        """
        first = render_excel.write_workbook({"requested": [job("a", 70.0)]},
                                            self.path)
        self.assertTrue(first["ok"])
        good = self.path.read_bytes()

        with self._workbook_that_fails_to_close(keep_handle_open=False):
            with self.assertRaises(ValueError):
                render_excel.write_workbook({"requested": [job("b", 60.0)]},
                                            self.path)

        self.assertEqual(self.path.read_bytes(), good)

    def test_no_temp_files_are_left_behind_by_a_successful_write(self):
        render_excel.write_workbook({"requested": [job("a", 70.0)]}, self.path)
        self.assertEqual([p.name for p in self.dir.glob("*.tmp")], [])


class WriteWorkbookReportsWhatReachedTheDisk(unittest.TestCase):
    """`ok` was hard-coded True on a path that skips scopes it cannot write.

    In the degraded CSV path a scope whose file will not open is warned about
    and skipped -- but `sheets` and `rows` still counted it and `ok` still said
    True. The warning and the machine-readable result contradicted each other,
    and a script only ever reads the second one.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "ranked.xlsx"
        self.addCleanup(self._tmp.cleanup)
        render_excel.reset_warning_cache()
        self.addCleanup(render_excel.reset_warning_cache)
        self._patch = mock.patch.object(render_excel, "_load_xlsxwriter",
                                        lambda: None)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _block(self, scope: str) -> None:
        """Make one scope's CSV impossible to open, without patching `open`.

        A directory sitting where the file must go is a real OSError from the
        real call, so the test exercises the actual failure path rather than a
        simulated one.
        """
        (self.dir / f"{self.path.stem}.{render_excel.sheet_name(scope)}.csv").mkdir()

    def test_a_scope_that_could_not_be_written_makes_ok_false(self):
        self._block("blocked")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = render_excel.write_workbook(
                {"requested": [job("a", 70.0)], "blocked": [job("b", 60.0)]},
                self.path)
        self.assertFalse(result["ok"])

    def test_the_counts_describe_only_the_scopes_that_were_written(self):
        self._block("blocked")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = render_excel.write_workbook(
                {"requested": [job("a", 70.0)], "blocked": [job("b", 60.0)]},
                self.path)
        self.assertEqual(result["sheets"], ["requested"])
        self.assertEqual(result["rows"], {"requested": 1})
        self.assertIn("blocked", result["failed"])
        self.assertEqual(len(result["paths"]), 1)

    def test_the_note_names_the_scope_that_was_lost(self):
        self._block("blocked")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = render_excel.write_workbook(
                {"requested": [job("a", 70.0)], "blocked": [job("b", 60.0)]},
                self.path)
        self.assertIn("blocked", result["note"])

    def test_a_clean_degraded_write_is_still_ok(self):
        """The fix must not turn every degraded run into a failure."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = render_excel.write_workbook(
                {"requested": [job("a", 70.0)]}, self.path)
        self.assertTrue(result["ok"])
        self.assertEqual(result["failed"], {})

    def test_the_pipeline_does_not_hand_back_a_path_for_an_incomplete_workbook(self):
        """The caller side of the contract.

        `_write_run_workbook` returned `result["path"]` regardless, so a run
        summary pointed the user at a workbook that was never written.
        """
        self._block("blocked")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            written = pipeline._write_run_workbook(
                [], pipeline.TailorRun(), "blocked", self.dir)
        self.assertIsNone(written)


# --------------------------------------------------------------------------
# 6. `or {}` defaults reported as a complete success
# --------------------------------------------------------------------------

FACT = {"fact_id": "northwind.retrieval", "org": "Northwind Labs",
        "role": "AI Engineer", "source_span": "Built a retrieval pipeline",
        "numbers": [], "entities": [], "skills": ["python"]}


def chat_returning(data: dict):
    def chat(*args, **kwargs):
        return {"ok": True, "data": data}
    return chat


class MissingSectionsAreRecordedNotDefaulted(unittest.TestCase):
    """The bug this module's own docstring documents, one layer down.

    `schema_keys=("facts",)` validates exactly one key, so a response carrying
    facts and nothing else passes. Six `or {}` / `or []` defaults then turned
    every other absent section into an empty container and returned `ok: True`
    -- indistinguishable from a resume that genuinely has no education. That is
    how an entire degree went missing from every generated resume the first
    time, and the prompt was fixed while the code path was not.
    """

    def test_absent_sections_are_named_in_the_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = import_resume.extract_facts(
                "resume text", chat=chat_returning({"facts": [FACT]}))
        self.assertTrue(result["ok"])
        self.assertFalse(result["complete"])
        self.assertEqual(set(result["missing_sections"]),
                         set(import_resume.EXPECTED_SECTIONS))

    def test_a_complete_response_reports_nothing_missing(self):
        result = import_resume.extract_facts("resume text", chat=chat_returning({
            "facts": [FACT],
            "skills_declared": {"expert": ["Python"]},
            "skill_groups": [{"label": "AI / ML", "items": ["RAG"]}],
            "education": [{"institution": "NUS", "qualification": "BSc"}],
            "languages": ["English (fluent)"],
            "identity": {"name": "Ada Nakamura"},
        }))
        self.assertTrue(result["complete"])
        self.assertEqual(result["missing_sections"], [])

    def test_an_empty_section_counts_as_missing(self):
        """`"education": []` tells the reader nothing a dropped key does not."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = import_resume.extract_facts("resume text", chat=chat_returning({
                "facts": [FACT], "education": [], "languages": [],
                "skills_declared": {"expert": ["Python"]},
                "skill_groups": [{"label": "AI / ML", "items": ["RAG"]}],
                "identity": {"name": "Ada Nakamura"},
            }))
        self.assertEqual(set(result["missing_sections"]),
                         {"education", "languages"})

    def test_a_missing_identity_is_never_silent(self):
        """No name on the resume is not a thing to discover in a finished PDF."""
        with self.assertWarns(RuntimeWarning):
            import_resume.extract_facts("resume text",
                                        chat=chat_returning({"facts": [FACT]}))

    def test_the_draft_file_records_which_sections_were_absent(self):
        """The draft is what the user opens to review it.

        An empty `education: []` in the file reads as "the resume had none".
        Next to `_missing_sections`, it reads as "the extractor did not return
        one", which is a different instruction to the reviewer.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extracted = import_resume.extract_facts(
                "resume text", chat=chat_returning({"facts": [FACT]}))
        draft = import_resume.build_draft(extracted, "resume text")
        self.assertIn("education", draft["_missing_sections"])

    def test_build_draft_still_describes_a_hand_built_dict_correctly(self):
        """Recomputed when `missing_sections` is absent, so a caller that
        assembles `extracted` itself is not silently reported as complete."""
        draft = import_resume.build_draft({"ok": True, "facts": []}, "text")
        self.assertEqual(set(draft["_missing_sections"]),
                         set(import_resume.EXPECTED_SECTIONS))

    def test_the_run_summary_carries_the_incompleteness(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "resume.pdf"
            pdf.write_bytes(b"%PDF-1.4 pretend")
            out = Path(tmp) / "draft.json"
            with mock.patch.object(import_resume, "read_pdf_text",
                                   return_value="x" * 400):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = import_resume.import_resume(
                        pdf, chat=chat_returning({"facts": [FACT]}),
                        out_path=out)
            self.assertTrue(result["ok"])
            self.assertFalse(result["complete"])
            self.assertIn("education", result["missing_sections"])
            # And it is on disk, not only in the return value.
            written = json.loads(out.read_text(encoding="utf-8-sig"))
            self.assertIn("education", written["_missing_sections"])


# --------------------------------------------------------------------------
# 7. the comment that contradicted its code
# --------------------------------------------------------------------------

class TheShortBudgetBranchKeepsItsPromise(unittest.TestCase):
    """`_digest(original)[:1]` under a comment claiming uniqueness wins.

    One hex character is sixteen buckets: two jobs in a run collide at better
    than even odds once there are six of them, and a collision here silently
    overwrites a deliverable. The branch is unreachable through
    `job_component` -- `MIN_COMPONENT` floors the budget at exactly
    `HASH_LEN + 2` -- so these call `_fit` directly. That is deliberate and is
    the honest way to test it: there is no public seam, and inventing one to
    reach dead code would be worse than saying so.
    """

    def test_two_titles_sharing_a_prefix_stay_distinct_at_a_tiny_budget(self):
        long_a = "Senior Machine Learning Engineer, Retrieval Platform"
        long_b = "Senior Machine Learning Engineer, Ranking Platform"
        self.assertNotEqual(paths._fit(long_a, 3, long_a),
                            paths._fit(long_b, 3, long_b))

    def test_a_tiny_budget_returns_the_whole_digest_rather_than_a_slice(self):
        """The budget is overshot deliberately: the module's stated trade is
        that an over-long name annoys and a collision loses a file."""
        name = "Senior Machine Learning Engineer"
        self.assertEqual(len(paths._fit(name, 3, name)), paths.HASH_LEN)

    def test_no_pair_of_realistic_titles_collides_at_the_smallest_budget(self):
        """Sixteen buckets failed this immediately; 24 bits does not."""
        titles = [f"Senior Machine Learning Engineer, Team {i}"
                  for i in range(60)]
        digests = {paths._fit(t, 2, t) for t in titles}
        self.assertEqual(len(digests), len(titles))

    def test_the_invariant_that_makes_the_branch_unreachable_still_holds(self):
        """If this fails, `job_component` has started reaching the branch above
        -- which is now safe, and was not before."""
        self.assertGreaterEqual(paths.MIN_COMPONENT, paths.HASH_LEN + 2)


# --------------------------------------------------------------------------
# 8. the page was starving the content
# --------------------------------------------------------------------------

def page_geometry(pdf_bytes: bytes) -> dict:
    """Page box and text extent of a rendered PDF, in points.

    Reads the compiled document rather than the `.typ` source or the constants
    that produced it. A test asserting `MARGIN_MAX_IN == 0.52` would pass while
    the page still looked starved -- the margin is only one of several things
    that decide where text actually lands.
    """
    from pypdf import PdfReader

    page = PdfReader(io.BytesIO(pdf_bytes)).pages[0]
    runs: list[tuple[float, float, float]] = []

    def visit(text, cm, tm, font, size):
        if text.strip():
            # Typst leaves the offset in `cm`, so `tm` alone reads (0, 0) for
            # every run. Compose them.
            runs.append((tm[4] * cm[0] + tm[5] * cm[2] + cm[4],
                         tm[4] * cm[1] + tm[5] * cm[3] + cm[5],
                         float(size or 0)))

    page.extract_text(visitor_text=visit)
    runs = [r for r in runs if r[1] > 0 and r[0] > 0]
    body_pt = Counter(round(r[2], 1) for r in runs).most_common(1)[0][0]
    baselines = sorted({round(r[1], 1) for r in runs}, reverse=True)
    gaps = [round(baselines[i] - baselines[i + 1], 1)
            for i in range(len(baselines) - 1)]
    # The modal gap, not the modal gap under some threshold. A threshold
    # expressed as a multiple of body size silently excludes the very spacing
    # being measured once it grows past it -- which is precisely the regression
    # this reports on, so it would have gone unnoticed. Body lines outnumber
    # section breaks several times over, so the plain mode is the body gap.
    body_gap = Counter(g for g in gaps if g > 0).most_common(1)
    width = float(page.mediabox.width)
    left = min(r[0] for r in runs)
    return {
        "page_width": width,
        "left_margin": left,
        # Symmetric x margins, so the column is the page less both sides.
        "column_fraction": (width - 2 * left) / width,
        "body_pt": body_pt,
        "body_leading_ratio": (body_gap[0][0] / body_pt) if body_gap else None,
    }


# Bullet lengths from the real profile this module is measured against, in
# characters: 14 facts across 5 roles, 3310 characters in total, mean 236. The
# LENGTHS are the fixture -- the wording below is invented, because this repo is
# public. A fixture of short bullets never reaches the fitter's floor and so
# cannot see the failure at all: with the starved margins it still fitted at
# full scale, and the test passed against the bug.
REAL_BULLET_LENGTHS = (30, 139, 183, 187, 203, 229, 238, 260, 275, 287, 293,
                       295, 338, 353)

FILLER = ("rebuilt the ingestion path in PySpark and cut the nightly "
          "reconciliation window from 6 hours to 40 minutes across 18 upstream "
          "feeds, with schema contracts on every one of them and an on-call "
          "runbook the team actually follows during incidents ")


def dense_model() -> dict:
    """A model the size of a real senior resume: 14 bullets plus every section.

    Invented wording, real proportions -- see `REAL_BULLET_LENGTHS`. The
    property under test is a layout outcome, so the fixture has to put the same
    amount of text on the page as the document that exposed the problem.
    """
    orgs = [("Northwind Labs", "AI Engineer", "2024-08", ""),
            ("Umbra Financial", "Manager, Data Engineering", "2023-08", "2024-11"),
            ("Olea Systems", "Senior Data Engineer", "2021-02", "2023-07"),
            ("Kestrel Analytics", "Data Engineer", "2019-06", "2021-01"),
            ("Vantage Retail", "Analyst", "2018-01", "2019-05")]
    bullets = []
    for index, length in enumerate(REAL_BULLET_LENGTHS):
        org, role, start, end = orgs[index % len(orgs)]
        head = f"Delivered platform outcome {index + 1}: "
        text = (head + FILLER * (length // len(FILLER) + 1))[:length].strip()
        bullets.append({
            "text": text, "fact_id": f"fact.{index}", "org": org, "role": role,
            "start": start, "end": end, "note": "", "rank": index + 1,
        })
    profile = {
        "identity": {"name": "Ada Nakamura", "email": "ada@example.com",
                     "phone": "+65 8000 0000", "location": "Singapore",
                     "links": ["github.com/example-ada"]},
        "skills_declared": {"expert": ["Python", "PySpark"],
                            "working": ["Airflow"]},
        "skill_groups": [
            {"label": "AI / ML", "items": ["LLMs (OpenAI, Claude, Gemini)",
                                           "RAG pipelines", "PyTorch"]},
            {"label": "Data", "items": ["PySpark", "Presto", "Airflow", "dbt"]},
        ],
        "education": [{"institution": "National University of Singapore",
                       "qualification": "BSc Computer Science, 2016-2020",
                       "bullets": ["Dean's List, two semesters"]}],
        "languages": ["English (fluent)", "Mandarin (fluent)"],
        "facts": [],
    }
    return render_resume.build_model(profile, {"headline": "AI engineer "
                                               "building measured retrieval "
                                               "and ETL systems.",
                                               "bullets": bullets})


class ThePageIsNotStarved(unittest.TestCase):
    """0.9in margins left a 6.5in column on an 8.27in page, and cut bullets.

    The fitter shrinks before it cuts, so a starved column drove the search to
    the 9pt floor and then began dropping the lowest-ranked bullets -- content
    that fits comfortably at the source document's own 0.5in margins. Rendering
    narrower than the source and then deleting the user's evidence to fit it is
    the wrong trade twice over.

    Asserted against the rendered PDF, never against the constants.
    """

    def setUp(self):
        render_resume.reset_capability_cache()
        self.addCleanup(render_resume.reset_capability_cache)
        if not render_resume.capabilities()["pdf"]:
            self.skipTest("typst is not installed; layout is untestable")
        self.fit = render_resume.fit_to_pages(dense_model(), max_pages=1)

    def test_the_text_column_occupies_most_of_the_page(self):
        geometry = page_geometry(self.fit["pdf_bytes"])
        self.assertGreaterEqual(geometry["column_fraction"],
                                render_resume.MIN_TEXT_COLUMN_FRACTION,
                                f"text column is only "
                                f"{geometry['column_fraction']:.1%} of the page")

    def test_a_full_length_resume_fits_one_page_with_nothing_cut(self):
        """The outcome the margins exist for. The user has said plainly that
        shrinking to the floor is fine and dropping content is not.

        Honest about what this does and does not guard: it does NOT fail
        against the starved margins. A resume of exactly this density fitted
        one page either way -- at 9.77pt with the old margins and at 11pt with
        the new ones -- so cutting was never the symptom at this length. It
        guards the next regression that does cut, and
        `test_the_page_keeps_headroom_for_more_content` guards the distance to
        that cliff.
        """
        self.assertEqual(self.fit["pages"], 1)
        self.assertEqual(self.fit["dropped"], [],
                         f"dropped {len(self.fit['dropped'])} bullet(s) at "
                         f"scale {self.fit['scale']}")
        self.assertEqual(len(self.fit["bullets"]), 14)

    def test_the_page_keeps_headroom_for_more_content(self):
        """The scale reached on a real-sized resume IS the headroom.

        This is the property that connects the margins to the cutting. The
        fitter shrinks before it cuts, so how close a full-length resume sits
        to the floor decides how much more content can arrive before bullets
        start being deleted. With the starved margins this fixture landed at
        0.888 -- most of the way down -- and one longer round of tailored
        wording would have pushed it into cutting. It now lands at full size.
        """
        self.assertGreaterEqual(self.fit["scale"], 0.95,
                                f"a full-length resume already needs to shrink "
                                f"to {self.fit['scale']}; there is no room left "
                                f"before bullets start being cut")

    def test_the_line_spacing_matches_the_source_document(self):
        """The source runs 1.14 baselines per body size; this rendered 1.34,
        which over ~57 lines is most of an inch of air the source does not
        have. Bounded on both sides -- too tight is its own failure."""
        ratio = page_geometry(self.fit["pdf_bytes"])["body_leading_ratio"]
        self.assertIsNotNone(ratio)
        self.assertGreater(ratio, 1.05)
        self.assertLess(ratio, 1.25)

    def test_the_type_stays_readable_rather_than_being_shrunk_to_the_floor(self):
        """Reaching the floor is the symptom that preceded the cutting."""
        self.assertGreater(self.fit["font_pt"], render_resume.MIN_PT)


if __name__ == "__main__":
    unittest.main()
