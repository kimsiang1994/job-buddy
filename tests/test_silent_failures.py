"""Fallback paths must announce themselves.

    py -m unittest tests.test_silent_failures

The repo's read paths deliberately cannot raise: a tool that dies because its
config is malformed is worse than one that warns and falls back. That
convention is only safe while the second half of it holds. A read path that
falls back *silently* is strictly worse than one that raises, because the run
completes, the numbers look plausible, and nothing points at the cause.

Every test here corrupts one input, takes the fallback, and asserts that
something was said about it. They are assertions about the WARNING, not about
the return value -- the return values were always correct.

Offline: no network, no API key, no cost.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path

from jobbuddy import (
    company_registry,
    fetcher,
    html_text,
    net,
    quota,
    skill_extract,
    source_aggregator,
    source_ats,
    source_hn,
)


@contextlib.contextmanager
def captured_warnings():
    """Collect what `net._warn` printed during the block.

    `_warn` dedupes by message for the life of the process, so the seen-set has
    to be cleared or the second test asserting on a given message gets nothing.
    """
    saved = set(net._warned)
    net._warned.clear()
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            yield buffer
    finally:
        net._warned.clear()
        net._warned.update(saved)


class CompanyRegistryReportsAnEmptyLoad(unittest.TestCase):
    """The bug: a corrupt registry loaded as {} and was then SAVED over itself.

    `observe()` calls `save(load())`, so one bad parse replaced the accumulated
    company -> ATS board map -- weeks of amortised discovery -- with an empty
    file, and printed nothing.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jobbuddy-registry-")
        self.path = Path(self.dir) / "companies.json"
        self.saved_env = os.environ.get(company_registry._PATH_ENV)
        os.environ[company_registry._PATH_ENV] = str(self.path)

    def tearDown(self):
        if self.saved_env is None:
            os.environ.pop(company_registry._PATH_ENV, None)
        else:
            os.environ[company_registry._PATH_ENV] = self.saved_env

    def test_unparseable_registry_says_it_is_starting_empty(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        with captured_warnings() as err:
            self.assertEqual(company_registry.load(), {})
        message = err.getvalue()
        self.assertIn("registry", message)
        self.assertIn("companies.json", message)
        self.assertIn("EMPTY", message)

    def test_warning_names_the_overwrite_risk(self):
        """The dangerous part is the next write, so the warning has to say so."""
        self.path.write_text("[]", encoding="utf-8")
        with captured_warnings() as err:
            self.assertEqual(company_registry.load(), {})
        self.assertIn("empty registry", err.getvalue())

    def test_a_healthy_registry_warns_about_nothing(self):
        self.path.write_text(
            json.dumps({"companies": {"acme": {"seen": 3}}}), encoding="utf-8")
        with captured_warnings() as err:
            self.assertEqual(company_registry.load(), {"acme": {"seen": 3}})
        self.assertEqual(err.getvalue(), "")


class SkillVocabularyReportsAnEmptyLoad(unittest.TestCase):
    """The bug: a corrupt vocabulary silently disabled the heaviest component.

    An empty vocabulary makes `extract` find nothing, so `skill_match` (weight
    30) returns None for every job from every source that publishes no
    structured skills. The run completes and the scores look reasonable. Worse,
    `harvest` writes the empty record straight back over the real file.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jobbuddy-vocab-")
        self.path = Path(self.dir) / "skill_vocab.json"

    def test_unparseable_vocab_says_extraction_is_running_empty(self):
        self.path.write_text("{{{", encoding="utf-8")
        with captured_warnings() as err:
            record = skill_extract._load_full(self.path)
        self.assertEqual(record["terms"], {})
        message = err.getvalue()
        self.assertIn("skills", message)
        self.assertIn("EMPTY", message)

    def test_wrong_toplevel_type_is_reported_not_assumed(self):
        self.path.write_text('["python", "sql"]', encoding="utf-8")
        with captured_warnings() as err:
            record = skill_extract._load_full(self.path)
        self.assertEqual(record["terms"], {})
        self.assertIn("not an", err.getvalue())

    def test_a_healthy_vocab_warns_about_nothing(self):
        self.path.write_text(
            json.dumps({"terms": {"python": 4}, "doc_freq": {}, "documents": 2}),
            encoding="utf-8")
        with captured_warnings() as err:
            record = skill_extract._load_full(self.path)
        self.assertEqual(record["terms"], {"python": 4})
        self.assertEqual(err.getvalue(), "")

    def test_unwritable_vocab_reports_the_lost_terms(self):
        """`harvest` discards save_vocab's return, so silence lost the harvest."""
        blocker = Path(self.dir) / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        with captured_warnings() as err:
            saved = skill_extract.save_vocab({"python": 1}, blocker / "vocab.json")
        self.assertFalse(saved)
        self.assertIn("not saved", err.getvalue())


class QuotaReportsAResetBudget(unittest.TestCase):
    """The bug: a corrupt usage file said "you have spent nothing this month".

    That re-arms the whole paid allowance for every caller, and the next
    `spend()` writes the fiction back. The failure mode is a bill, so it is the
    one fallback in the repo that must never be quiet.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jobbuddy-quota-")
        self.saved_path = quota.USAGE_PATH
        quota.USAGE_PATH = Path(self.dir) / "api_usage.json"

    def tearDown(self):
        quota.USAGE_PATH = self.saved_path

    def test_unparseable_usage_file_says_the_cap_is_unguarded(self):
        quota.USAGE_PATH.write_text("not json at all", encoding="utf-8")
        with captured_warnings() as err:
            self.assertEqual(quota.used("jsearch"), 0)
        message = err.getvalue()
        self.assertIn("quota", message)
        self.assertIn("ZERO", message)
        self.assertIn("unguarded", message)

    def test_usage_file_of_the_wrong_shape_is_reported(self):
        quota.USAGE_PATH.write_text('{"month": "2026-07"}', encoding="utf-8")
        with captured_warnings() as err:
            self.assertEqual(quota.used("jsearch"), 0)
        self.assertIn("not a usage record", err.getvalue())

    def test_unwritable_usage_file_says_spend_is_not_recorded(self):
        blocker = Path(self.dir) / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        quota.USAGE_PATH = blocker / "api_usage.json"
        with captured_warnings() as err:
            quota.spend("jsearch")
        message = err.getvalue()
        self.assertIn("NOT being recorded", message)
        self.assertIn("cap is not enforced", message)

    def test_a_new_month_resets_without_a_warning(self):
        """A genuine rollover is not a failure and must stay quiet."""
        quota.USAGE_PATH.write_text(
            json.dumps({"month": "1999-01", "counts": {"jsearch": 900}}),
            encoding="utf-8")
        with captured_warnings() as err:
            self.assertEqual(quota.used("jsearch"), 0)
        self.assertEqual(err.getvalue(), "")


class FailedErrorBodyReadIsReported(unittest.TestCase):
    """The bug: an unreadable HTTP error body was discarded with `except:`.

    For Adzuna and Careerjet the error BODY is the only place the real reason
    ("bad app_key") appears -- the status alone just says 403.
    """

    class _UnreadableHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("https://example.test/x", 403, "Forbidden", {}, None)

        def read(self, *_args, **_kwargs):
            raise OSError("connection reset while reading error body")

    class _Opener:
        def __init__(self, exc):
            self._exc = exc

        def open(self, request, timeout=None):
            raise self._exc

    class _FakeClock:
        """A fake sleep MUST advance a fake clock -- see net._throttle.

        A no-op sleep with a frozen clock turns the throttle into an infinite
        busy-spin, which is precisely the failure that docstring exists to warn
        about. Learned by hanging the suite on the first draft of this test.
        """

        def __init__(self):
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    def test_the_lost_error_detail_is_announced(self):
        opener = self._Opener(self._UnreadableHTTPError())
        clock = self._FakeClock()
        with captured_warnings() as err:
            result = net.fetch("https://example.test/x", opener=opener,
                               sleep=clock.sleep, clock=clock)
        self.assertFalse(result.ok)
        message = err.getvalue()
        self.assertIn("body unreadable", message)
        self.assertIn("error detail lost", message)


class ZyteEmptyResponseIsAFailure(unittest.TestCase):
    """The bug: `result.json() or {}` reported ok=True with a blank page.

    A quota notice or plan error from Zyte came back as a successful fetch of an
    empty document, so the caller reported "no jobs found" for a host we were
    being billed to read.
    """

    def setUp(self):
        self.saved_fetch = net.fetch
        self.saved_env = {k: os.environ.get(k)
                          for k in ("SCRAPING_PROVIDER", "SCRAPING_API_KEY")}
        os.environ["SCRAPING_PROVIDER"] = "zyte"
        os.environ["SCRAPING_API_KEY"] = "test-key"

    def tearDown(self):
        net.fetch = self.saved_fetch
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _respond_with(self, payload: bytes):
        def fake_fetch(url, **kwargs):
            return net.FetchResult(True, 200, url, body=payload,
                                   headers={"content-type": "application/json"})
        net.fetch = fake_fetch

    def test_missing_browser_html_is_not_reported_as_success(self):
        self._respond_with(b'{"error": "quota exceeded"}')
        result = fetcher._fetch_unblocker("https://www.glints.com/sg/jobs", 10)
        self.assertFalse(result.ok)
        self.assertIn("no browserHtml", result.error)

    def test_non_object_body_is_not_reported_as_success(self):
        self._respond_with(b'"upstream unavailable"')
        result = fetcher._fetch_unblocker("https://www.glints.com/sg/jobs", 10)
        self.assertFalse(result.ok)
        self.assertIn("non-object body", result.error)

    def test_a_real_page_still_succeeds(self):
        self._respond_with(b'{"browserHtml": "<html>jobs</html>"}')
        result = fetcher._fetch_unblocker("https://www.glints.com/sg/jobs", 10)
        self.assertTrue(result.ok)
        self.assertEqual(result.html, "<html>jobs</html>")


class MalformedWorkdayTokenIsReported(unittest.TestCase):
    """The bug: a broken registry entry looked like a company that is not hiring.

    Workday packs tenant|datacentre|site into one token. `token.split("|")`
    raising ValueError returned [] silently, so the counters showed a board
    fetched with zero postings, run after run.
    """

    def test_a_two_part_token_says_what_is_wrong_and_where(self):
        with captured_warnings() as err:
            self.assertEqual(source_ats._workday("acme|wd3", 0.0), [])
        message = err.getvalue()
        self.assertIn("malformed", message)
        self.assertIn("tenant|datacentre|site", message)
        self.assertIn("ats_boards.json", message)

    def test_an_empty_segment_is_also_rejected(self):
        """'acme||site' has three parts and used to build a nonsense hostname."""
        with captured_warnings() as err:
            self.assertEqual(source_ats._workday("acme||site", 0.0), [])
        self.assertIn("malformed", err.getvalue())


class RejectedJobsSayWhy(unittest.TestCase):
    """The bug: `if validate_job(job): continue` dropped records in silence.

    MCF and Workable both name the problems; HN threw the list away without
    even a counter, and the aggregators kept a count but no reason. When a
    vendor renames a field, every record starts failing on the same missing key
    -- which the count alone cannot tell you.
    """

    def setUp(self):
        self.saved = (source_hn.find_threads, source_hn.thread_comments,
                      source_hn.to_job)

    def tearDown(self):
        (source_hn.find_threads, source_hn.thread_comments,
         source_hn.to_job) = self.saved

    def test_hn_reports_the_validation_problems_and_counts_them(self):
        """`to_job` is stubbed because no HN *comment* can produce this record.

        `parse_posting` already rejects anything without a company and a title,
        so the only way a validation failure reaches `fetch_jobs` is a mapper
        that stops populating a field -- which is exactly the regression this
        guards: a field moves, every record starts failing on the same key, and
        the run reports it as "no Singapore roles in this thread".
        """
        source_hn.find_threads = lambda **_kw: [{"objectID": "1"}]
        source_hn.thread_comments = lambda _story, _ttl=0: iter([{"id": "99"}])

        def mapper_that_lost_a_field(comment):
            job = source_hn.job_schema.new_job("hn", comment["id"])
            job["title"] = "Machine Learning Engineer"
            job["company"] = "Acme"
            job["url"] = ""          # the field that moved
            job["jd_text"] = "machine learning in singapore"
            job["is_overseas"] = False
            return source_hn.job_schema.finalise(job)

        source_hn.to_job = mapper_that_lost_a_field
        with captured_warnings() as err:
            jobs, counters = source_hn.fetch_jobs("machine learning")
        self.assertEqual(jobs, [])
        self.assertEqual(counters["invalid"], 1)
        message = err.getvalue()
        self.assertIn("hn:", message)
        self.assertIn("url is empty", message)

    def test_aggregator_names_the_vendor_and_the_problem(self):
        saved = source_aggregator._jsearch_records
        saved_available = source_aggregator.available
        try:
            source_aggregator.available = lambda: {"jsearch": True, "adzuna": False}
            # A record with an id and a title but no apply link at all: the URL
            # is what validation rejects.
            source_aggregator._jsearch_records = lambda *_a, **_kw: [{
                "job_id": "abc123",
                "job_title": "ML Engineer",
                "employer_name": "Acme",
            }]
            with captured_warnings() as err:
                jobs, counters = source_aggregator.fetch_jobs("ml")
        finally:
            source_aggregator._jsearch_records = saved
            source_aggregator.available = saved_available
        self.assertEqual(jobs, [])
        self.assertEqual(counters["invalid"], 1)
        message = err.getvalue()
        self.assertIn("jsearch:", message)
        self.assertIn("url is empty", message)


class HtmlFallbackAnnouncesTheDegradation(unittest.TestCase):
    """The bug: the regex fallback silently lost block structure.

    Regex tag-stripping welds `<h3>Nice to have</h3>` onto the previous
    sentence, so `skill_extract` reads every optional skill as mandatory. A job
    could be scored on that worse extraction with nothing marking it out.
    """

    def setUp(self):
        self.saved = html_text.TextExtractor

    def tearDown(self):
        html_text.TextExtractor = self.saved

    def test_the_fallback_says_which_path_it_took_and_what_it_costs(self):
        class Exploding:
            def __init__(self, preserve_blocks=True):
                pass

            def feed(self, _html):
                raise RecursionError("nesting too deep")

        html_text.TextExtractor = Exploding
        with captured_warnings() as err:
            text = html_text.flatten_html("<p>Python</p><h3>Nice to have</h3>")
        self.assertIn("Python", text)
        message = err.getvalue()
        self.assertIn("RecursionError", message)
        self.assertIn("regex tag-stripping", message)
        self.assertIn("loses block structure", message)

    def test_ordinary_html_warns_about_nothing(self):
        with captured_warnings() as err:
            text = html_text.flatten_html("<p>Python</p><h3>Nice to have</h3>")
        self.assertIn("Nice to have", text)
        self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()


class AnInferredSalaryIsDistinguishableFromAStatedOne(unittest.TestCase):
    """`to_monthly_sgd` guesses annual-vs-monthly from magnitude when it does
    not recognise the period. The guess is defensible; making it silently was
    not, because it feeds the pay component of the score and a reader could not
    tell an inferred figure from an employer's stated one."""

    def test_an_unrecognised_period_is_reported_as_guessed(self):
        from jobbuddy import job_schema
        self.assertTrue(
            job_schema.salary_period_was_guessed(180000, "per annum-ish"))

    def test_a_recognised_period_is_not_reported_as_guessed(self):
        from jobbuddy import job_schema
        for period in ("Monthly", "Annually", "Hourly"):
            with self.subTest(period=period):
                self.assertFalse(
                    job_schema.salary_period_was_guessed(8000, period))

    def test_an_unusable_amount_is_not_reported_as_guessed(self):
        """No conversion happened, so there is nothing to disclose."""
        from jobbuddy import job_schema
        self.assertFalse(job_schema.salary_period_was_guessed(None, "wat"))
        self.assertFalse(job_schema.salary_period_was_guessed(0, "wat"))

    def test_the_guess_still_produces_the_sane_number(self):
        """Reporting must not change the conversion it reports on."""
        from jobbuddy import job_schema
        self.assertEqual(job_schema.to_monthly_sgd(180000, "unknown"), 15000)
        self.assertEqual(job_schema.to_monthly_sgd(8000, "unknown"), 8000)
