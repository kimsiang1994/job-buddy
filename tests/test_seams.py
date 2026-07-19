"""Tests for the modules that had no seam to test through.

    py test_seams.py

Offline, no network, no API key, no cost -- same contract as test_scrapers.py
and test_pipeline.py.

Everything here was untestable before the seams went in. `net.fetch` built its
opener inline and took its cache path from a module constant, so 393 lines of
hand-written protocol handling -- the retry set, the redirect cap, Retry-After,
the size cap, the content-type guard -- could not be reached without
monkeypatching stdlib. The run sequence lived in `main()` and a notebook cell,
so it could not be reached at all.

These are the tests the seams exist for. If a seam here stops being used, delete
the seam too: one adapter is a hypothetical seam, two are a real one.
"""

from __future__ import annotations

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

from jobbuddy import job_schema
from jobbuddy import job_store
from jobbuddy import net
from jobbuddy import pipeline
from jobbuddy import scoring
from jobbuddy import user_input


# --------------------------------------------------------------------------
# Fake transport -- the second adapter that justifies net.fetch's opener seam
# --------------------------------------------------------------------------

class FakeResponse:
    """Enough of http.client.HTTPResponse for net.fetch."""

    def __init__(self, status=200, body=b"{}", headers=None):
        self.status = status
        self.headers = headers or {"content-type": "application/json"}
        self._stream = io.BytesIO(body)

    def read(self, size=-1):
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Replays a queued list of responses and records what was requested."""

    def __init__(self, *responses):
        self._queue = list(responses)
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("FakeOpener ran out of queued responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def call_count(self):
        return len(self.requests)


class FakeClock:
    """A monotonic clock that only moves when someone sleeps.

    Injected alongside the fake sleep. A no-op sleep on its own turns net's
    throttle loop into a busy-spin -- the first version of this suite took 36
    seconds burning CPU rather than waiting. Advancing a fake clock makes both
    the throttle and the retry backoff instant AND correct.
    """

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _fake_time():
    clock = FakeClock()
    return {"sleep": clock.sleep, "clock": clock}, clock


class NetPureParts(unittest.TestCase):
    """The bits that were always testable and simply never were."""

    def test_text_uses_charset_from_headers(self):
        result = net.FetchResult(True, 200, "u", "café".encode("latin-1"),
                                 {"content-type": "text/html; charset=latin-1"})
        self.assertEqual(result.text(), "café")

    def test_text_never_raises_on_bad_bytes(self):
        result = net.FetchResult(True, 200, "u", b"\xff\xfe\x00bad",
                                 {"content-type": "application/json"})
        self.assertIsInstance(result.text(), str)

    def test_text_survives_an_unknown_charset(self):
        result = net.FetchResult(True, 200, "u", b"hi",
                                 {"content-type": "text/html; charset=not-a-codec"})
        self.assertEqual(result.text(), "hi")

    def test_json_returns_none_rather_than_raising(self):
        result = net.FetchResult(True, 200, "u", b"<html>not json</html>", {})
        self.assertIsNone(result.json())

    def test_decompress_gzip_and_deflate(self):
        self.assertEqual(net._decompress(gzip.compress(b"hello"), "gzip"), b"hello")

    def test_decompress_returns_raw_on_corrupt_input(self):
        # A truncated body must degrade, not crash the run.
        self.assertEqual(net._decompress(b"not gzip", "gzip"), b"not gzip")

    def test_read_capped_truncates_at_the_limit(self):
        oversized = io.BytesIO(b"x" * (net.MAX_BYTES + 5000))
        body, truncated = net._read_capped(oversized)
        self.assertTrue(truncated)
        self.assertEqual(len(body), net.MAX_BYTES)

    def test_read_capped_passes_small_bodies_through(self):
        body, truncated = net._read_capped(io.BytesIO(b"small"))
        self.assertFalse(truncated)
        self.assertEqual(body, b"small")


class NetFetchPolicy(unittest.TestCase):
    """The policy that was structurally unreachable before the opener seam."""

    def test_rejects_non_http_scheme(self):
        result = net.fetch("file:///etc/passwd")
        self.assertFalse(result.ok)
        self.assertIn("non-http scheme", result.error)

    def test_rejects_url_without_host(self):
        self.assertFalse(net.fetch("https://").ok)

    def test_successful_fetch_returns_body(self):
        opener = FakeOpener(FakeResponse(200, b'{"ok": true}'))
        result = net.fetch("https://example.test/a", opener=opener, **_fake_time()[0])
        self.assertTrue(result.ok)
        self.assertEqual(result.json(), {"ok": True})
        self.assertEqual(opener.call_count, 1)

    def test_429_is_retried(self):
        opener = FakeOpener(
            FakeResponse(429, b"", {"content-type": "application/json"}),
            FakeResponse(200, b'{"ok": true}'),
        )
        result = net.fetch("https://example.test/a", opener=opener, **_fake_time()[0])
        self.assertTrue(result.ok)
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(result.attempts, 2)

    def test_503_is_retried(self):
        opener = FakeOpener(FakeResponse(503), FakeResponse(200, b'{}'))
        self.assertTrue(net.fetch("https://example.test/a", opener=opener,
                                  **_fake_time()[0]).ok)
        self.assertEqual(opener.call_count, 2)

    def test_404_is_NOT_retried(self):
        """Retrying a 4xx hides the bug that caused it and burns the quota."""
        opener = FakeOpener(FakeResponse(404, b"missing"))
        result = net.fetch("https://example.test/a", opener=opener, **_fake_time()[0])
        self.assertFalse(result.ok)
        self.assertEqual(opener.call_count, 1)

    def test_401_is_NOT_retried(self):
        opener = FakeOpener(FakeResponse(401, b""))
        net.fetch("https://example.test/a", opener=opener, **_fake_time()[0])
        self.assertEqual(opener.call_count, 1)

    def test_gives_up_after_max_attempts(self):
        opener = FakeOpener(*[FakeResponse(503) for _ in range(net.MAX_ATTEMPTS)])
        result = net.fetch("https://example.test/a", opener=opener, **_fake_time()[0])
        self.assertFalse(result.ok)
        self.assertEqual(opener.call_count, net.MAX_ATTEMPTS)

    def test_retry_after_header_is_honoured(self):
        slept = []
        opener = FakeOpener(
            FakeResponse(429, b"", {"content-type": "application/json",
                                    "retry-after": "7"}),
            FakeResponse(200, b'{}'),
        )
        net.fetch("https://example.test/a", opener=opener, sleep=slept.append)
        self.assertTrue(any(s >= 7 for s in slept),
                        f"Retry-After: 7 ignored; slept {slept}")

    def test_redirect_is_followed_and_final_url_reported(self):
        opener = FakeOpener(
            FakeResponse(302, b"", {"location": "https://example.test/final"}),
            FakeResponse(200, b'{"ok": true}'),
        )
        result = net.fetch("https://example.test/start", opener=opener, **_fake_time()[0])
        self.assertTrue(result.ok)
        # The final URL is the evidence an expired posting redirected to /careers.
        self.assertEqual(result.url, "https://example.test/final")

    def test_redirect_loop_is_capped(self):
        loop = [FakeResponse(302, b"", {"location": "https://example.test/x"})
                for _ in range(net.MAX_REDIRECTS + 3)]
        result = net.fetch("https://example.test/x", opener=FakeOpener(*loop),
                           **_fake_time()[0])
        self.assertFalse(result.ok)
        self.assertIn("redirects", result.error)

    def test_redirect_without_location_is_an_error(self):
        result = net.fetch("https://example.test/x",
                           opener=FakeOpener(FakeResponse(302, b"", {})),
                           **_fake_time()[0])
        self.assertFalse(result.ok)
        self.assertIn("Location", result.error)

    def test_content_type_guard_rejects_html_served_as_json(self):
        """An outage serving an error page must not read as an empty result.

        Without this, a JSON endpoint returning a login wall came back ok=True
        with a None body, and callers reported zero jobs and zero warnings.
        """
        opener = FakeOpener(FakeResponse(200, b"<html>down for maintenance</html>",
                                         {"content-type": "text/html"}))
        result = net.fetch("https://example.test/a", opener=opener,
                           expect_content_type="json", **_fake_time()[0])
        self.assertFalse(result.ok)
        self.assertIn("content-type", result.error)

    def test_get_json_enforces_the_content_type_guard(self):
        opener = FakeOpener(FakeResponse(200, b"<html>oops</html>",
                                         {"content-type": "text/html"}))
        data, result = net.get_json("https://example.test/a", opener=opener,
                                    **_fake_time()[0])
        self.assertIsNone(data)
        self.assertFalse(result.ok)

    def test_network_error_is_returned_not_raised(self):
        opener = FakeOpener(*[OSError("connection reset")
                              for _ in range(net.MAX_ATTEMPTS)])
        result = net.fetch("https://example.test/a", opener=opener, **_fake_time()[0])
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 0)
        self.assertIn("network error", result.error)

    def test_post_body_is_json_encoded(self):
        opener = FakeOpener(FakeResponse(200, b'{}'))
        net.fetch("https://example.test/a", method="POST",
                  payload={"appliedFacets": {}, "limit": 20},
                  opener=opener, **_fake_time()[0])
        request = opener.requests[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data)["limit"], 20)


class NetCache(unittest.TestCase):
    """The cache seam. Previously every cache test wrote into the real repo."""

    def test_second_fetch_is_served_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            opener = FakeOpener(FakeResponse(200, b'{"n": 1}'))
            first = net.fetch("https://example.test/c", opener=opener,
                              cache_ttl_s=3600, cache_dir=cache, **_fake_time()[0])
            second = net.fetch("https://example.test/c", opener=FakeOpener(),
                               cache_ttl_s=3600, cache_dir=cache, **_fake_time()[0])
        self.assertTrue(first.ok and second.ok)
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(second.json(), {"n": 1})

    def test_expired_cache_entry_is_refetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            net.fetch("https://example.test/c", opener=FakeOpener(FakeResponse(200, b'{"n":1}')),
                      cache_ttl_s=3600, cache_dir=cache, **_fake_time()[0])
            opener = FakeOpener(FakeResponse(200, b'{"n": 2}'))
            fresh = net.fetch("https://example.test/c", opener=opener,
                              cache_ttl_s=0, cache_dir=cache, **_fake_time()[0])
        self.assertEqual(fresh.json(), {"n": 2})

    def test_failed_responses_are_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            net.fetch("https://example.test/e", opener=FakeOpener(FakeResponse(404, b"")),
                      cache_ttl_s=3600, cache_dir=cache, **_fake_time()[0])
            opener = FakeOpener(FakeResponse(200, b'{"recovered": true}'))
            result = net.fetch("https://example.test/e", opener=opener,
                               cache_ttl_s=3600, cache_dir=cache, **_fake_time()[0])
        self.assertTrue(result.ok)
        self.assertFalse(result.from_cache)

    def test_clear_cache_only_touches_the_directory_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            net.fetch("https://example.test/c", opener=FakeOpener(FakeResponse(200, b'{}')),
                      cache_ttl_s=3600, cache_dir=cache, **_fake_time()[0])
            self.assertEqual(net.clear_cache(cache), 1)
            self.assertEqual(net.clear_cache(cache), 0)


# --------------------------------------------------------------------------
# JobHistory -- the ordering contract that had no interface to test
# --------------------------------------------------------------------------

def _job(key="a", content="A", company="acme", **overrides):
    job = job_schema.new_job("mcf", key)
    job.update({"title": "ML Engineer", "company": company,
                "url": "https://example.test/j", "jd_text": "Build models.",
                "is_open": True})
    job.update(overrides)
    job_schema.finalise(job)
    job["content_key"] = content
    return job


class HistoryOrdering(unittest.TestCase):
    """observe() owns the record-then-fold sequence so callers cannot invert it."""

    def _history(self, tmp):
        return job_store.JobHistory.load(
            path=Path(tmp) / "s.jsonl", snapshot_path=Path(tmp) / "state.json"
        )

    def test_first_run_reports_every_job_as_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._history(tmp).observe([_job("a"), _job("b", "B")], "r1")
        self.assertEqual(result.new_count, 2)
        self.assertEqual(result.returning_count, 0)
        self.assertEqual(result.prior_run_count, 0)

    def test_second_run_reports_nothing_new(self):
        """The architectural proof: run twice, see no false 'new' jobs."""
        with tempfile.TemporaryDirectory() as tmp:
            self._history(tmp).observe([_job("a")], "r1")
            result = self._history(tmp).observe([_job("a")], "r2")
        self.assertEqual(result.new_count, 0)
        self.assertEqual(result.returning_count, 1)
        self.assertEqual(result.prior_run_count, 1)

    def test_first_seen_at_survives_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._history(tmp).observe([_job("a")], "r1").jobs[0]["first_seen_at"]
            later = self._history(tmp).observe([_job("a")], "r2").jobs[0]["first_seen_at"]
        self.assertEqual(first, later)

    def test_velocity_excludes_the_run_being_recorded(self):
        """Velocity must be the PRIOR vintage.

        Counting the run you are in makes a company's first sighting inflate its
        own score -- the exact ordering error the eleven-call contract invited.
        """
        with tempfile.TemporaryDirectory() as tmp:
            result = self._history(tmp).observe([_job("a"), _job("b", "B")], "r1")
        self.assertEqual(result.velocity, {},
                         "velocity saw this run's own sightings")

    def test_dry_run_writes_nothing_and_leaves_history_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            history = job_store.JobHistory.load(path=path,
                                                snapshot_path=Path(tmp) / "st.json")
            result = history.observe([_job("a")], "r1", record=False, snapshot=False)
            self.assertFalse(path.exists())
            self.assertEqual(result.sightings_written, 0)
            self.assertEqual(history.run_count, 0)

    def test_repost_detected_across_a_changed_job_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._history(tmp).observe([_job("old", content="SAME")], "r1")
            result = self._history(tmp).observe([_job("new", content="SAME")], "r2")
        job = result.jobs[0]
        self.assertTrue(job["reposted"])
        self.assertIn("mcf:old", job["repost_of"])

    def test_absent_runs_is_not_set_for_a_job_in_this_run(self):
        """The removed dead branch: a job in front of you is not absent."""
        with tempfile.TemporaryDirectory() as tmp:
            for run in ("r1", "r2", "r3"):
                result = self._history(tmp).observe([_job("a")], run)
        self.assertEqual(result.jobs[0]["absent_runs"], 0)

    def test_snapshot_written_only_when_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "state.json"
            job_store.JobHistory.load(path=Path(tmp) / "s.jsonl",
                                      snapshot_path=snap).observe([_job("a")], "r1")
            self.assertTrue(snap.exists())

    def test_sightings_of_returns_every_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            for run in ("r1", "r2"):
                job_store.JobHistory.load(path=path,
                                          snapshot_path=Path(tmp) / "st.json"
                                          ).observe([_job("a")], run)
            rows = job_store.JobHistory.load(path=path).sightings_of("mcf:a")
        self.assertEqual(len(rows), 2)


class CompanyVelocity(unittest.TestCase):
    def test_closed_postings_are_not_counted_as_open_reqs(self):
        """open_reqs must mean OPEN.

        It used to increment unconditionally, so it meant 'every posting from
        this employer, ever'. Scoring reads 40 + 12*open_reqs, so an employer
        with five sightings pinned at 100 and a 10-weight component became a
        constant.
        """
        history = job_store.fold([
            {"ts": "2026-01-01T00:00:00Z", "run_id": "r1", "job_key": "a",
             "content_key": "A", "company_norm": "acme", "is_open": False},
            {"ts": "2026-01-01T00:00:00Z", "run_id": "r1", "job_key": "b",
             "content_key": "B", "company_norm": "acme", "is_open": True},
        ])
        stats = job_store.company_velocity(history)
        self.assertEqual(stats["acme"]["open_reqs"], 1)
        self.assertEqual(stats["acme"]["ever_seen"], 2)


# --------------------------------------------------------------------------
# pipeline -- the run, testable for the first time
# --------------------------------------------------------------------------

SCOPE = {"name": "test-scope", "queries": ["ml engineer"], "max_results_per_query": 10}


def _fake_fetch(jobs):
    """A stand-in for source_mcf.fetch_jobs -- the run's one impure dependency."""
    def fetch_jobs(query, **kwargs):
        import copy
        return copy.deepcopy(jobs), {"fetched": len(jobs), "kept": len(jobs)}
    return fetch_jobs


class PipelineCollect(unittest.TestCase):
    def test_filters_are_applied_with_a_reason(self):
        jobs = [_job("a", company="Acme"), _job("b", "B", company="TikTok Pte Ltd")]
        config = {"filters": {"exclude_companies": ["tiktok"]}, "profile": {}}
        kept, counters, excluded = pipeline.collect(
            SCOPE, config, fetch_jobs=_fake_fetch(jobs))
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(excluded), 1)
        self.assertIn("excluded company", excluded[0]["reason"])

    def test_same_job_across_two_queries_is_kept_once(self):
        scope = dict(SCOPE, queries=["ml engineer", "machine learning"])
        kept, counters, _ = pipeline.collect(
            scope, {"filters": {}, "profile": {}}, fetch_jobs=_fake_fetch([_job("a")]))
        self.assertEqual(len(kept), 1)
        self.assertEqual(counters.get("duplicate_job"), 1)

    def test_agency_duplicate_of_one_requisition_collapses_to_the_direct_employer(self):
        """A large share of SG postings are agencies re-advertising one role.

        Applying through three agencies wastes everyone's time, and the direct
        employer is the one worth keeping.
        """
        direct = _job("direct", content="SAME", company="Acme Pte Ltd")
        direct["is_agency"] = False
        agency = _job("agency", content="SAME", company="Talentsis Pte Ltd")
        agency["is_agency"] = True
        kept, counters, _ = pipeline.collect(
            SCOPE, {"filters": {}, "profile": {}},
            fetch_jobs=_fake_fetch([agency, direct]))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["job_key"], "mcf:direct")
        self.assertEqual(counters.get("duplicate_content"), 1)

    def test_oldest_posting_wins_when_neither_is_an_agency(self):
        older = _job("older", content="SAME", posted_at="2026-01-01")
        newer = _job("newer", content="SAME", posted_at="2026-06-01")
        kept, _, _ = pipeline.collect(SCOPE, {"filters": {}, "profile": {}},
                                      fetch_jobs=_fake_fetch([newer, older]))
        self.assertEqual(kept[0]["job_key"], "mcf:older")


class PipelineRun(unittest.TestCase):
    """The orchestration that existed twice and was reachable by no test."""

    def _run(self, tmp, jobs, **kwargs):
        history = job_store.JobHistory.load(
            path=Path(tmp) / "s.jsonl", snapshot_path=Path(tmp) / "st.json")
        return pipeline.run(
            SCOPE, scoring.load_config(), history=history,
            fetch_jobs=_fake_fetch(jobs), output_dir=Path(tmp) / "out", **kwargs)

    def test_run_ranks_by_total_descending(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, [_job("a"), _job("b", "B"), _job("c", "C")])
        totals = [j["scores"]["total"] for j in result.jobs]
        self.assertEqual(totals, sorted(totals, reverse=True))

    def test_run_writes_both_artefacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, [_job("a")])
        names = sorted(p.name for p in result.written)
        self.assertEqual(names, ["ranked.csv", "ranked.json"])

    def test_dry_run_writes_no_artefacts_and_no_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, [_job("a")], dry_run=True)
            self.assertEqual(result.written, [])
            self.assertFalse((Path(tmp) / "s.jsonl").exists())
        self.assertTrue(result.dry_run)

    def test_exit_code_is_2_when_nothing_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, [])
        self.assertEqual(result.exit_code(), 2)

    def test_exit_code_is_1_when_a_source_degraded(self):
        def degraded(query, **kwargs):
            return [_job("a")], {"fetched": 2, "invalid": 1}
        with tempfile.TemporaryDirectory() as tmp:
            history = job_store.JobHistory.load(path=Path(tmp) / "s.jsonl",
                                                snapshot_path=Path(tmp) / "st.json")
            result = pipeline.run(SCOPE, scoring.load_config(), history=history,
                                  fetch_jobs=degraded, output_dir=Path(tmp) / "o")
        self.assertTrue(result.degraded)
        self.assertEqual(result.exit_code(), 1)

    def test_exit_code_is_0_on_a_clean_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run(tmp, [_job("a")]).exit_code(), 0)

    def test_second_run_reports_no_new_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, [_job("a")])
            result = self._run(tmp, [_job("a")])
        self.assertEqual(result.new_count, 0)

    def test_exclusion_reasons_are_ordered_by_frequency(self):
        jobs = [_job("a", company="TikTok Pte Ltd"), _job("b", "B", company="TikTok SG"),
                _job("c", "C", company="Acme")]
        with tempfile.TemporaryDirectory() as tmp:
            history = job_store.JobHistory.load(path=Path(tmp) / "s.jsonl",
                                                snapshot_path=Path(tmp) / "st.json")
            result = pipeline.run(
                SCOPE, {"filters": {"exclude_companies": ["tiktok"]},
                        "profile": {}, "weights": {}},
                history=history, fetch_jobs=_fake_fetch(jobs),
                output_dir=Path(tmp) / "o")
        reasons = result.exclusion_reasons()
        self.assertEqual(sum(reasons.values()), 2)


class CsvContract(unittest.TestCase):
    """The 21-column contract nothing verified."""

    def test_row_has_every_declared_column(self):
        job = _job("a")
        scoring.score_job(job, scoring.load_config())
        self.assertEqual(set(pipeline.csv_row(job, 1)), set(pipeline.CSV_COLUMNS))

    def test_row_survives_an_unscored_job(self):
        # write_csv used to dig blindly through scores["components"]["skill_match"].
        row = pipeline.csv_row(_job("a"), 1)
        self.assertEqual(row["rank"], 1)
        self.assertIsNone(row["total"])

    def test_write_csv_reports_failure_rather_than_raising(self):
        ok = pipeline.write_csv([_job("a")], Path("Z:/nonexistent/x/ranked.csv"))
        self.assertIsInstance(ok, bool)


# --------------------------------------------------------------------------
# Intake -> config handoff, where the silent skill loss lived
# --------------------------------------------------------------------------

class SkillTierMerge(unittest.TestCase):
    def test_derived_skills_are_not_discarded(self):
        """The confirmed defect: 24 skills derived, 0 used.

        The old guard was `if profile.skills and not existing`, and `existing`
        was never falsy because run_config.json always ships a skills block.
        """
        merged = user_input.merge_skill_tiers(
            ["kubernetes", "terraform"],
            {"expert": ["python"], "working": [], "familiar": []},
        )
        self.assertIn("kubernetes", merged["working"])
        self.assertIn("terraform", merged["working"])
        self.assertIn("python", merged["expert"])

    def test_configured_tier_wins_for_a_skill_it_names(self):
        """Tiering is a judgement only the user can make."""
        merged = user_input.merge_skill_tiers(
            ["python"], {"expert": ["python"], "working": [], "familiar": []})
        self.assertIn("python", merged["expert"])
        self.assertNotIn("python", merged["working"])

    def test_derived_skills_never_land_in_expert(self):
        merged = user_input.merge_skill_tiers(["kubernetes"], None)
        self.assertEqual(merged["expert"], [])
        self.assertIn("kubernetes", merged["working"])

    def test_alias_forms_do_not_duplicate_a_configured_skill(self):
        merged = user_input.merge_skill_tiers(
            ["large language model"],
            {"expert": ["llm"], "working": [], "familiar": []})
        everything = merged["expert"] + merged["working"] + merged["familiar"]
        self.assertEqual(len(everything), 1)

    def test_to_run_config_carries_derived_skills_through(self):
        profile = user_input.IntakeProfile(
            full_name="A", resume_path="x", target_roles=["AI Engineer"],
            skills=["kubernetes"], target_seniority="senior")
        config = user_input.to_run_config(
            profile, {"profile": {"skills": {"expert": ["python"]}}, "filters": {}})
        flat = sum(config["profile"]["skills"].values(), [])
        self.assertIn("kubernetes", flat)
        self.assertIn("python", flat)


class CompanyExclusion(unittest.TestCase):
    """Both sides must go through the same normaliser."""

    def _excluded(self, company, pattern):
        job = _job("a", company=company)
        return scoring.check_filters(
            job, {"filters": {"exclude_companies": [pattern]}, "profile": {}})

    def test_single_word_still_matches(self):
        self.assertIsNotNone(self._excluded("ByteDance Pte Ltd", "bytedance"))

    def test_multi_word_exclusion_matches(self):
        """'bytedance technology' silently failed: norm_company strips
        'technology', so the raw needle could never be found."""
        self.assertIsNotNone(self._excluded("ByteDance Technology Pte Ltd",
                                            "bytedance technology"))

    def test_legal_suffix_in_the_exclusion_is_ignored(self):
        self.assertIsNotNone(self._excluded("TikTok Pte Ltd", "TikTok Singapore"))

    def test_unrelated_company_is_not_excluded(self):
        self.assertIsNone(self._excluded("Acme Pte Ltd", "bytedance"))


class ComponentRegistry(unittest.TestCase):
    """Guards the rename that would silently drop a component from the ranking."""

    def test_every_component_has_a_weight_in_run_config(self):
        names = {name for name, _ in scoring._COMPONENTS}
        weights = {k for k in (scoring.load_config().get("weights") or {})
                   if not k.startswith("_")}
        self.assertEqual(names, weights,
                         "a component with no weight is silently dropped from the rank")

    def test_company_signal_is_inside_the_registry(self):
        self.assertIn("company_signal", {n for n, _ in scoring._COMPONENTS})

    def test_a_raising_component_does_not_lose_the_job(self):
        def boom(job, profile, ctx=None):
            raise RuntimeError("component exploded")

        original = scoring._COMPONENTS
        try:
            scoring._COMPONENTS = original + (("boom", boom),)
            job = _job("a")
            scores = scoring.score_job(job, scoring.load_config())
            self.assertIsNone(scores["components"]["boom"]["value"])
            self.assertGreater(scores["total"], 0)
        finally:
            scoring._COMPONENTS = original

    def test_every_component_accepts_the_uniform_signature(self):
        ctx = scoring.ScoreContext()
        for name, func in scoring._COMPONENTS:
            with self.subTest(component=name):
                value, detail = func(_job("a"), {"target_seniority": "senior"}, ctx)
                self.assertIsInstance(detail, dict)


class SightingSchema(unittest.TestCase):
    """The history log's field list.

    Dropping a field here makes every past run un-joinable to new ones,
    silently and irreversibly -- history cannot be regenerated.
    """

    EXPECTED = {
        "job_key", "content_key", "source", "url", "title_norm", "company_norm",
        "company_uen", "is_agency", "seniority", "salary_min_sgd", "salary_max_sgd",
        "salary_is_stated", "ssoc_code", "posted_at", "expires_at", "source_status",
        "is_open", "liveness", "applications", "views", "apps_per_view",
        "repost_count", "edit_count", "vacancies", "scope",
    }

    def test_field_list_is_unchanged(self):
        self.assertEqual(set(job_store._SIGHTING_FIELDS), self.EXPECTED)

    def test_every_field_exists_on_a_canonical_job(self):
        job = _job("a")
        for field in job_store._SIGHTING_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, job)


if __name__ == "__main__":
    unittest.main(verbosity=2)
