"""Regressions for the silent-failure / lazy-init audit.

Each test here pins a specific defect that was found and fixed. The thread
tests deliberately WIDEN the race window with a slow patched dependency: the
bug they guard is a lost-update window that spans a whole file read or module
import, and a test that merely calls the function twice never lands inside it.
A mutation that narrows the window to a nanosecond is not a faithful
reproduction of the original bug.
"""

import json
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from jobbuddy import (fact_guard, render_charts, render_report, render_resume,
                      user_input, verify_profile)
from jobbuddy.deepseek import model_config, token_budget


# --------------------------------------------------------------------------
# lazy caches must never expose a half-built state
# --------------------------------------------------------------------------

class LazyCacheThreadSafetyTests(unittest.TestCase):
    """The `loaded = True` before the work bug, in its two other homes.

    `token_budget.load_profiles()` had this and it failed ~75% of tailoring
    jobs. `model_config._load()` and `token_budget._backend_load()` had the
    same shape and are fixed the same way.
    """

    def setUp(self):
        model_config.reload()
        token_budget.reload()
        self.addCleanup(model_config.reload)
        self.addCleanup(token_budget.reload)

    def test_model_config_load_never_returns_none_under_threads(self):
        """No thread may observe `loaded` set while `config` is still None.

        With the defect, the first thread set the flag and then spent the whole
        file read populating the cache; every thread arriving in that window
        got None, silently took the hardcoded-fallback path in
        resolve_verbose() and reported "config unavailable" about a file that
        was perfectly readable.
        """
        self.assertTrue(Path(model_config.CONFIG_PATH).is_file(),
                        "this test is meaningless without a real models.json")

        real_load = json.load

        def slow_load(fh, *args, **kwargs):
            # Cover the entire read, which is the real window.
            time.sleep(0.05)
            return real_load(fh, *args, **kwargs)

        results, errors = [], []

        def worker():
            try:
                results.append(model_config._load())
            except Exception as exc:            # noqa: BLE001 - recorded, then asserted
                errors.append(exc)

        with mock.patch.object(json, "load", slow_load):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        self.assertEqual(errors, [], f"_load() raised under threads: {errors}")
        self.assertEqual(len(results), 8)
        for index, config in enumerate(results):
            self.assertIsInstance(
                config, dict,
                f"thread {index} observed a half-built cache (got {config!r})")
            self.assertIn("models", config)

    def test_backend_load_reports_one_consistent_backend_under_threads(self):
        """Every thread must see the same tokenizer backend.

        With the defect, `loaded` was set before the probe, so racing threads
        got the still-empty backend and counted with the char-ratio heuristic
        while the winner ended up on the official tokenizer. That does not
        crash -- it silently produces inconsistent token estimates and writes a
        wrong `estimator` field into usage_log.jsonl, which is the ground truth
        calibrate_budgets.py tunes against.
        """
        real_exists = os.path.exists

        def slow_exists(path, *args, **kwargs):
            # The flag used to be set before this call, so sleeping here
            # reproduces the original window faithfully.
            if str(path) == token_budget.TOKENIZER_JSON:
                time.sleep(0.05)
            return real_exists(path, *args, **kwargs)

        names, errors = [], []

        def worker():
            try:
                names.append(token_budget._backend_load()["name"])
            except Exception as exc:            # noqa: BLE001
                errors.append(exc)

        with mock.patch.object(token_budget.os.path, "exists", slow_exists):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        self.assertEqual(errors, [], f"_backend_load() raised: {errors}")
        self.assertEqual(len(names), 8)
        self.assertEqual(
            len(set(names)), 1,
            f"threads disagreed about the tokenizer backend: {sorted(set(names))}")

    def test_reload_then_load_still_yields_a_real_config(self):
        """reload() clears the flag first, so a racing reader re-reads."""
        model_config.reload()
        self.assertIsInstance(model_config._load(), dict)


# --------------------------------------------------------------------------
# fact_guard must fail closed on an unreadable date
# --------------------------------------------------------------------------

class YearsBetweenFailsClosedTests(unittest.TestCase):
    """A malformed end date meant "still employed today", inflating the span.

    The duration check compares a bullet's claim against this number, so
    failing open here let an overstated claim through the one gate that exists
    to catch it.
    """

    def test_malformed_end_is_unknown_not_today(self):
        self.assertIsNone(fact_guard._years_between("2015-01", "20I9-01"))

    def test_malformed_start_is_still_unknown(self):
        self.assertIsNone(fact_guard._years_between("2O15-01", "2019-01"))

    def test_absent_end_still_means_current_role(self):
        """The genuine "current role" case must keep working."""
        span = fact_guard._years_between("2015-01", None)
        self.assertIsNotNone(span)
        self.assertGreater(span, 5.0)

    def test_well_formed_dates_are_unchanged(self):
        self.assertAlmostEqual(
            fact_guard._years_between("2015-01", "2019-01"), 4.0, places=6)

    def test_bullet_claiming_years_against_a_malformed_end_is_rejected(self):
        """The end-to-end effect: the guard no longer passes an inflated claim.

        Without a total_years fallback, an unreadable span must not support a
        multi-year claim.
        """
        fact = {
            "fact_id": "f1",
            "start": "2017-01",
            "end": "20I9-01",          # typo: letter I for digit 1
            "phrasings": ["Built Kubernetes tooling"],
            "source_span": "Built Kubernetes tooling",
        }
        verdict = fact_guard.check_bullet(
            "Built Kubernetes tooling over 8 years", fact)
        self.assertFalse(verdict.ok)
        self.assertTrue(
            any(v.kind == "unsupported duration" for v in verdict.violations),
            f"expected an unsupported-duration violation, got {verdict.violations}")


# --------------------------------------------------------------------------
# user_input must say which fallback it took
# --------------------------------------------------------------------------

class UserInputReportsFallbacksTests(unittest.TestCase):

    def setUp(self):
        user_input._warned.clear()
        self.addCleanup(user_input._warned.clear)

    def test_fingerprint_failure_is_reported(self):
        """An unreadable resume silently disabled archiving. Now it warns."""
        with mock.patch.object(Path, "read_bytes",
                               side_effect=OSError("permission denied")):
            with mock.patch("sys.stderr") as stderr:
                self.assertEqual(user_input.resume_fingerprint("resume.pdf"), "")
        written = "".join(str(c.args[0]) for c in stderr.write.call_args_list
                          if c.args)
        self.assertIn("could not fingerprint", written)

    def test_missing_taxonomy_is_reported_not_read_as_an_empty_resume(self):
        """`derive_skills` returned [] on ImportError with no signal at all.

        Downstream that became "only 0 skills matched the taxonomy", blaming
        the user's resume for a missing module.
        """
        import jobbuddy

        # `from jobbuddy import skills_taxonomy` resolves via the package
        # attribute first and only then via sys.modules, so both have to go for
        # the import to actually fail. A None entry in sys.modules is what makes
        # the import machinery raise ImportError rather than re-import it.
        had_attr = hasattr(jobbuddy, "skills_taxonomy")
        saved = getattr(jobbuddy, "skills_taxonomy", None)
        if had_attr:
            delattr(jobbuddy, "skills_taxonomy")
        try:
            with mock.patch.dict("sys.modules",
                                 {"jobbuddy.skills_taxonomy": None}):
                with mock.patch("sys.stderr") as stderr:
                    skills = user_input.derive_skills("Python and Kubernetes")
        finally:
            if had_attr:
                setattr(jobbuddy, "skills_taxonomy", saved)

        self.assertEqual(skills, [])
        written = "".join(str(c.args[0]) for c in stderr.write.call_args_list
                          if c.args)
        self.assertIn("skills_taxonomy unavailable", written)

    def test_warn_does_not_mark_a_message_it_failed_to_print(self):
        """Marking first meant a failed print suppressed every later retry."""
        with mock.patch("sys.stderr") as broken:
            broken.write.side_effect = ValueError("stream detached")
            user_input._warn("boom")
        self.assertNotIn("boom", user_input._warned)

        with mock.patch("sys.stderr") as working:
            user_input._warn("boom")
        written = "".join(str(c.args[0]) for c in working.write.call_args_list
                          if c.args)
        self.assertIn("boom", written)


# --------------------------------------------------------------------------
# save_current must own its temp file
# --------------------------------------------------------------------------

class SaveCurrentAtomicityTests(unittest.TestCase):
    """A fixed `current_profile.tmp` broke atomicity under the thread pool.

    os.replace is atomic only for a writer that owns its source file. With one
    shared temp name, a second writer overwrites the first's temp between its
    write and its replace, and whichever replaces second raises
    FileNotFoundError because the first already consumed the file.
    """

    def test_concurrent_saves_all_succeed_and_leave_valid_json(self):
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmpdir:
            intake = Path(tmpdir)
            with mock.patch.object(user_input, "INTAKE_DIR", intake), \
                 mock.patch.object(user_input, "CURRENT_PROFILE",
                                   intake / "current_profile.json"):

                real_replace = os.replace

                def slow_replace(src, dst):
                    # Widen the window between write and replace, which is
                    # exactly where the shared temp name was clobbered.
                    time.sleep(0.03)
                    return real_replace(src, dst)

                results, errors = [], []

                def worker(index):
                    profile = user_input.IntakeProfile(
                        full_name=f"Person {index}",
                        resume_path="r.pdf",
                        target_roles=["engineer"])
                    try:
                        results.append(user_input.save_current(profile))
                    except Exception as exc:    # noqa: BLE001
                        errors.append(exc)

                with mock.patch.object(os, "replace", slow_replace):
                    threads = [threading.Thread(target=worker, args=(i,))
                               for i in range(6)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=30)

                self.assertEqual(errors, [], f"save_current raised: {errors}")
                self.assertTrue(all(results),
                                "a concurrent save reported failure")

                # The published file must be one writer's complete output,
                # never a torn mix of two.
                written = (intake / "current_profile.json").read_text(
                    encoding="utf-8")
                data = json.loads(written)
                self.assertIn("full_name", data)

                # No orphaned temp files left behind.
                leftovers = list(intake.glob("*.tmp"))
                self.assertEqual(leftovers, [],
                                 f"temp files were not cleaned up: {leftovers}")


# --------------------------------------------------------------------------
# absence must not be drawn as confidence
# --------------------------------------------------------------------------

class MissingConfidenceIsNotFullConfidenceTests(unittest.TestCase):
    """`_num(scores.get("confidence"), 1.0)` imputed 100% confidence.

    It also made the module's own `confidence is None` branches unreachable,
    so the "not measured" caption it was written to print could never appear.
    """

    def test_absent_confidence_is_reported_as_not_measured(self):
        svg = render_charts.component_bars(
            {"components": {"skills": {"value": 80, "weight": 1.0}}})
        self.assertIn("not measured", svg)
        self.assertNotIn("100%", svg)

    def test_unparseable_confidence_is_reported_as_not_measured(self):
        svg = render_charts.component_bars(
            {"components": {"skills": {"value": 80, "weight": 1.0}},
             "confidence": "n/a"})
        self.assertIn("not measured", svg)

    def test_real_low_confidence_still_warns(self):
        svg = render_charts.component_bars(
            {"components": {"skills": {"value": 80, "weight": 1.0}},
             "confidence": 0.2})
        self.assertIn("low confidence", svg)

    def test_real_high_confidence_still_reads_normally(self):
        svg = render_charts.component_bars(
            {"components": {"skills": {"value": 80, "weight": 1.0}},
             "confidence": 0.95})
        self.assertIn("0-100 per component", svg)
        self.assertNotIn("low confidence", svg)


class GuardCountersDistinguishZeroFromUnmeasuredTests(unittest.TestCase):
    """"Bullets rejected by fact_guard: 0" was printed for a guard that never
    ran -- an affirmative "nothing was lost" based on no measurement."""

    def test_absent_guard_is_not_measured(self):
        losses = render_report.silent_losses(None, None)
        self.assertFalse(losses["guard_measured"])

    def test_present_guard_is_measured(self):
        losses = render_report.silent_losses(
            {"guard": {"rejected": 0, "fell_back": 2}}, None)
        self.assertTrue(losses["guard_measured"])
        self.assertEqual(losses["guard_rejected"], 0)

    def test_report_says_not_measured_rather_than_zero(self):
        model = render_report.build_model({"job_key": "k", "title": "Engineer"})
        source = render_report.build_typst_source(model)
        # Typst-escaped: the underscore in fact_guard comes through as fact\_guard.
        rendered = [line for line in source.splitlines()
                    if "rejected by fact" in line]
        self.assertTrue(rendered, "the guard line is missing from the report")
        self.assertIn(render_report.NOT_MEASURED, rendered[0])
        self.assertNotRegex(rendered[0], r":\*\s*0\s")


class CorruptVerifiedProfileIsReportedTests(unittest.TestCase):
    """A corrupted profile returned {} in silence, which every caller reads as
    "no profile yet" -- so hand-done verification looked like it never
    happened. Still {}, still cannot raise, but now it says so."""

    def setUp(self):
        verify_profile._warned.clear()
        self.addCleanup(verify_profile._warned.clear)

    def test_malformed_verified_profile_warns(self):
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "master_profile.json"
            bad.write_text("{not json", encoding="utf-8")
            with mock.patch("sys.stderr") as stderr:
                self.assertEqual(verify_profile.load_verified(bad), {})
        written = "".join(str(c.args[0]) for c in stderr.write.call_args_list
                          if c.args)
        self.assertIn("unreadable", written)

    def test_wrong_shape_verified_profile_warns(self):
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "master_profile.json"
            bad.write_text('["a", "list"]', encoding="utf-8")
            with mock.patch("sys.stderr") as stderr:
                self.assertEqual(verify_profile.load_verified(bad), {})
        written = "".join(str(c.args[0]) for c in stderr.write.call_args_list
                          if c.args)
        self.assertIn("not an object", written)

    def test_absent_profile_is_silent(self):
        """An absent file is a normal state, not a fallback. It must NOT warn."""
        with mock.patch("sys.stderr") as stderr:
            self.assertEqual(
                verify_profile.load_verified(Path("no_such_profile.json")), {})
        written = "".join(str(c.args[0]) for c in stderr.write.call_args_list
                          if c.args)
        self.assertEqual(written, "")

    def test_valid_profile_still_loads(self):
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmpdir:
            good = Path(tmpdir) / "master_profile.json"
            good.write_text('{"facts": [{"fact_id": "f1"}]}', encoding="utf-8")
            self.assertEqual(len(verify_profile.load_verified(good)["facts"]), 1)


class CapabilityCacheProbeTests(unittest.TestCase):
    """`capabilities()` always returns a COMPLETE probe, never a partial one.

    Deliberately NOT a race test. The reset-between-check-and-read window in
    the unlocked version is about two bytecodes wide and did not reproduce
    under six readers, two resetters and sys.setswitchinterval(1e-6) for three
    seconds. A lock was added anyway because it is uncontended and makes the
    invariant structural, but no test here can fail on the unlocked version, so
    none claims to. What IS worth pinning is the property that matters: the
    dict is built before it is published, so a concurrent caller never sees a
    half-filled capability map.
    """

    def tearDown(self):
        render_resume.reset_capability_cache()

    def test_concurrent_callers_all_get_a_complete_map(self):
        real_loader = render_resume._load_typst

        def slow_loader():
            time.sleep(0.02)
            return real_loader()

        seen, errors = [], []

        def worker():
            try:
                seen.append(render_resume.capabilities())
            except Exception as exc:            # noqa: BLE001
                errors.append(exc)

        render_resume.reset_capability_cache()
        with mock.patch.object(render_resume, "_load_typst", slow_loader):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        self.assertEqual(errors, [], f"capabilities() raised: {errors}")
        self.assertEqual(len(seen), 8)
        for caps in seen:
            self.assertEqual(set(caps), {"pdf", "docx", "page_count"},
                             f"a caller saw a partial capability map: {caps}")


if __name__ == "__main__":
    unittest.main()
