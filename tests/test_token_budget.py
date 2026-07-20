"""Tests for task-profile budgeting.

    py -m unittest tests.test_token_budget

Offline, no API key, no network, no cost.
"""

from __future__ import annotations

import unittest

from jobbuddy.deepseek import token_budget

class ProfileCacheIsThreadSafe(unittest.TestCase):
    """The lazy cache set `loaded = True` before the file read that fills
    `data`, so a second thread arriving in that window got None back and every
    caller died on `profiles.get(...)`.

    It presented as an intermittent AttributeError that never reproduced on a
    single job, cost two wrong diagnoses, and was only found once the traceback
    was captured rather than just the exception message.
    """

    def setUp(self):
        token_budget.reload()
        self.addCleanup(token_budget.reload)

    def test_concurrent_first_calls_never_see_a_half_built_cache(self):
        import threading
        import time

        real_open = open
        opened = []

        def slow_open(*args, **kwargs):
            # Widen the window the race needs. Without the fix this fails
            # essentially every run; with it, never.
            opened.append(1)
            time.sleep(0.05)
            return real_open(*args, **kwargs)

        results = []
        errors = []

        def worker():
            try:
                results.append(token_budget.load_profiles())
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                errors.append(exc)

        import builtins
        builtins.open = slow_open
        try:
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            builtins.open = real_open

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        for value in results:
            self.assertIsNotNone(
                value, "a thread saw the cache flagged loaded but still empty")
            self.assertIsInstance(value, dict)

    def test_the_file_is_read_once_even_under_contention(self):
        import threading

        calls = []
        real_open = open

        def counting_open(*args, **kwargs):
            if str(args[0]).endswith("task_profiles.json"):
                calls.append(1)
            return real_open(*args, **kwargs)

        import builtins
        builtins.open = counting_open
        try:
            threads = [threading.Thread(target=token_budget.load_profiles)
                       for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            builtins.open = real_open

        self.assertEqual(len(calls), 1, f"read {len(calls)} times, expected 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
