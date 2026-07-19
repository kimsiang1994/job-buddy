"""Tests for the output tree. This is Windows, so the path logic is real logic.

    py -m unittest tests.test_paths

No filesystem is touched except in the one case that asserts a directory can
actually be created -- the rest is pure construction, which is why the rules are
testable at all.

Every case here is a real Windows failure mode, not a hypothetical:
MAX_PATH, illegal characters, trailing dots, reserved device names, and a title
that sanitises to nothing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jobbuddy import paths


class IllegalCharactersAreRemoved(unittest.TestCase):
    def test_every_character_windows_rejects_is_replaced(self):
        name = paths.sanitise_component('AI/ML <Lead>: "Data" | Ops? *now*\\here')
        for character in '<>:"/\\|?*':
            with self.subTest(character=character):
                self.assertNotIn(character, name)

    def test_control_characters_are_removed(self):
        self.assertNotIn("\n", paths.sanitise_component("Data\nEngineer\t2"))

    def test_a_trailing_dot_is_stripped(self):
        """The API accepts it and the shell strips it, so the two disagree."""
        self.assertEqual(paths.sanitise_component("Senior Engineer."),
                         "Senior Engineer")

    def test_a_trailing_space_is_stripped(self):
        self.assertEqual(paths.sanitise_component("Senior Engineer   "),
                         "Senior Engineer")

    def test_a_readable_title_survives_intact(self):
        """Guards against sanitising by mangling everything."""
        self.assertEqual(paths.sanitise_component("Senior AI Engineer - Umbra"),
                         "Senior AI Engineer - Umbra")


class ReservedDeviceNamesAreEscaped(unittest.TestCase):
    def test_each_reserved_name_is_escaped(self):
        for reserved in ("CON", "PRN", "AUX", "NUL", "COM1", "COM9",
                         "LPT1", "LPT9"):
            with self.subTest(reserved=reserved):
                self.assertNotEqual(
                    paths.sanitise_component(reserved).upper(), reserved)

    def test_a_reserved_name_is_still_reserved_with_an_extension(self):
        """NUL.txt is the null device, not a text file."""
        self.assertNotEqual(paths.sanitise_component("NUL.txt").upper(), "NUL.TXT")

    def test_the_check_is_case_insensitive(self):
        self.assertNotEqual(paths.sanitise_component("aux").lower(), "aux")

    def test_a_name_merely_containing_a_device_name_is_left_alone(self):
        """AUXILIARY is not a device, and mangling it would be a false positive."""
        self.assertEqual(paths.sanitise_component("Auxiliary Systems Lead"),
                         "Auxiliary Systems Lead")


class ANameThatSanitisesToNothingStillWorks(unittest.TestCase):
    def test_a_title_of_only_illegal_characters_gets_a_usable_name(self):
        name = paths.sanitise_component("///???***")
        self.assertTrue(name)
        self.assertEqual(name, paths.FALLBACK)

    def test_an_empty_title_gets_a_usable_name(self):
        self.assertTrue(paths.sanitise_component(""))

    def test_a_title_of_only_dots_gets_a_usable_name(self):
        self.assertTrue(paths.sanitise_component("..."))

    def test_the_fallback_directory_can_actually_be_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = paths.job_dir("scope", "2026-07-19_101500", "///", base=tmp)
            paths.ensure_dir(target)
            self.assertTrue(target.is_dir())


class LongPathsAreTruncatedAtTheJobNotTheRoot(unittest.TestCase):
    def setUp(self):
        self.base = Path("C:/Users/somebody/Desktop/job buddy/potential applications")
        self.stamp = "2026-07-19_101500"

    def test_a_very_long_title_produces_a_path_within_max_path(self):
        title = "Senior Staff Machine Learning Engineer " * 8
        target = paths.job_dir("ai-engineer-sg", self.stamp, title, base=self.base)
        self.assertLessEqual(len(str(target)) + paths.FILENAME_HEADROOM,
                             paths.MAX_PATH)

    def test_the_root_is_never_shortened(self):
        title = "Principal Engineer " * 20
        target = paths.job_dir("ai-engineer-sg", self.stamp, title, base=self.base)
        expected = paths.run_root("ai-engineer-sg", self.stamp, self.base)
        self.assertEqual(target.parent, expected)

    def test_two_long_titles_sharing_a_prefix_do_not_collide(self):
        """Truncation alone would have the second job overwrite the first."""
        shared = "Senior Machine Learning Engineer, Platform Infrastructure, "
        first = paths.job_dir("scope", self.stamp, shared + "Retrieval Systems",
                              base=self.base)
        second = paths.job_dir("scope", self.stamp, shared + "Ranking Systems",
                               base=self.base)
        self.assertNotEqual(first, second)

    def test_truncation_is_deterministic(self):
        """Two runs over the same job must land in the same directory name."""
        title = "Head of Data Platform Engineering " * 6
        self.assertEqual(paths.job_dir("scope", self.stamp, title, base=self.base),
                         paths.job_dir("scope", self.stamp, title, base=self.base))

    def test_a_root_that_eats_the_budget_still_yields_a_usable_name(self):
        deep = Path("C:/") / ("verylongdirectory" * 13)
        target = paths.job_dir("scope", self.stamp, "Data Engineer", base=deep)
        self.assertGreaterEqual(len(target.name), 1)
        self.assertEqual(target.parent,
                         paths.run_root("scope", self.stamp, deep))

    def test_a_short_title_is_not_truncated(self):
        target = paths.job_dir("scope", self.stamp, "Data Engineer", base=self.base)
        self.assertEqual(target.name, "Data Engineer")


class IdenticalTitlesAreNumberedRatherThanOverwritten(unittest.TestCase):
    def test_the_same_title_twice_produces_two_directories(self):
        taken: set[str] = set()
        first = paths.job_dir("scope", "2026-07-19_101500", "Data Engineer",
                              base="/tmp", taken=taken)
        second = paths.job_dir("scope", "2026-07-19_101500", "Data Engineer",
                               base="/tmp", taken=taken)
        self.assertNotEqual(first.name, second.name)

    def test_the_collision_check_ignores_case(self):
        """Windows would treat these as one directory."""
        taken: set[str] = set()
        paths.job_dir("scope", "2026-07-19_101500", "Data Engineer",
                      base="/tmp", taken=taken)
        second = paths.job_dir("scope", "2026-07-19_101500", "DATA ENGINEER",
                               base="/tmp", taken=taken)
        self.assertNotIn(second.name.casefold(), {"data engineer"})

    def test_without_a_taken_set_the_name_is_stable(self):
        first = paths.job_dir("scope", "2026-07-19_101500", "Data Engineer")
        second = paths.job_dir("scope", "2026-07-19_101500", "Data Engineer")
        self.assertEqual(first, second)


class TheTreeShapeIsWhatWasAskedFor(unittest.TestCase):
    def test_the_layout_is_scope_then_stamp_then_job(self):
        target = paths.job_dir("ai-engineer-sg", "2026-07-19_101500",
                               "AI Engineer", base="/out")
        self.assertEqual(target.parts[-3:],
                         ("ai-engineer-sg", "2026-07-19_101500", "AI Engineer"))

    def test_an_illegal_scope_name_is_sanitised_too(self):
        target = paths.run_root("AI / ML", "2026-07-19_101500", base="/out")
        self.assertNotIn("/", target.parts[-2])

    def test_the_stamp_sorts_lexicographically(self):
        early = paths.timestamp(__import__("datetime").datetime(2026, 7, 19, 9, 5, 1))
        late = paths.timestamp(__import__("datetime").datetime(2026, 7, 19, 10, 5, 1))
        self.assertLess(early, late)
        self.assertEqual(early, "2026-07-19_090501")

    def test_the_job_label_carries_the_company(self):
        """Three 'Data Engineer' directories tell the user nothing."""
        label = paths.job_label({"title": "Data Engineer",
                                 "company": "Umbra Financial"})
        self.assertEqual(label, "Data Engineer - Umbra Financial")

    def test_a_job_with_no_title_still_gets_a_label(self):
        self.assertTrue(paths.job_label({"company": "Northwind Labs"}))
        self.assertTrue(paths.job_label({}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
