"""Tests for the ranked workbook.

    py -m unittest tests.test_render_excel

Offline, no network, no API key. Every job, employer and salary below is
invented -- this repo is public and must never carry real CV or salary data.

The assertion that matters is that the rows are sorted **in the file**. XLSX can
store a sort state rather than sorted output, and a sheet written unsorted with
a sort state applied opens sorted in Excel and unsorted in every plain parser.
Asserting that `rank_jobs` was called would pass in exactly that broken case. So
these tests unzip the written workbook and read the cell values back in document
order, which is what a downstream reader actually sees. `openpyxl` is not a
dependency of this project, so the reader below is ~30 lines of stdlib zipfile
and ElementTree rather than a new wheel.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from jobbuddy import render_excel

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# --------------------------------------------------------------------------
# a minimal xlsx reader -- reads what was WRITTEN, not what Excel would show
# --------------------------------------------------------------------------

def _col_index(ref: str) -> int:
    """"C7" -> 2. Cells are placed by their `r` reference, never by position.

    A blank cell with no format is omitted from the XML entirely, so reading
    cells in document order shifts every value after it left by one -- which is
    exactly how a reader ends up believing a blank application count is a real
    number from the next column.
    """
    index = 0
    for char in ref:
        if not char.isalpha():
            break
        index = index * 26 + (ord(char.upper()) - 64)
    return index - 1


def read_sheets(path: Path) -> dict[str, list[list]]:
    """{sheet name: rows of values}, in the order the file stores them."""
    with zipfile.ZipFile(path) as book:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.iter(f"{NS}t"))
                      for item in root.findall(f"{NS}si")]

        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        names = [s.get("name") for s in workbook.iter(f"{NS}sheet")]

        sheets: dict[str, list[list]] = {}
        for index, name in enumerate(names, start=1):
            root = ET.fromstring(book.read(f"xl/worksheets/sheet{index}.xml"))
            width = len(render_excel.COLUMNS)
            rows: list[list] = []
            for row in root.iter(f"{NS}row"):
                cells: list = [None] * width
                for cell in row.findall(f"{NS}c"):
                    index = _col_index(cell.get("r") or "A1")
                    value = cell.find(f"{NS}v")
                    if value is None or value.text is None:
                        continue
                    if cell.get("t") == "s":
                        cells[index] = shared[int(value.text)]
                    elif cell.get("t") == "str":
                        cells[index] = value.text
                    else:
                        cells[index] = float(value.text)
                rows.append(cells)
            sheets[name] = rows
    return sheets


def sheet_xml(path: Path, index: int = 1) -> str:
    with zipfile.ZipFile(path) as book:
        return book.read(f"xl/worksheets/sheet{index}.xml").decode("utf-8")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def job(key: str, adjusted, total=None, confidence=1.0, **overrides) -> dict:
    data = {
        "job_key": key,
        "title": f"Engineer {key}",
        "company": "Umbra Financial",
        "source": "mcf",
        "seniority": "manager",
        "salary_min_sgd": 10000,
        "salary_max_sgd": 14000,
        "applications": 12,
        "vacancies": 1,
        "age_days": 5,
        "location": "Singapore",
        "url": f"https://example.test/{key}",
        "scores": {
            "adjusted": adjusted,
            "total": total if total is not None else adjusted,
            "confidence": confidence,
            "components": {
                "skill_match": {"value": 80.0, "weight": 30, "detail": {}},
                "competition": {"value": 40.0, "weight": 20, "detail": {}},
                "freshness": {"value": 95.0, "weight": 5, "detail": {}},
                "comp_signal": {"value": None, "weight": 15,
                                "detail": {"reason": "salary not stated"}},
            },
        },
    }
    data.update(overrides)
    return data


def unsorted_jobs() -> list[dict]:
    """Deliberately out of order, and with `total` disagreeing with `adjusted`.

    The disagreement is the point: a workbook ranked on `total` would put "c"
    first, which is the bug -- a job scores highly on `total` precisely because
    little about it could be measured.
    """
    return [
        job("a", adjusted=61.0, total=61.0, confidence=1.0),
        job("c", adjusted=52.4, total=94.0, confidence=0.3),
        job("b", adjusted=83.5, total=83.5, confidence=1.0),
        job("d", adjusted=70.2, total=70.2, confidence=0.9),
    ]


class WorkbookCase(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="jobbuddy-xlsx-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        render_excel.reset_warning_cache()
        self.addCleanup(render_excel.reset_warning_cache)
        self.path = self.dir / "ranked.xlsx"


class SortedInTheFile(WorkbookCase):
    """Read the rows back out and assert the order there."""

    def test_rows_are_written_in_descending_adjusted_order(self):
        render_excel.write_workbook({"requested": unsorted_jobs()}, self.path)
        rows = read_sheets(self.path)["requested"]
        headers = rows[0]
        adjusted = headers.index("adjusted")
        values = [row[adjusted] for row in rows[1:]]
        self.assertEqual(values, [83.5, 70.2, 61.0, 52.4])
        self.assertEqual(values, sorted(values, reverse=True))

    def test_the_rank_column_agrees_with_the_row_order_in_the_file(self):
        render_excel.write_workbook({"requested": unsorted_jobs()}, self.path)
        rows = read_sheets(self.path)["requested"]
        rank = rows[0].index("rank")
        self.assertEqual([row[rank] for row in rows[1:]], [1.0, 2.0, 3.0, 4.0])

    def test_ranking_uses_adjusted_not_the_raw_total(self):
        """`c` has the highest `total` and must not lead: it scored highly
        because almost nothing about it could be measured."""
        render_excel.write_workbook({"requested": unsorted_jobs()}, self.path)
        rows = read_sheets(self.path)["requested"]
        title = rows[0].index("title")
        self.assertEqual(rows[1][title], "Engineer b")
        self.assertEqual(rows[-1][title], "Engineer c")

    def test_the_file_carries_no_sort_state_that_could_reorder_it(self):
        """A stored sortState would make readers disagree about the ranking."""
        render_excel.write_workbook({"requested": unsorted_jobs()}, self.path)
        self.assertNotIn("sortState", sheet_xml(self.path))

    def test_rank_jobs_puts_an_unscored_job_last_not_first(self):
        ranked = render_excel.rank_jobs(
            [job("x", adjusted=None), job("y", adjusted=0.0),
             job("z", adjusted=50.0)])
        self.assertEqual([j["job_key"] for j in ranked], ["z", "y", "x"])


class Tabs(WorkbookCase):

    def test_one_tab_per_scope_in_the_order_given(self):
        result = render_excel.write_workbook({
            "requested": unsorted_jobs()[:2],
            "adjacent ml": unsorted_jobs()[2:],
            "adjacent platform": [],
        }, self.path)
        self.assertEqual(result["sheets"],
                         ["requested", "adjacent ml", "adjacent platform"])
        self.assertEqual(list(read_sheets(self.path)),
                         ["requested", "adjacent ml", "adjacent platform"])

    def test_an_empty_scope_still_gets_a_tab_with_headers(self):
        render_excel.write_workbook({"empty": []}, self.path)
        rows = read_sheets(self.path)["empty"]
        self.assertEqual(len(rows), 1)
        self.assertIn("adjusted", rows[0])

    def test_a_scope_name_illegal_in_excel_is_sanitised(self):
        result = render_excel.write_workbook(
            {"AI / ML [senior]": unsorted_jobs()[:1]}, self.path)
        self.assertEqual(result["sheets"], ["AI - ML -senior-"])
        self.assertIn("AI - ML -senior-", read_sheets(self.path))

    def test_a_scope_name_over_31_characters_is_truncated(self):
        long_name = "senior artificial intelligence engineering singapore"
        result = render_excel.write_workbook({long_name: []}, self.path)
        self.assertLessEqual(len(result["sheets"][0]), 31)

    def test_two_scopes_that_sanitise_to_the_same_name_stay_distinct(self):
        result = render_excel.write_workbook(
            {"AI/ML": [], "AI:ML": []}, self.path)
        self.assertEqual(len(set(result["sheets"])), 2)


class WhyThisRank(unittest.TestCase):

    def test_it_names_the_top_two_and_bottom_two_contributors(self):
        why = render_excel.why_this_rank(job("a", 61.0)["scores"])
        self.assertIn("skill match 80", why)
        self.assertIn("competition 40", why)

    def test_it_ranks_by_contribution_not_by_raw_value(self):
        """freshness scores 95 on weight 5 and did not move this job."""
        why = render_excel.why_this_rank(job("a", 61.0)["scores"])
        up = why.split(";")[0]
        self.assertIn("skill match", up)
        self.assertNotIn("freshness", up)

    def test_it_names_what_could_not_be_measured(self):
        why = render_excel.why_this_rank(job("a", 61.0)["scores"])
        self.assertIn("not measured: comp signal", why)

    def test_a_job_with_nothing_scored_says_so_rather_than_returning_blank(self):
        why = render_excel.why_this_rank({"components": {
            "skill_match": {"value": None, "weight": 30}}})
        self.assertIn("no component could be scored", why)
        self.assertIn("not measured: skill match", why)

    def test_it_survives_a_missing_scores_block(self):
        self.assertTrue(render_excel.why_this_rank(None))
        self.assertTrue(render_excel.why_this_rank({}))


class Formatting(WorkbookCase):

    def test_the_sheet_has_an_autofilter_over_the_used_range(self):
        render_excel.write_workbook({"requested": unsorted_jobs()}, self.path)
        self.assertIn("<autoFilter", sheet_xml(self.path))

    def test_the_score_column_carries_conditional_formatting(self):
        render_excel.write_workbook({"requested": unsorted_jobs()}, self.path)
        xml = sheet_xml(self.path)
        self.assertIn("conditionalFormatting", xml)
        self.assertIn("colorScale", xml)

    def test_confidence_is_a_column_so_a_sparse_job_is_visibly_sparse(self):
        render_excel.write_workbook({"requested": unsorted_jobs()}, self.path)
        rows = read_sheets(self.path)["requested"]
        confidence = rows[0].index("confidence")
        by_title = {row[rows[0].index("title")]: row[confidence]
                    for row in rows[1:]}
        self.assertEqual(by_title["Engineer c"], 0.3)
        self.assertEqual(by_title["Engineer b"], 1.0)


class NeverCrashes(WorkbookCase):

    def test_missing_and_none_fields_do_not_break_the_write(self):
        sparse = {"job_key": "k", "scores": {"adjusted": 12.0}}
        result = render_excel.write_workbook({"requested": [sparse]}, self.path)
        self.assertTrue(result["ok"])
        rows = read_sheets(self.path)["requested"]
        self.assertEqual(rows[1][rows[0].index("adjusted")], 12.0)

    def test_a_job_with_no_scores_at_all_still_writes_a_row(self):
        result = render_excel.write_workbook(
            {"requested": [{"job_key": "k", "title": "Engineer k"}]}, self.path)
        self.assertTrue(result["ok"])
        rows = read_sheets(self.path)["requested"]
        self.assertEqual(rows[1][rows[0].index("title")], "Engineer k")

    def test_no_scopes_at_all_writes_an_empty_workbook_rather_than_raising(self):
        result = render_excel.write_workbook({}, self.path)
        self.assertTrue(result["ok"])
        self.assertEqual(result["sheets"], [])

    def test_none_instead_of_a_scope_dict_is_tolerated(self):
        self.assertTrue(render_excel.write_workbook(None, self.path)["ok"])

    def test_a_blank_cell_is_written_blank_and_never_as_zero(self):
        blank = job("a", 61.0)
        blank["applications"] = None
        render_excel.write_workbook({"requested": [blank]}, self.path)
        rows = read_sheets(self.path)["requested"]
        self.assertIsNone(rows[1][rows[0].index("applications")])


class Degradation(WorkbookCase):
    """No xlsxwriter writes one CSV per tab, in the same sorted order."""

    def _without_xlsxwriter(self):
        original = render_excel._load_xlsxwriter
        render_excel._load_xlsxwriter = lambda: None
        self.addCleanup(setattr, render_excel, "_load_xlsxwriter", original)

    def test_it_writes_one_csv_per_scope_and_says_which_degradation(self):
        self._without_xlsxwriter()
        with self.assertWarns(RuntimeWarning):
            result = render_excel.write_workbook(
                {"requested": unsorted_jobs(), "adjacent": []}, self.path)
        self.assertTrue(result["ok"])
        self.assertEqual(result["degraded"], "csv")
        self.assertIn("xlsxwriter missing", result["note"])
        self.assertEqual(len(result["paths"]), 2)
        for path in result["paths"]:
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".csv")

    def test_the_csv_rows_are_sorted_exactly_as_the_workbook_would_be(self):
        self._without_xlsxwriter()
        with self.assertWarns(RuntimeWarning):
            result = render_excel.write_workbook(
                {"requested": unsorted_jobs()}, self.path)
        import csv

        with open(result["paths"][0], encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["adjusted"] for r in rows],
                         ["83.5", "70.2", "61.0", "52.4"])
        self.assertEqual([r["rank"] for r in rows], ["1", "2", "3", "4"])

    def test_the_degraded_path_never_raises_for_the_missing_wheel(self):
        self._without_xlsxwriter()
        with self.assertWarns(RuntimeWarning):
            render_excel.write_workbook({"requested": []}, self.path)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
