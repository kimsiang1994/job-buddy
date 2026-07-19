"""The ranked workbook: one tab per scope, best job first.

**Ranked on `adjusted`, not `total`.** They are different numbers and only one
of them is the ranking. `total` is the weighted mean of whatever happened to be
measurable, which on a live run put nine jobs with no published salary at 90-95
and a job with a stated 10-20k range at 82 -- knowing less about a job made it
look better. `adjusted` prices that ignorance by shrinking the score toward
neutral in proportion to how little was measured. Sorting a workbook by `total`
would reintroduce the exact bug `scoring.py` fixed, one layer further out, so
`total` is shown as a column and never used as the key.

**The sort happens in Python, before a single cell is written.** XLSX stores a
sort *state* -- a record of what a sheet was sorted by -- not sorted output.
Apply a sort state to rows written in arbitrary order and Excel will honour it,
LibreOffice will honour it on refresh, and a plain parser (pandas, an importer,
`openpyxl`) will read the rows exactly as written: unsorted. The workbook then
looks right to the person who made it and wrong to everyone downstream. Sorting
the list in Python makes the file's byte order the ranking, so every reader
agrees. `tests/test_render_excel.py` reads the rows back out of the written file
and asserts the order there, rather than asserting that a sort was called.

**Sparse jobs are visibly sparse.** `confidence` is a column, not a footnote,
and a low-confidence row is formatted so the eye catches it. A workbook is
skimmed top-down and a top-ranked row carries authority; if that row was graded
on a third of the evidence, the reader has to be able to see it without opening
the JSON.

**`why this rank` is built from the score, not written by hand.** Top two
contributors and bottom two, by contribution to the weighted mean rather than by
raw value -- a 100 on a weight-5 component is not what moved a job up the list.
Components that could not be scored are named, because "not measured" is a
reason for a rank too.

Degrades: no `xlsxwriter` writes one CSV per tab, in the same sorted order, and
says which degradation was taken.
"""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from typing import Any

# Below this share of the scoring weight, a row is mostly prior rather than
# evidence. Same threshold the charts use, for the same reason.
LOW_CONFIDENCE = 0.5

COLUMNS: tuple[tuple[str, str, int], ...] = (
    # (header, job key or computed name, column width)
    ("rank", "_rank", 6),
    ("adjusted", "_adjusted", 10),
    ("total", "_total", 8),
    ("confidence", "_confidence", 11),
    ("why this rank", "_why", 62),
    ("title", "title", 38),
    ("company", "company", 26),
    ("source", "source", 10),
    ("seniority", "seniority", 14),
    ("salary min", "salary_min_sgd", 11),
    ("salary max", "salary_max_sgd", 11),
    ("applications", "applications", 12),
    ("vacancies", "vacancies", 10),
    ("age days", "age_days", 9),
    ("location", "location", 20),
    ("url", "url", 44),
    ("job key", "job_key", 22),
    # Set by the tailoring stage, blank when it did not run or did not reach
    # this job. It carries the FAILURE reason as well as the successes: a
    # workbook that lists only the jobs that rendered lies by omission, and the
    # omission is invisible to the person reading it.
    ("tailoring", "tailoring", 44),
)

_ILLEGAL_SHEET = re.compile(r"[\[\]:*?/\\]")
_warned: set[str] = set()


def _load_xlsxwriter():
    """The xlsxwriter module, or None. Patched in tests to simulate absence."""
    try:
        import xlsxwriter

        return xlsxwriter
    except ImportError:
        return None


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def reset_warning_cache() -> None:
    """Forget which warnings have been emitted. For tests only."""
    _warned.clear()


# --------------------------------------------------------------------------
# ranking and explanation -- pure, so they are testable without a workbook
# --------------------------------------------------------------------------

def _adjusted(job: dict[str, Any]) -> float:
    """The ranking key. A job that failed to score sorts last, not first.

    Missing becomes -1 rather than 0 so it cannot tie with a genuine zero, and
    negative so it never outranks one.
    """
    value = (job.get("scores") or {}).get("adjusted")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return -1.0
    return float(value)


def rank_jobs(jobs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Jobs, best `adjusted` first. Ties broken by title for a stable file.

    A stable order matters more than it looks: two runs over the same data
    should produce byte-comparable workbooks, or a diff is useless.
    """
    return sorted(list(jobs or []),
                  key=lambda j: (-_adjusted(j), str(j.get("title") or ""),
                                 str(j.get("job_key") or "")))


def why_this_rank(scores: dict[str, Any] | None) -> str:
    """Top two and bottom two contributors, plus what could not be measured.

    Ranked by contribution (value x weight), not by raw value. The difference is
    the whole point of the column: a perfect score on a weight-5 component did
    not move this job, and naming it as a reason would mislead.
    """
    scores = scores or {}
    components: dict[str, Any] = scores.get("components") or {}
    scored = [
        (name, float(c["value"]), float(c.get("weight") or 0))
        for name, c in components.items()
        if isinstance(c, dict)
        and isinstance(c.get("value"), (int, float))
        and not isinstance(c.get("value"), bool)
        and float(c.get("weight") or 0) > 0
    ]
    missing = [str(name).replace("_", " ") for name, c in components.items()
               if isinstance(c, dict) and c.get("value") is None]

    if not scored:
        return ("no component could be scored"
                + (f"; not measured: {', '.join(missing)}" if missing else ""))

    by_contribution = sorted(scored, key=lambda x: x[1] * x[2], reverse=True)
    up = ", ".join(f"{n.replace('_', ' ')} {v:.0f}"
                   for n, v, _ in by_contribution[:2])
    down = ", ".join(f"{n.replace('_', ' ')} {v:.0f}"
                     for n, v, _ in sorted(scored, key=lambda x: x[1])[:2])
    tail = f"; not measured: {', '.join(missing)}" if missing else ""
    return f"up: {up}; down: {down}{tail}"


def row_for(job: dict[str, Any], rank: int) -> dict[str, Any]:
    """One row. Separate from the writer so the column contract is testable."""
    scores = job.get("scores") or {}
    computed = {
        "_rank": rank,
        "_adjusted": scores.get("adjusted"),
        "_total": scores.get("total"),
        "_confidence": scores.get("confidence"),
        "_why": why_this_rank(scores),
    }
    row: dict[str, Any] = {}
    for header, key, _ in COLUMNS:
        row[header] = computed[key] if key.startswith("_") else job.get(key)
    return row


def sheet_name(scope: str, taken: set[str] | None = None) -> str:
    """A legal, unique sheet name.

    Excel rejects `[]:*?/\\` and anything over 31 characters, and silently
    refuses a duplicate. A scope called "AI / ML engineering" would otherwise
    kill the whole write at the last tab.
    """
    name = _ILLEGAL_SHEET.sub("-", str(scope or "jobs").strip()) or "jobs"
    name = name[:31]
    taken = taken if taken is not None else set()
    if name.lower() not in {t.lower() for t in taken}:
        return name
    for suffix in range(2, 100):
        candidate = f"{name[:31 - len(str(suffix)) - 1]}-{suffix}"
        if candidate.lower() not in {t.lower() for t in taken}:
            return candidate
    return name[:28] + "-xx"


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def write_workbook(jobs_by_scope: dict[str, list[dict[str, Any]]] | None,
                   out_path: Path) -> dict[str, Any]:
    """One tab per scope, each sorted descending by `adjusted`.

    The first key is the requested scope by convention -- dicts preserve
    insertion order -- and the rest are the adjacent scopes, so tab order
    matches the order the caller asked for.

    Never raises for a missing wheel: without `xlsxwriter` it writes one CSV per
    tab and reports `degraded="csv"`.
    """
    jobs_by_scope = jobs_by_scope or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = {scope: rank_jobs(jobs) for scope, jobs in jobs_by_scope.items()}

    xlsxwriter = _load_xlsxwriter()
    if xlsxwriter is None:
        _warn_once("xlsxwriter",
                   "xlsxwriter is not installed -- writing one CSV per scope "
                   "instead of a workbook: py -m pip install -e .[tailoring]")
        paths = _write_csvs(ranked, out_path)
        return {
            "ok": True, "degraded": "csv", "path": out_path.parent,
            "paths": paths, "sheets": list(ranked),
            "rows": {scope: len(jobs) for scope, jobs in ranked.items()},
            "note": "xlsxwriter missing; wrote CSV per scope instead of .xlsx",
        }

    book = xlsxwriter.Workbook(str(out_path), {"constant_memory": False})
    try:
        header_fmt = book.add_format({"bold": True, "bg_color": "#eeeeee",
                                      "bottom": 1})
        number_fmt = book.add_format({"num_format": "0.0"})
        percent_fmt = book.add_format({"num_format": "0%"})
        # A sparse row must be visible at a glance. Same threshold as the
        # charts: below it, the rank is mostly prior rather than evidence.
        sparse_fmt = book.add_format({"num_format": "0%", "italic": True,
                                      "font_color": "#993333"})
        wrap_fmt = book.add_format({"text_wrap": True, "valign": "top"})

        names: set[str] = set()
        sheets: list[str] = []
        for scope, jobs in ranked.items():
            name = sheet_name(scope, names)
            names.add(name)
            sheets.append(name)
            sheet = book.add_worksheet(name)
            _write_sheet(sheet, jobs, header_fmt, number_fmt, percent_fmt,
                         sparse_fmt, wrap_fmt)
        book.close()
    except Exception:
        # A half-written .xlsx is unopenable and looks like a corrupt file
        # rather than a failed run.
        out_path.unlink(missing_ok=True)
        raise

    return {
        "ok": True, "degraded": None, "path": out_path, "paths": [out_path],
        "sheets": sheets,
        "rows": {scope: len(jobs) for scope, jobs in ranked.items()},
        "note": "",
    }


def _write_sheet(sheet: Any, jobs: list[dict[str, Any]], header_fmt: Any,
                 number_fmt: Any, percent_fmt: Any, sparse_fmt: Any,
                 wrap_fmt: Any) -> None:
    """Write one already-sorted sheet.

    `jobs` arrives sorted. This function does not sort and must not: the file's
    row order IS the ranking, and a sort applied here would hide whether the
    caller ranked at all.
    """
    headers = [h for h, _, _ in COLUMNS]
    for index, (header, _, width) in enumerate(COLUMNS):
        sheet.write(0, index, header, header_fmt)
        sheet.set_column(index, index, width,
                         wrap_fmt if header == "why this rank" else None)

    adjusted_col = headers.index("adjusted")
    confidence_col = headers.index("confidence")

    for row_index, job in enumerate(jobs, start=1):
        row = row_for(job, row_index)
        confidence = row["confidence"]
        low = (isinstance(confidence, (int, float))
               and not isinstance(confidence, bool)
               and confidence < LOW_CONFIDENCE)
        for col_index, header in enumerate(headers):
            value = row[header]
            if header == "confidence":
                fmt = sparse_fmt if low else percent_fmt
            elif header in ("adjusted", "total"):
                fmt = number_fmt
            else:
                fmt = None
            if value is None:
                # Blank, never 0. A zero here is a measurement.
                sheet.write_blank(row_index, col_index, None, fmt)
            elif isinstance(value, bool):
                sheet.write_boolean(row_index, col_index, value, fmt)
            elif isinstance(value, (int, float)):
                sheet.write_number(row_index, col_index, value, fmt)
            else:
                sheet.write_string(row_index, col_index, str(value), fmt)

    last_row = max(len(jobs), 1)
    sheet.autofilter(0, 0, last_row, len(headers) - 1)
    sheet.freeze_panes(1, 0)
    if jobs:
        sheet.conditional_format(1, adjusted_col, last_row, adjusted_col, {
            "type": "3_color_scale",
            "min_color": "#f4b6b6", "mid_color": "#ffe9a8",
            "max_color": "#b7d7bf",
        })
        sheet.conditional_format(1, confidence_col, last_row, confidence_col, {
            "type": "cell", "criteria": "<", "value": LOW_CONFIDENCE,
            "format": sparse_fmt,
        })


def _write_csvs(ranked: dict[str, list[dict[str, Any]]],
                out_path: Path) -> list[Path]:
    """The degraded path: one CSV per scope, in the same sorted order.

    utf-8-sig because Excel on Windows renders a plain-utf-8 CSV as mojibake,
    and the whole point of this fallback is that a human can still open it.
    """
    written: list[Path] = []
    headers = [h for h, _, _ in COLUMNS]
    for scope, jobs in ranked.items():
        path = out_path.parent / f"{out_path.stem}.{sheet_name(scope)}.csv"
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                for index, job in enumerate(jobs, start=1):
                    writer.writerow(row_for(job, index))
            written.append(path)
        except OSError as exc:
            _warn_once(f"csv:{path.name}",
                       f"could not write {path.name} ({exc})")
    return written
