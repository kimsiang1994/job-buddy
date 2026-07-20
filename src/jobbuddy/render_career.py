"""`career_paths.pdf` -- the career analysis as a document, plus its charts.

**Fully deterministic. No LLM call anywhere in this module, and there must
never be one.** The same constraint as `career_paths`, for the same reason: the
numbers upstream are arithmetic over postings we hold, and a model asked to
narrate them would produce confident prose about somebody's career with nothing
gating it. This renderer only formats what `career_paths.analyse` computed, and
it is not allowed to conclude anything the analysis did not.

Three rules it inherits rather than invents:

  **Degrade, never raise.** No `typst` wheel means the `.typ` source is written
  next to where the PDF would have gone, with the compile command in the note,
  exactly as `render_resume` does. A missing optional dependency must cost you
  a file format, not the run.

  **Thin samples must look thin.** Cluster coverage is drawn through
  `render_charts.component_bars`, whose low-confidence state already exists for
  this purpose -- a hatched, pale bar for a measurement made on little
  evidence. Clusters at or under the thin threshold are rendered as a separate
  chart at a confidence derived from their own size, so they physically cannot
  be drawn as solidly as a 34-posting cluster. That is structural: no caption
  is doing the work.

  **The caveats are printed, not footnoted.** `analyse` puts the causation and
  coverage warnings in its return value so a renderer cannot drop them. This
  one prints them near the top, before the table anybody will act on, because a
  warning underneath a ranked list is a warning nobody read.

Typst plumbing is imported from `render_resume` rather than copied -- one
`_compile`, one escape function, one capability check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import render_charts
from .render_resume import (
    _compile,
    _escape,
    _load_typst,
    _quote,
    _warn_once,
)

STEM = "career_paths"

# Above this share of `thin_cluster_n` a cluster is drawn solid. Expressed as
# a fraction so it feeds `render_charts`' existing confidence machinery instead
# of a second, parallel notion of "reliable" -- the chart module already draws
# anything below `LOW_CONFIDENCE` hatched, and sample size is exactly the kind
# of thing that state was built to express.
_CONFIDENCE_DIVISOR = 2.0


def _confidence(cluster_ns: list[int], thin_n: int) -> float | None:
    """Map a group's sample sizes onto the chart module's confidence scale.

    Median cluster size over twice the thin threshold, capped at 1.0. A group
    whose typical cluster sits at the thin threshold lands exactly on
    `render_charts.LOW_CONFIDENCE`, and anything smaller falls below it and is
    drawn hatched. Nothing here is a magic number: the two inputs are the
    thin threshold the analysis already chose and the group's own sizes.
    """
    if not cluster_ns:
        return None
    ordered = sorted(cluster_ns)
    middle = ordered[len(ordered) // 2]
    ceiling = max(1.0, float(thin_n) * _CONFIDENCE_DIVISOR)
    return min(1.0, middle / ceiling)


def _coverage_scores(clusters: list[dict[str, Any]], thin_n: int) -> dict[str, Any]:
    """A `scoring.score_job`-shaped block, so `component_bars` can read it."""
    components: dict[str, Any] = {}
    for cluster in clusters:
        coverage = cluster.get("coverage") or {}
        components[f"{cluster['label']} (n={cluster['n']})"] = {
            "value": coverage.get("median_pct"),
            "weight": cluster.get("n", 0),
            "detail": {"reason": coverage.get("reason")},
        }
    return {
        "confidence": _confidence([c.get("n", 0) for c in clusters], thin_n),
        "components": components,
    }


def _salary_evidence_scores(clusters: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of each cluster's postings that stated a salary, as a 0-100 bar.

    This chart exists because a median with a hidden denominator is the single
    most misleading number the analysis can produce. A cluster whose median was
    suppressed for too few stated salaries renders as `value=None` -- the
    hatched empty track, with the reason -- rather than as a low bar, because a
    low bar would read as "this pays badly" when it means "we do not know".
    """
    components: dict[str, Any] = {}
    for cluster in clusters:
        pay = cluster.get("pay") or {}
        suppressed = pay.get("median_sgd") is None
        components[f"{cluster['label']} (n={cluster['n']})"] = {
            "value": None if suppressed else round(
                float(pay.get("stated_fraction") or 0.0) * 100.0, 1),
            "weight": pay.get("stated_n", 0),
            "detail": {"reason": pay.get("reason")},
        }
    return {"confidence": 1.0, "components": components}


def charts(analysis: dict[str, Any]) -> dict[str, str]:
    """Every SVG for the analysis, keyed by filename stem. Never raises."""
    analysis = analysis or {}
    clusters = [c for c in (analysis.get("clusters") or []) if isinstance(c, dict)]
    thin_n = int((analysis.get("settings") or {}).get(
        "thin_cluster_n", 8) or 8)

    solid = [c for c in clusters if not c.get("thin")]
    thin = [c for c in clusters if c.get("thin")]

    out = {
        "coverage": render_charts.component_bars(_coverage_scores(solid, thin_n)),
        "salary_evidence": render_charts.component_bars(
            _salary_evidence_scores(clusters)),
    }
    # Only when there are any -- an empty second chart invites the reader to
    # think something was measured and came back blank.
    if thin:
        out["coverage_thin"] = render_charts.component_bars(
            _coverage_scores(thin, thin_n))
    return out


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------

def _pay_line(pay: dict[str, Any], current: int | None) -> str:
    stated_n = pay.get("stated_n", 0)
    of_n = pay.get("of_n", 0)
    if pay.get("median_sgd") is None:
        return f"{_escape(pay.get('reason') or 'no salary data')} of {of_n} postings"

    line = (f"median S${pay['median_sgd']:,}/month "
            f"(stated by {stated_n} of {of_n} postings)")
    if pay.get("delta_pct") is not None and current:
        delta = pay["delta_vs_current_sgd"]
        sign = "+" if delta >= 0 else ""
        line += f" — {sign}S${delta:,} ({sign}{pay['delta_pct']:g}%) vs your S${current:,}"
    return _escape(line)


def _cluster_block(cluster: dict[str, Any], current: int | None) -> list[str]:
    coverage = cluster.get("coverage") or {}
    lines: list[str] = []

    thin = " #text(fill: rgb(\"#a33\"))[— thin sample]" if cluster.get("thin") else ""
    lines.append(f"=== {_escape(cluster.get('label') or 'unnamed')} "
                 f"({cluster.get('n', 0)} postings){thin}")
    lines.append("")

    median = coverage.get("median_pct")
    if median is None:
        lines.append(f"Requirement coverage: {_escape(coverage.get('reason') or 'not measured')}")
    else:
        lines.append(f"Requirement coverage: {median:g}% "
                     f"(median over {coverage.get('measured_n', 0)} of "
                     f"{coverage.get('of_n', 0)} postings)")
    lines.append("")
    lines.append(f"Pay: {_pay_line(cluster.get('pay') or {}, current)}")
    lines.append("")

    mix = cluster.get("seniority_mix") or {}
    if mix:
        rendered = ", ".join(f"{_escape(k)} {v}" for k, v in mix.items())
        lines.append(f"Seniority of these postings: {rendered}")
        lines.append("")

    gap = cluster.get("skills_gap") or []
    if gap:
        lines.append("Recurring requirements your profile does not cover:")
        lines.append("")
        for entry in gap:
            lines.append(f"- {_escape(entry.get('skill'))} — "
                         f"{entry.get('postings', 0)} of {entry.get('of_n', 0)} postings")
    else:
        lines.append("No requirement recurs across this cluster that your profile "
                     "does not already cover. With this few postings that is as "
                     "likely to mean too little data as a clean sheet.")
    lines.append("")
    return lines


def _movement_block(movement: dict[str, Any]) -> list[str]:
    lines = ["== Movement over time", ""]
    if movement.get("status") != "measured":
        lines.append(_escape(movement.get("status") or "insufficient history"))
        lines.append("")
        if movement.get("note"):
            lines.append(f"_{_escape(movement['note'])}_")
            lines.append("")
        lines.append("No trend is estimated from this. An imputed trend is worse "
                     "than no trend, because it looks like a finding.")
        lines.append("")
        return lines

    lines.append(_escape(movement.get("basis") or ""))
    lines.append("")
    for entry in movement.get("clusters") or []:
        change = entry.get("change", 0)
        sign = "+" if change >= 0 else ""
        flag = " (thin)" if entry.get("thin") else ""
        lines.append(f"- {_escape(entry.get('label'))}: "
                     f"{entry.get('postings_prior', 0)} → "
                     f"{entry.get('postings_recent', 0)} postings "
                     f"({sign}{change}){flag}")
    lines.append("")
    skills = movement.get("skills") or {}
    if skills.get("status") != "measured":
        lines.append(f"Skills movement: {_escape(skills.get('reason') or 'unavailable')}")
        lines.append("")
    return lines


def build_typst_source(analysis: dict[str, Any]) -> str:
    """The `.typ` document. Pure: same analysis in, same bytes out."""
    analysis = analysis or {}
    clusters = [c for c in (analysis.get("clusters") or []) if isinstance(c, dict)]
    caveats = analysis.get("caveats") or {}
    current = analysis.get("current_salary_sgd_monthly")

    lines: list[str] = [
        "#set page(paper: \"a4\", margin: (x: 0.6in, y: 0.7in))",
        "#set text(font: (\"Helvetica\", \"Arial\"), size: 10pt)",
        "#set par(justify: false, leading: 0.6em)",
        f"#set document(title: {_quote('Career paths')})",
        "",
        "= Where to go next",
        "",
        f"Computed from {analysis.get('n_postings', 0)} postings in "
        f"{analysis.get('n_clusters', 0)} role clusters"
        + (f", with {analysis['unclustered_postings']} postings whose titles named "
           "no recognisable role and which were left out rather than forced "
           "somewhere" if analysis.get("unclustered_postings") else "")
        + ".",
        "",
        f"Skills read from: {_escape(analysis.get('profile_basis') or 'unknown')}.",
        "",
        "Method: " + _escape(analysis.get("method") or "") + ".",
        "",
        "== Read these first",
        "",
    ]

    # Printed above the ranked list, deliberately. A caveat below the table is a
    # caveat nobody reaches.
    for key in ("causation", "coverage", "coverage_metric", "thin_samples",
                "salary_denominator", "selection"):
        text = caveats.get(key)
        if text:
            lines.append(f"- *{_escape(key.replace('_', ' ').title())}.* {_escape(text)}")
    lines.append("")

    lines.append("== Adjacent roles")
    lines.append("")
    if not clusters:
        lines.append("No clusters. Either there were no postings, or no title "
                     "named a role often enough to group. Nothing is inferred "
                     "from an empty corpus.")
        lines.append("")
    else:
        lines.append("Ranked by median requirement coverage times number of "
                     "postings — read as roughly how many of these postings you "
                     "already meet the stated requirements for. It is not a "
                     "score out of anything.")
        lines.append("")
        for cluster in clusters:
            lines.extend(_cluster_block(cluster, current))

    lines.extend(_movement_block(analysis.get("movement") or {}))

    lines.append("#line(length: 100%)")
    lines.append("")
    lines.append("_Generated deterministically from collected job postings. "
                 "No language model was involved in producing any figure or "
                 "any sentence in this document._")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def render(analysis: dict[str, Any], out_dir: Path,
           stem: str = STEM) -> dict[str, Any]:
    """Write `career_paths.pdf` and its SVGs. Never raises.

    With Typst absent it writes `career_paths.typ` instead and says so in
    `note`, matching `render_resume.render_pdf` -- the charts are written
    either way, since they are hand-rolled SVG with no dependency at all.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name, svg in charts(analysis).items():
        path = out_dir / f"{stem}.{name}.svg"
        path.write_text(svg, encoding="utf-8")
        written.append(path)

    source = build_typst_source(analysis)

    if _load_typst() is None:
        _warn_once("pdf", "typst is not installed -- emitting .typ source "
                          "instead of a PDF: py -m pip install -e .[tailoring]")
        typ_path = out_dir / f"{stem}.typ"
        typ_path.write_text(source, encoding="utf-8")
        return {
            "ok": True,
            "degraded": "typ",
            "path": typ_path,
            "charts": written,
            "note": (f"typst missing; compile the emitted .typ with: "
                     f"typst compile {stem}.typ {stem}.pdf"),
            "source": source,
        }

    pdf_path = out_dir / f"{stem}.pdf"
    try:
        pdf_path.write_bytes(_compile(source))
    except Exception as exc:  # pragma: no cover - depends on the typst build
        # A compile failure must not lose the analysis. The source goes to disk
        # so the user still has the document, and the error is reported rather
        # than swallowed.
        typ_path = out_dir / f"{stem}.typ"
        typ_path.write_text(source, encoding="utf-8")
        _warn_once("career-typst-compile",
                   f"typst could not compile the career document ({exc}); "
                   f"wrote {typ_path.name} instead")
        return {"ok": False, "degraded": "typ", "path": typ_path,
                "charts": written, "note": f"typst compile failed: {exc}",
                "source": source}

    return {"ok": True, "degraded": None, "path": pdf_path,
            "charts": written, "note": None, "source": source}
