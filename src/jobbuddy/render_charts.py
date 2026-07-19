"""Hand-rolled SVG charts for the per-job report. Pure functions, stdlib only.

**No matplotlib, deliberately.** Six chart shapes do not justify ~60MB of wheel
plus NumPy in a project whose core is stdlib-only and whose one hard rule is
that a missing wheel degrades instead of raising. Matplotlib would also draw in
its own type and its own palette, so the report would read as two documents
stapled together. Writing the SVG by hand keeps the charts in the report's
typography and keeps them trivially testable -- every function here takes data
and returns a string, so a test asserts on structure rather than on pixels.

**A chart must never imply precision the data lacks.** This is the whole reason
this module has opinions.

`scoring.py` already fixed this bug once at the number level: a component with
no data returns None and its weight is *removed from the denominator*, because
imputing the neutral 50 would quietly claim knowledge nobody has. It fixed it a
second time with the confidence adjustment, after a live run where nine jobs
with no published salary out-ranked a job with a stated 10-20k range -- knowing
less made a job look better.

A bar chart re-introduces exactly that bug at the picture level, and worse,
because a picture is not read sceptically. A missing component drawn at the
middle of the axis is a confident claim of "average". A missing component drawn
at zero is a confident claim of "bad". Both are lies about evidence that does
not exist, and no caption underneath undoes them.

So the rule here is structural, not cosmetic:

  - `value is None` never produces a bar. It produces a hatched, empty track
    labelled "not measured". The reader's eye finds nothing to compare.
  - Where `scores["confidence"]` is low, every measured bar is drawn hatched and
    pale rather than solid, so a job graded on a third of the evidence cannot
    look as certain as one graded on all of it.
  - Both states are carried in `data-state` on the group, so the distinction is
    machine-checkable and cannot be quietly dropped by a later restyle.

Ids inside `<defs>` are constant. Two of these SVGs in one HTML page share a
pattern id, which is harmless because the definitions are identical; as separate
files or separate PDF images -- which is how the report uses them -- it never
arises.
"""

from __future__ import annotations

from typing import Any, Sequence

# Below this share of the scoring weight, "measured" overstates it. 0.5 is the
# point where at least as much of the ranking is prior as is evidence.
LOW_CONFIDENCE = 0.5

WIDTH = 520
LABEL_W = 152
GUTTER = 12
ROW_H = 24
BAR_H = 12
PAD = 14
FONT = "11px sans-serif"
SMALL = "9px sans-serif"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
TRACK = "#e6e6e6"
STRONG = "#2f6f4f"
WEAK = "#a33"
PALE = "#9dbfae"
MARK = "#1a1a1a"

NOT_MEASURED = "not measured"

# Hatching, not a lighter solid. A pale solid bar still reads as a measurement;
# a hatched one reads as provisional, which is the honest impression.
_DEFS = (
    '<defs>'
    '<pattern id="jb-hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
    'patternTransform="rotate(45)">'
    f'<rect width="6" height="6" fill="#ffffff"/>'
    f'<line x1="0" y1="0" x2="0" y2="6" stroke="{PALE}" stroke-width="3"/>'
    '</pattern>'
    '<pattern id="jb-empty" width="6" height="6" patternUnits="userSpaceOnUse" '
    'patternTransform="rotate(45)">'
    '<rect width="6" height="6" fill="#ffffff"/>'
    f'<line x1="0" y1="0" x2="0" y2="6" stroke="{TRACK}" stroke-width="2"/>'
    '</pattern>'
    '</defs>'
)

MEASURED = "measured"
LOW = "low-confidence"
UNMEASURED = "not-measured"


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def _esc(text: Any) -> str:
    """XML-escape. Company names carry `&` and job titles carry `<` often enough."""
    return (str("" if text is None else text)
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _num(value: Any, default: float | None = None) -> float | None:
    """A float, or None. Never raises -- a renderer that dies on a bad cell is
    worse than one that says it could not measure something."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _state(value: float | None, confidence: float | None) -> str:
    """The three honesty states. See the module docstring."""
    if value is None:
        return UNMEASURED
    conf = _num(confidence, 1.0)
    return LOW if conf is not None and conf < LOW_CONFIDENCE else MEASURED


def _fill(state: str) -> str:
    if state == UNMEASURED:
        return "url(#jb-empty)"
    if state == LOW:
        return "url(#jb-hatch)"
    return STRONG


def _open(width: int, height: int, title: str, desc: str = "") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(title)}">',
        f"<title>{_esc(title)}</title>",
        f"<desc>{_esc(desc)}</desc>" if desc else "",
        _DEFS,
    ]


def _text(x: float, y: float, content: str, *, fill: str = INK,
          font: str = FONT, anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="font: {font}" '
            f'fill="{fill}" text-anchor="{anchor}">{_esc(content)}</text>')


def _svg(parts: Sequence[str]) -> str:
    return "\n".join(p for p in parts if p) + "\n</svg>\n"


# --------------------------------------------------------------------------
# 1. horizontal bars -- the score components
# --------------------------------------------------------------------------

def component_bars(scores: dict[str, Any] | None,
                   width: int = WIDTH) -> str:
    """One horizontal bar per scoring component, 0-100.

    Reads `scores` exactly as `scoring.score_job` writes it. A component whose
    `value` is None gets an empty hatched track and the words "not measured" --
    never a bar at 50 and never a bar at 0. When the run-level `confidence` is
    below `LOW_CONFIDENCE` every measured bar is hatched too, because the
    ranking those bars explain is itself mostly prior at that point.
    """
    scores = scores or {}
    components: dict[str, Any] = scores.get("components") or {}
    confidence = _num(scores.get("confidence"), 1.0)

    rows = list(components.items())
    height = PAD * 2 + 22 + max(len(rows), 1) * ROW_H
    track_w = width - LABEL_W - GUTTER - PAD - 44

    parts = _open(width, height, "Score components",
                  f"{len(rows)} components; confidence "
                  f"{'not measured' if confidence is None else f'{confidence:.0%}'}")
    y = PAD + 12
    if confidence is not None and confidence < LOW_CONFIDENCE:
        parts.append(_text(PAD, y,
                           f"low confidence: only {confidence:.0%} of the scoring "
                           "weight could be measured",
                           fill=WEAK, font=SMALL))
    else:
        parts.append(_text(PAD, y, "0-100 per component", fill=MUTED, font=SMALL))
    y += 14

    if not rows:
        parts.append(_text(PAD, y + ROW_H / 2, "no components scored", fill=MUTED))
        return _svg(parts)

    for name, component in rows:
        component = component if isinstance(component, dict) else {}
        value = _num(component.get("value"))
        weight = _num(component.get("weight"), 0.0) or 0.0
        state = _state(value, confidence)
        top = y + (ROW_H - BAR_H) / 2

        label = str(name).replace("_", " ")
        parts.append(f'<g data-component="{_esc(name)}" data-state="{state}">')
        parts.append(_text(PAD, top + BAR_H - 2, label))
        parts.append(_text(LABEL_W - 6, top + BAR_H - 2, f"w{weight:g}",
                           fill=MUTED, font=SMALL, anchor="end"))
        parts.append(f'<rect x="{LABEL_W + GUTTER}" y="{top:.1f}" '
                     f'width="{track_w}" height="{BAR_H}" fill="{TRACK}"/>')
        if state == UNMEASURED:
            # The track is drawn hatched and EMPTY. No bar means no comparison,
            # which is the only honest picture of an absent measurement.
            parts.append(f'<rect x="{LABEL_W + GUTTER}" y="{top:.1f}" '
                         f'width="{track_w}" height="{BAR_H}" '
                         f'fill="{_fill(state)}" stroke="{TRACK}"/>')
            reason = (component.get("detail") or {}).get("reason") if isinstance(
                component.get("detail"), dict) else None
            parts.append(_text(LABEL_W + GUTTER + 4, top + BAR_H - 2,
                               NOT_MEASURED + (f" ({reason})" if reason else ""),
                               fill=MUTED, font=SMALL))
        else:
            filled = track_w * _clamp(value) / 100.0
            parts.append(f'<rect x="{LABEL_W + GUTTER}" y="{top:.1f}" '
                         f'width="{filled:.1f}" height="{BAR_H}" '
                         f'fill="{_fill(state)}" stroke="{PALE}"/>')
            parts.append(_text(width - PAD, top + BAR_H - 2, f"{value:.0f}",
                               anchor="end", font=SMALL))
        parts.append("</g>")
        y += ROW_H

    return _svg(parts)


# --------------------------------------------------------------------------
# 2. bullet chart -- fit against the bar the JD states
# --------------------------------------------------------------------------

def fit_gauge(value: float | None, target: float | None = None,
              confidence: float | None = None, *,
              label: str = "Fit", width: int = WIDTH) -> str:
    """A bullet chart: the score, against the bar the job states it wants.

    `target` is the JD's own stated requirement expressed on the same 0-100
    scale (e.g. the share of core skills it calls mandatory). When it is None
    the tick is simply absent -- an invented target would turn "they did not
    say" into a pass or a fail.

    `value is None` renders an explicit insufficient-data state rather than an
    empty gauge, because an empty gauge reads as a zero.
    """
    value = _num(value)
    target = _num(target)
    state = _state(value, confidence)

    height = 74
    track_x, track_w = PAD, width - PAD * 2
    top = 34

    parts = _open(width, height, f"{label} gauge",
                  "insufficient data" if state == UNMEASURED else f"{value:.0f} of 100")
    parts.append(f'<g data-chart="gauge" data-state="{state}">')
    parts.append(_text(PAD, 18, label))
    parts.append(f'<rect x="{track_x}" y="{top}" width="{track_w}" height="18" '
                 f'fill="{TRACK}"/>')

    if state == UNMEASURED:
        parts.append(f'<rect x="{track_x}" y="{top}" width="{track_w}" '
                     f'height="18" fill="url(#jb-empty)" stroke="{TRACK}"/>')
        parts.append(_text(track_x + 6, top + 13, "insufficient data - "
                           + NOT_MEASURED, fill=MUTED, font=SMALL))
    else:
        filled = track_w * _clamp(value) / 100.0
        parts.append(f'<rect x="{track_x}" y="{top}" width="{filled:.1f}" '
                     f'height="18" fill="{_fill(state)}" stroke="{PALE}"/>')
        parts.append(_text(track_x + track_w, 18, f"{value:.0f}", anchor="end"))
        if state == LOW:
            parts.append(_text(track_x, height - 4,
                               "hatched: scored on partial evidence",
                               fill=WEAK, font=SMALL))

    if target is not None:
        tick = track_x + track_w * _clamp(target) / 100.0
        parts.append(f'<line data-role="target" x1="{tick:.1f}" y1="{top - 5}" '
                     f'x2="{tick:.1f}" y2="{top + 23}" stroke="{MARK}" '
                     'stroke-width="2"/>')
        parts.append(_text(tick, top - 8, f"asks {target:.0f}", anchor="middle",
                           font=SMALL, fill=MUTED))
    else:
        parts.append(_text(track_x + track_w, height - 4,
                           "no stated bar - " + NOT_MEASURED, anchor="end",
                           fill=MUTED, font=SMALL))
    parts.append("</g>")
    return _svg(parts)


# --------------------------------------------------------------------------
# 3. timeline -- how long this has been on the market
# --------------------------------------------------------------------------

def posting_timeline(job: dict[str, Any] | None, width: int = WIDTH,
                     halflife_days: float = 21.0) -> str:
    """Posted -> today -> closes, on a day axis.

    The halflife marker is drawn because "21 days old" means nothing on its own
    and "past the point where most applications have already landed" means
    something. With no posting date there is no axis to draw, so it says so
    instead of drawing a zero-length bar from an assumed today.
    """
    job = job or {}
    age = _num(job.get("age_days"))
    days_left = _num(job.get("days_left"))
    if days_left is None:
        days_left = _num((job.get("scores") or {}).get("days_left"))

    height = 82
    axis_y = 46
    x0, axis_w = PAD, width - PAD * 2

    if age is None:
        parts = _open(width, height, "Posting age", NOT_MEASURED)
        parts.append('<g data-chart="timeline" data-state="not-measured">')
        parts.append(_text(PAD, 18, "Time on market"))
        parts.append(f'<rect x="{x0}" y="{axis_y - 9}" width="{axis_w}" '
                     f'height="18" fill="url(#jb-empty)" stroke="{TRACK}"/>')
        parts.append(_text(PAD + 6, axis_y + 5, "no posting date - " + NOT_MEASURED,
                           fill=MUTED, font=SMALL))
        parts.append("</g>")
        return _svg(parts)

    span = max(age + max(days_left or 0.0, 0.0), halflife_days, 1.0)
    def at(day: float) -> float:
        return x0 + axis_w * _clamp(day / span, 0.0, 1.0)

    state = MEASURED
    parts = _open(width, height, "Posting age",
                  f"{age:.0f} days on market")
    parts.append(f'<g data-chart="timeline" data-state="{state}">')
    parts.append(_text(PAD, 18, "Time on market"))
    parts.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x0 + axis_w}" '
                 f'y2="{axis_y}" stroke="{TRACK}" stroke-width="6"/>')
    parts.append(f'<line data-role="elapsed" x1="{x0}" y1="{axis_y}" '
                 f'x2="{at(age):.1f}" y2="{axis_y}" stroke="{STRONG}" '
                 'stroke-width="6"/>')
    parts.append(f'<circle data-role="posted" cx="{x0}" cy="{axis_y}" r="4" '
                 f'fill="{MARK}"/>')
    parts.append(_text(x0, axis_y - 12, "posted", font=SMALL, fill=MUTED))
    parts.append(f'<circle data-role="today" cx="{at(age):.1f}" cy="{axis_y}" '
                 f'r="4" fill="{MARK}"/>')
    parts.append(_text(at(age), axis_y + 20, f"today, day {age:.0f}",
                       anchor="middle", font=SMALL))

    if halflife_days <= span:
        parts.append(f'<line data-role="halflife" x1="{at(halflife_days):.1f}" '
                     f'y1="{axis_y - 10}" x2="{at(halflife_days):.1f}" '
                     f'y2="{axis_y + 10}" stroke="{WEAK}" stroke-dasharray="3 2"/>')
        parts.append(_text(at(halflife_days), height - 4,
                           f"{halflife_days:.0f}d: most applications in",
                           anchor="middle", font=SMALL, fill=WEAK))

    if days_left is not None:
        parts.append(f'<circle data-role="closes" cx="{at(age + days_left):.1f}" '
                     f'cy="{axis_y}" r="4" fill="none" stroke="{MARK}"/>')
        parts.append(_text(x0 + axis_w, axis_y - 12,
                           f"closes in {days_left:.0f}d", anchor="end",
                           font=SMALL, fill=MUTED))
    else:
        parts.append(_text(x0 + axis_w, axis_y - 12,
                           "closing date " + NOT_MEASURED, anchor="end",
                           font=SMALL, fill=MUTED))
    parts.append("</g>")
    return _svg(parts)


# --------------------------------------------------------------------------
# 4. distribution strip -- where this job sits among the run's others
# --------------------------------------------------------------------------

def score_distribution(value: float | None, population: Sequence[Any] | None,
                       *, label: str = "Adjusted score vs this run",
                       width: int = WIDTH) -> str:
    """A one-dimensional strip: every job in the run, this one marked.

    A rank ("7th of 40") hides whether 7th is a near-miss or a cliff. The strip
    shows the shape, which is the part a number cannot carry.

    Non-numeric entries in `population` are dropped rather than coerced -- a job
    that failed to score is not a job that scored zero.
    """
    value = _num(value)
    points = [p for p in (_num(v) for v in (population or [])) if p is not None]

    height = 76
    strip_y = 40
    x0, strip_w = PAD, width - PAD * 2

    def at(score: float) -> float:
        return x0 + strip_w * _clamp(score) / 100.0

    state = UNMEASURED if value is None else MEASURED
    parts = _open(width, height, label, f"{len(points)} jobs in this run")
    parts.append(f'<g data-chart="distribution" data-state="{state}" '
                 f'data-points="{len(points)}">')
    parts.append(_text(PAD, 18, label))
    parts.append(f'<rect x="{x0}" y="{strip_y - 8}" width="{strip_w}" '
                 f'height="16" fill="{TRACK}"/>')

    for point in points:
        parts.append(f'<line data-role="peer" x1="{at(point):.1f}" '
                     f'y1="{strip_y - 8}" x2="{at(point):.1f}" '
                     f'y2="{strip_y + 8}" stroke="{PALE}"/>')

    if not points:
        parts.append(_text(PAD + 6, strip_y + 5, "no other jobs in this run",
                           fill=MUTED, font=SMALL))

    if value is None:
        parts.append(_text(PAD + 6, height - 6,
                           "this job: " + NOT_MEASURED, fill=MUTED, font=SMALL))
    else:
        parts.append(f'<line data-role="this-job" x1="{at(value):.1f}" '
                     f'y1="{strip_y - 14}" x2="{at(value):.1f}" '
                     f'y2="{strip_y + 14}" stroke="{MARK}" stroke-width="2"/>')
        better = sum(1 for p in points if p > value)
        # Stated as a count out of the population actually scored. With one job
        # the sentence is "1 of 1", which is honest and unimpressive -- exactly
        # the impression a one-job population should leave.
        parts.append(_text(at(value), height - 6,
                           f"{value:.0f} - ranks {better + 1} of "
                           f"{max(len(points), 1)}",
                           anchor="middle", font=SMALL))
    parts.append("</g>")
    return _svg(parts)


# --------------------------------------------------------------------------
# everything a report needs, from one job
# --------------------------------------------------------------------------

def charts_for_job(job: dict[str, Any] | None,
                   population: Sequence[Any] | None = None,
                   target: float | None = None) -> dict[str, str]:
    """Every chart for one job, keyed by name. Never raises on a partial job."""
    job = job or {}
    scores = job.get("scores") or {}
    return {
        "components": component_bars(scores),
        "fit": fit_gauge(scores.get("adjusted"), target,
                         scores.get("confidence")),
        "timeline": posting_timeline(job),
        "distribution": score_distribution(scores.get("adjusted"), population),
    }
