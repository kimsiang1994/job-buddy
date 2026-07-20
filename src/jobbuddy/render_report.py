"""The per-job analysis PDF: what this role is, and what it would cost you.

The resume is what you send. This is what you read before deciding to send it,
so its failure mode is the opposite one. A resume that omits something is
weaker; an analysis that omits something is *wrong*, because the reader will
fill the hole with an assumption and act on it.

**Nothing here is imputed. Ever.** Where the pipeline did not measure something
the report prints `not measured` and, where the reason is knowable, why. The
temptation this resists is small and constant: a salary midpoint from a national
average, a competition estimate from a view count, an application count guessed
from the posting's age. Each would make the page look complete. `scoring.py`
already paid for this lesson twice -- once when a missing component imputed at
50, and once when sparse jobs out-ranked well-documented ones because the
missing evidence was silently forgiven. A report that quietly fills gaps is
worse than one with visible gaps, because the visible gap is actionable and the
filled one is not.

Competition is the sharpest case. MyCareersFuture publishes a REAL
`totalNumberJobApplication`; Workable, the ATS boards and HN publish nothing. So
this reads the real number where it exists and prints "not published by this
source" where it does not. It never converts age into an estimated applicant
count.

**The gap section is the one worth the page.** `tailor()` returns `unaddressed`
-- the JD requirements no bullet answered -- but that list conflates two
completely different situations, and the difference is the whole value:

  - you HAVE this and it was cut for space. Actionable tonight: put it back, or
    lead with it in the cover note.
  - you genuinely LACK this. Not a tailoring bug, a development plan.

That split is `hr_panel.attribute_gaps`, and this module calls it rather than
reimplementing it, because a second implementation would drift and start telling
someone they have experience they do not have.

**Silent losses are surfaced.** What `fact_guard` rejected and what
`render_resume` cut to fit the page both appear here. A cut nobody is told about
is how a resume loses its best bullet.

Charts come from `render_charts` and the Typst plumbing comes from
`render_resume`, both imported rather than copied. No typst wheel degrades to
the `.typ` source plus the command that compiles it, exactly as the resume
renderer does -- one degradation pattern, not two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jobbuddy import render_charts
from jobbuddy import render_resume

NOT_MEASURED = "not measured"
NOT_PUBLISHED = "not published by this source"

# Reused so the report degrades exactly the way the resume renderer does.
_escape = render_resume._escape
_quote = render_resume._quote


# --------------------------------------------------------------------------
# helpers -- every one of them returns NOT_MEASURED rather than a zero
# --------------------------------------------------------------------------

def _value(raw: Any, fmt: str = "{}", *, missing: str = NOT_MEASURED) -> str:
    """A formatted value, or the missing marker. 0 and False are real values."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return missing
    try:
        return fmt.format(raw)
    except (ValueError, IndexError, KeyError):
        return str(raw)


def salary_line(job: dict[str, Any]) -> str:
    """The stated range, or why there is none.

    Never a midpoint from anywhere but the posting itself. MCF's advertising
    mandate makes salary mandatory for most SG postings but exempts anything
    over S$22,500/month -- which is exactly the band a senior search cares
    about, so "not stated" here is common and must not be papered over.
    """
    if not job.get("salary_is_stated"):
        return f"{NOT_MEASURED} (salary not stated in the posting)"
    low, high = job.get("salary_min_sgd"), job.get("salary_max_sgd")
    if low is None and high is None:
        return f"{NOT_MEASURED} (marked stated but no figures parsed)"
    if high is None or high == low:
        return f"SGD {low:,.0f}/month"
    return (f"SGD {low:,.0f} - {high:,.0f}/month "
            f"(midpoint {int((low + high) / 2):,})")


def competition_line(job: dict[str, Any]) -> dict[str, Any]:
    """Applications, per vacancy, and where the number came from.

    Splits "nobody has applied" from "nobody publishes the count". Those look
    identical in a blank cell and mean opposite things.
    """
    applications = job.get("applications")
    published = isinstance(applications, int) and not isinstance(applications, bool)
    vacancies = job.get("vacancies")
    detail = ((job.get("scores") or {}).get("components") or {}).get(
        "competition") or {}
    basis = (detail.get("detail") or {}).get("basis") if isinstance(
        detail.get("detail"), dict) else None

    per_vacancy: str = NOT_MEASURED
    if published and isinstance(vacancies, int) and vacancies > 0:
        per_vacancy = f"{applications / vacancies:.1f} per vacancy"
    elif published:
        per_vacancy = f"{applications} (vacancy count {NOT_MEASURED})"

    return {
        "applications": (str(applications) if published
                         else f"{NOT_PUBLISHED} ({job.get('source') or 'unknown'})"),
        "published": published,
        "per_vacancy": per_vacancy,
        "views": _value(job.get("views")),
        "apps_per_day": _value(job.get("apps_per_day"), "{:.2f}"),
        "basis": basis or (NOT_MEASURED if not published else "published count"),
        "vacancies": _value(vacancies),
    }


def market_time(job: dict[str, Any]) -> dict[str, str]:
    """How long it has been up, and how long is left."""
    age = job.get("age_days")
    freshness = ((job.get("scores") or {}).get("components") or {}).get(
        "freshness") or {}
    days_left = (freshness.get("detail") or {}).get("days_left") if isinstance(
        freshness.get("detail"), dict) else None
    return {
        "posted_at": _value(job.get("posted_at")),
        "age_days": _value(age, "{} days"),
        "expires_at": _value(job.get("expires_at")),
        "days_left": _value(days_left, "{} days"),
        "reposts": _value(job.get("repost_count"), "{}"),
        "seen_count": _value(job.get("seen_count"), "{}"),
    }


def apply_link(job: dict[str, Any]) -> dict[str, str]:
    """The apply URL, validated by shape only.

    Shape only, and it says so: reaching the URL is a network call and this
    module is offline. An `http(s)` absolute URL is reported as `validated:
    shape`; anything else is flagged rather than printed as if it were
    clickable, because a report that hands over a broken link has wasted the
    one action it exists to prompt.
    """
    url = str(job.get("url") or "").strip()
    if not url:
        return {"url": NOT_MEASURED, "status": "no URL on this posting"}
    if not url.lower().startswith(("http://", "https://")):
        return {"url": url, "status": "not a usable absolute URL - check the source"}
    return {"url": url, "status": "validated: shape (not fetched -- offline)"}


def company_signals(job: dict[str, Any]) -> dict[str, str]:
    """Hiring posture, from the company_signal component. Never inferred."""
    component = ((job.get("scores") or {}).get("components") or {}).get(
        "company_signal") or {}
    detail = component.get("detail") if isinstance(
        component.get("detail"), dict) else {}
    detail = detail or {}
    if component.get("value") is None:
        return {
            "status": NOT_MEASURED,
            "reason": str(detail.get("reason") or "no company history recorded"),
            "open_reqs": _value(detail.get("open_reqs")),
        }
    return {
        "status": f"score {component['value']}",
        "open_reqs": _value(detail.get("open_reqs")),
        "new_in_window": _value(detail.get("new_in_window")),
        "history_days": _value(detail.get("history_days")),
        "reason": "",
    }


def career_prospects(job: dict[str, Any]) -> dict[str, str]:
    """Seniority direction. Read off the seniority_fit component, not guessed."""
    component = ((job.get("scores") or {}).get("components") or {}).get(
        "seniority_fit") or {}
    detail = component.get("detail") if isinstance(
        component.get("detail"), dict) else {}
    detail = detail or {}
    if component.get("value") is None:
        return {"direction": NOT_MEASURED,
                "reason": str(detail.get("reason") or "seniority not determined"),
                "job_level": _value(job.get("seniority")),
                "years": NOT_MEASURED}
    return {
        "direction": str(detail.get("direction") or NOT_MEASURED),
        "job_level": _value(detail.get("job_level")),
        "target_level": _value(detail.get("target_level")),
        "years": _value(detail.get("years")),
        "basis": _value(detail.get("basis")),
        "reason": "",
    }


def strengths(job: dict[str, Any]) -> dict[str, Any]:
    """What the skill match actually found. Counts, not adjectives."""
    component = ((job.get("scores") or {}).get("components") or {}).get(
        "skill_match") or {}
    detail = component.get("detail") if isinstance(
        component.get("detail"), dict) else {}
    detail = detail or {}
    if component.get("value") is None:
        return {"measured": False,
                "reason": str(detail.get("reason") or "skills not comparable"),
                "matched": [], "missing_core": []}
    return {
        "measured": True,
        "matched": list(detail.get("matched") or []),
        "missing_core": list(detail.get("missing_core") or []),
        "matched_count": detail.get("matched_count"),
        "total_count": detail.get("total_count"),
        "core_score": detail.get("core_score"),
        "reason": "",
    }


# --------------------------------------------------------------------------
# the gap split -- delegated, never reimplemented
# --------------------------------------------------------------------------

def gap_analysis(tailored: dict[str, Any] | None,
                 profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Split `unaddressed` into "cut for space" and "genuinely lacking".

    Delegates to `hr_panel.attribute_gaps`. It is deterministic set matching
    against the verified facts, and a second copy of that logic here would
    eventually disagree with the first -- at which point the report starts
    telling someone they already have experience they do not have, which is the
    failure this codebase spends the most effort preventing.

    With no facts to match against, every unaddressed requirement stays
    unclassified. It is NOT defaulted to "genuinely lacking": with no evidence
    either way, claiming a gap is as much an invention as claiming coverage.
    """
    unaddressed = list((tailored or {}).get("unaddressed") or [])
    if not unaddressed:
        return {"unaddressed": [], "have_but_cut": [], "genuinely_lacking": [],
                "unclassified": [], "explanation": "no unaddressed requirements"}

    facts = [f for f in ((profile or {}).get("facts") or []) if f.get("verified")]
    if not facts:
        return {
            "unaddressed": unaddressed,
            "have_but_cut": [], "genuinely_lacking": [],
            "unclassified": unaddressed,
            "explanation": f"{len(unaddressed)} unaddressed requirements; "
                           f"cut-vs-lacking {NOT_MEASURED} (no verified facts "
                           "available to match against)",
        }

    from jobbuddy import hr_panel

    split = hr_panel.attribute_gaps(unaddressed, facts)
    return {
        "unaddressed": unaddressed,
        "have_but_cut": split.get("have_but_cut") or [],
        "genuinely_lacking": split.get("genuinely_lacking") or [],
        "unclassified": [],
        "explanation": split.get("explanation") or "",
    }


def silent_losses(tailored: dict[str, Any] | None,
                  resume_render: dict[str, Any] | None) -> dict[str, Any]:
    """What the guard rejected and what the page fit cut.

    Both are decisions made on the candidate's behalf by a machine, and neither
    is visible in the document that results. A cut nobody is told about is how a
    resume loses its best bullet.
    """
    # `guard_measured` distinguishes "the guard ran and rejected nothing" from
    # "the guard never ran". Both used to render as a flat `0`, so a run where
    # the tailoring stage failed before the guard printed "Bullets rejected by
    # fact_guard: 0" -- an affirmative claim that nothing was lost, made on the
    # basis of no measurement at all. That is the one thing this module's
    # docstring forbids, in the section it calls the one worth the page.
    guard_raw = (tailored or {}).get("guard")
    guard = guard_raw or {}
    dropped = list((resume_render or {}).get("dropped") or [])
    return {
        "guard_measured": isinstance(guard_raw, dict) and bool(guard_raw),
        "guard_rejected": int(guard.get("rejected") or 0),
        "guard_fell_back": int(guard.get("fell_back") or 0),
        "guard_by_kind": dict(guard.get("by_kind") or {}),
        "guard_examples": list(guard.get("examples") or []),
        "cut_to_fit": dropped,
        "cut_for_length": int((tailored or {}).get("dropped_for_length") or 0),
        "unknown_fact_ids": list((tailored or {}).get("unknown_fact_ids") or []),
    }


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_model(job: dict[str, Any],
                tailored: dict[str, Any] | None = None,
                profile: dict[str, Any] | None = None,
                resume_render: dict[str, Any] | None = None,
                population: list[Any] | None = None) -> dict[str, Any]:
    """Everything the report prints, resolved once.

    Separate from rendering so every "not measured" decision is assertable
    without compiling a PDF, and so the same model can feed a second output
    format later without a second set of those decisions.
    """
    job = job or {}
    scores = job.get("scores") or {}
    return {
        "title": _value(job.get("title")),
        "company": _value(job.get("company")),
        "location": _value(job.get("location")),
        "source": _value(job.get("source")),
        "seniority": _value(job.get("seniority")),
        "employment_types": list(job.get("employment_types") or []),
        "adjusted": scores.get("adjusted"),
        "total": scores.get("total"),
        "confidence": scores.get("confidence"),
        "explanation": str(scores.get("explanation") or NOT_MEASURED),
        "apply": apply_link(job),
        "competition": competition_line(job),
        "salary": salary_line(job),
        "market_time": market_time(job),
        "strengths": strengths(job),
        "gaps": gap_analysis(tailored, profile),
        "company_signals": company_signals(job),
        "career": career_prospects(job),
        "losses": silent_losses(tailored, resume_render),
        "charts": render_charts.charts_for_job(job, population),
    }


# --------------------------------------------------------------------------
# Typst source -- pure, so it exists whether or not the compiler does
# --------------------------------------------------------------------------

def _bullets(items: list[str]) -> list[str]:
    return [f"- {_escape(i)}" for i in items] or [f"- {NOT_MEASURED}"]


def _kv(label: str, value: Any) -> str:
    """One labelled line. The trailing `\\` is a forced Typst linebreak -- without
    it consecutive short lines collapse into one run-on paragraph."""
    return f"*{_escape(label)}:* {_escape(value)} \\"


def build_typst_source(model: dict[str, Any]) -> str:
    """The `.typ` analysis document. No charts embedded -- see `render()`.

    Charts are written beside the PDF as standalone `.svg` rather than inlined,
    because Typst's SVG support varies by build and a chart that fails to
    compile must not take the analysis down with it. The report is text that
    stands alone; the charts illustrate it.
    """
    lines = [
        f"#set document(title: {_quote(str(model.get('title')) + ' analysis')})",
        '#set page(paper: "a4", margin: 0.8in)',
        "#set text(size: 10pt, hyphenate: false)",
        "#set par(leading: 0.65em, justify: false)",
        '#show heading: it => block(above: 1.1em, below: 0.5em)'
        '[#text(size: 12pt, weight: "bold", upper(it.body))]',
        "",
        f"#align(center)[#text(size: 16pt, weight: \"bold\")"
        f"[{_escape(model.get('title'))}]\\\n{_escape(model.get('company'))}]",
        "",
        "= Role",
        "",
    ]
    lines += [
        _kv("Company", model.get("company")),
        _kv("Location", model.get("location")),
        _kv("Seniority", model.get("seniority")),
        _kv("Source", model.get("source")),
        _kv("Employment", ", ".join(model.get("employment_types") or [])
            or NOT_MEASURED),
        _kv("Adjusted score", _value(model.get("adjusted"))),
        _kv("Confidence", _value(model.get("confidence"))),
        "",
        _escape(model.get("explanation")),
        "",
        "= Apply",
        "",
        _kv("Link", model["apply"]["url"]),
        _kv("Status", model["apply"]["status"]),
        "",
        "= Competition",
        "",
    ]
    competition = model["competition"]
    lines += [
        _kv("Applications", competition["applications"]),
        _kv("Per vacancy", competition["per_vacancy"]),
        _kv("Vacancies", competition["vacancies"]),
        _kv("Views", competition["views"]),
        _kv("Applications per day", competition["apps_per_day"]),
        _kv("Basis", competition["basis"]),
        "",
        "= Pay",
        "",
        _kv("Stated range", model["salary"]),
        "",
        "= Time on market",
        "",
    ]
    market = model["market_time"]
    lines += [
        _kv("Posted", market["posted_at"]),
        _kv("Age", market["age_days"]),
        _kv("Closes", market["expires_at"]),
        _kv("Days left", market["days_left"]),
        _kv("Reposts", market["reposts"]),
        "",
        "= Strengths for this role",
        "",
    ]
    strength = model["strengths"]
    if not strength.get("measured"):
        lines.append(f"{NOT_MEASURED}: {_escape(strength.get('reason'))}")
    else:
        lines.append(_kv("Skills matched",
                         f"{strength.get('matched_count')} of "
                         f"{strength.get('total_count')}"))
        lines.append(_kv("Core requirement score",
                         _value(strength.get("core_score"))))
        lines.append("")
        lines += _bullets([str(m) for m in strength.get("matched") or []])
    lines += ["", "= Gaps", ""]

    gaps = model["gaps"]
    if not gaps["unaddressed"]:
        lines.append("No stated requirement went unaddressed.")
    else:
        lines.append(_escape(gaps["explanation"]))
        lines += ["", "== You have this -- it was cut, not missing", ""]
        cut = [f"{g.get('requirement')} (covered by {g.get('fact_id')}, "
               f"matched {g.get('matched')})" for g in gaps["have_but_cut"]]
        lines += _bullets(cut) if cut else ["- none"]
        lines += ["", "== You genuinely lack this", ""]
        lines += (_bullets(list(gaps["genuinely_lacking"]))
                  if gaps["genuinely_lacking"] else ["- none"])
        if gaps["unclassified"]:
            lines += ["", "== Unclassified", "",
                      f"Cut-vs-lacking {NOT_MEASURED} for these:", ""]
            lines += _bullets(list(gaps["unclassified"]))

    lines += ["", "= What was dropped on your behalf", ""]
    losses = model["losses"]
    # Print the counts only when the guard actually ran. Printing 0 for a guard
    # that never executed tells the reader nothing was lost, which is a claim
    # about evidence that does not exist.
    if losses.get("guard_measured"):
        guard_rejected = losses["guard_rejected"]
        guard_fell_back = losses["guard_fell_back"]
    else:
        guard_rejected = guard_fell_back = NOT_MEASURED
    lines += [
        _kv("Bullets rejected by fact_guard", guard_rejected),
        _kv("Bullets that fell back to approved phrasing", guard_fell_back),
        _kv("Bullets cut to fit the page", len(losses["cut_to_fit"])),
        "",
    ]
    if losses["cut_to_fit"]:
        lines += _bullets([f"{d.get('text')} -- {d.get('reason')}"
                           for d in losses["cut_to_fit"]])
    if losses["guard_examples"]:
        lines += ["", "Rejected examples:", ""]
        lines += _bullets([f"{e.get('bullet')} -- "
                           f"{'; '.join(e.get('reasons') or [])}"
                           for e in losses["guard_examples"]])

    lines += ["", "= Company signals", ""]
    company = model["company_signals"]
    lines += [_kv("Hiring posture", company["status"])]
    if company.get("reason"):
        lines.append(_kv("Why", company["reason"]))
    lines.append(_kv("Open requisitions", company.get("open_reqs")))

    lines += ["", "= Career prospects", ""]
    career = model["career"]
    lines += [
        _kv("Direction", career["direction"]),
        _kv("Job level", career.get("job_level")),
        _kv("Experience asked", career.get("years")),
    ]
    if career.get("reason"):
        lines.append(_kv("Why", career["reason"]))

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def write_charts(model: dict[str, Any], out_dir: Path,
                 stem: str = "report") -> list[Path]:
    """Write each chart as a standalone `.svg`. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, svg in (model.get("charts") or {}).items():
        path = out_dir / f"{stem}.{name}.svg"
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    return written


def render(model: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Write the analysis PDF, or the `.typ` source when Typst is missing.

    Same degradation contract as `render_resume.render_pdf`: never raises for a
    missing wheel, always names which degradation was taken.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = build_typst_source(model)
    charts = write_charts(model, out_path.parent, out_path.stem)

    if not render_resume.capabilities()["pdf"]:
        render_resume._warn_once(
            "pdf", "typst is not installed -- emitting .typ source instead of "
                   "a PDF: py -m pip install -e .[tailoring]")
        typ_path = out_path.with_suffix(".typ")
        typ_path.write_text(source, encoding="utf-8")
        return {"ok": True, "degraded": "typ-source", "path": typ_path,
                "charts": charts, "source": source,
                "note": "typst missing; compile the emitted .typ with: "
                        f"typst compile {typ_path.name} {out_path.name}"}

    pdf = render_resume._compile(source)
    out_path.write_bytes(pdf)
    return {"ok": True, "degraded": None, "path": out_path, "charts": charts,
            "source": source, "note": "", "bytes": len(pdf)}
