# Handover

State at handover: **branch `feat/resume-tailor`, pushed. 608 tests pass.
Search works. Tailoring works on one job at a time and fails on most jobs under
concurrency.**

Read [FINDINGS.md](FINDINGS.md) before changing anything about how tactics are
evaluated — it records what was measured and, more importantly, what those
measurements do not support.

---

## Start here: the one open bug

The tailoring stage fails on roughly three quarters of jobs when several run at
once.

    FAILED_AT_tailor: AttributeError: 'NoneType' object has no attribute 'get'

**What is known:**

| Observation | Detail |
|---|---|
| Fails under concurrency | 3 jobs -> 3 ok · 2 jobs -> 1 failed · 4 jobs -> 3 failed |
| Succeeds in isolation | Calling `tailor.tailor()` directly on a job that just failed works every time, repeatedly |
| Degrades safely | Failure isolation holds: the run continues, each failure is recorded, the workbook is still written |
| Possibly related | Every run prints `[model_config] config unavailable` at startup |

The pattern — clean in isolation, worse with more parallelism — points at shared
mutable state on the threaded path. `deepseek_client` and `model_config` are the
first places to look. `_prepare_job` runs in a `ThreadPoolExecutor` and is
documented as touching no shared state, so either that documentation is wrong or
something underneath it is.

**How to see it:**

    py -m jobbuddy.cli --scope ai-engineer-sg --tailor --tailor-top 4 --max-cost 0.60

The CLI now prints the last six traceback frames under any failed job, so the
first run should name the file and line. That printing was added specifically
because the previous session burned several cycles on an exception message with
no location attached.

**Do not** "fix" this by lowering `TAILOR_WAVE` to 1. That hides it, and the
stage is meant to run over dozens of jobs.

---

## What works, verified live

    py -m jobbuddy.cli --scope ai-engineer-sg --tailor --tailor-top 3 --max-cost 0.50

Produces, in ONE directory:

    potential applications/<scope>/<run_id>/
        ranked.csv  ranked.json  ranked.xlsx
        <job>/  resume.pdf  resume.docx  report.pdf  + 4 SVG charts

Measured on real runs: **328 jobs ranked** from 5 sources; resume PDF **25 KB,
one page at 11pt** against Taleo's 100 KB limit; **$0.006 per job**.

The profile import path also works: 16 facts extracted from the real resume, 10
auto-verified by literal span matching, 6 flagged with specific reasons.

---

## The pipeline, and what guards it

    import_resume  ->  verify_profile  ->  tailor  ->  render_resume  ->  report/excel
                            |                 |            |
                       span matching     fact_guard   resume_rules

- **`fact_guard`** — every bullet cites a `fact_id`; numbers, entities and
  durations must trace to it. Rejected bullets fall back to the fact's approved
  phrasing. The pipeline may produce a blander resume, never a false one.
- **`verify_profile`** — proves a fact was COPIED from the resume, not invented.
  It explicitly does not prove the resume is accurate; only you can.
- **`resume_rules`** — deterministic. Personal-data leaks and hidden text are
  ERRORS and block the render entirely. Singapore field suppression (no photo,
  NRIC, DOB, gender, race, religion, marital status) is government guidance, not
  preference, and is the highest-confidence rule in the whole research set.
- **`ab_harness`** — the noise floor is a required argument with no default,
  because an optional guard is skipped exactly on the run where it mattered.

---

## Next, in order

1. **Fix the concurrency bug above.** Everything else is blocked behind running
   the stage at volume.
2. **Expose the flow in `JobBuddy.ipynb`.** It is the user-facing interface and
   currently exposes none of the tailoring work.
3. **Bullet length.** The only rule warning on a real render: bullets wrap to
   ~3 lines against a 2-line cap. The resume's own phrasings are long, so this
   needs either shorter approved phrasings in the profile or a selection
   preference for concise facts.
4. **Re-run the tactic experiment across several jobs.** The current result is
   n=1 job, and the length control truncated arms to 2 bullets, which is a weak
   test. See FINDINGS.md §2 for exactly how weak.
5. **The yield finding.** Tactics changed how many bullets survive selection by
   4x (8 vs 2 from the same facts) with no measurable quality difference. That
   may matter more than the quality question. Not acted on: n=1.

---

## Traps, each of which cost real time

- **Do not commit a file another agent is editing.** `git add <path>` stages
  whatever is on disk at that instant. Commit `8e4ee22` captured the tests and
  not the code because a subagent had reverted the file to measure a baseline at
  that moment. Everything looked right and the suite passed, because the working
  tree held what the commit did not. **Verify the committed blob**, not the
  working tree: `git show HEAD:path | grep <thing>`.
- **Statistical machinery does not protect against measuring the wrong thing.**
  The first experiment reported a tactic winning 0.92 to 0.08 with a bootstrap
  interval nowhere near chance. It was measuring bullet count. The noise floor,
  paired ordering, intervals and power check all passed, because a confound is a
  real difference in the inputs.
- **This repo is PUBLIC.** Use the fictional Alex Tan / Umbra Financial /
  Northwind Labs fixtures. Real CV fragments reached the remote once and had to
  be scrubbed; they remain in history, which would need a force-push to clear.
  `experiments/` is gitignored because graders quote the resume in their
  reasoning.
- **A false warning is worse than a missing rule.** `resume_rules` fired "no
  contact details" on every correct render because it accepted a dict and a
  string but not the list `render_resume` emits. A rule that cries wolf sits
  next to the personal-data check that must never be skimmed past.
- **A guard that rejects true statements gets switched off.** `fact_guard` once
  rejected "APIs" against an entity list saying "API", and rejected the approved
  phrasings too, so verified content vanished while the run reported success.

---

## Things I did not verify

Stated so nobody inherits them as fact:

- **Whether `references/` mattered.** It was untracked at session start and is
  gone. Git never had it, there are no stashes, and no deletion appears in the
  transcript. Most likely removed by the earlier approved cleanup, but that is
  inference.
- **The research provenance.** Reddit was unfetchable throughout, so the
  practitioner consensus in RESEARCH.md is Hacker News only and skews startup.
  The search layer confabulated fluent, correctly-attributed, fabricated content
  three times. Re-open any specific quote before acting on it.
- **The 5-test discrepancy.** A subagent measured a 549-test baseline where I
  measured 544, and could not reconcile it. Possibly a stale environment
  snapshot. Unresolved.
