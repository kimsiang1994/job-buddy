# Next session

State at handover: **12 commits on `feat/resume-tailor`, unpushed. 224 tests
pass. 162 jobs ranked from 5 live sources.**

---

## The honest headline

**The product is half built.** Everything so far is the *search* half — find
jobs, score them, track competition. The half you actually asked for on day one
is untouched:

> a tailored one-page resume per job, plus a per-job analysis PDF with charts,
> in a dated folder tree, plus a ranked Excel workbook

None of that exists. No resume is generated, no PDF is written, no workbook.
The pipeline currently ends at `ranked.csv`.

That is the biggest gap by a wide margin, and it should probably come before
any further source work.

---

## 1. Resume tailoring — the actual deliverable

Build in this order. The order matters: the gate goes in before the thing it
gates, or the gate never gets built.

1. **`fact_guard.py` + adversarial fixtures.** The deterministic validator that
   rejects any generated bullet whose numbers, entities or durations do not
   trace to a verified fact. Build and test this FIRST, against fixtures that
   try to sneak a fabricated percentage past it.
2. **`import_resume.py`** — read `input/Resume_*.pdf` with pypdf, LLM-extract
   to `profile/master_profile.draft.json` with every fact `verified: false`
   and a `source_span` quoting the resume verbatim.
3. **You hand-verify the draft** and promote it to `master_profile.json`. The
   only manual gate in the pipeline, and the thing standing between the tool
   and a resume that lies.
4. **`tailor.py`** — selection, not generation. The LLM chooses and reorders
   pre-approved phrasings; `fact_guard` rejects anything else.
5. **`render_resume.py`** — Typst → PDF and python-docx → DOCX, one data model.
   Hard one-page enforcement via binary search on a scale variable, floored at
   ~9pt, with a deterministic bullet-drop and a record of what was cut.
6. **`render_report.py` + `render_charts.py`** — the per-job analysis PDF.
7. **`render_excel.py`** — xlsxwriter, tabs per scope, autofilter, sorted in
   Python before writing (XLSX stores a sort *state*, not sorted output).

The `deepseek_client` gaps from the original plan are still open and block all
of this: JSON output mode with a repair call, HTTP retry with backoff, and a
`threading.Lock` on `_log_usage`.

---

## 2. Libraries to evaluate (user asked)

- **python-jobspy** — claims LinkedIn/Indeed/Glassdoor/ZipRecruiter with no API
  keys. Verify that claim rather than accept it: "no keys" for LinkedIn almost
  certainly means scraping underneath, which puts you back where the paid
  vendors were the answer. Measure what it actually returns for Singapore.
- **Crawl4AI** — right shape for company-board discovery, which currently does
  two plain fetches and misses JS-rendered careers pages.
- **Firecrawl** — mostly a paid service with an OSS core; small free tier.

---

## 3. Untested adapters — assume broken

Every paid tier that ran for the first time this session had a bug. Three
remain unexercised:

| Adapter | Needs | Status |
|---|---|---|
| Careerjet | free key, careerjet.com/partners/api/ | never run |
| Adzuna | free key, developer.adzuna.com | never run |
| unblocker tier | paid key (ScrapingBee/Zyte/Bright Data) | never fetched a page |

Do not assume any of these work because the code reads correctly. The CLI
not loading `.env` looked exactly like a broken key and had nothing to do with
the key at all.

---

## 4. Source work still open

- **JSON-LD adapter — premise unvalidated.** careers.gov.sg and Michael Page
  carried no `JobPosting` markup on the pages checked. Fetch a real job DETAIL
  page from an agency site before building anything on this.
- **Tech in Asia** — measured serving a plain browser (200, 786KB, strong job
  signal). Adapter never written.
- **Company registry** — 16 companies queued for board discovery, drains in ~2
  runs. It compounds on its own; just let it run.
- **Workday** — only OCBC and DBS resolved. UOB, GIC, Temasek, Standard
  Chartered still unmapped; feed their careers URLs to `discover_board`.

---

## 5. Housekeeping

- **12 commits unpushed.** Decide whether to push `feat/resume-tailor` or merge
  to `main` first.
- `NEXT_SESSION.md` (this file) should be deleted once its contents are done.
- Salary is now set correctly: 8,000/month, floor auto-derived at 7,200.

---

## Things this codebase has learned the hard way

Worth reading before changing any of it — each cost a real bug:

- **A component with no data must not score as average.** Renormalising alone
  let sparse sources outrank rich ones; the rank shrinks toward neutral in
  proportion to how little was measurable.
- **Frequency alone cannot identify filler.** Python appears in 73% of AI job
  ads because it is required, not because it is noise.
- **A false skill match is worse than a missed one.** It inflates fit and could
  later justify a resume bullet claiming something untrue.
- **Feed absence is not closure.** Only a job's own endpoint can say it closed.
- **robots.txt is per-agent.** The `User-agent: *` group governs this tool; a
  rule naming someone else's crawler does not.
- **Untested code is not working code.**
