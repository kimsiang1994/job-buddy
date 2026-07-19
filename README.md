# job-buddy

Finds Singapore jobs, scores them against your resume, and tracks how the
competition builds over time. The search half runs for free — no API key, no
LLM, stdlib only.

## Start here

```
pip install -e ".[notebook]"
```

Then open **`JobBuddy.ipynb`** and run the cells top to bottom. Upload a resume,
say what you're looking for, and it searches.

That notebook is the whole interface. Everything else is implementation.

```
JobBuddy.ipynb        the interface
run_config.json       the one file you edit by hand
config/               machine-managed data (models.json, task_profiles.json)
src/jobbuddy/         the code
  deepseek/           the LLM layer -- unused by search, needed for tailoring
tests/                offline suites: no network, no key, no cost
```

Command line, if you prefer:

```
jobbuddy --scope ai-engineer-sg          # after pip install -e .
jobbuddy --all --limit 40
jobbuddy --dry-run                       # score and print, write nothing
jobbuddy --explain mcf:59501ac0...       # one job's full history
```

> **Use `py`, not `python`.** This machine has two Pythons — 3.13 under
> `Programs\Python\Python313` (where `pip` installs go) and a Microsoft Store
> 3.11 under `WindowsApps`. No script here carries a `#!` line, deliberately:
> `py.exe` *reads* shebangs and would silently route to the Store 3.11.

## Where the jobs come from

**MyCareersFuture** (`api.mycareersfuture.gov.sg/v2/jobs`) — keyless, and
`robots.txt` disallows nothing. Every employer hiring in Singapore must post
there under the Fair Consideration Framework, so it reaches TikTok, Shopee and
Grab, none of which expose a public careers API.

It publishes three things almost nothing else does:

| Field | Why it matters |
|---|---|
| `salary` | Mandatory and structured, by law |
| `totalNumberJobApplication` | **The real count of submitted applications** |
| `ssocCode`, `uen` | Join keys to MOM wage tables and the ACRA registry |

That application count is the competition signal. It is better than LinkedIn's
"X applicants", which counts click-throughs rather than applications — and
which is scraping-only, so obtaining it would put your own account at risk.

## How a job is scored

Seven components, each 0–100, weights in `run_config.json`. **A component with
no data returns nothing and its weight leaves the denominator** — imputing an
average would claim knowledge we don't have.

| Component | Weight | Reads |
|---|---|---|
| `skill_match` | 30 | your skills vs the job's, via the alias taxonomy |
| `competition` | 20 | applications per vacancy, arrival rate, reposts |
| `seniority_fit` | 15 | the level you **want**, not the one you hold |
| `comp_signal` | 15 | stated range vs your reference point |
| `company_signal` | 10 | open reqs and hiring velocity |
| `application_friction` | 5 | direct employer vs agency |
| `freshness` | 5 | age, with a 21-day half-life |

Every score carries its inputs and a sentence explaining itself, and the CSV has
a "why" column. A ranking you can't audit is one you can't trust.

**You are matched against the level you want.** An `Ambition` setting (same
level / one up / stretch) drives `target_seniority`. Defaulting it to your
current level made the scorer rank staying-put roles top, which is the opposite
of why anyone runs a job search.

## What it records, and what never leaves

| Path | Contents |
|---|---|
| `intake/` | your profile, submission log, archived resumes |
| `state/sightings.jsonl` | every job seen, every run — the history behind "new since last time" |
| `potential applications/` | ranked output per run |

All gitignored. This repo is public and those hold a real name, phone number and
salary expectations.

**Salary never crosses the process boundary.** `user_input.redact_for_llm()` is
an allowlist, so a confidential field added later is excluded by default rather
than leaking until someone remembers a blocklist — and prompts are retained by
providers, which makes that kind of leak permanent. Salary isn't in the
allowlist at all: scoring turns it into a ratio locally, and a ratio is not a
salary.

## Tests

```
py -m unittest discover -s tests -t .     # 163 tests, offline, free
py -m tests.test_deepseek                  # live smoke test, needs a key
```

CI runs the offline suites on every push. They previously ran nowhere — only the
scraper tests ran, inside a monthly cron — which is how 86 tests came to exist
without ever having been executed by CI.

Every fixture is either real captured data or the exact shape that broke
something once. **Add to them rather than rewriting them**; each one is a
failure mode someone already paid for.

## Model selection — never hardcode a model id

```python
from jobbuddy.deepseek import model_config
model = model_config.resolve("fast")     # or "quality"
```

`config/models.json` maps tiers to concrete ids, refreshed by
`py -m jobbuddy.deepseek.update_models`. Scraping is *enrichment, never
authority*: a failed scrape keeps the last known-good values rather than
changing which model you call. `resolve()` never raises — its fallback chain
ends at a hardcoded constant, so a corrupt config cannot take the app down.

**Note:** `deepseek-chat` and `deepseek-reasoner` were deprecated 2026-07-24.
This repo resolves to `deepseek-v4-flash` / `deepseek-v4-pro`.

## Design rules that earn their keep

- **One writer per file.** Two writers eventually clobber each other.
- **The read path cannot raise.** A tool that dies because its config is
  malformed is worse than one that warns and falls back.
- **Enrichment is never authority.** A failed scrape marks data stale; it never
  overwrites good data with degraded data.
- **Feed absence is not closure.** A job vanishing from search results means the
  results shifted. Only its own endpoint can say it's closed.
- **A false skill match is worse than a missed one.** It inflates fit, and
  downstream it could justify a resume bullet claiming something you can't back
  up.

## Not built yet

Resume tailoring. It comes after `fact_guard.py` — the deterministic validator
that rejects any generated bullet containing a number, entity or duration that
doesn't trace to a verified fact. The gate gets built before the thing it gates.
