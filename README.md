# job-buddy

Foundation for the Job Buddy project: a DeepSeek integration that keeps its own
model choice and token budgets up to date instead of hardcoding them.

## Quick start

    py test_deepseek.py          # verify the API key works end to end
    py update_models.py          # refresh models.json from the API + docs
    py calibrate_budgets.py --report

> **Use `py`, not `python`.** This machine has two Pythons — 3.13 under
> `Programs\Python\Python313` (where `pip` installs go) and a Microsoft Store
> 3.11 under `WindowsApps`. None of the scripts here carry a `#!` line, and that
> is deliberate: `py.exe` *reads* shebangs and would silently route
> `#!/usr/bin/env python3` to the Store 3.11, which has no packages installed.

## Secrets

The API key lives in a local **`.env`** file that is **git-ignored** and never
committed. `.env.example` is the committed template.

    copy .env.example .env
    # then edit .env and paste your real key

Get a key from the DeepSeek console: https://platform.deepseek.com/

## Model selection — never hardcode a model id

Application code asks for a *capability tier*, not a model name:

```python
import model_config
model = model_config.resolve("fast")     # or "quality"
```

`models.json` maps tiers to concrete ids. When DeepSeek renames or retires a
model, only that file changes — your code does not.

| Tier | Today | Why |
|---|---|---|
| `fast` | `deepseek-v4-flash` | lowest input price in the current generation |
| `quality` | `deepseek-v4-pro` | highest input price in the current generation |

Price is used as the capability proxy: within one vendor generation the price
ladder *is* the vendor's own capability ordering, and it is the only signal that
is published, numeric and self-updating.

`py update_models.py` refreshes that file from three sources, in trust order:

1. `GET /models` — authoritative for availability (ids only).
2. the public pricing page — best-effort scrape.
3. the public changelog — best-effort scrape for deprecation notices.

Scraping is *enrichment*, never authority. A failed or nonsensical scrape marks
data stale and keeps the last known-good values rather than changing which model
you call. Specific guards:

- **5× anomaly gate** — a scraped price wildly different from the stored one is
  treated as parser drift, not a price change.
- **Sticky deprecation** — a notice can set `deprecated`, but a failed scrape can
  never clear it. Silently calling a dead model is the worse failure.
- **Stale availability halts selection** — if `/models` does not answer, tiers are
  not re-resolved at all.
- **New generations are quarantined** — an unrecognised id lands in `pending` and
  must pass six gates (parses, known family, known price, 7-day soak, 3 sightings,
  sane price) *and* an explicit `--accept-new-generation`.

`model_config.resolve()` never raises. Its fallback chain is
`$DEEPSEEK_MODEL` → tier pin → configured model → re-derived → hardcoded constant,
so a missing or corrupt `models.json` cannot take the app down.

Exit codes: `0` clean · `1` a source failed · `2` action needed.

## Token budgeting

```python
import deepseek_client
result = deepseek_client.complete("Is this a job ad? yes/no", profile="classify")
print(result["text"])
```

**The dominant lever is `thinking`, not `max_tokens`.** On v4 models thinking is
*enabled by default*, reasoning tokens are billed inside `completion_tokens`, and
they are produced *before* the answer — so an undersized budget returns an empty
reply with `finish_reason: "length"`. Measured on the same prompt:

| Setting | completion tokens | cost |
|---|---|---|
| thinking enabled | 35 (30 of them reasoning) | $0.0000115 |
| thinking disabled | 4 | $0.0000028 |

`task_profiles.json` sets both knobs per task type:

| Profile | thinking | max_tokens |
|---|---|---|
| `classify` | disabled | 64 |
| `extract` | disabled | 512 |
| `summarize` | disabled | 1024 |
| `analyze` | enabled / high | 4096 |
| `deep` | enabled / max | 16384 |

If a reply is truncated, the client **retries once at double the budget**, which
makes any estimation error self-correcting.

### Token counting

`token_budget.estimate_tokens()` uses DeepSeek's official tokenizer when present
and falls back to the documented char-ratio heuristic (0.3 tok/char EN,
0.6 CJK) otherwise, so it always works with zero dependencies.

    py fetch_tokenizer.py            # downloads tokenizer/tokenizer.json
    pip install -r requirements.txt  # the lightweight `tokenizers` loader

Every call logs its predicted token count next to the API's real one, so the
estimator's accuracy is a **measured number, not an assumption** — which matters
because the only tokenizer DeepSeek publishes is the *v3* one while we call *v4*
models. Current measurement:

    heuristic  mean error 23.3%   worst 30.8%
    official   mean error  0.0%   worst  0.0%

Check it yourself any time with `py calibrate_budgets.py --report`.

### Calibration

`usage_log.jsonl` (git-ignored) records every call. `py calibrate_budgets.py`
retunes each profile's `max_tokens` to p95 × 1.25 of observed usage, but only
once a profile has ≥30 samples — below that, p95 is noise. Until real traffic
accumulates the seeded defaults stand, which is the intended behaviour.

## Automated docs watch

`.github/workflows/watch-deepseek-docs.yml` runs monthly and opens an issue when
DeepSeek's public docs change. It needs **no API key**: pricing and deprecation
notices are public, and only `/models` requires auth, which that job skips.

It is a **sensor, not a mutator** — it never writes `models.json`. Without
`/models` it cannot verify availability, so writing partial config would be
unsafe; the authoritative refresh stays a local `py update_models.py` run.

Monthly rather than weekly because DeepSeek changed models ~5 times in 11 months
and gives ~90 days' deprecation notice — weekly polling would be >90% no-op runs,
and no-op automation is how the one alert that matters gets ignored.

Note GitHub disables scheduled workflows after 60 days of repo inactivity, so the
primary safety net is local: `model_config` warns whenever `models.json` is more
than 30 days stale.

## Files

| File | Role |
|---|---|
| `deepseek_common.py` | shared `.env` loading + HTTP plumbing |
| `model_config.py` | tier → model resolver, plus the selection heuristic |
| `update_models.py` | refreshes `models.json` (the only writer) |
| `models.json` | model facts, pricing, deprecations, tier mapping |
| `token_budget.py` | token estimation + profile → budget |
| `task_profiles.json` | task profiles (written only by `calibrate_budgets.py`) |
| `deepseek_client.py` | the call path: resolve → call → log → retry |
| `calibrate_budgets.py` | retunes profiles from logged usage |
| `fetch_tokenizer.py` | downloads the optional official tokenizer |
| `test_deepseek.py` | end-to-end key + inference check |

Each config file has exactly one writer, so two scripts can never clobber the
same file.
