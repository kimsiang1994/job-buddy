# Measured findings

Things this project established by running an experiment rather than by
repeating advice. Each entry says what was measured, on what, and what it does
NOT support — a finding with its limits stripped off becomes folklore within
two retellings, which is how most resume advice got the way it is.

---

## 1. The evaluator is noisier than any tactic it measures

**Measured here, on this pipeline.** The HR panel scored an *identical* resume
against an *identical* job 20 times:

    mean 47.8   sd 8.85   range 31.7 – 66.7

A 35-point spread produced by nothing at all. The input did not move.

This independently reproduces the published HackerRank result (one unchanged
resume, 100 runs of their own open-source screener at temperature 0.1, scores
spanning 66–99, failing an 85-point cutoff 65% of the time).

**Consequences, and these are not optional:**

- Any single-run score is uninterpretable. A resume "scoring 82" means nothing.
- Any tactic claiming a 5-point improvement is claiming something smaller than
  the measurement error.
- Optimising a resume toward a score threshold is optimising toward a coin flip.
- Comparisons must be *paired forced choice*, repeated, with a bootstrap
  interval and a noise floor underneath. `ab_harness` enforces this; the floor
  is a required argument precisely so it cannot be skipped on the run where it
  matters.

**What it does not support:** that the graders are worthless. Their *relative*
judgement over many paired trials still carries signal. It is the absolute
score that is noise.

---

## 2. XYZ formula: no detectable effect on quality

**Arms:** `baseline`, `xyz_formula`, `technical_decision`, one real Singapore
ML job, 60 trials, 360 paired comparisons per arm, length held fixed.

    baseline             win 0.557   CI [0.539, 0.576]
    xyz_formula          win 0.418   CI [0.393, 0.442]
    technical_decision   win 0.525   CI [0.504, 0.546]

**Verdict: no detectable difference.** No arm's win rate clears both its
bootstrap interval and the noise floor.

This is the most universally repeated piece of resume advice — "Accomplished X
as measured by Y by doing Z". Bock titled it *"My Personal Formula"*, cited no
study, and supported it with hand-picked examples. Tested properly, it does not
move the needle.

**What it does not support:** that structure is worthless, or that XYZ is
*harmful*. The intervals overlap the floor; that is an absence of evidence at
this power, on one job.

**The real limitation, stated plainly:** the length control truncates every arm
to the *shortest*, which here was 2 bullets. Comparing 2-bullet resumes is a
weak test — there is little room for a tactic to show anything. Read this as
"no effect detectable under these conditions", not "proven equivalent". A
better design truncates to a fixed count and drops arms that cannot reach it.

---

## 3. The first version of that experiment produced a fake result

Worth recording because it was convincing.

    xyz_formula  win 0.92   baseline  win 0.08   verdict: significant

Emphatic, bootstrap interval nowhere near chance, and **meaningless**. That arm
had emitted 8 bullets against baseline's 3. The graders were shown a fuller
resume and a thinner one and asked which was stronger. Nothing about XYZ
structure was being measured.

The harness had a noise floor, paired ordering, bootstrap intervals and a
power check — and none of them catch a confound, because a confound is a real
difference in the inputs. **Statistical machinery does not protect against
measuring the wrong thing.** Length is now held fixed by default.

---

## 4. Tactics change yield fourfold, which is not what they claim

From the same verified facts, on the same job:

    xyz_formula          2 bullets
    baseline             3 bullets
    technical_decision   8 bullets

Quality showed no measurable difference; *how much material survives selection*
differs by 4x. That is a real and reproducible effect, and it is not the effect
any of these tactics advertises.

It plausibly matters more than the quality question: more surviving verified
bullets means more for the renderer to rank and cut against a page limit.

**Not yet acted on.** This is n=1 job. Changing a default on one observation is
how the folklore this project exists to resist gets made. Run it across several
jobs first.

---

## 5. Evidence corrections to earlier assumptions

These were in the original plan and are wrong:

| Claim | Reality |
|---|---|
| "75% of resumes are auto-rejected by ATS" | Traces to Preptel, a vendor defunct since 2013, no study ever published. Circulates as 70/75/88 — a real statistic has one value |
| The HBS/Accenture "88%" | An *opinion survey question* about employer-configured filter criteria, not resume parsing. Miscited industry-wide |
| Parse failure means rejection | Greenhouse's own docs: a resume that fails to parse **still creates a candidate record**. Real auto-rejection is recruiter-configured knockout questions on the form |
| One page is correct | Institutional backing is *undergraduate* material — Harvard's guide is titled "Undergraduate Resource Series"; CMU's says "Students with less than 10 years". Five volume screeners say two pages is fine at 5+ years |
| Shorter is safer | Wilson & Caliskan (AIES 2024) measured that **shortening resumes increased biased outcomes by 22.2%** |

**The durable strategic finding:** Cui et al. (5.5M cover letters) found
employers did not detect AI-written applications — they *repriced* them.
Tailoring's correlation with callbacks fell 51% while verifiable-history
signals rose. As everyone automates tailoring, prose stops carrying
information and only hard-to-fake signal does: shipped systems, named
production impact, public artifacts.

---

## Research provenance and its limits

The evidence review behind §5 carries real caveats and they should travel with it:

- **Reddit was unfetchable throughout.** The r/EngineeringResumes wiki came via
  its GitHub mirror; no individual threads were read by anyone. So there is no
  r/recruiting content and no agency-recruiter voice — the practitioner
  consensus is Hacker News only, which skews startup and away from enterprise.
- **The search layer confabulated three times**, returning fluent,
  correctly-attributed, entirely fabricated content — including a plausible
  false version of the wiki's position and a tidy table whose citations
  resolved to unrelated discussions. These were caught and discarded, but the
  base rate is not zero.
- Several practitioner quotes were never re-opened from their primary URL.
  Re-check before acting on any specific one.
