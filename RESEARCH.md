# Resume-Tailoring Tactics: Graded Evidence Review

**Purpose:** input to an automated resume-tailoring pipeline that will A/B test tactics for senior AI/ML engineering roles.

**Coverage:** All nine questions researched. Q8 (Singapore) and Q9 (AI detection) landed late and are fully incorporated; §1's verb-list rationale and both ranked lists were revised as a result.

**Attribution convention.** This report merges my own direct fetches with two delegated research streams. Every claim is tagged:
- **[verified]** — I fetched the primary source myself and read it
- **[agent]** — reported by a research subagent with a URL, **not independently re-fetched by me**

This distinction matters more than usual here: one subagent reported that the search layer in this environment **returned fluent, correctly-attributed, entirely fabricated content on at least three occasions**, including a plausible-but-false version of the r/EngineeringResumes wiki position. That agent discarded those and re-verified by direct fetch. I cannot fully audit that. Treat **[agent]** claims as good leads, not settled facts.

**Grading scale**
- **A — measured**: study, controlled experiment, or a platform publishing its own data
- **B — practitioner consensus**: multiple independent people who actually screen, agreeing
- **C — folklore**: widely repeated, no traceable primary source

---

## 0. THE FINDING THAT GOVERNS YOUR TEST DESIGN

Read this before designing any A/B test. It determines whether your measurements can detect anything at all.

### Establish an evaluator noise floor before attributing any score change to a tactic

**Grade:** A (single well-documented experiment, n=150 runs, method fully described; not independently replicated) **[verified]**

**Source:** Dan Kinsky, "HackerRank open sourced its ATS. My resume scored 90/100. Oh wait 74/100. No — 88/100. Actually 83/100." — https://danunparsed.com/p/hackerrank-open-source-ats
Tool under test: HackerRank's own open-sourced screener, https://github.com/interviewstreet/hiring-agent (MIT licence, runs offline via Ollama, or hosted Gemini).

**Full provenance:**

| Parameter | Value |
|---|---|
| Tool | `interviewstreet/hiring-agent` (HackerRank, MIT licence) |
| Input | One unchanged resume PDF |
| Model 1 | `gemma3:4b` (local, via Ollama), **temperature 0.1** |
| Runs, model 1 | **100** |
| Total-score spread | **66–99** (of 100 + 20 bonus), clustering 78–88 |
| Consequence | At an 85-point cutoff the identical resume **fails 65% of the time** |
| Model 2 | `gemini-3.1-flash-lite` (hosted) |
| Runs, model 2 | **50** |
| Total-score spread | **48–64** |
| Consequence | At a 60-point cutoff, fails **28% of the time** |

**Spread was reported both on total score and per-dimension.** The per-dimension breakdown is the more useful result:

| Dimension | Weight | Behaviour across 100 runs |
|---|---|---|
| Open source | 35 pts | not individually reported |
| Personal projects | 30 pts | **"HUGE variation"** — dominant noise source |
| Work experience | 25 pts | **25/25 on every run** — pathologically stable |
| Technical skills | 10 pts | **8/10 in 98 of 100 runs** — stable |
| Bonus | up to 20 | not individually reported |

Kinsky's diagnosis: subjective categories (project quality) lack rubric anchors and wander; oversimplified rubrics (experience) produce *false* consistency — 25/25 every time is not reliability, it is the evaluator failing to discriminate. He also notes the rubric places 65% of weight on GitHub/open-source signal, structurally penalising engineers whose best work is private — **directly relevant to senior AI/ML candidates from industry labs.**

**Implementable as:**
1. Run the **unmodified baseline** through the evaluator N≥30 times; compute mean and SD per dimension. That is your noise floor.
2. Treat any variant-vs-baseline delta under ~2 SD as **no effect**, however large it looks on one run.
3. Never optimise toward a threshold. Optimise the distribution mean.
4. Weight trust per dimension: changes to "projects" text measure mostly noise; changes moving "technical skills" or "experience" are either real or the rubric is blind to them.
5. Log model name, version and temperature with every score. **Temperature 0.1 was not sufficient for stability** — low temperature ≠ deterministic.

**Measurable by:** coefficient of variation on repeated identical runs. If CV on total score exceeds ~10%, single-shot scoring is not a valid instrument and you need many-run averaging or a deterministic metric (parse-field extraction success).

**Conflicts with:** every tactic proposing an ATS-score success metric. Doesn't invalidate them; sets the minimum sample size at which they become testable.

**Caveat on record:** one person, one tool, unreplicated. It does not prove *commercial* ATS scorers are equally noisy — it proves one shipped, open-sourced, LLM-based screener is. Given HackerRank published it as a reference implementation I treat it as representative of the current design pattern, but **that is my inference, not a measured claim.**

---

## 1. Bullet construction

### Do not treat XYZ as proven — the competition is three untested formulas

**Grade:** **C** for "XYZ outperforms alternatives"; **B** for "structure bullets deliberately with outcome + method" **[verified]**

**Source:** Laszlo Bock, "My Personal Formula for a Better Résumé", LinkedIn Pulse, 29 Sept 2014 — https://www.linkedin.com/pulse/20140929001534-24454816-my-personal-formula-for-a-better-resume

I fetched the original and checked specifically for an evidence base. **There is none.** No data, no study, no A/B test, no internal Google hiring numbers. Support is: personal anecdote, before/after examples Bock wrote himself, and his authority as ex-SVP People Ops at Google. The title says "**My Personal** Formula" — he is not claiming it was measured.

Independently corroborated by the recruiter research stream **[agent]**: Recruiter In Your Pocket (https://www.recruiterinyourpocket.com/research/quantifying-impact) traces the formula to the same 2014 post and concedes it "makes assertions without direct comparative evidence."

**I initially graded this B on Bock's credibility. On reflection the subagent's C is better reasoned and I've adopted it** — the claim in circulation is "XYZ *works better*", and that claim has never been tested. Bock's credibility supports "this is a sensible heuristic", not "this beats alternatives."

**The critical reframe [agent]:** MIT's PAR and Stanford's C-A-R have **identical** evidentiary status. Nobody has ever compared XYZ against PAR, CAR, or plain prose. You are not choosing between a proven method and unproven ones — you are choosing among three untested conventions. **That makes bullet structure an unusually good A/B candidate, because the field genuinely does not know.**

**Implementable as:** template with three slots — outcome, quantified measure, method — plus a validator flagging missing slots. Run XYZ, CAR and method-first as **three arms**.

**Measurable by:** proportion of bullets containing (a) leading past-tense action verb, (b) ≥1 numeric token, (c) a by/using/through method clause. Then A/B the arms with §0 noise discipline.

**Conflicts with:** the metrics-density findings below, and with Chip Huyen's ML-specific position.

---

### For AI/ML roles, require a technical decision — not just a metric

**Grade:** B (single credible domain practitioner; snippets only, primary not opened) **[partially verified — see gaps]**

**Source:** Chip Huyen — "What we look for in a resume" (LinkedIn, Jan 2023, https://www.linkedin.com/posts/chiphuyen_what-we-look-for-in-a-resume-activity-7023868353047928832-31qT) and *Introduction to Machine Learning Interviews* §2.3.1.1 — https://huyenchip.com/ml-interviews-book/contents/2.3.1.1-background-and-resume.html

Her position per search snippets: metrics without context are meaningless; wants **less "what", more "how"** — hardest challenge faced and what was learned; ML resumes are highly homogeneous so differentiation matters; "4,500 hours on Python" is negative signal; depth in one language beats breadth; SQL underrated; interviewers "can easily spot talkers from doers."

**Implementable as:** for AI/ML roles require each bullet to name a *technical decision or constraint* (latency budget, data scarcity, drift, serving cost), not only an outcome number. Flag bullets whose only distinguishing content is a percentage.

**Measurable by:** proportion of bullets containing a domain-specific noun phrase alongside the metric. Downstream: interview-conversion, not screener score — a metric-stuffed bullet may score *well* on an LLM screener while failing a human ML hiring manager.

**Conflicts with:** XYZ-as-usually-applied and Harvard-style "quantify everything." **Genuine, unresolvable disagreement:** Bock optimises for legibility to a generalist screener, Huyen for credibility to a domain expert. Which is right depends on who reads first, which differs by company. **Test as separate arms; do not blend.**

---

### Metrics: few, only where you drove the outcome, only where a baseline exists

**Grade:** B (multiple independent screeners on each side; the reconciliation is well-supported) **[agent]**

**Critical negative finding [agent]:** **no experiment has ever randomised "resume with numbers" vs "identical resume without numbers" and measured callbacks.** The academic audit literature randomises names, gaps and unemployment duration — never bullet phrasing. ResumeGo, the one firm running real field experiments, has no quantification study. This is a checked absence.

**Two famous stats, both traced, both phantom [agent]:**
- **"3.2x more likely — Harvard Business Review."** hbr.org searched directly; **no such study or figure exists.** Laundered onto HBR's name. **Grade C.**
- **"Up to 40% — Resume Worded"** (https://resumeworded.com/how-to-quantify-resume-key-advice) cites only TheLadders eye-tracking, which measures scan time, not quantification. **No source for its own headline number. Grade C.**
- **Cultivated Culture's 125,000-resume dataset** (https://cultivatedculture.com/resume-statistics/) is real and finds 36% of resumes have zero metrics — but scores with their own tool and never measures callbacks. **Grade B as description, worthless as causation.**

**Against metrics [agent]:**
- **jiveturkey** — https://news.ycombinator.com/item?id=34562575: *"When I see these kind of metrics on every other bullet (for L5 and below) it tells me a lot about the candidate and **not** that they have an appreciation for the business impact of their technical skills."* The objection is **density at mid-level**, not metrics per se.
- **cfiggers**, hiring manager — https://news.ycombinator.com/item?id=44784060: *"99% of the time I have no frame of reference and therefore no way to evaluate claimed accomplishments. Oh, you managed accounts totaling $24MM…? Sounds impressive, but what if every one of your peers were managing $30–40MM…? Descriptive statistics do very little for me."* **A number without a denominator or peer baseline is uninterpretable.**
- **saltybytes** — https://news.ycombinator.com/item?id=46639626: *"Everybody is trying to 'quantify' their resume but hiring managers are calling the bluff."*
- **hoosieree** (39446181): quantification "falls apart when you invent something, create a new revenue stream." **ryandrake** (47280306): it "penalizes hard working, talented devs who don't happen to be working in areas where wins are easily quantifiable."

**For metrics [agent]:**
- **jedberg** — https://news.ycombinator.com/item?id=34534717: *"It tells you that the candidate at least understood the business impact… and it gives you something to talk about in the interview. 'How was that issue identified? Did you find it or were you assigned the task?' … So many good questions to spawn from that one line."*

**The reconciliation [agent], which I find convincing:** the pro camp values a metric as an **interview hook**, not a screening signal. The anti camp objects to **density and unearned attribution**. Both are consistent with: *few metrics, only on work you genuinely drove, only where a baseline exists.*

**Implementable as:** cap metric density (e.g. ≤40% of bullets carry a number) rather than maximising it; require that metric-bearing bullets also carry an ownership verb; suppress metrics where no baseline is expressible.

**Measurable by:** metric-density ratio per role as a tunable A/B parameter. **This is a strong test candidate precisely because the advice industry pushes density toward 100% and screeners say that backfires.**

**Conflicts with:** XYZ applied uniformly, and Harvard-style "quantify every bullet."

---

### Bullet length and ordering

**Grade:** B for length (four independent sources converge); **C for bullets-per-role**; **C for ordering**

**Length — the most convergent finding in the report [verified for the wiki, agent for the rest]:**
- r/EngineeringResumes wiki: *"Bullet points should be 1-2 lines long."* / *"Aim for **1** sentence per bullet"* **[verified — cloned from GitHub mirror]**
- CMU SCS: *"no more than two lines per bullet point"* **[agent]**
- MIT: *"keeping each statement to 1-2 lines"* **[agent]**
- r/EngineeringResumes checklist: *"rarely more than two lines"* **[verified]**

Note this is convergence among *advice-givers*, not a measured result.

**Bullets per role — near-total evidence vacuum [agent]:** **no screener in any accessible corpus stated a count.** Only Berkeley Fung's MEng student guide commits ("At least 3 bullets; max to 5"). CMU templates print 3. Harvard, MIT, Stanford, Georgia Tech silent. **Treat any specific number as folklore.**

**Ordering within a role — plausible but unmeasured [agent]:** only Georgia Tech Scheller's undergraduate template addresses it directly. The wiki says *"Most relevant/impressive first"* **[verified]**, reasoning that "some hiring managers only have time to read the first." **Two targeted searches for screeners describing whether they read past the first bullet returned zero hits. Grade C** — plausible given lettergram's "I won't read past the first page," but nobody has measured it.

**Implementable as:** deterministic validators — line-count, leading-verb check, pronoun detector, orphan-wrap detector, numeral normaliser — plus a JD-relevance re-ranker.

**Measurable by:** all statically checkable on the rendered document. Highest-signal, lowest-cost automated checks available: unambiguous, no LLM needed.

---

### Ban the inflated-verb list

**Grade:** B (community standard) **[verified]**

**Source:** r/EngineeringResumes wiki, "Action Verbs" (GitHub mirror).

Banned as "superfluous/awkward/unnecessarily complex": *amplified, conceptualized, crafted, elevated, employed, engaged, engineered, enhanced, ensured, fostered, headed, honed, innovated, mastered, orchestrated, perfected, pioneered, revolutionized, **spearheaded**, transformed*. Plus *leverage, enhance, utilize*. Banned adjectives: *excellent, innovative, expert, revolutionary, disruptive*. Banned adverbs: *creatively, diligently, meticulously, strategically, successfully, independently, innovatively, excellently, expertly*.

Preferred: *analyzed, architected, automated, built, created, decreased, designed, developed, implemented, improved, optimized, published, reduced, refactored*.

**Correction:** an earlier draft claimed this blocklist doubles as an LLM-tell list. **§9 research disproved that.** Only *delve, underscore, meticulous, crucial* have academic corroboration as LLM markers, and of those only *meticulous* appears here. **"Spearheaded" — the canonical alleged tell — is explicitly folklore.** The blocklist stands on community-consensus style grounds alone; the AI-tell rationale is withdrawn.

**Caveat [verified]:** the wiki's own sample bullets don't all comply — several Electrical and Software examples carry no metric, and at least two use "leveraging" despite banning "leverage." Treat the sample bank as weaker than the stated rules.

**Measurable by:** deny-list token count per document (target 0). Deterministic.

---

## 2. Keyword matching and ATS mechanics

**Headline [agent]: the "ATS auto-rejects on resume keywords" model is not supported by a single vendor's own documentation.** Every vendor documenting automatic rejection ties it to **employer-configured questions**, never to resume text.

### Do not build keyword auto-reject countermeasures

**Grade:** A (multiple vendors' own docs) **[agent]** — corroborated by **[verified]** practitioner sourcing

**Vendor documentation [agent]:**
- **Oracle Taleo** — the only platform with documented instant auto-exit, and it fires on questions: *"A disqualification question… A candidate not meeting the required response can be instantly exited from the application process."* (https://docs.oracle.com/en/cloud/saas/taleo-enterprise/21b/otrec/candidate-prescreening.html). Taleo's ACE ranking is computed from **question and competency answers, not resume text.**
- **Greenhouse** — *"With auto-reject, your organization can set up application rules so that, based on an applicant's answer to a question, they will automatically be rejected"* (https://support.greenhouse.io/hc/en-us/articles/360000653472-Auto-reject). Rules operate only on Yes/No, single- and multi-select questions. **No mention anywhere of resume scanning as a rejection trigger.** Strong negative evidence: Greenhouse's entire "Resumes" doc section has four articles — non-English parsing, bulk download, virus scanning, unsuccessful parse. **No scoring or ranking article exists.**
- **Ashby** — explicitly denies AI rejection: *"it is up to the reviewer to advance or reject"*; *"it's ultimately the qualified humans on your team making an advance/reject decision"* (https://www.ashbyhq.com/product-updates/ai-assisted-application-review). All AI features opt-in; PII redacted before resumes reach the model.
- **iCIMS** — negative evidence from a legal filing, the most trustworthy kind. Under NYC Local Law 144 they disclosed **one** feature meeting the automated-employment-decision-tool definition: Candidate Ranking, which *"surfaces the highest-matching candidates."* It surfaces; it does not reject. (https://www.icims.com/blog/how-icims-supports-the-nyc-automated-employment-decision-tools-law/)
- **Workday** — **undocumented either way. Grade C on any claim.** Public pages never address auto-rejection; Community docs are login-gated.

**Practitioner corroboration [verified]:** Gergely Orosz, *The Tech Resume Inside Out*, "ATS Myths Busted" (https://thetechresume.com/samples/ats-myths-busted), who interviewed technical recruiters at Amazon/Google/Microsoft: *"do ATSes reject resumes? Spoiler: **they do not. Humans do.**"* The ATS is a human-workflow organiser; the only automated filtering is **knockout questions** answered on the form, not parsed from the resume. *"ATSes are still far more simple than even to understand what programming languages you've listed on your resume."*

**Implementable as:** deprioritise resume-side keyword gaming. If the pipeline can fill application-form fields, **knockout answers are the real automated gate** and deserve more engineering attention than keyword density.

**Conflicts with:** the entire keyword-optimisation product category (Jobscan, Resume Worded). They sell products premised on the opposite.

---

### The "75%" claim — DO NOT IMPLEMENT AGAINST

**Grade:** C (folklore; no primary source has ever existed) **[verified]**

**Source:** https://unchartedcareer.com/blog/the-75-of-resumes-are-auto-rejected-myth-traced-to-its-source

- Originates in **Preptel** marketing, ~2012, vendor of "ResumeterPro."
- Preptel **shut down August 2013** and "never published a study, a dataset, or a method."
- Chain runs through an uncited Forbes mention, then CIO, CNBC, and a wall of resume blogs — all pointing back to the same defunct vendor.
- Diagnostic tell: the number **drifts between 70, 75 and 88 percent**. "A real statistic has one value and one method. This one has three values and none."

**Why it persists:** commercially load-bearing. Every resume-optimisation product needs a threat to sell against. It also flatters the rejected applicant — a machine misjudged you rather than a human judging you accurately.

**What replaces it [agent]:**
- **Jan Tegze** (recruiter, informal survey): *"90–95%+ of all applications are viewed by a human."* Remaining 5–10% attributed to recruiter workload and technical failures. **Grade B.**
- **Enhancv survey**, 25 US recruiters, Sept–Oct 2025: only 2 of 25 (8%) said their system auto-rejects beyond knockout questions; 92% said systems do **not** auto-reject for formatting or design; 100% use knockout filters. (https://enhancv.com/blog/does-ats-reject-resumes/) **Grade B, not A** — N=25, self-selected, and Enhancv sells resumes. This is the actual source of the "8%" now circulating unattributed.

**The real study the myth gets confused with:** HBS/Accenture *Hidden Workers: Untapped Talent* (2021), about **recruiter-configured hard filters** (years of experience, degrees, gaps), not keyword parsing. **Caution [agent]: the canonical HBS PDF 404'd and the landing page carries no statistics. Do not cite it until someone verifies what it measured** — it is the most-miscited source in this space.

---

### Matching mechanics: scoring is per-vendor and often separately licensed

**Grade:** A for existence, C for defaults **[agent]**

| Vendor | Resume→JD score? | Default? | Grade |
|---|---|---|---|
| Greenhouse | None documented | — | A (negative evidence) |
| Lever | Historically none; "Talent Fit AI" ~2025 | Employer-enabled | C — help doc failed to load |
| Ashby | No numeric score — "Meets"/"Does not Meet" | Opt-in | A |
| Taleo | ACE — from **question answers**, not resume text | Built-in | A |
| Workday | A/B/C/D grades via **HiredScore** | **Separate purchase** | A |
| iCIMS | Candidate Ranking / Role Fit | Unknown | A exists / C defaults |
| SmartRecruiters | SmartAssistant Match Score | Unknown | A exists / C defaults |
| BambooHR | None found — recruiter-initiated keyword search only | — | B |

**SmartRecruiters — most mechanically specific disclosure [agent]:** Match Score is *"a confidence interval of a candidate's fit… based on the **skills overlap with the Job Ad**"*, with skills normalised against the **EU ESCO taxonomy** (~14,000 skills). **Implication: non-standard skill phrasing may fail to map to a node.** Explicitly excludes personal data from the algorithm. (https://ta.smartrecruiters.com/rs/664-NIC-529/images/SR-Product-Sheet-SmartAssistant.pdf)

**Textkernel (powers many ATS back-ends) — term-based, not embeddings [agent]:** *"Match is a term-based matching engine."* Configurable per-criterion weights, recency and cross-field boosting. Admits opacity: scoring includes *"semantic signals… not always visible in the UI."* (https://developer.textkernel.com/SearchMatch/master/Matching/Scoring/)

**On exact-match vs stemming vs synonym expansion: no vendor publishes tokenisation or expansion rules [agent].** Claims that "Workday understands PM = project management" or "exact matches score higher than semantic equivalents" trace **only to SEO content farms**. **Grade C — unverified.** What *is* documented is taxonomy normalisation (ESCO at SmartRecruiters, Skills Cloud at Workday), implying ontology expansion rather than raw string matching — but no vendor states the mechanism.

**Implementable as:** prefer **canonical skill names** over idiosyncratic phrasing ("PyTorch" not "Torch-based DL"), since taxonomy mapping is the one documented mechanism. Keep a canonical-form lookup for AI/ML terms.

**Measurable by:** proportion of Skills entries that map to a known ESCO/canonical node.

---

### Do not implement white-text or prompt injection

**Grade:** B **[verified]** + mechanism analysis **[agent]**

**Source [verified]:** HN thread 40489596 — https://news.ycombinator.com/item?id=40489596
- ATS product manager: *"we use OCR in our resume parsing, so white text on white backgrounds won't get picked up."*
- Tester: *"I've tried this over and over with different methods… and in no case has it affected the results at all, using GPT-4o."*
- Same PM's legitimate alternative: *"you could probably accomplish almost the same thing without any subterfuge by just adding a Skills section."*

**Mechanism note [agent], which cuts the other way and is worth knowing:** for *text-layer* extraction (not OCR), **colour is not part of extracted text** — white-on-white extracts identically to black. So some parsers see it, **and so does any recruiter who selects-all or opens the parsed-text pane.** Two documented deterrents: Textkernel code **303** fires when a section is "longer than the WORK HISTORY and EDUCATION sections combined" (a stuffed keyword block trips this), and code **323** flags "multiple sections of the same type."

Beyond not working reliably, this is deception directed at an employer and disqualifying if found.

**Implementable as:** a guardrail — refuse to emit invisible text, off-canvas text, or instruction-like strings.

**Measurable by:** static check that all rendered text has non-white fill and lies within page bounds.

---

### Keyword stuffing as perceived by the human — EVIDENCE NOT FOUND

Everything found on where JD-mirroring flips from helping to hurting was **vendor content marketing** (Jobscan, InterviewPal, Resumefast, Resume Pilots, Story.CV) asserting "recruiters spot it instantly" with no study and no named screener. **Grade C.**

**No credible evidence in either direction.** Treat the stuffing threshold as an **empirical question for your own A/B test** — measure JD-token-overlap ratio against real callback outcomes and find your own inflection point. That is a better use of the pipeline than any advice currently available.

---

## 3. Ordering and layout

### Section order: Experience before Education once employed

**Grade:** C (contested; no screener data) **[verified for wiki, agent for institutions]**

- **r/EngineeringResumes** (experienced professionals): *Work Experience > Skills > Education*, or *Skills > Work Experience > Education*. Senior: *"Move your education section to the bottom."* **[verified]**
- **But both technical institutional templates put Skills ABOVE Experience** — CMU SCS (Education → Skills → Professional Experience) and Berkeley Fung. **[agent]**

**These contradict, and the resolution is audience [agent]:** both institutional templates are written for students with thin experience sections. Neither says what happens when experience is substantial. For a senior engineer, r/EngineeringResumes is better-matched. **Grade C either way — no screener data exists.**

CMU's one transferable rule **[agent]**: *"List skills in order of proficiency… **Do not include soft skills such as 'teamwork' or 'leadership' in this section.**"* — matching the wiki **[verified]**.

---

### Summary for senior IC: weakest-supported area in the report

**Grade:** C **[mixed]**

**Against, verified screeners [agent]:**
- **encoderer** (11678542): *"I never read the cover letters, nor the 'objective' on a resume… my choice to move you to a phone screen will never hinge on those details."*
- **scarface74**, ~20 yrs, makes hiring decisions (19351185): *"Objectives on resumes are useless fluff."*
- **leeny**, hires engineers (4236902): *"No objective (in 99% of resumes this is bullshit fluff), no summary."*

**Important caveat that weakens two of three [agent]:** encoderer and scarface74 say **"objective"** — a distinct, more-derided artifact than a senior summary. Only leeny rejects both.

**For:** **no verified screener quote in favour exists.** Institutional sources split; MIT's sample pack carries a SUMMARY on exactly **two of ~18 samples** — the PhD and the Alum, zero undergraduates — an implicit seniority signal expressed only through sample design **[agent]**.

**The position best matching your case [verified]:** r/EngineeringResumes: *"Do **not** include a summary/profile unless you're a **senior/staff engineer or above**, making a career change, or addressing an unemployment gap."* Senior section: *"Consider including a brief summary (**<2 sentences**)."* **Explicitly seniority-gated, and it gates in your favour at senior title — but carries no reasoning and no evidence.**

**Note the internal tension [verified]:** the wiki gates on *title* (senior/staff+) while its senior section is written for *10+ YoE*. At senior title with ~5 YoE the wiki is genuinely ambiguous.

**The best operational test, from two independent sources:**
- **CMU [agent]:** *"If your objective/summary isn't adding clarity and advancing your purpose and resume, remove it."*
- **Mike Peditto, Director of Talent at Teal [verified, snippet]:** *"if you can copy and paste your summary onto anyone else's resume and it still makes sense, that's not a good summary."*

**Implementable as:** emit only when (title ≥ senior) OR (career change) OR (gap); cap at 2 sentences; run an automated **copy-paste test**.

**Measurable by:** the copy-paste test is mechanisable — strip proper nouns and role-specific terms and measure similarity to a corpus of generic senior-engineer summaries; or require ≥2 claims that could not be true of a randomly chosen senior engineer.

**Conflicts with:** page budget — a summary costs space the one-page rule cannot spare.

---

## 4. What gets a resume binned fast

**Caveat [agent]: a proper frequency ranking is not deliverable.** The source that would have produced one was **confabulated by the search layer** and discarded. Below is what was individually verified by the subagent.

### The single biggest factor is not your resume at all

**Grade:** A for the mechanism **[agent]**

- **conductr** (48721959): *"There might be 9900/10000 resumes you never even looked at."*
- **Bartoš et al., "Attention Discrimination," AER 2016** (https://www.aeaweb.org/articles?id=10.1257/aer.20140571): employers **endogenously allocate attention**, deciding how hard to read before reading.

**Most rejection is never being opened.** Content optimisation operates downstream of a gate you do not control. **This should temper expected effect sizes for the entire pipeline** — if 90%+ of applications are never opened, even a large content improvement moves a small denominator.

### One practitioner's ranked list

**Grade:** B (single well-credentialed source) **[agent]**

**jacurtis** — https://news.ycombinator.com/item?id=34462897 — *"hiring manager… thousands of resumes/CVs and hired about 50-60 engineers."*

1. Under-qualified for senior roles
2. Too many technologies, claiming expertise in all
3. Nonsensical titles or unrealistic progression
4. Overly colourful designs, excessive whitespace
5. Personal photos (auto-reject, bias concerns)
6. Exceeding three pages
7. Consistent short employment stints
8. **Lacking specific accomplishments in bullets**
9. No visible career progression
10. Unclear career direction

Explicitly **doesn't** matter: home address, mission statements, **minor typos**. Replies contest #5 (country-dependent) and #3 (Scandinavian flat title structures).

### Independently corroborated knockouts [agent]

- **Failing hard requirements.** **jpp**, CTO, ~1000:1 application-to-hire (37884109): *"We have a few screening questions that literally confirm the required skills listed in the job posting and use these to auto-reject applicants."* Also *"About a third are just outright unqualified."* **Grade B.**
- **Exaggeration, caught downstream.** jpp: most who advance then fail *"generally because they've exaggerated on their resumes."* Matches lhorie: *"pretty easy to catch buzzword-slinging w/ a phone screen."* **Two independent sources — and a direct warning for a generative pipeline.**
- **Filler reads junior.** **lhorie** (10833804): *"If you add a hobbies section to fill space, it'll make you look very junior."*
- **Technology-list noise.** **encoderer** (11678542): *"it's simply about signal to noise. Adding every technology you've ever touched doesn't add much value."* Matches jacurtis #2.

**Note the tension on typos:** jacurtis says minor typos don't matter; the wiki says they "can easily cause your resume to be ignored" **[verified]**. Unresolved. Cheap enough to fix that it doesn't need resolving.

**Implementable as:** hard grammar/spelling gate; personal-details scrubber; skills-list length cap; drop hobbies/filler sections; **never generate a claim the candidate cannot defend in a phone screen.**

---

## 5. The one-page rule — RESOLVED AGAINST, for your case

**This is the clearest finding in the report and it cuts against the one-page rule.**

### Five independent screeners say two pages is fine

**Grade:** B — the strongest practitioner consensus found, 2014–2023, none selling anything **[agent]**

- **briandear**, hired ~15 developers (8810452), states your hypothesis outright: *"**The one page resume 'standard' is a myth. Unless you're a new grad**, one page is extremely difficult to do… I have hired about 15 developers over the past few years and **never once did I say 'a two page resume!! Next!'** … Page count is arbitrary."*
- **lettergram**, reviews hundreds per quarter (20730787), **directly on point for ~5 years**: *"**People with 5+ years of experience will often hit two pages, which IMO is fine.** I just expect everything to be on point… So I expect 2 pages or less, but don't penalize too much for large resumes. **I just won't read past the first page.**"*
- **rm999**, interviews senior data scientists/engineers (7250570) — a case where one-page compression actively *harmed* a senior candidate: *"more than half the resumes are two pages… by no means a deal-killer. **The last guy I interviewed had seven jobs and three degrees crammed into a single page — I had to waste 25% of the interview just asking him what he did at each job because his resume didn't tell me**… This only hurt him because I had less time to build a case to hire him."*
- **jacurtis**, 50-60 engineers hired (34462897): *"**Most resumes should be 2 pages.** If you are just getting into the industry and only have < 3 jobs then you should use 1 page… Never go above 3 pages."*
- **lhorie** (10833804): *"I wouldn't reject an applicant based on whether their resume has two pages."*

### The measured counterpart

**Grade:** A design with two material caveats **[verified]**

ResumeGo, "Settling the Debate" (2018) — https://www.resumego.net/research/one-or-two-page-resumes/
- **482 participants** with recruiting experience; ~7,700 reviews
- Paired 1-page (350–500 words) and 2-page (700–850 words) resumes, matched candidates
- Preference for two pages: entry **1.4×**, **mid-level 2.6×**, managerial **2.9×**, overall **2.3×**
- Credential-summary score 8.6 vs 7.1; time spent 4m05s vs 2m24s

**Caveats:** authors state it was *"only a simulation"*, not hiring data; ResumeGo sells resume writing (longer = more billable); the time result may be tautological.

### The wiki's opposing rule, and why it doesn't apply to you

**[verified]** *"One page long, unless you have some 10+ years of experience. The rule of thumb is 1 page per decade of experience."*

Their stated reasoning: *"The majority of resumes that are posted on this sub don't need a second page. Once it gets cleaned up… a lot of people will have trouble populating 1 page with relevant, technical content."*

**Read that carefully — it is a claim about their submission population, not about screeners.** The sub is dominated by students and early-career posters. The rule is calibrated to people who genuinely cannot fill a page, then applied universally. **At 5 years you are exactly where their empirical premise stops holding and their arbitrary 10-year threshold hasn't yet released you.**

Provenance point **[verified]**: the one-page rule's institutional backing is largely **undergraduate** material — Harvard's guide is titled "*Undergraduate* Resource Series." CMU SCS's version reads *"**Students** with less than 10 years of experience should have a one-page resume"* — still "Students" **[agent]**. This supports your hypothesis that it is inherited new-grad guidance.

### The resolution

The screener consensus and the wiki rule are less opposed than they look, because of lettergram's refinement: **two pages is permitted, but page 2 is not read.** The operative constraint is not page count — it is that **everything decisive must be on page 1.** A second page is free storage that costs nothing and earns nothing.

**Implementable as:** allow 2 pages; enforce a **page-1 sufficiency check** — the most JD-relevant role, the top 3 bullets, and the skills block must all render above the page-1 break.

**Measurable by:** deterministic — compute which content lands above the page break and score its JD relevance against the whole document's. **This is a better-specified and more testable tactic than "one page vs two."**

### On "6 seconds" — Grade C dressed as Grade A [agent]

TheLadders, 2012: **30 recruiters, 10 weeks, 6.25 seconds.** 2018 follow-up: **7.4 seconds, sample size never disclosed.** Never peer-reviewed. Susan Adams' original 2012 Forbes coverage: *"As with many studies by outfits in the career business, TheLadders' findings conclude that job seekers should buy one of its wares."*

**Susan Gygax**, working recruiter, in ERE (https://www.ere.net/articles/is-the-6-second-resume-scan-a-myth): *"The majority of these articles 1) were not written by recruiters or 2) were selling something."* The 2018 report *"does not specify the types of positions or lengths of resumes… It also doesn't state how many recruiters were in the study."*

**CareerBuilder data: only 17% of recruiters spend under 30 seconds.** The useful reframe: these studies measured an *initial skim*, not a decision — matching what screeners describe (5–30 seconds, top of page 1, deciding only whether to look again).

---

## 6. Tailoring per job

### The cleanest experiment — and its most useful result is not the headline

**Grade:** A design, vendor-run **[agent]**

ResumeGo cover-letter field experiment — https://www.resumego.net/research/cover-letters/
- **7,287 fictitious applications to real postings**, Jul 2019 – Jan 2020
- Three arms: none / generic / **tailored**
- Tailored → **+53% callback** vs none
- **The load-bearing result: generic cover letters showed minimal advantage over sending nothing at all.** The entire effect came from **customisation**, not from the artifact existing.
- Survey of 236 hiring professionals: 81% significantly preferred tailored to generic

Caveat: tests *cover letter* tailoring, not resume tailoring, and ResumeGo sells resume writing.

**The famous "31% more likely" tailored-resume stat is a phantom [agent, corroborating my own finding]:** attributed to ResumeGo across the advice web; **ResumeGo's research index lists no resume-tailoring study.** I independently confirmed **[verified]** that their index publishes tailoring and cover-letter headline findings **without methodology or sample size**, in contrast to their page-length study where methodology *is* disclosed. **Grade C. Do not encode the number.**

### Largest dataset, weakest design

**Grade:** B-minus **[agent]**

Huntr (https://huntr.co/research/job-search-trends-q2-2025): 461k applications, Q2 2025. Tailored **5.75%** application→interview vs generic **2.68%** = **+115%**. Covered by Forbes.

**Fatal caveat, confirmed by fetching the report: observational, not randomised.** Users self-selected into tailoring and outcomes are self-reported. People who tailor are engaged, selective, and applying to jobs they're plausibly qualified for. Huntr doesn't control for this and sells an AI Resume Tailor.

### The one honest individual A/B test — and it reframes the metric

**Grade:** B-minus, n=50, uncontrolled **[agent]**

Jasmeet, DEV Community (https://dev.to/jasmeet7015/should-you-customize-your-resume-for-every-job-heres-what-i-learned-after-applying-to-100-roles-4bo3):
- **Tailored: 45 min each, 37.5 hrs → 6 interviews (12%)**
- **Generic: 15 min each, 12.5 hrs → 4 interviews (8%)**

**Tailoring won on rate and lost on throughput.** The 25 extra hours would have funded ~100 more generic applications → ~8 more interviews at base rate.

**This is the most important framing in the section for your pipeline.** An automated tailoring system is valuable *precisely because it collapses the time cost that made tailoring a losing trade for a human*. Your pipeline's advantage is not that tailoring works — it is that tailoring at near-zero marginal cost changes the arithmetic. **Track interviews per unit of effort, not just conversion rate.**

### What tailoring actually changes: nobody has measured the decomposition

**No screener describes detecting tailoring as such [agent]** — only what makes a resume readable fast. Most concrete, **lostcolony** (7428411), screening at ~3 seconds: *"Highlight what niche you can fill (yes, preferably tailored for the company), and make that -obvious- in your resume."* He contrasts *"Old coder, part of a large team that did…some stuff that isn't spelled out clearly"* against *"Perl, C, and Linux expert, extensive application development experience."* **Mechanism implied is placement and legibility, not keyword density.**

**Explicit absence [agent]: no study decomposes tailoring into keyword-matching vs bullet-reordering vs summary-rewriting.** Huntr has the data and doesn't. **Grade C.**

**Notably [agent], the r/EngineeringResumes wiki is silent on tailoring** — its job-search page argues networking over volume and never mentions per-job customisation. My own read **[verified]** found tailoring advice in the Work Experience section (*"**Tailor your resume** for each application"*), so I'd call this partially contradicted; the wiki mentions it but does not elaborate.

**What good tailorers change [verified, wiki]:** reorder roles and bullets by relevance; align Skills to the JD while keeping adjacent honest skills; ensure Skills↔bullets consistency (*"Repeat things that you use in your bullet points"*); cut technical-but-irrelevant content.

**Implementable as:** per-application transform — (a) re-rank bullets by JD relevance, (b) reorder/filter Skills against JD terms, (c) enforce Skills↔bullets consistency, (d) drop lowest-relevance content to fit page budget. **All four are lossless reorderings/selections — no fabrication.** Safest high-value part of the pipeline.

**Measurable by:** Skills↔bullets consistency is deterministic. JD-relevance lift = mean bullet-relevance before vs after. **Because the decomposition has never been measured, your pipeline is positioned to generate genuinely novel evidence — run (a)–(d) as separate arms rather than one bundle.**

---

## 7. Formatting mechanics that break parsers

### Multi-column layout — the one formatting failure with hard measured evidence

**Grade:** A **[agent]**

**Textkernel published internal measured data** (https://www.textkernel.com/learn-support/blog/improving-extraction-from-column-resumes/): *"at least **15% of CV documents use a column layout**."* Under their old rule-based renderer only **62%** of CVs rendered well; replacing rules with a gradient-boosting model raised that to **90%**. Visual-gap classification accuracy 82%→91% overall, 60%→82% on column-separator cases. Contact-info fill rates rose 4–10 points across 12,000 random CVs.

**Independently corroborated academically:** Zhu et al. (Alibaba), *Layout-Aware Parsing Meets Efficient LLMs*, arXiv:2510.09722, deployed in Alibaba's HR platform. **~20% of resumes use non-linear multi-column layouts**; layout-aware preprocessing lifts F1 from **0.919 → 0.959** with Claude-4. Long-text fields (job descriptions) benefit most: **0.136 → 0.846 F1**.

**The nuance that matters:** multi-column is a **real** failure, **substantially fixed** in modern parsers, and the residual ~10% is still the largest single formatting risk. "Your resume will be shredded" is a stale 2010s claim; "single column is measurably safer" is current and true.

### Vendor-documented hard constraints

**Grade:** A **[agent]**

**Greenhouse** (https://support.greenhouse.io/hc/en-us/articles/200989175-Unsuccessful-resume-parse): files over **2.5 MB**; "a resume with spaces between the letters" (kerning tricks); graphics, photos, word art; image-format resumes; "complex resumes with tables, headers, and footers"; contact info in headers/footers/text boxes; columned layouts; abbreviated job titles ("Sr. Account Exec"); company names missing "Inc."/"LLC".

**Taleo** (https://docs.oracle.com/cloud/latest/taleo/OTREC/_candidate_user_fmx.htm): **resume cannot exceed 100 kilobytes** (configurable default). Accepted: .doc, .docx, .wpd, .txt, .rtf, .html, .pdf, .xls, .odt. Image formats accepted as uploads but **"will not be parsed."** And notably: *"Resume Parsing has no impact on the formatting of a text (bold, italics, bullets)"* — **styling is discarded, not fatal.** The 100 KB cap is the least-known and most actionable constraint found.

**Textkernel Resume Quality API** — a vendor-authored machine-readable list of what breaks parsing, in four severity tiers. Codes include **408** document truncated before parsing, **412** no sections found, **441** neither email nor phone found, **303** a section longer than Work History and Education combined, **323** multiple sections of the same type, **132** multiple email addresses. (https://developer.textkernel.com/tx-platform/v10/resume-parser/overview/parser-output/) Also a **22.5-second hard parse timeout**.

**The single most useful diagnostic found [agent]:** Textkernel — *"The vast majority of problems in parsing are not from processing the plain text, but from **conversion to plain text**, so when you find a mistake in the output, look at the converted text and see if it reads logically."* Plus: *"always send the original file, not the result of copy/paste, not a conversion by some other software, not a scanned image."*

**Implementable as:** the copy-paste test is your highest-value parser check — extract text from the rendered PDF and assert it reads in logical order. Enforce single column, no images, contact info in the body (never headers/footers), file size <100 KB, full job titles, company legal suffixes.

**Measurable by:** all deterministic — extraction-order check, column count, image object count, byte size, regex for abbreviated titles.

### PDF vs DOCX — genuinely contested, and the popular numbers are fabricated

- **Recruiter position [verified]:** José Marchena, via Orosz — *"the best CV format being PDF. With Word documents, you risk ruining your format."*
- **Jobscan says PDF [agent]:** *"most applicant tracking systems read and parse PDF resumes more accurately."* **Grade B at best** — no methodology, sample size, or per-ATS breakdown, and **they reversed their historic .docx recommendation without publishing data behind either position.** (I separately found the older .docx recommendation still circulating **[verified]** — so Jobscan has publicly held both positions. Treat them as unreliable on this question.)
- **Textkernel flags "PDF format" as a Major Issue (300-series) in its own quality codes [agent]** — a vendor implicitly ranking PDF below DOCX.
- **Every "2026 data" PDF-vs-DOCX statistic circulating is fabricated [agent]** — "97% vs 76% vs 53%", "DOCX beat PDF in 6 of 8 systems", "Workday autofill 34% error rate" trace exclusively to AI-generated SEO sites naming no ATS instance, sample, or date. **Grade C. Do not repeat.**

**Honest read [agent]: no credible controlled PDF-vs-DOCX study exists.** The defensible statement is Textkernel's — what breaks parsing is the **text-extraction layer, not the container.** A text-based PDF exported from a word processor and a DOCX both extract fine; a design-tool PDF, a scan, or an image does not.

**Implementable as:** emit both; let the form decide; or make it an A/B arm. Assert the PDF has a real text layer.

### Real but narrower, and stale folklore

**Non-standard section headings — Grade A by architecture [agent]:** Textkernel *"systematically divides resumes into sections, then applies specialized sub-parsers"*, and code 412 fires when none are found. A creative heading means the sub-parser never runs. **Use conventional headings.** (Matches the wiki's insistence on "Experience", "Skills", "Projects" verbatim **[verified]**.)

**Ligatures/font encoding — Grade B [agent]:** real PDF-level phenomenon where "fi"/"fl" and accents extract as garbage. **No ATS vendor states it affects them and nobody has measured frequency.** The copy-paste test detects it instantly.

**Stale or folklore [agent]:**
- *"Bold/italics/bullets break parsing"* — **contradicted directly by Oracle.** Grade C.
- *"Use only Arial/Times New Roman"* — **no vendor says this anywhere.** Grade C. The real concern is font *embedding/encoding*, not choice.
- *"Never use any table"* — Greenhouse names tables only within "complex resumes with tables, headers, and footers." A simple two-cell table is not that. Overstated.

---

## 8. Singapore specifics

**Researched.** All sourcing in this section is **[agent]** — I did not independently re-fetch any of it. But the sourcing is unusually strong: these are government primary sources (MOM, TAFEP, PDPC, AGC), not commentary.

### Omit personal particulars entirely — this is regulatory, not stylistic

**Grade:** A (official Singapore government guidance) **[agent]**

**TAFEP job application form guidance** — https://www.tal.sg/tafep/employment-practices/recruitment/preparing-job-application-forms — explicit "should not ask for" list: **age (e.g. NRIC, date of birth), gender, race, religion, marital status and family responsibilities, disability, photographs, National Service liability**, plus mental-health declarations. Where genuinely needed, photos and NRIC *"should be requested for these at the point of job offer."*

**TGFEP** — https://www.tal.sg/tafep/getting-started/fair/tripartite-guidelines — protected: age, race, gender, religion, marital status and family responsibilities, disability ("not exhaustive"). Application forms should *"request only job-relevant information"* and *"avoid requesting photographs or NRIC numbers unless justified."*

**PDPC NRIC Advisory Guidelines** (issued 31 Aug 2018, in force 1 Sep 2019) — https://www.pdpc.gov.sg/-/media/files/pdpc/pdf-files/advisory-guidelines/advisory-guidelines-for-nric-numbers---310818.pdf — extracted verbatim from the PDF by the agent. Para 3.1: organisations *"are generally not allowed to collect, use or disclose NRIC numbers (or copies of NRIC)"* except where required by law or to verify identity *"to a high degree of fidelity."* Worked example 3.9 is decisive:

> *"Benny wishes to apply for a job with Organisation XYZ and fills in a job application form. **The application form does not require Benny to provide his NRIC number.** … **There is no requirement under the law to ask for NRIC numbers for the purpose of job applications.**"*

NRIC is legitimate **at hire** (Employment Act s.95 record-keeping), **not at application**. Same treatment extends to **FIN, Work Permit, Birth Certificate and passport numbers** (paras 1.4–1.5).

**On the 2024/25 update:** following the **MDDI statement of 13 Dec 2024**, the guidelines *"will be updated. In the meantime, these guidelines remain valid."* That update concerns NRIC as a **password/authenticator** (post the ACRA/Bizfile exposure), **not collection at application**. **Net effect on resumes: unchanged.**

| Field | Answer | Grade |
|---|---|---|
| **Photo** | **Omit.** TAFEP lists photographs for removal; permitted only at job-offer point | A |
| **NRIC / FIN** | **Omit.** No legal basis to collect at application stage | A |
| **Age / DOB** | **Omit.** TAFEP names "Age (e.g. NRIC, date of birth)" directly | A |
| **Race, religion, gender, marital status** | **Omit.** All protected, all on the removal list | A |
| **National Service liability** | **Omit.** On the removal list | A |
| **Nationality** | Nuanced — see COMPASS below | A |
| **CV length** | **2–3 pages, not the US one-pager** | B |
| **Referees** | Not expected; sources split between "available upon request" and omitting entirely | B |

**Implementable as:** a Singapore locale profile that hard-suppresses photo, NRIC/FIN, DOB/age, gender, race, religion, marital status and NS liability, and raises the page budget to 2–3.

**Measurable by:** deterministic field-presence assertions. **This is the highest-confidence, lowest-ambiguity rule set in the entire report** — it is government guidance, not preference.

### CV length is materially different from the US

**Grade:** B (three independent SG recruiters converging) **[agent]**

Michael Page SG: *"Aim for 2-3 pages"* (https://www.michaelpage.com.sg/advice/career-advice/resume-and-cover-letter/how-to-write-winning-resume). Randstad SG: *"should not exceed two pages."* JobStreet SG: one page early-career, two acceptable with extensive experience.

**For senior/tech in SG, 2–3 pages is right.** Note this compounds with §5 — the one-page rule is even less applicable in Singapore than in the US.

### The "Personal Particulars block" convention — significant negative finding

**Grade:** C (folklore) **[agent]**

The belief that SG CVs conventionally carry a Personal Particulars header with photo, current/expected salary and notice period **has no Grade A or B support in anything reachable.** Michael Page, Randstad and JobStreet **all omit** current salary, expected salary, notice period and work-pass status from resume guidance entirely. **Not one recommends a Personal Particulars header.**

The convention is real in **job ads and application forms** — SG employers routinely demand current/expected salary *there* — but the inference "therefore put it on your CV" is folklore hardened by SEO resume-builder content targeting SG keywords.

**Implementable as:** do not emit salary expectations or notice period on the resume. Handle them as form fields if the pipeline fills forms.

**Salary-history legal position: NOT VERIFIED.** Singapore likely has no US-state-style ban on salary-history questions, but the agent could not source this before the budget ran out. **Do not ship as fact.**

### Workplace Fairness Act — not yet in force, and a date correction

**Grade:** A **[agent]**

https://www.tal.sg/tafep/workplace-fairness · statute https://sso.agc.gov.sg/Act/WFA2025

Passed 8 Jan 2025; Dispute Resolution Bill 4 Nov 2025. **Takes effect end-2027 — NOT in force as of now.** Protected: age; **nationality**; sex/marital status/pregnancy/caregiving; race/religion/language; disability and mental health. Employers with 25+ staff first; penalties to **SGD 50,000**.

**Correction worth carrying:** a widely-syndicated blog (asanify.com) asserts a **July 2026** commencement. TAFEP's own page says **end-2027**. Trust TAFEP. Note **"nationality" is protected under the WFA but is not in the current TGFEP list** — a real expansion, and it changes the calculus below from 2027.

### COMPASS: citizenship/PR status is worth stating, but this is inference

**Grade:** A for the framework; **the resume recommendation is inference, not sourced advice** **[agent]**

https://www.mom.gov.sg/passes-and-permits/employment-pass/eligibility · https://www.mom.gov.sg/-/media/mom/documents/work-passes-and-permits/compass/compass-booklet.pdf

In force 1 Sep 2023. Candidate needs **40 points** across C1 salary vs sector benchmark, C2 qualifications, **C3 nationality diversity** (candidate's nationality share of the firm's PMETs: 20 pts if <5%, 10 if 5–25%, **0 if ≥25%**), C4 local PMET share, C5 Shortage Occupation List bonus, C6 strategic priorities. **Exempt above SGD 22,500/month** fixed salary.

EP qualifying salary (age-graduated 23→45+): **SGD 5,600–10,700** general, **6,200–11,800** financial services. **From 1 Jan 2027: 6,000–11,500 and 6,600–12,700.** There is a **5-year EP for experienced tech professionals with shortage skills** — https://www.mom.gov.sg/passes-and-permits/employment-pass/experienced-tech-professionals-with-skills-in-shortage — directly relevant to senior AI/ML.

**The reasoning:** C3 and C4 are *firm* attributes, not candidate paperwork. PRs count as local under C4, but their nationality is taken from passport under C3. So "Singapore Citizen" or "PR" on a CV signals **no work-pass overhead and no COMPASS hit** — a genuine asymmetric advantage.

**But flag clearly:** this is the agent's inference from MOM's Grade A rules, **not a recruiter recommendation — no SG recruiter guide reached advises it.** It also sits in tension with the WFA making nationality protected from end-2027: employer *demands* for nationality are getting riskier even as candidate *volunteering* of it stays advantageous. The r/EngineeringResumes wiki independently advises listing citizenship/visa status where work eligibility isn't obvious **[verified]**, which points the same way from a different market.

**Implementable as:** for SG applications, emit citizenship/PR status near the name **only when true and only as work-eligibility signal** — never age, race or NRIC. Treat as an A/B arm given it is inference.

---

## 9. AI-written resume detection

**Researched, and the headline result is not what the framing of the question assumes.** All **[agent]**-sourced.

### The finding that reframes the problem: repricing, not detection

**Grade:** A **[agent]**

**Cui, Dias & Ye, "Signaling in the Age of AI: Evidence from Cover Letters"** — https://arxiv.org/abs/2509.25054. Platform: **Freelancer.com** ("AI Bid Writer", launched 19 Apr 2023). **5,499,707 cover letters, 106,714 jobs, 264,082 workers**, Jan–Sep 2023, difference-in-differences.

- Access to the tool: **+0.43pp callbacks (51% relative lift off a 7.02% baseline)** — significant at 10%, **not at 5%** — and **effects tapered after ~2 months**.
- **The load-bearing result: correlation between cover-letter tailoring and callbacks fell 51%**, while correlation between callbacks and platform review scores **rose 5%**.
- **Time spent editing the AI draft correlates positively with hiring success.**

**Synthesis: employers did not detect AI. They repriced the signal** — shifting weight onto things AI cannot fake (ratings, verifiable work history). **The mechanism is devaluation, not punishment.**

**This directly answers the distinction your brief raised** — penalising *detected AI text* vs penalising *generic text*. The measured answer is the second, and more precisely: the penalty is not applied to the text at all, it is **withdrawn from the channel**. Tailored text stopped earning credit because tailoring stopped carrying information.

**Implication for the pipeline, and it is a strategic one:** if tailoring is being repriced toward zero as everyone automates it, the durable advantage is not better-tailored prose. It is **making verifiable, hard-to-fake signals legible** — shipped systems, named production impact, public artifacts, references. Your pipeline should treat prose tailoring as table stakes and surface-verifiable-signal as the differentiator.

**Evidence absent [agent]:** **no field experiment or audit study measures callbacks for human- vs LLM-written resumes sent to real corporate employers.** Both Grade A papers study platform-provided assistance on labour marketplaces. Every "AI resumes get X% fewer callbacks" claim traces to a vendor survey, not a measurement.

### Writing assistance measurably helped — but check what was actually tested

**Grade:** A, with a caveat almost all coverage drops **[agent]**

**Wiles, Munyikwa, Horton & van Inwegen** — https://arxiv.org/abs/2301.08083 · NBER w30886 · published ***Management Science* 71(12), Dec 2025**. Randomised field experiment, online labour market, **~500,000 jobseekers**. Treated jobseekers **hired 8% more often at 10% higher wages**, no evidence employers were less satisfied.

**The caveat:** the abstract specifies **"nongenerative algorithmic writing assistance"** — grammar and clarity editing, **not an LLM writing your resume.** It is routinely miscited as "ChatGPT resumes get you hired." **It is not that.** I flagged this study in the previous draft on weaker information; the correction is material and worth carrying.

### No major ATS ships AI-authorship detection

**Grade:** A (vendor primary, fetched and checked by the agent) **[agent]**

**Greenhouse "Real Talent"** (3 Jun 2025) — https://www.greenhouse.com/blog/introducing-greenhouse-real-talent — detects *"bots, fake job applicants, mass applications and patterns that suggest deception or impersonation."* **It makes no claim to detect AI-written text.** The May 2026 update adds **CLEAR identity verification** — again identity, not authorship. Greenhouse also states it *"never uses AI to rate candidates or auto-reject applications"* (corroborating §2).

**The revealed judgement is the finding:** facing an AI-application flood, the largest ATS vendor built **identity verification and spam scoring, not a text classifier.**

**Workday, Lever, iCIMS, Ashby: evidence absent.** Every claim either way traced to SEO content marketing by resume-tool companies. **Grade C.**

### Why detection isn't viable — and why this matters especially for Singapore

**Grade:** A **[agent]**

**Liang, Yuksekgonul, Mao, Wu & Zou, "GPT detectors are biased against non-native English writers," *Patterns* 4(7):100779, 2023** — https://arxiv.org/abs/2304.02819. **91 non-native TOEFL essays × 7 commercial detectors: 61.3% flagged as machine-written, vs ~5.1% for native samples.** Mechanism: detectors score on perplexity, and non-native writing has lower lexical variability.

**This compounds directly with §8.** A screening gate with ~61% false positives on non-native English writers is a disparate-impact liability, and **Singapore's WFA makes nationality and language ability protected characteristics from end-2027.** An SG employer running a detector would be building a legal problem.

**Two limits [agent]:** Liang tested 2023-era detectors on **essays**; **nobody has measured detector performance on resume-length text** — shorter and more formulaic, so false positives should go **up**. Evidence absent.

### Linguistic tells: four words have evidence, the rest is folklore

**Grade:** A for the aggregate finding; **C for the popular tell-list** **[agent]**

**Kobak, González-Márquez, Horvát & Lause**, ***Science Advances* 11(27), 2 Jul 2025** — https://www.science.org/doi/10.1126/sciadv.adt3813 · preprint https://arxiv.org/abs/2406.07016. **~15 million PubMed abstracts, 2010–2024**, excess-vocabulary method (no detector, no ground truth needed). Lower bound **≥13.5% of 2024 abstracts LLM-processed**, up to 40% in some subcorpora. Excess words: **delve, underscore, potential, findings, crucial** — stylistic verbs/adjectives, where pre-2023 excess words were content nouns. Effect exceeds Covid's impact on scientific vocabulary. Corroborated independently by https://pmejournal.org/articles/10.5334/pme.1929 (delve, underscore, primarily, **meticulous**).

**Crucial limitation:** this is an **aggregate frequency** finding. It says nothing about classifying a single document. On one 300-word cover letter, "delve" is weak evidence — **the base-rate error is severe.**

**Folklore, Grade C — no measurement behind any of these [agent]:** *tapestry, seamless, robust, **spearheaded**, showcasing*, em-dash frequency, tricolon/rule-of-three, uniform bullet length, "not only X but also Y." Every source naming them was resume-tool SEO content written by companies with a commercial interest in you believing your resume is detectable.

**This corrects my §1 note.** I earlier suggested the r/EngineeringResumes banned-verb list doubles as an LLM tell-list. **Only *delve, underscore, meticulous, crucial* have academic corroboration** — and of those, only *meticulous* appears on the wiki's list. **"Spearheaded", the canonical alleged tell, is explicitly folklore.** The wiki's blocklist is well-founded **as a style rule** and should still be implemented; my AI-tell rationale for it was wrong and I've withdrawn it.

**Implementable as:** filter *delve, underscore, meticulous, crucial* as a small evidence-based deny-list on top of the wiki's style blocklist. Do not build a detector-evasion feature — there is nothing credible to evade.

**Measurable by:** token counts. But note this is cheap insurance against a weakly-evidenced risk, not a high-value tactic.

### Recruiter surveys: mostly marketing, and they measure fraud not authorship

**Greenhouse 2025 AI in Hiring Report — Grade A-minus [agent]** — https://www.greenhouse.com/newsroom/an-ai-trust-crisis-70-of-hiring-managers-trust-ai-to-make-faster-and-better-hiring-decisions-only-8-of-job-seekers-call-it-fair — n=4,136 (2,900 seekers, 1,236 recruiters/HMs), US/UK/IE/DE. 65% of hiring managers caught AI-assisted deception: scripted AI 32%, **hidden resume prompt injections 22%**, deepfakes 18%. 41% of US seekers admit using prompt injections. *No research firm or fielding dates named — downgraded.*

**Read this carefully:** it measures **deception and fraud**, not "used ChatGPT to write my cover letter." Greenhouse's framing conflates them and downstream coverage conflates them further. **The 22% prompt-injection detection rate is independent corroboration for the §2 "do not implement" ruling** — recruiters are actively catching it.

**Resume.io "49% of AI resumes automatically dismissed" — Grade C, do not use [agent].** n=3,000 but no survey date, no panel provider, no weighting; measures stated intent not behaviour; vendor interest. Its state-level breakdown (Iowa 71% vs New Hampshire 20%) implies ~50–60 respondents per state — **noise presented as finding, which discredits the instrument.** An aggregator claim that "three independent surveys converge on 49%" could not be verified even to a second source — **treat that framing as fabricated.**

**Resume Now "62% reject AI resumes without personalization"** — https://www.resume-now.com/job-resources/careers/ai-applicant-report — n=925, fielded 28 Mar 2025. **Grade C+.** **The conditional is the whole finding:** the objection is to *unpersonalised* output, not AI use — consistent with Cui et al.

**Platform volume:** LinkedIn **~11,000 applications/minute, +45% YoY**, via NYT (Jun 2025) — **Grade A for the number, B for attribution** (no LinkedIn-published page located). Other circulating LinkedIn figures — **Grade C.** Indeed and Workday publish nothing.

### The integrity constraint still binds

Two independent screeners (jpp, lhorie) report exaggeration is **caught at phone screen** **[agent, §4]**. Combined with Cui et al., the generative pipeline's real risk is **not** "sounds like AI" — it is **generating claims the candidate cannot defend in conversation**, and **producing text whose signal value has already been arbitraged away.**

---

## RANKED: IMPLEMENT FIRST

1. **Singapore locale field-suppression profile** (§8, A). Hard-suppress photo, NRIC/FIN, DOB/age, gender, race, religion, marital status, NS liability; page budget 2–3. **This is the single highest-confidence rule set in the report** — Singapore government guidance, not preference — and it is trivially implementable and deterministically checkable.
2. **Establish an evaluator noise floor before measuring anything** (§0, A). Prerequisite, not a tactic. N≥30 identical runs; discard deltas under ~2 SD. Without this every downstream number is uninterpretable.
3. **The copy-paste extraction test** (§7, A). Textkernel's own diagnostic: most parse failures come from text conversion, not text processing. Extract from the rendered PDF and assert logical reading order. Catches column bleed, ligature corruption and header/footer loss at once.
4. **Deterministic static validators** (§1, §7, B): bullet line-count, leading past-tense verb, no pronouns, no orphan wraps, deny-list verbs/adjectives, single column, no images, conventional section headings, contact info in body, **file size <100 KB** (Taleo), full job titles.
5. **Two pages allowed + a page-1 sufficiency check** (§5, B consensus + A simulation; §8 raises this to 2–3 for SG). Five independent screeners say 2 pages is fine at 5+ YoE; lettergram adds that page 2 isn't read. Enforce that the most JD-relevant role, top 3 bullets and skills block render above the page-1 break.
6. **Surface verifiable, hard-to-fake signals over polished prose** (§9, A). Cui et al. measured tailoring's signal value **falling 51%** while verifiable-history signals rose. As everyone automates tailoring, prose quality is being arbitraged away. Prioritise shipped systems, named production impact, public artifacts, references. **This is the most strategically important finding in the report and the least obvious.**
7. **Re-rank bullets and roles by JD relevance** (§3, §6, B). Lossless, no invented content.
8. **Skills↔bullets consistency + canonical skill names** (§2, §6, B/A). Canonical naming is the one documented matching mechanism (ESCO/Skills Cloud taxonomy normalisation).
9. **Cap metric density rather than maximising it** (§1, B). Screeners report density at mid-level reads as padding. The advice industry pushes toward 100%; test 30–40%.
10. **Never generate a claim the candidate cannot defend in a phone screen** (§4, §9, B). Two independent screeners report exaggeration is caught there. Primary integrity constraint for a generative pipeline — a hard rule, not a preference.
11. **Summary gated on seniority + copy-paste test** (§3, C). Weakly supported, but cheap and mechanisable.
12. **Grammar/spelling gate, personal-details scrubber, skills-length cap, no hobbies** (§4, B).
13. **Small evidence-based LLM-vocabulary deny-list** (§9): *delve, underscore, meticulous, crucial*. Cheap insurance against a weakly-evidenced risk — implement, but don't over-invest.
14. **Run tailoring components as separate arms, and track interviews per unit effort** (§6). Nobody has decomposed tailoring into keyword-matching vs reordering vs summary-rewriting — you can. Jasmeet's result shows automation's real value is collapsing tailoring's time cost, so throughput belongs in the metric.

## RANKED: DO NOT IMPLEMENT

1. **Anything premised on "75% of resumes are auto-rejected by ATS"** (§2, C). Traces to Preptel, defunct 2013, no study, three different values. Persists because it is commercially necessary to resume-tool vendors and emotionally comfortable to rejected applicants. **Vendor documentation from Greenhouse, Taleo, Ashby and iCIMS all confirm auto-reject fires on form questions, never resume text.**
2. **White-text / invisible keyword injection and prompt injection** (§2, B against). An ATS PM states they OCR so it isn't read; independently tested against GPT-4o with zero effect; visible to any recruiter who selects-all; trips Textkernel codes 303/323. Also deceptive toward an employer — the pipeline should actively guard against emitting it.
3. **Keyword-density maximisation as an objective function** (§2). The threat it defends against is folklore; the human-side cost is unmeasured. Optimising hard against a phantom while risking a real penalty.
4. **Optimising toward a specific ATS score threshold** (§0, A against). An identical resume crosses and re-crosses an 85-point cutoff 65% of the time.
5. **Citing "31% more interviews from tailoring", "3.2x from quantification (HBR)", or "40% (Resume Worded)"** (§1, §6, C). All three traced to nothing. The HBR figure does not exist on hbr.org at all. Tailor and quantify anyway — but don't encode the numbers.
6. **Any "2026 data" PDF-vs-DOCX statistic** (§7, C). Every circulating figure traces to AI-generated SEO sites naming no ATS, sample or date. No credible controlled study exists.
7. **Avoiding bold/italics/bullets, or restricting to Arial/Times New Roman** (§7, C). Oracle states parsing is unaffected by styling; no vendor anywhere specifies fonts.
8. **Blanket table avoidance** (§7). Greenhouse names tables only within "complex resumes with tables, headers, and footers." Overstated in popular advice.
9. **Treating labour-economics audit studies as evidence for tailoring** (§6). They measure demographic and gap effects. Grade A for their findings, C for this use.
10. **Trusting the r/EngineeringResumes success-story archive as evidence its advice works** (C). ~60 uncontrolled testimonials, total survivorship bias.
11. **A "Personal Particulars" block, photo, or salary expectations on a Singapore CV** (§8, C). No SG recruiter source recommends it; TAFEP actively lists most of those fields for removal. The convention is real in **application forms**, and the leap to CVs is SEO folklore.
12. **Any AI-detector-evasion feature** (§9). No major ATS ships authorship detection — Greenhouse built identity verification and spam scoring instead. Detectors that do exist flag non-native English writers at **61.3%**, so they are legally radioactive, especially under Singapore's WFA. There is nothing credible to evade.
13. **Citing "49% of AI resumes are automatically dismissed" (Resume.io)** (§9, C). No survey date, no panel provider, no weighting; state-level cells of ~50 respondents presented as findings; vendor interest. The claim that three surveys converge on it could not be verified to even a second source.
14. **Citing NBER w30886 as "ChatGPT resumes get you hired"** (§9). The study tested **nongenerative** grammar/clarity assistance. Widely miscited, including in my own earlier draft.

---

## CONFIDENCE AND GAPS

### Reddit was unfetchable

reddit.com, old.reddit.com and a jina.ai proxy all returned access errors, for me and for both subagents (WebSearch also rejects `allowed_domains: [reddit.com]` on a user-agent block).

**All r/EngineeringResumes content here comes from the official GitHub mirror** (https://github.com/r-engineeringresumes/subreddit-wiki), which I cloned and read from source files **[verified]**. One subagent independently retrieved the same wiki via raw.githubusercontent and its quotes match mine — genuine independent corroboration on the wiki specifically.

**But no individual Reddit threads were read by anyone.** So there is **no r/recruiting content, no r/cscareerquestions verified-recruiter answers, and no agency-recruiter voice** in this report at all. The practitioner consensus in §4 and §5 is **entirely Hacker News**, which skews toward startup/product engineering hiring managers and away from agency and enterprise recruiters. That is a real population bias in the strongest finding of the report (§5, page count).

### The confabulation problem

One subagent reported that the search layer **returned fluent, correctly-attributed, entirely fabricated content on at least three occasions** — including a plausible false version of the r/EngineeringResumes wiki position, and a tidy 19-row "red flags" table whose HN comment IDs resolved to unrelated discussions when checked individually. It discarded these and re-verified by direct fetch.

**I cannot audit that agent's verification.** I did spot-check that the wiki quotes it reported match my own clone, which is reassuring. But every HN quote in §1, §3, §4, §5 and §6 is **[agent]**-sourced and **I did not re-open a single one of those threads.** If you are going to act on a specific quote, re-open its URL first. This is the largest verification debt in the report.

### Claims resting on search snippets only, never an opened primary

- **Chip Huyen's positions (§1).** Her huyenchip.com post 404'd; LinkedIn not fetchable. Confident about the general thrust because it recurred across independent snippets and matches her book's structure, but **I have not read her words in context and have verified no quote.** This matters because she is my only AI/ML-domain-specific source.
- **Mike Peditto / Teal copy-paste heuristic (§3)** — snippet only.
- **Harvard OCS specifics (§1, §5).** PDF downloaded but couldn't render (no poppler). **The only thing I verified is its title.** The "Harvard rules" in my search results came largely from a vendor blog I don't trust and haven't repeated as fact.

### Gradings I am least sure about

- **§5 ResumeGo page-length study, graded A.** Shakiest A. Real disclosed methodology and numbers, which clears the bar — but a forced-choice **simulation**, not field data, run by a company profiting from longer resumes. A reasonable reviewer would grade it B. It now matters less than it did, because the five-screener consensus points the same way from an independent direction.
- **§0 HackerRank variance, graded A.** Clean method, fully documented, but n=1 investigator, unreplicated, personal blog. Graded A on the experiment's quality. My **extrapolation** to commercial screeners is explicitly not Grade A and is labelled inference.
- **§1 XYZ.** I originally graded B on Bock's credibility; the subagent argued C; **I changed it to C.** I think C is right for the claim in circulation ("XYZ works better") but a case exists for B on the weaker claim ("deliberate structure helps"). I split it accordingly, which is a judgement call.
- **§1 Chip Huyen, graded B** on credibility and snippet consistency, not on a source I read. If the snippets mischaracterised her, the grade is wrong.
- **§2 Gergely Orosz, graded B.** He interviewed named-company recruiters, which is strong, but he also sells a resume book.
- **§7 Textkernel PDF-as-Major-Issue.** The subagent reported retrieving this from a page summary but **could not pull the exact code number or wording.** It is doing real work in my PDF-vs-DOCX conclusion and is the weakest link in §7.

### Reported by a subagent, not independently verified by me

To be explicit, since you asked: **essentially all of §2, §4, §6, §7, §8 and §9, plus the screener quotes in §1, §3 and §5.** Specifically —

**Singapore (§8) — 100% agent-sourced, none re-fetched by me.** All TAFEP/TGFEP/PDPC/MOM/AGC citations, the verbatim PDPC example 3.9, the WFA end-2027 date and the asanify correction, all COMPASS point values and EP salary thresholds, and the Michael Page/Randstad/JobStreet length guidance. Mitigating factor: these are government primary sources, which are less likely to be confabulated than forum quotes and are easy for you to spot-check at the URLs given. Aggravating factor: **the specific numbers (SGD 22,500 exemption, 40-point threshold, salary bands, the 2027 revisions) are exactly the kind of detail that degrades silently.** Verify before encoding any threshold.

**AI detection (§9) — 100% agent-sourced.** Both arXiv papers and their figures, the *Science Advances* and *Patterns* studies, the Greenhouse Real Talent pages and survey, Resume.io and Resume Now. The Cui et al. paper (arXiv:2509.25054) is doing the most work of anything in the report — it drives implement-first item #6 — and **I have not opened it.** If you act on one thing from §9, open that paper first.

**Everything else previously listed —**

- All ATS vendor documentation quotes (Oracle/Taleo, Greenhouse, Ashby, iCIMS, Workday, SmartRecruiters, Textkernel), including the 100 KB Taleo cap and 2.5 MB Greenhouse limit
- All Textkernel measured figures (15% column usage, 62%→90% render rate, 12,000-CV sample, 22.5s timeout, quality codes 132/303/323/408/412/441)
- The Alibaba arXiv:2510.09722 figures (~20% multi-column, F1 0.919→0.959, 0.136→0.846)
- Every Hacker News quote and username in the report
- ResumeGo cover-letter study details (7,287 applications, +53%, generic≈nothing)
- Huntr (461k applications, +115%), Jasmeet (n=50, 12% vs 8%)
- NBER w30886 (+8% hired), AEJ reference letters (+60%), AER 2016 attention discrimination, Enhancv N=25
- The negative traces: that hbr.org contains no "3.2x" study; that ResumeGo's index has no tailoring study; that TheLadders 2018 never disclosed sample size

I verified independently: the HackerRank variance post, Bock's 2014 LinkedIn post, the full r/EngineeringResumes wiki (cloned), ResumeGo's page-length methodology page and research index, Orosz's ATS-myths sample, HN thread 40489596, the Preptel trace, and Harvard OCS's title.

### Remaining evidence gaps, stated plainly

- **No Reddit or forum data anywhere in the report**, for any topic. r/askSingapore, r/singaporefi, r/recruiting, r/recruitinghell all hard-blocked. SG salary-expectation sentiment and recruiter tell-spotting threads are genuinely unfulfilled.
- **Robert Walters SG, Hays SG, NodeFlair, Glints and Tech in Asia all returned 403.** NodeFlair is the loss that matters most — the only SG-tech-specific source on the brief.
- **Singapore's legal position on salary-history questions: not verified.** Do not ship as fact.
- **Detector performance on resume-length text: never measured by anyone.** Liang tested essays. Shorter, more formulaic text should push false positives up, but that is inference.
- **No audit study of human- vs LLM-written resumes to real corporate employers exists.** Both Grade A papers study labour marketplaces. This is the central gap in §9 and nobody has closed it.
- **No study decomposes tailoring** into keyword-matching vs reordering vs summary-rewriting (§6).
- **No experiment isolates quantification** (§1).
- **No screener data on bullets-per-role or within-role bullet order** (§1).
- **Nothing distinguishes AI/ML resume screening from general SWE screening** in any accessible source — Chip Huyen is my only domain-specific voice and I verified none of her words.

### Other constraints

The 200-call web-search budget was exhausted across all streams. Workday/Greenhouse parser docs are login-gated; Lever's help centre threw a persistent CSS error. All three research streams did eventually return — the Singapore/AI-detection stream landed after the report was first written, and §8, §9, both ranked lists and the §1 verb-list rationale were revised to incorporate it. **One earlier claim of mine was falsified in that revision** (the banned-verb list as LLM tell-list) and is marked as withdrawn in §1 rather than quietly deleted.
