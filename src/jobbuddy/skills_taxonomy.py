"""Skill term normalisation and matching. Offline, deterministic, no LLM.

Job boards do not share a vocabulary. MyCareersFuture runs machine extraction
over the job description, which produces a mix of three things:

  real skills        'Machine Learning', 'PyTorch', 'AWS'
  odd casings/forms  'Ai', 'Fine Tuning', 'LLMs', 'Docker Container'
  extraction noise   'Ship Building', 'scientific discipline',
                     'developed software systems', 'technological knowledge'

Naive string matching fails all three ways: 'Ai' never equals 'artificial
intelligence', 'Fine Tuning' never equals 'fine-tuning', and the noise inflates
the denominator so a perfect candidate scores near zero. That is not a
hypothetical -- an AI engineering role scored 0/16 against an AI engineer's
profile before this module existed.

So matching happens in four passes, cheapest first: exact, alias, containment,
token overlap. And noise terms are dropped before scoring rather than counted
as skills the candidate lacks.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Canonical form -> every surface form that means it. Written as
# alias -> canonical at import time. Curated deliberately: a generated synonym
# list would introduce false matches, and a false skill match is a lie on a CV.
_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "artificial intelligence": ("ai", "a i", "artificial intelligence application",
                                "ai models", "ai agents", "intelligent reasoning",
                                "ai application", "applied ai"),
    "machine learning": ("ml", "machine learning engineering", "applied machine learning",
                         "statistical learning", "supervised learning",
                         "unsupervised learning", "predictive modelling",
                         "predictive modeling"),
    "deep learning": ("dl", "neural networks", "neural network",
                      "representation learning"),
    "large language model": ("llm", "llms", "language model", "language models",
                             "large language models", "foundation model",
                             "foundation models"),
    "generative ai": ("genai", "gen ai", "generative artificial intelligence",
                      "generative ai application development and deployment",
                      "generative ai application development"),
    "retrieval augmented generation": ("rag", "retrieval-augmented generation"),
    "prompt engineering": ("prompting", "prompt design"),
    "fine tuning": ("finetuning", "fine-tuning", "model fine tuning",
                    "supervised fine tuning", "sft"),
    "natural language processing": ("nlp", "text mining", "text analytics"),
    "computer vision": ("cv", "image processing", "image recognition"),
    "reinforcement learning": ("rl", "rlhf",
                               "reinforcement learning from human feedback"),
    "model deployment": ("model serving", "mlops", "ml ops", "ml engineering",
                         "model operationalisation", "model operationalization"),
    "vector database": ("vector db", "vector store", "embeddings",
                        "embedding", "vector search", "semantic search"),
    "python": ("python programming", "python3"),
    "sql": ("structured query language", "t-sql", "pl/sql"),
    "pyspark": ("spark", "apache spark", "spark sql"),
    "airflow": ("apache airflow", "workflow orchestration"),
    "aws": ("amazon web services", "amazon aws"),
    "gcp": ("google cloud platform", "google cloud"),
    "azure": ("microsoft azure",),
    "kubernetes": ("k8s", "kubernetes cluster"),
    "docker": ("docker container", "containerisation", "containerization",
               "containers"),
    "etl": ("etl pipeline", "etl pipelines", "extract transform load",
            "data pipeline", "data pipelines", "elt"),
    "data warehousing": ("data warehouse", "dwh", "data warehouse design"),
    "data engineering": ("data engineer", "data pipelines", "pipelines"),
    # Added after the frequency filter removed these as "too common". They are
    # common because this field genuinely wants them, not because they are
    # filler -- and once a real skill leaves the vocabulary, a job asking for it
    # stops counting as a match the candidate actually has.
    "data science": ("data scientist", "data analytics", "analytics"),
    "security": ("application security", "secure coding", "infosec",
                 "security engineering"),
    "scalability": ("scalable systems", "distributed systems", "high availability"),
    "system design": ("architecture", "solution architecture", "technical design"),
    "data governance": ("governance", "data quality", "data compliance",
                        "data stewardship"),
    "statistics": ("statistical analysis", "statistical modelling",
                   "statistical modeling", "stochastic"),
    "bayesian modelling": ("bayesian", "bayesian statistics", "bayesian inference",
                           "bayesian modeling"),
    "experimentation": ("ab testing", "a/b testing", "causal inference",
                        "incrementality"),
    "api": ("apis", "rest api", "rest apis", "restful api", "api development"),
    "huggingface": ("hugging face", "transformers", "huggingface transformers"),
    "pytorch": ("torch",),
    "tensorflow": ("tf", "keras"),
    "stakeholder management": ("stakeholder engagement", "cross functional",
                               "cross functional project management",
                               "cross-functional collaboration"),
    "software engineering": ("software development", "programming",
                             "software development tools", "coding",
                             "software verification", "developed software systems"),
}

# Built once: surface form -> canonical.
ALIASES: dict[str, str] = {}
for _canonical, _forms in _ALIAS_GROUPS.items():
    ALIASES[_canonical] = _canonical
    for _form in _forms:
        ALIASES[_form] = _canonical

# Terms that are not skills. MCF's extractor emits these regularly; counting
# them as requirements the candidate fails to meet is simply wrong, and it
# drags every score toward zero.
_NOISE_PATTERNS = (
    r"^(scientific|technological|technical)\s+(discipline|knowledge|skills?)$",
    r"^(developed|proven|strong|excellent|good|relevant)\b",
    r"^(computer|information)\s+(systems?|science|engineering)$",
    r"^(ship building|manufacturing workflow management|smart manufacturing)$",
    r"^(online training|theoretical models|logical reasoning)$",
    r"^(testing practices|reliability improvement|end-to-end design)$",
    r"^(product and production engineering|supply and transportation)$",
    r"^(ecosystems|open source technology|distributed platforms)$",
    r"^(scalable applications|real-time production|decision systems)$",
    r"^(data-driven decision making|structured data analysis)$",
    r"^(web application deployment|simulation software|simulation)$",
    r"^(architect|digital design|anomaly detection)$",
    r"\b(mindset|attitude|ability|willingness|passion|team player)\b",

    # Soft attributes. Real qualities, but not things a resume can be matched
    # on -- and MCF's extractor emits them constantly, so every candidate
    # "lacked" them and every score was dragged down by words no job is
    # actually about.
    r"^(collaboration|communication|teamwork|interpersonal( skills)?|"
    r"analytical( skills| thinking)?|problem[- ]solving|critical thinking|"
    r"attention to detail|adaptability|flexibility|creativity|initiative|"
    r"leadership|self[- ]motivat\w*|time management|organisational skills|"
    r"organizational skills|work ethic|multitasking|fast[- ]paced|"
    r"detail[- ]oriented|proactive|independent|resourceful)$",

    # Generic activity nouns that name no capability on their own.
    r"^(running|onboarding|efficiency|innovation|excellence|quality|"
    r"delivery|execution|planning|coordination|documentation|reporting|"
    r"maintenance|troubleshooting|support|training|mentoring|"
    r"stakeholders?|customers?|clients?|users?|teams?|projects?)$",

    # Fragments and acronyms too short or ambiguous to mean anything. 'COM'
    # was matching as a skill; it is the tail of a domain name.
    r"^(com|net|org|inc|ltd|pte|www|http|api\d|the|and|for|with)$",

    r"^.{0,2}$",           # one or two characters is never a usable skill term
    r"^\W+$",
)
_NOISE_RE = tuple(re.compile(p, re.I) for p in _NOISE_PATTERNS)

# --------------------------------------------------------------------------
# Compound advert phrases.
#
# The patterns above are anchored against single generic words, so every
# multi-word advert phrase walked straight past them: 'Liaising with cross
# functional teams' (126 postings), 'Data Solutions', 'Analytical and
# Problem-Solving Skills', 'Extensive Work Experience'. Measured on the real
# 331-job corpus, the top 'missing skills' for the largest cluster were about
# half phrases of this shape, sitting at the same document frequency as AI
# Evaluation and CI/CD -- so the development plan could not be read.
#
# The distinction being encoded: a SKILL is a thing you can learn, name and be
# tested on ('Kubernetes', 'Bayesian modelling', 'CI/CD'). An ADVERT PHRASE
# describes a behaviour, a soft quality, or a generic activity ('liaising with
# stakeholders', 'attention to detail', 'design' as a bare noun).
#
# Every rule below is a curated list rather than a productive heuristic,
# because the error directions are not symmetric -- see `is_advert_phrase`.
# --------------------------------------------------------------------------

# 1. Verb- and gerund-led phrases. The head is an ACTION the post-holder
#    performs, not a capability they hold. Matched on the FIRST word only, so
#    'Model Training' and 'Data Wrangling' are untouched.
#
#    'design' and 'test' are deliberately absent: 'Design Patterns' and 'Test
#    Automation' are real, and the bare nouns are handled separately.
_ACTIVITY_VERBS = (
    "liais(?:e|es|ed|ing)", "collaborat(?:e|es|ed|ing)", "engag(?:e|es|ed|ing)",
    "communicat(?:e|es|ed|ing)", "interfac(?:e|es|ed|ing)", "partner(?:s|ed|ing)?",
    "coordinat(?:e|es|ed|ing)", "manag(?:e|es|ed|ing)", "work(?:s|ed|ing)",
    "build(?:s|ing)?", "creat(?:e|es|ed|ing)", "gather(?:s|ed|ing)?",
    "driv(?:e|es|ing)", "improv(?:e|es|ed|ing)", "interpret(?:s|ed|ing)?",
    "meet(?:s|ing)?", "optimis(?:e|es|ed|ing)", "optimiz(?:e|es|ed|ing)",
    "resolv(?:e|es|ed|ing)", "solv(?:e|es|ed|ing)", "translat(?:e|es|ed|ing)",
    "provid(?:e|es|ed|ing)", "deliver(?:s|ed|ing)?", "implement(?:s|ed|ing)?",
    "lead(?:s|ing)?", "assist(?:s|ed|ing)?", "ensur(?:e|es|ed|ing)",
    "perform(?:s|ed|ing)?", "conduct(?:s|ed|ing)?", "participat(?:e|es|ed|ing)",
    "contribut(?:e|es|ed|ing)", "articulat(?:e|es|ed|ing)",
)
_ACTIVITY_LED_RE = re.compile(
    r"^(?:" + "|".join(_ACTIVITY_VERBS) + r")\s+\S", re.I)

# 2. Soft-quality vocabulary. Anywhere in the phrase, these mark a disposition
#    rather than a skill. The single-word forms are already covered above; this
#    is what catches them inside a compound.
_SOFT_QUALITY_RE = re.compile(
    r"\b(?:attention to detail|interpersonal|teamwork|team player|"
    r"soft skills?|work ethic|track record|fast[- ]paced|enthusiasm|"
    r"proficiency|aptitude|multitask\w*|dependability|accountability|"
    r"self[- ]starter|can[- ]do|hands[- ]on|eagerness|willing\w*|"
    r"ambiguity)\b", re.I)
# 'integrity', 'autonomy' and 'diligence' were tried here and removed: they
# cost 'Data Integrity' and 'Autonomy iManage', both real, and bought nothing
# the bare-noun list below does not already catch.

# 3. '<adjective> skills' -- the adjective never makes it nameable.
_SOFT_SKILLS_RE = re.compile(
    r"\b(?:soft|presentation|communication|interpersonal|analytical|people|"
    r"organisational|organizational|mathematical|technical|computer|"
    r"leadership|problem[- ]solving|writing|listening|numerical|"
    r"developing|transferable)\s+skills?\b|\bskillset\b", re.I)

# 4. Filler HEAD nouns. A phrase whose final word is one of these names a
#    quantity of something rather than the something: 'Data Solutions',
#    'Solution Delivery', 'Technology Expertise', 'Industry Experience'.
#    Only fires when the phrase carries no technical anchor (see below), so
#    'Enterprise Security Solutions' survives on the strength of 'security'.
_FILLER_HEADS = frozenset({
    "solution", "solutions", "delivery", "capability", "capabilities",
    "experience", "knowledge", "expertise", "proficiency", "approach",
    "environment", "environments", "mindset", "culture", "awareness",
    "exposure", "understanding", "familiarity", "background",
})

# 5. '<something> of <something>' constructions: 'knowledge of data
#    networking', 'analysis of business problems'. The leading noun is the
#    filler; whatever follows is a topic, not a named skill.
_OF_PHRASE_RE = re.compile(
    r"^(?:knowledge|understanding|awareness|appreciation|familiarity|"
    r"experience|exposure|analysis|application)\s+of\b", re.I)

# 6. Bare abstract nouns. Curated ONE BY ONE rather than derived, because the
#    productive version of this rule is what deleted Python and AWS from an
#    earlier build. Each entry is a word that names no capability standing
#    alone, and each was read off the real corpus before being added.
#
#    Notably ABSENT, and left in on purpose: 'debugging', 'monitoring',
#    'orchestration', 'optimization', 'classification', 'segmentation',
#    'reproducibility', 'traceability', 'vision', 'writing', 'research',
#    'testing', 'deployment', 'evaluation', 'validation'. Every one of those
#    is arguable, and arguable means keep.
_BARE_ABSTRACT_NOUNS = frozenset({
    "design", "development", "management", "operations", "adoption",
    "solutions", "solution", "budget", "certifications", "certification",
    "conferences", "journals", "internships", "education", "tenure",
    "registration", "diversity", "lifestyle", "autonomy", "accountability",
    "enthusiasm", "professionalism", "seniority", "responsibilities",
})

# A phrase this long has stopped naming a skill and started describing a job.
# Counted in MEANINGFUL tokens and only applied to a term with no technical
# anchor, which is what keeps 'Software as a Service (SaaS)' and 'BIRT
# (Business Intelligence and Reporting Tools)' -- both real, both long. Set at
# six rather than the five originally tried for exactly that reason: at five it
# was eating product names, and this rule is a backstop, not a workhorse.
_TOO_MANY_WORDS = 6

# Taxonomy canonicals that are ALSO bare abstract nouns.
#
# These are the one place the protect-known-terms rule is deliberately
# overridden, and only for the bare single-word surface form. 'Scalability'
# alone is advert copy -- nobody can act on "learn Scalability" -- but
# 'Scalable Systems', 'Distributed Systems' and 'High Availability' all
# canonicalise to it and ARE nameable, so they are matched on their surface
# form and survive. Keep this set tiny and keep it explicit.
_ABSTRACT_TAXONOMY_TERMS = frozenset({"scalability"})

_PUNCT_RE = re.compile(r"[^a-z0-9+#]+")
_STOPWORDS = frozenset({
    "and", "or", "the", "a", "an", "of", "for", "with", "in", "on", "to",
    "using", "based", "related", "various", "other", "etc",
})


def surface(term: Any) -> str:
    """Normalised text of a term WITHOUT alias substitution.

    `canon` folds 'High Availability' onto 'scalability', which is right for
    matching and wrong for judging the shape of what the board actually wrote.
    Anything reasoning about the words in front of it wants this.
    """
    text = _PUNCT_RE.sub(" ", str(term or "").lower().strip())
    return re.sub(r"\s+", " ", text).strip()


def canon(term: Any) -> str:
    """Canonical form of a skill term: lowercase, depunctuated, de-aliased.

    'Fine-Tuning', 'fine tuning' and 'SFT' all collapse to 'fine tuning'.
    """
    text = str(term or "").lower().strip()
    # Keep + and # so 'c++' and 'c#' survive; everything else becomes a space.
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return ALIASES.get(text, text)


def has_technical_anchor(term: Any) -> bool:
    """True when some run of words in the term is a known taxonomy entry.

    Checked over contiguous n-grams, longest first, so 'Enterprise Security
    Solutions' anchors on 'security' and 'Generative AI Application
    Development and Deployment in Financial Services' anchors on 'generative
    ai'. A term with an anchor is naming something real even when it is
    wrapped in advert packaging, so the shape-based rules stand down.

    Deliberately NOT the same as "every word appears somewhere in the
    taxonomy": 'data' appears inside 'data engineering', which would have
    anchored 'Data Solutions' and defeated the whole filter. Only whole
    taxonomy keys count.
    """
    words = canon(term).split()
    for size in range(len(words), 0, -1):
        for start in range(len(words) - size + 1):
            if " ".join(words[start:start + size]) in ALIASES:
                return True
    return False


def is_advert_phrase(term: Any) -> bool:
    """True when a term is job-advert copy rather than a nameable skill.

    A skill is a thing you can learn, name and be tested on -- 'Kubernetes',
    'Bayesian modelling', 'CI/CD'. An advert phrase describes a behaviour, a
    soft quality, or a generic activity -- 'Liaising with cross functional
    teams', 'attention to detail', 'Design' as a bare noun.

    BIASED TOWARD KEEPING, on purpose, and the bias is not symmetric. A term
    wrongly dropped disappears from the denominator, so the candidate's
    coverage reads HIGHER than it is and a genuine gap silently vanishes from
    the development plan -- a wrong answer nobody can see. A term wrongly kept
    is merely a piece of visible clutter in a list. So anything arguable stays,
    every rule here is a curated list rather than a productive heuristic, and
    anything carrying a known taxonomy term is exempt from all of them but one
    (see `_ABSTRACT_TAXONOMY_TERMS`).
    """
    canonical = canon(term)
    if not canonical:
        return True

    words = canonical.split()

    # The single documented override of taxonomy protection: a bare abstract
    # noun that happens to head an alias group. Tested on the SURFACE form, so
    # 'Scalability' goes and 'High Availability' -- which canonicalises onto it
    # -- stays.
    if surface(term) in _ABSTRACT_TAXONOMY_TERMS:
        return True

    # Protection. A known taxonomy term is a skill by construction, whatever
    # shape it arrives in. An earlier filter dropped Python, AWS and API on
    # frequency alone; this is the guard against repeating that.
    if canonical in ALIASES:
        return False

    if canonical in _BARE_ABSTRACT_NOUNS:
        return True

    if _SOFT_QUALITY_RE.search(canonical) or _SOFT_SKILLS_RE.search(canonical):
        return True

    if len(words) > 1:
        if _ACTIVITY_LED_RE.search(canonical) or _OF_PHRASE_RE.search(canonical):
            return True

    anchored = has_technical_anchor(canonical)
    if anchored:
        return False

    if len(words) > 1 and words[-1] in _FILLER_HEADS:
        return True

    return len(tokens(canonical)) >= _TOO_MANY_WORDS


def is_noise(term: Any) -> bool:
    """True when a term is extraction garbage rather than a real skill."""
    text = str(term or "").strip()
    if not text:
        return True
    if any(pattern.search(text) for pattern in _NOISE_RE):
        return True
    return is_advert_phrase(text)


def tokens(term: str) -> frozenset[str]:
    """Meaningful tokens of a canonical term."""
    return frozenset(
        t for t in canon(term).split()
        if t and t not in _STOPWORDS and len(t) > 1
    )


def match(job_term: str, owned: dict[str, float]) -> tuple[float, str, str]:
    """Match one job skill against owned skills.

    `owned` maps canonical skill -> proficiency weight.
    Returns (weight, matched_skill, how). weight is 0.0 when nothing matched.

    Four passes, strongest evidence first, so a weak token overlap can never
    outrank an exact hit.
    """
    target = canon(job_term)
    if not target:
        return 0.0, "", "none"

    # 1. exact, after canonicalisation
    if target in owned:
        return owned[target], target, "exact"

    target_tokens = tokens(target)
    if not target_tokens:
        return 0.0, "", "none"

    best = (0.0, "", "none")

    for skill, weight in owned.items():
        skill_tokens = tokens(skill)
        if not skill_tokens:
            continue

        # 2. containment: the owned skill's words all appear in the job term.
        #    'generative ai' inside 'generative ai application development'.
        if skill_tokens <= target_tokens:
            candidate = (weight * 0.95, skill, "contains")
            if candidate[0] > best[0]:
                best = candidate
            continue

        # 3. reverse containment: job term is narrower than the owned skill.
        if target_tokens <= skill_tokens:
            candidate = (weight * 0.85, skill, "narrower")
            if candidate[0] > best[0]:
                best = candidate

    return best

# There is deliberately no fuzzy "token overlap" pass. It was tried and removed:
# with a 50% threshold, any single shared head-noun produced a match --
# 'Model Deployment' matched 'large language model', 'Agentic Memory Management'
# matched 'stakeholder management', 'Data Science' matched 'data engineering'.
#
# A false skill match is strictly worse than a missed one. It inflates the fit
# score, and downstream it could justify a resume bullet asserting a skill the
# candidate does not have -- the exact failure the whole design exists to
# prevent. When a real synonym is missing, add it to _ALIAS_GROUPS, where the
# decision is explicit and reviewable.


def build_owned(skill_tiers: dict[str, Iterable[str]],
                tier_weights: dict[str, float]) -> dict[str, float]:
    """Flatten a tiered profile skill list into canonical skill -> weight.

    Highest tier wins when a skill appears twice.
    """
    owned: dict[str, float] = {}
    for tier, weight in tier_weights.items():
        for skill in skill_tiers.get(tier, []) or []:
            key = canon(skill)
            if key and weight > owned.get(key, 0.0):
                owned[key] = weight
    return owned


def clean_job_skills(terms: Iterable[Any]) -> list[str]:
    """Drop noise and duplicates from a board's skill list, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for term in terms or []:
        text = str(term or "").strip()
        if not text or is_noise(text):
            continue
        key = canon(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
