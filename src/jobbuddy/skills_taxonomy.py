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
    r"^.{0,2}$",           # one or two characters is never a usable skill term
    r"^\W+$",
)
_NOISE_RE = tuple(re.compile(p, re.I) for p in _NOISE_PATTERNS)

_PUNCT_RE = re.compile(r"[^a-z0-9+#]+")
_STOPWORDS = frozenset({
    "and", "or", "the", "a", "an", "of", "for", "with", "in", "on", "to",
    "using", "based", "related", "various", "other", "etc",
})


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


def is_noise(term: Any) -> bool:
    """True when a term is extraction garbage rather than a real skill."""
    text = str(term or "").strip()
    if not text:
        return True
    return any(pattern.search(text) for pattern in _NOISE_RE)


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
