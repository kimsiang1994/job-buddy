"""Token estimation and per-task budgeting.

Two knobs matter, and `thinking` is the bigger one:

  * thinking disabled -> no reasoning tokens at all, so tiny budgets are fine.
  * thinking enabled  -> reasoning is billed inside completion_tokens and is
    produced BEFORE the answer, so an under-sized max_tokens returns an empty
    reply with finish_reason == "length".

Token counting has a pluggable backend: DeepSeek's official tokenizer when it has
been fetched, otherwise the documented char-ratio heuristic. The heuristic keeps
this module usable with zero dependencies.
"""

import json
import os
import sys

from jobbuddy.deepseek import model_config

# src/jobbuddy/deepseek/<file> -> four levels up is the repo root.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PROFILES_PATH = os.path.join(REPO_DIR, "config", "task_profiles.json")
TOKENIZER_JSON = os.path.join(REPO_DIR, "tokenizer", "tokenizer.json")

# DeepSeek's published offline ratios.
TOKENS_PER_CHAR_EN = 0.3
TOKENS_PER_CHAR_CJK = 0.6

DEFAULT_PROFILE = "classify"

# Used only when a model has no declared limits in models.json.
FALLBACK_MAX_OUTPUT = 8192
FALLBACK_CONTEXT = 65536

_warned = set()
_backend = {"loaded": False, "encode": None, "name": "heuristic"}
_profiles_cache = {"loaded": False, "data": None}


def _warn(message):
    if message in _warned:
        return
    _warned.add(message)
    print(f"[token_budget] {message}", file=sys.stderr)


# --------------------------------------------------------------------------
# token counting
# --------------------------------------------------------------------------

def _cjk_share(text):
    if not text:
        return 0.0
    cjk = sum(1 for ch in text
              if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"
              or "぀" <= ch <= "ヿ")
    return cjk / len(text)


def _heuristic_tokens(text):
    """DeepSeek's documented ratios, blended by how much CJK the text contains."""
    if not text:
        return 0
    share = _cjk_share(text)
    per_char = TOKENS_PER_CHAR_EN * (1 - share) + TOKENS_PER_CHAR_CJK * share
    return max(1, int(len(text) * per_char + 0.5))


def _backend_load():
    if _backend["loaded"]:
        return _backend
    _backend["loaded"] = True
    if not os.path.exists(TOKENIZER_JSON):
        return _backend
    try:
        from tokenizers import Tokenizer  # optional dependency
        tokenizer = Tokenizer.from_file(TOKENIZER_JSON)
        _backend["encode"] = lambda text: len(tokenizer.encode(text).ids)
        _backend["name"] = "official"
    except ImportError as err:
        # Report the real exception -- "not installed" is only one possible cause
        # (a broken wheel or a shadowing local file look identical otherwise).
        _warn(f"tokenizer/tokenizer.json found but `from tokenizers import "
              f"Tokenizer` failed ({err.__class__.__name__}: {err}) - falling "
              "back to the char-ratio heuristic. If the package is missing: "
              "pip install -r requirements.txt")
    except Exception as err:                      # noqa: BLE001 - never fatal
        _warn(f"could not load the official tokenizer ({err}) - using heuristic")
    return _backend


def backend_name():
    """'official' or 'heuristic'. Useful for reporting estimator accuracy."""
    return _backend_load()["name"]


def estimate_tokens(text):
    """Best available token count for `text`. Never raises."""
    if not text:
        return 0
    backend = _backend_load()
    if backend["encode"]:
        try:
            return backend["encode"](text)
        except Exception as err:                  # noqa: BLE001
            _warn(f"tokenizer failed ({err}) - using heuristic for this call")
    return _heuristic_tokens(text)


def estimate_messages(messages):
    """Rough prompt size for a chat payload, including light per-message overhead."""
    total = 0
    for message in messages or []:
        total += estimate_tokens(message.get("content") or "")
        total += 4  # role/delimiter overhead, approximately
    return total


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------

def load_profiles():
    if _profiles_cache["loaded"]:
        return _profiles_cache["data"]
    _profiles_cache["loaded"] = True
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        _profiles_cache["data"] = data.get("profiles") or {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError) as err:
        _warn(f"task_profiles.json unusable ({err}) - using built-in defaults")
        _profiles_cache["data"] = {}
    return _profiles_cache["data"]


def reload():
    _profiles_cache["loaded"] = False
    _profiles_cache["data"] = None


def profile_names():
    return sorted(load_profiles())


def thinking_payload(profile):
    """-> the `thinking` request object for a profile, or None to omit it."""
    mode = profile.get("thinking")
    if mode not in ("enabled", "disabled"):
        return None
    payload = {"type": mode}
    effort = profile.get("reasoning_effort")
    if mode == "enabled" and effort:
        payload["reasoning_effort"] = effort
    return payload


def budget_for(profile_name=DEFAULT_PROFILE, messages=None, model=None, tier="fast"):
    """Resolve a task profile into concrete request settings.

    -> {profile, model, max_tokens, thinking, estimated_prompt_tokens, clamped}
    """
    profiles = load_profiles()
    profile = profiles.get(profile_name)
    if profile is None:
        _warn(f"unknown profile {profile_name!r} - falling back to {DEFAULT_PROFILE!r}")
        profile = profiles.get(DEFAULT_PROFILE) or {
            "thinking": "disabled", "max_tokens": 512}
        profile_name = DEFAULT_PROFILE

    model = model or model_config.resolve(tier)
    info = model_config.model_info(model) or {}
    max_output = info.get("max_output_tokens") or FALLBACK_MAX_OUTPUT
    context = info.get("context_window") or FALLBACK_CONTEXT

    prompt_tokens = estimate_messages(messages)
    wanted = int(profile.get("max_tokens") or 512)

    # Clamp to the model's own output cap and to whatever context is left.
    ceiling = min(max_output, max(1, context - prompt_tokens))
    granted = max(1, min(wanted, ceiling))

    return {
        "profile": profile_name,
        "model": model,
        "max_tokens": granted,
        "thinking": thinking_payload(profile),
        "estimated_prompt_tokens": prompt_tokens,
        "clamped": granted < wanted,
    }


if __name__ == "__main__":
    print(f"backend: {backend_name()}")
    sample = "Summarise this job description in one sentence."
    print(f"estimate_tokens({sample!r}) = {estimate_tokens(sample)}")
    for name in profile_names():
        plan = budget_for(name, [{"role": "user", "content": sample}])
        print(f"  {name:10} max_tokens={plan['max_tokens']:<6} "
              f"thinking={plan['thinking']}")
