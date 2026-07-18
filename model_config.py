"""Runtime resolver: capability tier -> concrete DeepSeek model id.

Import this from application code:

    import model_config
    model = model_config.resolve("fast")

Guarantees: never raises, never writes, never touches the network. If models.json
is missing, stale or corrupt it degrades down a fallback chain ending in a
hardcoded constant, so a config problem can never take the app down.

This module also owns the tier-selection heuristic (`select_tier`), which
update_models.py imports -- one implementation, not two.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(REPO_DIR, "models.json")

TIERS = ("fast", "quality")

# Last-resort values, immune to any problem with models.json.
HARDCODED_FALLBACK = {
    "fast": "deepseek-v4-flash",
    "quality": "deepseek-v4-pro",
}

# Families we are willing to auto-select. An unrecognised suffix (say a future
# "deepseek-v5-turbo") stays unselectable until a human adds it here -- that is
# the point, not an oversight.
KNOWN_FAMILIES = ("flash", "pro")
FAMILY_RANK = {"flash": 10, "pro": 20}

STALE_AFTER_DAYS = 30

# Anything not matching this is permanently ineligible for auto-selection.
# It is what makes the legacy `deepseek-chat` / `deepseek-reasoner` names inert.
MODEL_ID_RE = re.compile(r"^deepseek-v(\d+)(?:\.(\d+))?-([a-z]+)$")

_warned = set()
_cache = {"loaded": False, "config": None}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _warn(message):
    """Warn to stderr once per process, so a stale config nags without spamming."""
    if message in _warned:
        return
    _warned.add(message)
    print(f"[model_config] {message}", file=sys.stderr)


def _parse_dt(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _aware(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _is_past(date_str):
    dt = _parse_dt(date_str)
    return dt is not None and _aware(dt) <= datetime.now(timezone.utc)


def parse_model_id(model_id):
    """-> {'generation': int, 'minor': int, 'family': str} or None."""
    match = MODEL_ID_RE.match(model_id or "")
    if not match:
        return None
    return {
        "generation": int(match.group(1)),
        "minor": int(match.group(2)) if match.group(2) else 0,
        "family": match.group(3),
    }


def is_usable(info):
    """Can an already-chosen model still be called?

    Available, and not *past* its deprecation date. A future-dated deprecation is
    still usable -- it only warrants a warning. (Selection uses a stricter rule:
    see select_tier, which refuses to newly pick anything deprecated at all.)
    """
    if not isinstance(info, dict):
        return False
    if info.get("available") is False:
        return False
    if info.get("deprecated") and _is_past(info.get("deprecation_date")):
        return False
    return True


def _input_price(info):
    price = (info.get("pricing_usd_per_1m") or {}).get("input_cache_miss")
    return price if isinstance(price, (int, float)) else None


# --------------------------------------------------------------------------
# tier selection (shared with update_models.py)
# --------------------------------------------------------------------------

def select_tier(models, tier):
    """Pick the best model for `tier` from a models dict. -> (model_id|None, reason).

    Price is used as the capability proxy: within one vendor generation the price
    ladder *is* the vendor's own capability ordering, and it is the only signal
    that is published, numeric and self-updating. Falls back to a static family
    rank when pricing is unavailable.
    """
    candidates = []
    for mid, info in (models or {}).items():
        if not isinstance(info, dict):
            continue
        # Stricter than is_usable(): never newly select something on death row.
        if info.get("available") is False or info.get("deprecated"):
            continue
        parsed = parse_model_id(mid)
        if not parsed or parsed["family"] not in KNOWN_FAMILIES:
            continue
        candidates.append((parsed, mid, info))

    if not candidates:
        return None, "no eligible models"

    top = max(c[0]["generation"] for c in candidates)
    pool = [c for c in candidates if c[0]["generation"] == top]

    priced = [c for c in pool
              if not c[2].get("pricing_stale") and _input_price(c[2]) is not None]
    if priced:
        picker = min if tier == "fast" else max
        # Tie-break on id so re-runs never churn the file.
        best = picker(priced, key=lambda c: (_input_price(c[2]), c[1]))
        kind = "lowest" if tier == "fast" else "highest"
        return best[1], f"{kind} input price in generation {top}"

    picker = min if tier == "fast" else max
    best = picker(pool, key=lambda c: (FAMILY_RANK.get(c[0]["family"], 15), c[1]))
    return best[1], f"family rank (pricing unavailable) in generation {top}"


# --------------------------------------------------------------------------
# config loading
# --------------------------------------------------------------------------

def _load():
    if _cache["loaded"]:
        return _cache["config"]
    _cache["loaded"] = True
    try:
        # utf-8-sig, not utf-8: Notepad and PowerShell's -Encoding utf8 both write
        # a BOM, which json.load rejects. This reads fine with or without one.
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as fh:
            config = json.load(fh)
        if not isinstance(config, dict):
            raise ValueError("top level is not a JSON object")
        _cache["config"] = config
    except FileNotFoundError:
        _warn("models.json not found - using built-in defaults. "
              "Run: py update_models.py")
    except (json.JSONDecodeError, ValueError, OSError) as err:
        _warn(f"models.json unreadable ({err}) - using built-in defaults. "
              "Run: py update_models.py")
    return _cache["config"]


def reload():
    """Drop the cached config. Only needed by long-running processes."""
    _cache["loaded"] = False
    _cache["config"] = None


def _check_staleness(config):
    last = _parse_dt(config.get("last_checked"))
    if last is None:
        return
    age = (datetime.now(timezone.utc) - _aware(last)).days
    if age > STALE_AFTER_DAYS:
        _warn(f"models.json is {age} days stale - run: py update_models.py")


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def resolve_verbose(tier="fast"):
    """-> {'model': str, 'tier': str, 'source': str, 'warnings': [str]}."""
    if tier not in TIERS:
        _warn(f"unknown tier {tier!r} - treating as 'fast'")
        tier = "fast"

    # 1. Absolute override, for debugging. Bypasses everything.
    override = os.environ.get("DEEPSEEK_MODEL")
    if override:
        return {"model": override, "tier": tier, "source": "env", "warnings": []}

    config = _load()
    if config is None:
        return {"model": HARDCODED_FALLBACK[tier], "tier": tier,
                "source": "hardcoded", "warnings": ["config unavailable"]}

    _check_staleness(config)
    warnings = []
    models = config.get("models") or {}
    entry = (config.get("tiers") or {}).get(tier) or {}

    # 2. Human pin wins over everything in the file.
    pin = entry.get("pin")
    if pin:
        if not is_usable(models.get(pin)):
            warnings.append(f"pinned model {pin!r} is unavailable or retired")
        return {"model": pin, "tier": tier, "source": "pin", "warnings": warnings}

    # 3. The value the updater resolved.
    chosen = entry.get("model")
    if chosen and is_usable(models.get(chosen)):
        info = models.get(chosen) or {}
        if info.get("deprecated"):
            warnings.append(
                f"{chosen} is deprecated (retires {info.get('deprecation_date')})")
        return {"model": chosen, "tier": tier, "source": "config",
                "warnings": warnings}
    if chosen:
        warnings.append(f"configured model {chosen!r} unusable - re-deriving")

    # 4. Re-derive live, covering a hand-edited or partially stale file.
    derived, reason = select_tier(models, tier)
    if derived:
        return {"model": derived, "tier": tier, "source": f"derived ({reason})",
                "warnings": warnings}

    # 5. Last resort.
    warnings.append("no usable model in config")
    return {"model": HARDCODED_FALLBACK[tier], "tier": tier,
            "source": "hardcoded", "warnings": warnings}


def resolve(tier="fast"):
    """-> model id string. Always returns something callable."""
    result = resolve_verbose(tier)
    for message in result["warnings"]:
        _warn(message)
    return result["model"]


def model_info(model_id):
    """-> the model's config dict, or None."""
    config = _load()
    if not config:
        return None
    info = (config.get("models") or {}).get(model_id)
    return info if isinstance(info, dict) else None


def estimate_cost(model_id, in_tokens, out_tokens, cache_hit=False):
    """-> estimated USD for a call, or None if pricing is unknown."""
    info = model_info(model_id)
    if not info:
        return None
    pricing = info.get("pricing_usd_per_1m") or {}
    in_rate = pricing.get("input_cache_hit" if cache_hit else "input_cache_miss")
    out_rate = pricing.get("output")
    if not isinstance(in_rate, (int, float)) or not isinstance(out_rate, (int, float)):
        return None
    return (in_tokens / 1_000_000) * in_rate + (out_tokens / 1_000_000) * out_rate


if __name__ == "__main__":
    for _tier in TIERS:
        print(f"{_tier:8} -> {json.dumps(resolve_verbose(_tier), indent=2)}")
