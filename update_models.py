"""Refresh models.json from DeepSeek's API and public docs.

    py update_models.py                      # full authenticated refresh
    py update_models.py --dry-run --verbose  # show the diff, write nothing
    py update_models.py --docs-only --check  # CI sensor mode: no key, no writes
    py update_models.py --pin fast=deepseek-v4-flash
    py update_models.py --accept-new-generation

Data sources, in order of trust:
  1. GET /models          - authoritative for availability. Ids only.
  2. docs pricing page    - best-effort. Public, no auth.
  3. docs changelog page  - best-effort. Public, no auth.

Design rule: scraping is *enrichment*. A scrape that fails or returns nonsense
must never change which model the app calls -- it marks data stale and keeps the
last known-good values instead.

Exit codes:  0 clean  |  1 a source failed (config still valid)  |  2 action needed
"""

import argparse
import difflib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

import deepseek_common as common
import model_config

CONFIG_PATH = model_config.CONFIG_PATH

PRICING_URL = os.environ.get(
    "DEEPSEEK_PRICING_URL", "https://api-docs.deepseek.com/quick_start/pricing")
UPDATES_URL = os.environ.get(
    "DEEPSEEK_UPDATES_URL", "https://api-docs.deepseek.com/updates")

# Promotion gates for a newly-seen model.
SOAK_DAYS = 7
MIN_SIGHTINGS = 3
PRICE_CEILING = {"fast": 1.5, "quality": 3.0}

# A scraped price this far from the stored one is treated as a parser fault,
# not a price change. This is the main guard against a drifting parser silently
# redirecting tier selection.
ANOMALY_FACTOR = 5.0

PRICE_MIN, PRICE_MAX = 0.0001, 100.0

SEED_CONFIG = {
    "schema_version": 1,
    "generated_by": "update_models.py",
    "last_checked": None,
    "sources": {},
    "models": {
        "deepseek-v4-flash": {
            "available": True, "generation": 4, "family": "flash",
            "context_window": 1000000, "max_output_tokens": 384000,
            "pricing_usd_per_1m": {"input_cache_hit": 0.0028,
                                   "input_cache_miss": 0.14, "output": 0.28},
            "pricing_stale": False, "deprecated": False, "deprecation_date": None,
            "notes": "thinking + non-thinking; thinking is the default",
            "provenance": {},
        },
        "deepseek-v4-pro": {
            "available": True, "generation": 4, "family": "pro",
            "context_window": 1000000, "max_output_tokens": 384000,
            "pricing_usd_per_1m": {"input_cache_hit": 0.003625,
                                   "input_cache_miss": 0.435, "output": 0.87},
            "pricing_stale": False, "deprecated": False, "deprecation_date": None,
            "notes": "thinking + non-thinking; ~3.1x flash cost",
            "provenance": {},
        },
        "deepseek-chat": {
            "available": True, "generation": None, "family": "legacy-alias",
            "pricing_usd_per_1m": None, "pricing_stale": True,
            "deprecated": True, "deprecation_date": "2026-07-24",
            "notes": "alias for v4-flash non-thinking; retires 2026-07-24",
            "provenance": {},
        },
        "deepseek-reasoner": {
            "available": True, "generation": None, "family": "legacy-alias",
            "pricing_usd_per_1m": None, "pricing_stale": True,
            "deprecated": True, "deprecation_date": "2026-07-24",
            "notes": "alias for v4-flash thinking; retires 2026-07-24",
            "provenance": {},
        },
    },
    "tiers": {
        "fast": {"model": "deepseek-v4-flash", "pin": None,
                 "chosen_by": "seed", "reason": "seed default", "decided_at": None},
        "quality": {"model": "deepseek-v4-pro", "pin": None,
                    "chosen_by": "seed", "reason": "seed default", "decided_at": None},
    },
    "pending": {},
    "warnings": [],
}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# HTML -> flat text
# --------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self):
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def flatten_html(html):
    """Regexing flattened *text* rather than the DOM survives a theme change;
    it only breaks if the page's actual wording changes."""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


# --------------------------------------------------------------------------
# scrapers (best-effort, never fatal)
# --------------------------------------------------------------------------

def _pricing_valid(row):
    values = [row["input_cache_hit"], row["input_cache_miss"], row["output"]]
    if not all(PRICE_MIN < v < PRICE_MAX for v in values):
        return False
    return row["output"] >= row["input_cache_miss"] >= row["input_cache_hit"]


# The pricing table is TRANSPOSED: models are columns, prices are grouped by
# row-label, e.g. "1M INPUT TOKENS (CACHE HIT) $0.0028 $0.003625". So we read the
# column order from the header, then map each label's values positionally.
# The leading "1M" is load-bearing: it distinguishes a table row label from the
# same words in prose. Without it, "OUTPUT TOKENS" also matches the intro
# sentence "...the total number of input and output tokens by the model", which
# sits *before* the table and silently poisons the positional mapping.
PRICE_LABELS = (
    ("input_cache_hit",
     re.compile(r"1M\s+INPUT TOKENS\s*\(\s*CACHE HIT\s*\)", re.I)),
    ("input_cache_miss",
     re.compile(r"1M\s+INPUT TOKENS\s*\(\s*CACHE MISS\s*\)", re.I)),
    ("output",
     re.compile(r"1M\s+OUTPUT TOKENS", re.I)),
)

REQUIRED_PRICE_FIELDS = {"input_cache_hit", "input_cache_miss", "output"}


def _column_order(text, model_ids):
    """Model ids in the order they appear in the table header."""
    header = re.search(r"\bMODEL\b(.{0,400}?)BASE URL", text, re.I | re.S)
    if not header:
        return []
    segment = header.group(1)
    seen = [(segment.find(mid), mid) for mid in model_ids if segment.find(mid) != -1]
    return [mid for _, mid in sorted(seen)]


def scrape_pricing(text, model_ids):
    """-> {model_id: {input_cache_hit, input_cache_miss, output}} for columns that parse."""
    order = _column_order(text, model_ids)
    if not order:
        return {}

    # Locate every label first so each window can be bounded by the next label --
    # that stops one row's values bleeding into the next when a value is missing.
    hits = []
    for field, pattern in PRICE_LABELS:
        match = pattern.search(text)
        if not match:
            return {}
        hits.append((match.start(), match.end(), field))
    hits.sort()

    columns = {mid: {} for mid in order}
    for index, (_start, end, field) in enumerate(hits):
        stop = hits[index + 1][0] if index + 1 < len(hits) else min(len(text), end + 400)
        amounts = re.findall(r"\$\s*([0-9]*\.?[0-9]+)", text[end:stop])
        # One value per model column, or we do not trust the mapping.
        if len(amounts) != len(order):
            continue
        for mid, amount in zip(order, amounts):
            columns[mid][field] = float(amount)

    found = {}
    for mid, row in columns.items():
        if set(row) == REQUIRED_PRICE_FIELDS and _pricing_valid(row):
            found[mid] = row
    return found


MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}

DEPRECATION_WORDS = re.compile(r"deprecat|retire|sunset|discontinu", re.I)


def _find_date(sentence):
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", sentence)
    if match:
        return "%04d-%02d-%02d" % tuple(int(g) for g in match.groups())
    match = re.search(r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})", sentence)
    if match:
        month = MONTHS.get(match.group(1)[:3].lower())
        if month:
            return "%04d-%02d-%02d" % (int(match.group(3)), month,
                                       int(match.group(2)))
    return None


# How far before/after a retirement word we look for the model being retired.
DEPRECATION_LOOKBEHIND = 160
DEPRECATION_LOOKAHEAD = 200


# The gap between the retired model and the retirement word must read like a verb
# phrase: " will be ", " (to be ", " , will be ". Anything with real content means
# the nearby id is not the subject.
VERB_GAP_RE = re.compile(
    r"^[\s,;:()\-]*"
    r"(?:\b(?:and|or|will|shall|may|might|to|are|is|be|been|being|soon|now|both|"
    r"also|the|model|models|name|names)\b[\s,;:()\-]*)*$", re.I)

# Ids chain only across an EXPLICIT conjunction. Whitespace alone does not chain:
# "deepseek-v4-flash deepseek-v4-pro deepseek-chat" is a LIST of separate models,
# whereas "deepseek-chat and deepseek-reasoner" is genuinely one subject.
CHAIN_RE = re.compile(r"^\s*(?:,|;|and|or|&|,\s*and)\s*$", re.I)


def _attributed_ids(before, model_ids):
    """Which ids a retirement word actually applies to.

    The nearest id preceding the word, plus any joined to it by an explicit
    conjunction. Returns [] when the preceding text does not read like a subject.
    """
    spans = []
    for mid in model_ids:
        for match in re.finditer(re.escape(mid), before):
            spans.append((match.start(), match.end(), mid))
    if not spans:
        return []
    spans.sort()
    # Drop any span fully contained in another (one id being a prefix of another).
    trimmed = []
    for span in spans:
        if not any(o[0] <= span[0] and span[1] <= o[1] and o is not span
                   for o in spans):
            trimmed.append(span)
    spans = trimmed

    if not VERB_GAP_RE.match(before[spans[-1][1]:]):
        return []

    chosen = [spans[-1][2]]
    for index in range(len(spans) - 1, 0, -1):
        if not CHAIN_RE.match(before[spans[index - 1][1]:spans[index][0]]):
            break
        chosen.append(spans[index - 1][2])
    return chosen


def scrape_deprecations(text, model_ids):
    """-> {model_id: date_str} for models a notice actually retires.

    Deliberately conservative, because deprecation is *sticky*: a false positive
    would permanently disqualify a live model from tier selection, while a false
    negative merely means finding out later (and /models still catches an actual
    removal). Three rules:

    1. The id must PRECEDE the retirement word -- it is the subject. In
       "deepseek-chat ... deprecated ... they correspond to deepseek-v4-flash",
       v4-flash is the replacement, not the victim.
    2. Only the nearest id counts, unless others are chained by an explicit
       conjunction. The docs also render deprecations as a list --
       "deepseek-v4-flash deepseek-v4-pro deepseek-chat (to be deprecated ...)"
       -- where a plain proximity window would retire the two live models too.
    3. A parseable date is required; a retirement word with no date attached is
       far more likely to be prose than an actual notice.
    """
    found = {}
    for match in DEPRECATION_WORDS.finditer(text):
        before = text[max(0, match.start() - DEPRECATION_LOOKBEHIND):match.start()]
        boundary = max(before.rfind(". "), before.rfind("! "), before.rfind("? "))
        if boundary != -1:
            before = before[boundary + 1:]

        after = text[match.end():match.end() + DEPRECATION_LOOKAHEAD]
        date = _find_date(before + " " + after)
        if not date:
            continue
        for mid in _attributed_ids(before, model_ids):
            found.setdefault(mid, date)
    return found


# --------------------------------------------------------------------------
# config io
# --------------------------------------------------------------------------

def load_config(warnings):
    try:
        # utf-8-sig tolerates a BOM from a Notepad edit; writing stays plain utf-8.
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as fh:
            config = json.load(fh)
        if not isinstance(config, dict) or "models" not in config:
            raise ValueError("missing 'models' block")
        for key, value in SEED_CONFIG.items():
            config.setdefault(key, json.loads(json.dumps(value)))
        return config
    except FileNotFoundError:
        return json.loads(json.dumps(SEED_CONFIG))
    except (json.JSONDecodeError, ValueError, OSError) as err:
        warnings.append(f"models.json unusable ({err}) - rebuilt from seed")
        return json.loads(json.dumps(SEED_CONFIG))


def write_config(config):
    """Write via a temp file + os.replace so an interrupted run cannot leave
    a half-written config behind (os.replace is atomic on Windows too)."""
    payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
    os.replace(tmp, CONFIG_PATH)


def note_source(config, name, url, ok, error=None, **extra):
    entry = config.setdefault("sources", {}).setdefault(name, {})
    entry["url"] = url
    entry["last_attempt"] = now_iso()
    if ok:
        entry["last_success"] = now_iso()
        entry["consecutive_failures"] = 0
        entry["last_error"] = None
    else:
        entry["consecutive_failures"] = int(entry.get("consecutive_failures") or 0) + 1
        entry["last_error"] = error
    entry.update(extra)


# --------------------------------------------------------------------------
# main steps
# --------------------------------------------------------------------------

def step_availability(config, warnings, verbose):
    """-> True if /models answered. On failure nothing is marked unavailable."""
    key = common.get_api_key()
    if not key:
        warnings.append("DEEPSEEK_API_KEY missing - availability not refreshed")
        note_source(config, "models_api", common.BASE_URL + "/models", False,
                    "no api key")
        return False

    status, body = common.api_request("GET", "/models", key)
    if status != 200 or not isinstance(body, dict) or not isinstance(
            body.get("data"), list):
        warnings.append(f"/models unreachable (HTTP {status}) - availability data "
                        "kept from last success")
        note_source(config, "models_api", common.BASE_URL + "/models", False,
                    f"HTTP {status}: {str(body)[:200]}")
        return False

    live = {m.get("id") for m in body["data"] if isinstance(m, dict) and m.get("id")}
    if verbose:
        print(f"  /models returned {len(live)} ids: {sorted(live)}")

    for mid, info in config["models"].items():
        was = info.get("available")
        info["available"] = mid in live
        if was and not info["available"]:
            # A model already known to be retiring dropping off the list is
            # expected housekeeping; an unannounced disappearance is not.
            if info.get("deprecated"):
                warnings.append(f"{mid} is no longer listed by /models "
                                "(expected - already deprecated)")
            else:
                warnings.append(f"{mid} has VANISHED from /models with no notice")
        info.setdefault("provenance", {})["availability"] = {
            "source": "models_api", "at": now_iso()}

    for mid in sorted(live - set(config["models"])):
        track_pending(config, mid, verbose)

    note_source(config, "models_api", common.BASE_URL + "/models", True,
                model_count=len(live))
    return True


def track_pending(config, mid, verbose):
    """Quarantine a newly-seen id. It never enters `models` or `tiers` here."""
    entry = config.setdefault("pending", {}).setdefault(mid, {
        "first_seen": now_iso(), "seen_count": 0})
    entry["seen_count"] = int(entry.get("seen_count") or 0) + 1
    parsed = model_config.parse_model_id(mid)
    entry["generation"] = parsed["generation"] if parsed else None
    entry["family"] = parsed["family"] if parsed else None
    if verbose:
        print(f"  new model id seen: {mid} (sighting #{entry['seen_count']})")


def step_pricing(config, warnings, verbose):
    ids = list(config["models"]) + list(config.get("pending") or {})
    try:
        text = flatten_html(common.fetch_text(PRICING_URL))
    except Exception as err:                      # noqa: BLE001 - never fatal
        warnings.append(f"pricing scrape failed ({err}) - prices marked stale")
        for info in config["models"].values():
            info["pricing_stale"] = True
        note_source(config, "pricing_docs", PRICING_URL, False, str(err))
        return False

    scraped = scrape_pricing(text, ids)
    if verbose:
        print(f"  pricing rows parsed: {sorted(scraped)}")

    if not scraped:
        # The page fetched fine but yielded nothing -- that is parser rot, and it
        # must be visible rather than silently degrading to the family-rank path.
        warnings.append("pricing page fetched but NO rows parsed - the scraper "
                        "likely needs updating for a page change")
        for info in config["models"].values():
            if info.get("pricing_usd_per_1m"):
                info["pricing_stale"] = True
        note_source(config, "pricing_docs", PRICING_URL, False, "no rows parsed")
        return False

    for mid, info in config["models"].items():
        row = scraped.get(mid)
        if row is None:
            if info.get("pricing_usd_per_1m"):
                info["pricing_stale"] = True
                warnings.append(f"no pricing row parsed for {mid} - kept stored value")
            continue

        stored = info.get("pricing_usd_per_1m") or {}
        old = stored.get("input_cache_miss")
        new = row["input_cache_miss"]
        if isinstance(old, (int, float)) and old > 0 and (
                new > old * ANOMALY_FACTOR or new < old / ANOMALY_FACTOR):
            info["pricing_stale"] = True
            warnings.append(
                f"price anomaly for {mid}: scraped {new} vs stored {old} - "
                "kept stored value, verify manually")
            continue

        info["pricing_usd_per_1m"] = row
        info["pricing_stale"] = False
        info.setdefault("provenance", {})["pricing"] = {
            "source": "pricing_docs", "at": now_iso()}

    for mid in (config.get("pending") or {}):
        if mid in scraped:
            config["pending"][mid]["pricing_usd_per_1m"] = scraped[mid]

    note_source(config, "pricing_docs", PRICING_URL, True, rows_parsed=len(scraped))
    return True


def step_deprecations(config, warnings, verbose):
    ids = list(config["models"]) + list(config.get("pending") or {})
    try:
        text = flatten_html(common.fetch_text(UPDATES_URL))
    except Exception as err:                      # noqa: BLE001 - never fatal
        warnings.append(f"changelog scrape failed ({err}) - deprecations unchanged")
        note_source(config, "changelog_docs", UPDATES_URL, False, str(err))
        return False

    found = scrape_deprecations(text, ids)
    if verbose:
        print(f"  deprecation notices parsed: {found}")

    # Sticky on purpose: a found notice sets the flag, but a missing notice never
    # clears one. Silently using a dead model is worse than using a pricier one.
    for mid, date in found.items():
        info = config["models"].get(mid)
        if not info:
            continue
        if not info.get("deprecated"):
            warnings.append(f"NEW deprecation notice for {mid}"
                            + (f" (retires {date})" if date else ""))
        info["deprecated"] = True
        if date:
            info["deprecation_date"] = date
        info.setdefault("provenance", {})["deprecation"] = {
            "source": "changelog_docs", "at": now_iso()}

    note_source(config, "changelog_docs", UPDATES_URL, True,
                notices_parsed=len(found))
    return True


def evaluate_pending(config, warnings):
    """Score each quarantined model against the promotion gates.

    -> list of (model_id, tier) that pass every gate and would change a tier.
    """
    ready = []
    for mid, entry in (config.get("pending") or {}).items():
        parsed = model_config.parse_model_id(mid)
        price = (entry.get("pricing_usd_per_1m") or {}).get("input_cache_miss")
        first_seen = model_config._parse_dt(entry.get("first_seen"))
        soaked = bool(first_seen) and (
            datetime.now(timezone.utc) - model_config._aware(first_seen)
            >= timedelta(days=SOAK_DAYS))

        tier = None
        if parsed and parsed["family"] == "flash":
            tier = "fast"
        elif parsed and parsed["family"] == "pro":
            tier = "quality"

        incumbent_price = None
        if tier:
            current = (config["tiers"].get(tier) or {}).get("model")
            info = config["models"].get(current) or {}
            incumbent_price = (info.get("pricing_usd_per_1m") or {}).get(
                "input_cache_miss")

        price_sane = (
            isinstance(price, (int, float)) and isinstance(incumbent_price, (int, float))
            and price <= incumbent_price * PRICE_CEILING.get(tier or "fast", 1.5))

        gates = {
            "id_parses": bool(parsed),
            "known_family": bool(parsed) and parsed["family"] in model_config.KNOWN_FAMILIES,
            "pricing_known": isinstance(price, (int, float)),
            "soak_days": soaked,
            "seen_count": int(entry.get("seen_count") or 0) >= MIN_SIGHTINGS,
            "price_sane": price_sane,
        }
        entry["gates"] = gates
        entry["blocked_on"] = [name for name, ok in gates.items() if not ok]

        if all(gates.values()) and tier:
            ready.append((mid, tier))
        elif gates["id_parses"] and not gates["known_family"]:
            warnings.append(
                f"{mid} has unrecognised family {parsed['family']!r} - "
                "add it to model_config.KNOWN_FAMILIES to make it selectable")
    return ready


def step_tiers(config, warnings, ready, accept_new):
    """Re-resolve tier -> model. Only called when availability data is fresh."""
    promoted = []
    if accept_new:
        for mid, _tier in ready:
            entry = config["pending"].pop(mid)
            config["models"][mid] = {
                "available": True,
                "generation": entry.get("generation"),
                "family": entry.get("family"),
                "pricing_usd_per_1m": entry.get("pricing_usd_per_1m"),
                "pricing_stale": False,
                "deprecated": False,
                "deprecation_date": None,
                "notes": "promoted via --accept-new-generation",
                "provenance": {"availability": {"source": "models_api",
                                                "at": now_iso()}},
            }
            promoted.append(mid)
        if promoted:
            print(f"Promoted: {', '.join(promoted)}")

    for tier in model_config.TIERS:
        entry = config["tiers"].setdefault(tier, {})
        if entry.get("pin"):
            continue
        chosen, reason = model_config.select_tier(config["models"], tier)
        if not chosen:
            warnings.append(f"tier {tier!r}: {reason} - keeping {entry.get('model')}")
            continue
        if chosen != entry.get("model"):
            print(f"tier {tier}: {entry.get('model')} -> {chosen}  ({reason})")
        entry.update({"model": chosen, "chosen_by": "auto", "reason": reason,
                      "decided_at": now_iso()})
    return promoted


def step_health(config, warnings):
    """-> exit code contribution: 2 if a tier's effective model is dead."""
    code = 0
    for tier in model_config.TIERS:
        entry = config["tiers"].get(tier) or {}
        mid = entry.get("pin") or entry.get("model")
        info = config["models"].get(mid)
        if not info:
            print(f"ERROR: tier {tier!r} -> {mid!r} which is not in the config")
            warnings.append(f"tier {tier} points at unknown model {mid}")
            code = 2
            continue
        if info.get("available") is False:
            print(f"ERROR: tier {tier!r} -> {mid} has vanished from /models")
            warnings.append(f"{mid} vanished but is still selected for {tier}")
            code = 2
        if info.get("deprecated"):
            date = info.get("deprecation_date")
            if date and model_config._is_past(date):
                print(f"ERROR: tier {tier!r} -> {mid} was retired on {date}")
                warnings.append(f"{mid} retired {date} but is still selected")
                code = 2
            elif date:
                left = (model_config._aware(model_config._parse_dt(date))
                        - datetime.now(timezone.utc)).days
                print(f"WARNING: tier {tier!r} -> {mid} retires in {left} days ({date})")
                warnings.append(f"{mid} retires in {left} days")
    return code


# --------------------------------------------------------------------------
# docs-only check mode (CI sensor)
# --------------------------------------------------------------------------

def run_docs_check(config, verbose):
    """Compare public docs against the committed config. Writes nothing.

    Runs with no API key at all, which is why CI needs no secret. It cannot see
    availability, so it never touches tiers -- it only reports.
    """
    ids = list(config["models"]) + list(config.get("pending") or {})
    changes = []
    pricing_text = ""
    updates_text = ""

    try:
        pricing_text = flatten_html(common.fetch_text(PRICING_URL))
        scraped = scrape_pricing(pricing_text, ids)
    except Exception as err:                      # noqa: BLE001
        print(f"pricing scrape failed: {err}")
        scraped = {}

    for mid, row in scraped.items():
        stored = (config["models"].get(mid, {}).get("pricing_usd_per_1m") or {})
        old = stored.get("input_cache_miss")
        new = row["input_cache_miss"]
        if isinstance(old, (int, float)) and abs(new - old) > 1e-9:
            changes.append(f"price change {mid}: input_cache_miss {old} -> {new}")

    try:
        updates_text = flatten_html(common.fetch_text(UPDATES_URL))
        found = scrape_deprecations(updates_text, ids)
    except Exception as err:                      # noqa: BLE001
        print(f"changelog scrape failed: {err}")
        found = {}

    for mid, date in found.items():
        info = config["models"].get(mid)
        if info is None:
            changes.append(f"deprecation notice for unknown model {mid}")
        elif not info.get("deprecated"):
            changes.append(f"NEW deprecation notice: {mid}"
                           + (f" (retires {date})" if date else ""))
        elif date and info.get("deprecation_date") != date:
            changes.append(f"deprecation date moved for {mid}: "
                           f"{info.get('deprecation_date')} -> {date}")

    # Any model id mentioned in the docs that the config has never heard of.
    # This is how a v5 announcement reaches you without CI holding a key.
    known = set(ids)
    mentioned = set(re.findall(r"deepseek-v\d+(?:\.\d+)?-[a-z]+",
                               pricing_text + " " + updates_text))
    for mid in sorted(mentioned - known):
        changes.append(f"docs mention unknown model id: {mid}")

    if verbose:
        print(f"checked {len(ids)} known ids against the public docs")

    if changes:
        print("DOCS CHANGED:")
        for line in sorted(set(changes)):
            print(f"  - {line}")
        print("\nRun `py update_models.py` locally to refresh models.json.")
        return 2
    print("No material changes in the public docs.")
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def run_docs_debug():
    """Dump exactly what the scrapers see. Diagnoses CI-vs-local differences,
    which are real: the docs CDN can serve different content by region."""
    for name, url in (("pricing", PRICING_URL), ("updates", UPDATES_URL)):
        print(f"\n=== {name}: {url} ===")
        try:
            raw = common.fetch_text(url)
        except Exception as err:                  # noqa: BLE001
            print(f"  FETCH FAILED: {err}")
            continue
        text = flatten_html(raw)
        print(f"  raw {len(raw)} bytes -> flattened {len(text)} chars")
        mentioned = sorted(set(re.findall(r"deepseek-[a-z0-9.\-]+", text)))
        print(f"  model ids mentioned: {mentioned}")
        hits = list(DEPRECATION_WORDS.finditer(text))
        print(f"  retirement-word hits: {len(hits)}")
        for match in hits:
            before = text[max(0, match.start() - DEPRECATION_LOOKBEHIND):match.start()]
            boundary = max(before.rfind(". "), before.rfind("! "), before.rfind("? "))
            trimmed = before[boundary + 1:] if boundary != -1 else before
            after = text[match.end():match.end() + DEPRECATION_LOOKAHEAD]
            print(f"\n    word={match.group(0)!r} @ {match.start()}"
                  f"  sentence_boundary_found={boundary != -1}")
            print(f"    before(trimmed) = {trimmed[-220:]!r}")
            print(f"    after           = {after[:140]!r}")
            print(f"    date            = {_find_date(trimmed + ' ' + after)}")
            print(f"    ids in before   = "
                  f"{[m for m in mentioned if m in trimmed]}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the diff, write nothing")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--docs-only", action="store_true",
                        help="skip the authenticated /models call (no key needed)")
    parser.add_argument("--debug-docs", action="store_true",
                        help="dump what the scrapers see, then exit")
    parser.add_argument("--check", action="store_true",
                        help="report differences and exit; never write")
    parser.add_argument("--accept-new-generation", action="store_true",
                        help="promote quarantined models that pass every gate")
    parser.add_argument("--pin", metavar="TIER=MODEL", action="append", default=[])
    parser.add_argument("--unpin", metavar="TIER", action="append", default=[])
    parser.add_argument("--clear-deprecation", metavar="MODEL", action="append",
                        default=[],
                        help="undo a deprecation flag set in error; it is sticky "
                             "otherwise and would survive every later run")
    return parser.parse_args()


def main():
    common.enable_utf8_stdout()
    args = parse_args()
    if args.debug_docs:
        return run_docs_debug()
    warnings = []
    config = load_config(warnings)
    before = json.dumps(config, indent=2, ensure_ascii=False)

    # CI sensor mode: public docs only, no key, no writes.
    if args.docs_only and args.check:
        return run_docs_check(config, args.verbose)

    for spec in args.pin:
        tier, _, mid = spec.partition("=")
        if tier not in model_config.TIERS or not mid:
            print(f"bad --pin {spec!r}; expected TIER=MODEL")
            return 2
        config["tiers"].setdefault(tier, {})["pin"] = mid
        print(f"pinned {tier} -> {mid}")
    for tier in args.unpin:
        if tier in config.get("tiers", {}):
            config["tiers"][tier]["pin"] = None
            print(f"unpinned {tier}")

    for mid in args.clear_deprecation:
        info = config["models"].get(mid)
        if not info:
            print(f"cannot clear deprecation: {mid!r} is not in the config")
            return 2
        info["deprecated"] = False
        info["deprecation_date"] = None
        print(f"cleared deprecation flag on {mid}")

    print("1) availability  (GET /models)")
    availability_ok = False if args.docs_only else step_availability(
        config, warnings, args.verbose)
    if args.docs_only:
        print("   skipped (--docs-only)")

    print("2) pricing       (docs)")
    pricing_ok = step_pricing(config, warnings, args.verbose)

    print("3) deprecations  (docs)")
    deprecation_ok = step_deprecations(config, warnings, args.verbose)

    print("4) tier selection")
    ready = evaluate_pending(config, warnings)
    if availability_ok:
        step_tiers(config, warnings, ready, args.accept_new_generation)
    else:
        print("   skipped - availability data is not fresh")
        warnings.append("tier selection skipped: /models did not answer")

    exit_code = 0
    if not (availability_ok and pricing_ok and deprecation_ok):
        exit_code = 1

    # A generation bump is never silent, even with every gate green.
    pending_gen_bump = [
        (mid, tier) for mid, tier in ready
        if not args.accept_new_generation
        and (model_config.parse_model_id(mid) or {}).get("generation")
        != (model_config.parse_model_id(
            (config["tiers"].get(tier) or {}).get("model") or "") or {}).get("generation")
    ]
    for mid, tier in pending_gen_bump:
        current = (config["tiers"].get(tier) or {}).get("model")
        print(f"\nACTION NEEDED: {mid} passes all gates and would replace\n"
              f"  {tier}: {current} -> {mid}\n"
              f"Run:  py update_models.py --accept-new-generation")
        warnings.append(f"{mid} awaiting acceptance for tier {tier}")
        exit_code = 2

    print("\n5) health")
    exit_code = max(exit_code, step_health(config, warnings))

    config["last_checked"] = now_iso()
    config["warnings"] = warnings

    after = json.dumps(config, indent=2, ensure_ascii=False)
    if args.dry_run or args.check:
        diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                         "models.json (current)",
                                         "models.json (proposed)", lineterm=""))
        print("\n--- diff ---")
        print("\n".join(diff) if diff else "(no changes)")
        print("(nothing written)")
    else:
        write_config(config)
        print(f"\nwrote {CONFIG_PATH}")

    if warnings:
        print("\nwarnings:")
        for line in warnings:
            print(f"  - {line}")
    print(f"\nexit {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
