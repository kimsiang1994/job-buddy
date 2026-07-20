"""Properties every module must hold, enforced by reflection rather than example.

    py -m unittest tests.test_invariants

Every other test in this repo is a regression: it pins one bug that was found,
which means coverage equals "bugs we happened to hit". Two audits found twenty
silent-failure defects that had sat for weeks, and not one had a test, because
nobody had been bitten by it yet.

The tests here are the other kind. They walk `src/jobbuddy/` with `importlib`
and `ast` and assert properties over EVERY module found, so a module added
tomorrow is covered the day it lands rather than the day it breaks something.
The shared shape of the defects they encode is that all of them RETURNED A
PLAUSIBLE VALUE -- an empty registry, an `ok=True` page with no html, a clean
rule report from a check that read nothing. A test asserting "the return looks
right" would have passed on every one of them, so these assert on structure
instead: what the code is allowed to be shaped like.

Where a rule has a legitimate exception it goes in that rule's ALLOW map with a
written reason. An exemption should be a decision somebody made and signed,
never a rule quietly narrowed until it stopped complaining.

Offline: parses and imports source, touches no network and no API key.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
PACKAGE_DIR = SRC_ROOT / "jobbuddy"


# --------------------------------------------------------------------------
# discovery -- shared by every test below
# --------------------------------------------------------------------------

def iter_source_files() -> list[Path]:
    """Every .py file in the package, including subpackages."""
    return sorted(p for p in PACKAGE_DIR.rglob("*.py")
                  if "__pycache__" not in p.parts)


def module_name(path: Path) -> str:
    """`src/jobbuddy/deepseek/token_budget.py` -> `jobbuddy.deepseek.token_budget`."""
    parts = path.relative_to(SRC_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parsed_modules() -> list[tuple[str, Path, ast.Module]]:
    out = []
    for path in iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        out.append((module_name(path), path, tree))
    return out


MODULES = parsed_modules()


def functions_in(tree: ast.Module):
    """(function node, qualified name) for every def, including nested ones."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, node.name


def where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(SRC_ROOT.parent)}:{getattr(node, 'lineno', '?')}"


class InvariantTestCase(unittest.TestCase):
    """Shared reporting: a failure names every offender, not just the first."""

    def assertNoViolations(self, violations: list[str], rule: str) -> None:
        if violations:
            listing = "\n  ".join(violations)
            self.fail(f"{rule}\n\n  {listing}\n\n"
                      f"{len(violations)} violation(s). If one of these is a "
                      f"deliberate exception, add it to that rule's ALLOW map "
                      f"with a reason -- do not widen the rule.")


# --------------------------------------------------------------------------
# 0. the reflection itself has to be sound
# --------------------------------------------------------------------------

class EveryModuleIsReachable(InvariantTestCase):
    """Reflection over a module that will not import silently covers nothing.

    This runs first because every test below draws its subject list from the
    same walk. A module that fails to import, or that the walk cannot see,
    would simply be skipped by all of them -- the tests would still pass, and
    the new module would have no coverage at all. That is the exact failure
    mode this file exists to prevent, so it is asserted rather than assumed.
    """

    def test_the_walk_finds_the_whole_package(self):
        walked = {name for name, _, _ in MODULES}
        declared = {f"jobbuddy.{m.name}" if not m.ispkg else f"jobbuddy.{m.name}"
                    for m in pkgutil.iter_modules([str(PACKAGE_DIR)])}
        missing = {d for d in declared if d not in walked
                   and not any(w.startswith(d + ".") for w in walked)}
        self.assertFalse(missing, f"pkgutil sees modules the file walk missed: {missing}")

    def test_every_module_imports(self):
        broken = []
        for name, path, _ in MODULES:
            try:
                importlib.import_module(name)
            except Exception as exc:              # noqa: BLE001 - reporting all
                broken.append(f"{path.name}: {type(exc).__name__}: {exc}")
        self.assertNoViolations(
            broken, "Every module must import cleanly, or reflection silently "
                    "skips it and every invariant below passes vacuously.")

    def test_the_walk_is_not_empty(self):
        # A glob typo would make every test in this file pass over nothing.
        self.assertGreater(len(MODULES), 20,
                           "suspiciously few modules found -- check PACKAGE_DIR")


# --------------------------------------------------------------------------
# 1. no bare except
# --------------------------------------------------------------------------

class NoBareExcept(InvariantTestCase):
    """`except:` swallows KeyboardInterrupt, SystemExit and the typo above it.

    There are none today. The rule is here so the first one is rejected on the
    way in, which is the only time it is cheap to argue about.
    """

    # (module, function) -> reason
    ALLOW: dict[tuple[str, str], str] = {}

    def test_no_bare_except_anywhere(self):
        violations = []
        for name, path, tree in MODULES:
            for fn, fn_name in functions_in(tree):
                for node in ast.walk(fn):
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        if (path.stem, fn_name) in self.ALLOW:
                            continue
                        violations.append(f"{where(path, node)} in {fn_name}()")
        self.assertNoViolations(
            violations,
            "A bare `except:` catches KeyboardInterrupt and SystemExit, and "
            "hides the bug you will be asked to find next week. Name the "
            "exceptions, or use `except Exception` and say why.")


# --------------------------------------------------------------------------
# 2. loaders that fall back must say so
# --------------------------------------------------------------------------

_LOADER_PREFIXES = ("load", "_load", "read", "_read")

_EMPTY_CONSTANTS = (None, "", 0, 0.0, False)


def _is_empty_literal(node: ast.AST) -> bool:
    """True for the defaults a loader falls back to: None, {}, [], "", 0, False."""
    if isinstance(node, ast.Constant):
        return node.value is None or node.value in _EMPTY_CONSTANTS
    if isinstance(node, (ast.Dict, ast.List, ast.Set)):
        return not (getattr(node, "keys", None) or getattr(node, "elts", None))
    if isinstance(node, ast.Tuple):
        return not node.elts
    return False


def _handler_is_silent(handler: ast.ExceptHandler) -> bool:
    """True when this handler recovers without recording anything.

    "Recording" is read generously on purpose, because the point is to catch
    handlers that say NOTHING rather than to dictate how they speak. Any call
    (a warn, a print, a logger), any raise, any assignment or counter bump
    counts as a record. What is left over is a handler whose entire body is
    `pass`, `continue`, `break`, or a return of the empty default -- which is
    indistinguishable, from the caller's side, from the file being legitimately
    empty. `company_registry.load()` was exactly this, and the caller then
    saved the empty dict over the real registry.
    """
    for sub in ast.walk(handler):
        if sub is handler:
            continue
        if isinstance(sub, (ast.Call, ast.Raise, ast.Assert, ast.Assign,
                            ast.AugAssign, ast.AnnAssign, ast.With, ast.Yield)):
            return False
    for stmt in handler.body:
        if isinstance(stmt, (ast.Pass, ast.Continue, ast.Break)):
            continue
        if isinstance(stmt, ast.Return) and (stmt.value is None
                                             or _is_empty_literal(stmt.value)):
            continue
        return False
    return True


class LoadersReportTheirFallback(InvariantTestCase):
    """A read path may not raise. It may not stay quiet either.

    The repo convention is that a tool must not die because its config is
    malformed, and that is right. The half that keeps it safe is that the
    fallback names itself: `{}` from a corrupt file and `{}` from an absent one
    are the same value and completely different situations.

    Scoped to `load*`/`read*` deliberately. A pure coercer returning None
    (`job_schema.to_monthly_sgd`, `parse_date`) is NOT this bug -- None is a
    documented "cannot be trusted" that every caller tests for. An empty
    container from a loader is the bug, because nothing distinguishes it from
    real data.
    """

    ALLOW: dict[tuple[str, str], str] = {
        ("render_resume", "_load_typst"):
            "Capability probe, not a data read. `except ImportError: return "
            "None` IS the answer -- and `capabilities()` reports the resulting "
            "absence to the user, so the fallback is announced one level up.",
        ("render_resume", "_load_docx"):
            "Same: optional-dependency probe reported by `capabilities()`.",
        ("render_resume", "_load_pypdf"):
            "Same: optional-dependency probe reported by `capabilities()`.",
        ("render_excel", "_load_xlsxwriter"):
            "Same: optional-dependency probe. The degraded CSV path stamps "
            "`degraded: True` into its own result, so the caller is told.",
    }

    def test_loader_fallbacks_are_announced(self):
        violations = []
        for name, path, tree in MODULES:
            for fn, fn_name in functions_in(tree):
                if not fn_name.startswith(_LOADER_PREFIXES):
                    continue
                for node in ast.walk(fn):
                    if not isinstance(node, ast.ExceptHandler):
                        continue
                    if not _handler_is_silent(node):
                        continue
                    if (path.stem, fn_name) in self.ALLOW:
                        continue
                    violations.append(
                        f"{where(path, node)} in {fn_name}() recovers to a "
                        f"default without recording anything")
        self.assertNoViolations(
            violations,
            "A load*/read* function that catches an exception and returns a "
            "default must emit something -- a warn, a print, a counter the "
            "caller reports. Otherwise a corrupt input and an absent one are "
            "the same return value, and the next write persists the wrong one.")

    def test_the_rule_can_actually_fire(self):
        # A structural rule that matches nothing is not a passing test, it is a
        # broken detector. Prove the detector on a known-bad sample.
        bad = ast.parse("def load_x():\n"
                        "    try:\n        return read()\n"
                        "    except OSError:\n        return {}\n")
        handler = next(n for n in ast.walk(bad) if isinstance(n, ast.ExceptHandler))
        self.assertTrue(_handler_is_silent(handler))

        good = ast.parse("def load_x():\n"
                         "    try:\n        return read()\n"
                         "    except OSError as e:\n"
                         "        _warn(e)\n        return {}\n")
        handler = next(n for n in ast.walk(good) if isinstance(n, ast.ExceptHandler))
        self.assertFalse(_handler_is_silent(handler))


# --------------------------------------------------------------------------
# 3. an `ok` result may not contradict itself
# --------------------------------------------------------------------------

_ERROR_FIELDS = {"error", "errors", "problems", "failures", "violations"}

# Fields whose emptiness means the result carries nothing, whatever `ok` says.
# `fetcher` returned PageResult(ok=True, html="") for a Zyte quota error: the
# caller saw success, got no page, and scored the job on an empty description.
_PAYLOAD_FIELDS = {"html", "text", "body", "data", "content", "rows"}


def _dataclass_field_order(tree: ast.Module) -> dict[str, list[str]]:
    """class name -> annotated field names, in declaration order."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            fields = [s.target.id for s in node.body
                      if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
            if fields:
                out[node.name] = fields
    return out


ALL_FIELD_ORDERS: dict[str, list[str]] = {}
for _n, _p, _t in MODULES:
    ALL_FIELD_ORDERS.update(_dataclass_field_order(_t))


def _named_arguments(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """(name, value) pairs for a dict literal or a constructor call.

    Positional arguments are resolved against the class's declared field order,
    because `PageResult(True, url, "")` is the same assertion as
    `PageResult(ok=True, html="")` and a keyword-only check would miss the
    entire fetcher module, where every construction site is positional.
    """
    if isinstance(node, ast.Dict):
        return [(k.value, v) for k, v in zip(node.keys, node.values)
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        pairs = [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
        order = ALL_FIELD_ORDERS.get(name or "")
        if order:
            pairs += list(zip(order, node.args))
        return pairs
    return []


class OkNeverContradictsItself(InvariantTestCase):
    """`ok: True` is a claim, and it must not be made next to its own refutation.

    Two shapes, both of which shipped:

      `ok=True` with a non-empty error   -- says success and failure at once;
                                            callers branch on `ok` and lose.
      `ok=True` with an empty payload    -- `PageResult(ok=True, html="")` for a
                                            Zyte quota error. Nothing downstream
                                            could tell that page from a real one.

    Checked at the construction site rather than on the returned value, because
    that is where the two facts are visible together and where a reviewer can
    see the contradiction without running anything.
    """

    ALLOW: dict[tuple[str, str], str] = {}

    def test_ok_true_is_never_paired_with_its_own_refutation(self):
        violations = []
        for name, path, tree in MODULES:
            for node in ast.walk(tree):
                pairs = _named_arguments(node)
                if not pairs:
                    continue
                ok_values = [v for k, v in pairs if k == "ok"]
                if not ok_values:
                    continue
                ok = ok_values[0]
                if not (isinstance(ok, ast.Constant) and ok.value is True):
                    continue
                for key, value in pairs:
                    if key in _ERROR_FIELDS and isinstance(value, ast.Constant) \
                            and value.value not in (None, ""):
                        violations.append(
                            f"{where(path, node)} asserts ok=True beside a "
                            f"non-empty {key!r}")
                    if key in _PAYLOAD_FIELDS and _is_empty_literal(value):
                        violations.append(
                            f"{where(path, node)} asserts ok=True with an empty "
                            f"{key!r} -- success carrying no payload")
        self.assertNoViolations(
            violations,
            "A result that says ok=True must not carry an error, and must not "
            "be empty. Both shapes produce a plausible value that every caller "
            "reads as success.")

    def test_the_rule_can_actually_fire(self):
        sample = ast.parse('r = {"ok": True, "error": "quota exhausted"}')
        node = next(n for n in ast.walk(sample) if isinstance(n, ast.Dict))
        pairs = dict(_named_arguments(node))
        self.assertIn("error", pairs)
        self.assertTrue(isinstance(pairs["ok"], ast.Constant) and pairs["ok"].value is True)

    def test_positional_construction_is_resolved(self):
        # The fetcher bug was positional. If field-order resolution regresses,
        # this rule goes blind to the module it was written for.
        self.assertIn("PageResult", ALL_FIELD_ORDERS)
        self.assertEqual(ALL_FIELD_ORDERS["PageResult"][0], "ok")
        sample = ast.parse('PageResult(True, url, "")')
        node = next(n for n in ast.walk(sample) if isinstance(n, ast.Call))
        pairs = dict(_named_arguments(node))
        self.assertIn("html", pairs, "positional args were not mapped to fields")


# --------------------------------------------------------------------------
# 4. lazy caches: flag last, and under a lock
# --------------------------------------------------------------------------

_FLAG_KEYS = {"loaded", "cached", "ready", "initialised", "initialized"}


def _module_level_caches(tree: ast.Module) -> dict[str, tuple[str, list[str]]]:
    """name -> (flag key, data keys) for module-level dicts used as lazy caches."""
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        keys = [k.value for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        flags = [k for k in keys if k in _FLAG_KEYS]
        if not flags:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = (flags[0], [k for k in keys if k not in flags])
    return out


def _subscript_assignments(fn: ast.AST, cache: str):
    """(key, node) for every `cache["key"] = ...` inside this function."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == cache
                    and isinstance(target.slice, ast.Constant)):
                yield target.slice.value, node


def _enclosing_with_lock(fn: ast.AST, node: ast.AST) -> bool:
    """True when `node` sits inside a `with <something-lock>:` block in `fn`."""
    for candidate in ast.walk(fn):
        if not isinstance(candidate, ast.With):
            continue
        names = []
        for item in candidate.items:
            expr = item.context_expr
            names.append(getattr(expr, "id", "") or getattr(expr, "attr", ""))
        if not any("lock" in n.lower() for n in names if n):
            continue
        start = candidate.body[0].lineno
        end = max(getattr(s, "end_lineno", s.lineno) for s in candidate.body)
        if start <= node.lineno <= end:
            return True
    return False


class LazyCachesArePublishedSafely(InvariantTestCase):
    """The defect that failed roughly three in four tailoring jobs, in one rule.

    Three modules independently grew the same lazy cache, and all three set the
    flag BEFORE the work that fills it:

        if _cache["loaded"]:
            return _cache["data"]    # None, for a moment
        _cache["loaded"] = True      # set too early
        ... read the file ...
        _cache["data"] = data

    A second thread arriving inside that window sees `loaded` true and receives
    `None`. It does not crash here; it crashes somewhere else, intermittently,
    and never on a single job. It was blamed on thread safety in the wrong
    module and then on a transient API response before anyone captured the
    traceback.

    Two structural requirements, both of which kill it:

      the flag is assigned LAST   -- no thread can observe the flag without
                                     the data it promises
      the assignment holds a lock -- and the check inside the lock is repeated,
                                     so the window closes rather than narrows

    `reload()` clearing the flag to False is a different operation and is not
    matched: only assignments of True publish a cache.
    """

    ALLOW: dict[tuple[str, str], str] = {}

    def test_the_flag_is_set_after_the_data(self):
        violations = []
        for name, path, tree in MODULES:
            caches = _module_level_caches(tree)
            for fn, fn_name in functions_in(tree):
                for cache, (flag, data_keys) in caches.items():
                    publishes = [n for key, n in _subscript_assignments(fn, cache)
                                 if key == flag and isinstance(n.value, ast.Constant)
                                 and n.value.value is True]
                    if not publishes:
                        continue
                    flag_line = min(n.lineno for n in publishes)
                    for key, node in _subscript_assignments(fn, cache):
                        if key in data_keys and node.lineno > flag_line:
                            violations.append(
                                f"{where(path, node)} {fn_name}(): "
                                f"{cache}[{key!r}] is filled AFTER "
                                f"{cache}[{flag!r}] = True (line {flag_line})")
        self.assertNoViolations(
            violations,
            "A lazy cache must publish its flag last. Setting it first hands a "
            "racing thread a half-built cache that reads as fully loaded.")

    def test_publishing_the_flag_holds_a_lock(self):
        violations = []
        for name, path, tree in MODULES:
            caches = _module_level_caches(tree)
            if not caches:
                continue
            for fn, fn_name in functions_in(tree):
                for cache, (flag, _data) in caches.items():
                    for key, node in _subscript_assignments(fn, cache):
                        if key != flag:
                            continue
                        if not (isinstance(node.value, ast.Constant)
                                and node.value.value is True):
                            continue
                        if (path.stem, fn_name) in self.ALLOW:
                            continue
                        if not _enclosing_with_lock(fn, node):
                            violations.append(
                                f"{where(path, node)} {fn_name}(): "
                                f"{cache}[{flag!r}] = True outside any lock")
        self.assertNoViolations(
            violations,
            "Ordering alone narrows the race; it does not close it. The "
            "publish must happen under a lock, with the flag re-checked inside "
            "it, or two threads still both do the work and one wins.")

    def test_a_module_with_a_lazy_cache_declares_a_lock(self):
        violations = []
        for name, path, tree in MODULES:
            if not _module_level_caches(tree):
                continue
            source = path.read_text(encoding="utf-8")
            if "threading.Lock()" not in source and "Lock()" not in source:
                violations.append(f"{path.name} has a lazy cache and no Lock")
        self.assertNoViolations(
            violations, "A module with a lazy cache must own a lock for it.")

    def test_the_rule_can_actually_fire(self):
        # Detector proof: the original buggy shape must be detected as buggy.
        buggy = ast.parse(
            '_c = {"loaded": False, "data": None}\n'
            'def load():\n'
            '    _c["loaded"] = True\n'
            '    _c["data"] = read()\n')
        caches = _module_level_caches(buggy)
        self.assertIn("_c", caches)
        fn = next(n for n in ast.walk(buggy) if isinstance(n, ast.FunctionDef))
        flag, data_keys = caches["_c"]
        flag_line = min(n.lineno for k, n in _subscript_assignments(fn, "_c")
                        if k == flag)
        late = [k for k, n in _subscript_assignments(fn, "_c")
                if k in data_keys and n.lineno > flag_line]
        self.assertEqual(late, ["data"], "detector missed the flag-first bug")

    def test_the_caches_it_guards_are_still_there(self):
        # If the caches are renamed or restructured, this whole class silently
        # stops guarding anything. Pin the three it was written for.
        found = set()
        for name, path, tree in MODULES:
            for cache in _module_level_caches(tree):
                found.add(f"{path.stem}.{cache}")
        for expected in ("token_budget._profiles_cache", "token_budget._backend",
                         "model_config._cache"):
            self.assertIn(expected, found,
                          "a known lazy cache vanished from the scan -- either "
                          "it was removed, or the detector stopped seeing it")


# --------------------------------------------------------------------------
# 5. config reads survive a BOM
# --------------------------------------------------------------------------

class ConfigReadsTolerateABOM(InvariantTestCase):
    """Every text read of a config file uses `utf-8-sig`.

    Notepad and PowerShell's `Set-Content -Encoding utf8` both write a byte
    order mark. Read as plain `utf-8` the BOM survives into the string, and
    `json.loads` rejects it -- which lands in the module's own except handler
    and degrades to "no config", so a file that is present and correct reads as
    absent. Every hand-edited file in this repo (the .env holding the API key,
    the draft profile a human verifies fact by fact) is exposed to it.

    `utf-8-sig` reads correctly with OR without a BOM, so there is no case for
    plain `utf-8` on a read path.
    """

    ALLOW: dict[tuple[str, str], str] = {}

    def test_text_reads_use_utf8_sig(self):
        violations = []
        for name, path, tree in MODULES:
            for fn, fn_name in functions_in(tree):
                for node in ast.walk(fn):
                    if not isinstance(node, ast.Call):
                        continue
                    is_open = isinstance(node.func, ast.Name) and node.func.id == "open"
                    is_read_text = (isinstance(node.func, ast.Attribute)
                                    and node.func.attr == "read_text")
                    if not (is_open or is_read_text):
                        continue

                    mode = "r"
                    if is_open and len(node.args) > 1 and \
                            isinstance(node.args[1], ast.Constant):
                        mode = str(node.args[1].value)
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode = str(kw.value.value)
                    if any(c in mode for c in "waxb+"):
                        continue      # writing, or binary: no encoding applies

                    encoding = None
                    for kw in node.keywords:
                        if kw.arg == "encoding" and isinstance(kw.value, ast.Constant):
                            encoding = kw.value.value
                    if encoding == "utf-8-sig":
                        continue
                    if (path.stem, fn_name) in self.ALLOW:
                        continue
                    violations.append(
                        f"{where(path, node)} in {fn_name}() reads text as "
                        f"{encoding!r}, not 'utf-8-sig'")
        self.assertNoViolations(
            violations,
            "A text read must use encoding='utf-8-sig'. Plain 'utf-8' turns a "
            "BOM into a parse error, which every read path here catches and "
            "reports as a missing file -- so correct config reads as absent.")


if __name__ == "__main__":
    unittest.main()
