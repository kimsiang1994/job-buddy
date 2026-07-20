"""Where a run's deliverables go, and why that is harder than it looks.

The tree is `potential applications/<scope>/<YYYY-MM-DD_HHMMSS>/<job>/`, and
every part of the last component is attacker-shaped input: a job title written
by whoever posted it, on Windows, where the filesystem has opinions.

Four failures, all of which have happened to somebody:

  **MAX_PATH.** Windows refuses a path over 260 characters unless long paths are
  explicitly enabled, and the failure surfaces as a `FileNotFoundError` on a
  directory that plainly exists. So the job component is truncated to fit --
  and only the job component. The root is not ours to shorten: truncating it
  would move the whole run somewhere the caller did not ask for, which is worse
  than a long name. Where the root alone eats the budget, this returns a
  minimum-length name and lets the OS complain, because a silently relocated
  deliverable is not a fix.

  **Truncation collides.** "Senior Machine Learning Engineer, Retrieval..." and
  "Senior Machine Learning Engineer, Ranking..." share a 40-character prefix.
  Truncating alone would have the second job overwrite the first's resume, and
  the user would never learn that it happened. Every truncated name therefore
  carries a short digest of the FULL original, so distinct titles stay distinct.

  **Illegal characters and trailing punctuation.** `<>:"/\\|?*` are rejected
  outright; a trailing dot or space is accepted by the API and then silently
  stripped by the shell, so `mkdir` and the later `open` disagree about the
  name.

  **Reserved device names.** `CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9` and
  `LPT1`-`LPT9` are devices, not files, and they stay reserved WITH an
  extension -- `NUL.txt` is still the null device. A job at a company called
  "Aux" is not hypothetical enough to ignore.

Nothing here touches the disk except `ensure_dir`. Path construction is pure so
the rules above are testable without a filesystem.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_DIR / "potential applications"

# The classic Windows limit. Not raised even when long paths are enabled on the
# machine, because the deliverables get opened by Explorer, Word and Excel, and
# those are not uniformly long-path aware.
MAX_PATH = 260

# Room left inside a job directory for the longest file it will hold. The
# current worst case is `report.distribution.svg` (24); 40 leaves headroom for
# another artefact without re-deriving every path.
FILENAME_HEADROOM = 40

# Below this a name carries no information at all, so the budget stops here even
# when that means exceeding MAX_PATH.
MIN_COMPONENT = 8

# Long enough that a collision between two real job titles is not a thing that
# happens; short enough to leave the readable prefix room.
HASH_LEN = 6

FALLBACK = "job"

_ILLEGAL = re.compile(r'[<>:"/\\|?*]|[\x00-\x1f\x7f]')
_WHITESPACE = re.compile(r"\s+")
_DASH_RUN = re.compile(r"-{2,}")

# Reserved with or without an extension. `NUL.txt` is the null device.
RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{digit}" for digit in "123456789"}
    | {f"LPT{digit}" for digit in "123456789"}
)


def sanitise_component(name: str, *, fallback: str = FALLBACK) -> str:
    """One path component Windows will actually accept.

    Never returns an empty string: a title that sanitises away entirely (a
    Chinese-only title mangled by an encoding, or a string of slashes) still
    needs a directory to put the deliverable in, and failing the whole job over
    a name is a worse outcome than a generic one.
    """
    text = _ILLEGAL.sub("-", str(name or ""))
    text = _WHITESPACE.sub(" ", text)
    text = _DASH_RUN.sub("-", text)
    # Leading and trailing dots and spaces both: a trailing one is stripped by
    # the shell after `mkdir` accepted it, so the directory that exists is not
    # the one whose name we recorded.
    text = text.strip(" .-")
    if not text:
        return fallback

    # The device check is on the stem, because the reservation survives an
    # extension. Prefixed rather than replaced so the title stays readable.
    if text.split(".", 1)[0].strip().upper() in RESERVED_NAMES:
        text = "_" + text
    return text


def timestamp(when: datetime | None = None) -> str:
    """The run stamp. Sorts lexicographically, which is the whole point."""
    return (when or datetime.now()).strftime("%Y-%m-%d_%H%M%S")


def _digest(original: str) -> str:
    return hashlib.sha1(
        str(original).encode("utf-8", "surrogatepass")).hexdigest()[:HASH_LEN]


def _fit(component: str, budget: int, original: str) -> str:
    """`component` shortened to `budget`, kept unique by a digest of `original`.

    The digest is of the ORIGINAL title rather than of the truncated prefix,
    which is the only version that distinguishes two titles sharing a prefix --
    the case that would otherwise have one job overwrite another.

    Below `HASH_LEN + 2` there is no room for a readable prefix, and the budget
    is deliberately overshot: the digest is returned WHOLE. The previous version
    of this branch returned `_digest(original)[:max(1, budget)]` under a comment
    claiming "uniqueness beats legibility" -- which the code then contradicted,
    because one hex character is sixteen buckets and two jobs in the same run
    collide at better than even odds once there are six of them. A collision
    here silently overwrites a deliverable, which is the one outcome this whole
    module exists to prevent, so the name is allowed to exceed the budget by a
    few characters instead. That is the same trade the module docstring already
    makes for the root: exceed MAX_PATH and let the OS complain, rather than
    quietly lose a file.

    `job_component` cannot currently reach this: `MIN_COMPONENT` (8) floors the
    budget at exactly `HASH_LEN + 2`. It is kept, honest, rather than deleted,
    because it is the branch that makes the invariant safe to change -- lowering
    `MIN_COMPONENT` should shorten names, not start losing resumes.
    """
    if len(component) <= budget:
        return component
    if budget < HASH_LEN + 2:
        return _digest(original)
    keep = budget - HASH_LEN - 1
    prefix = component[:keep].rstrip(" .-") or component[:keep]
    return f"{prefix}-{_digest(original)}"


def job_component(job_label: str, parent: Path,
                  taken: set[str] | None = None) -> str:
    """The directory name for one job under `parent`, sanitised and made to fit.

    `taken` is the set of names already used in this parent. Two jobs with the
    SAME title genuinely collide -- the digest cannot separate them because the
    input is identical -- so they are numbered. Pass the same set across a run
    and it is updated in place.
    """
    name = sanitise_component(job_label)
    budget = max(MIN_COMPONENT,
                 MAX_PATH - len(str(parent)) - 1 - FILENAME_HEADROOM)
    candidate = base = _fit(name, budget, str(job_label))

    if taken is not None:
        lowered = {t.casefold() for t in taken}
        counter = 2
        while candidate.casefold() in lowered:
            suffix = f"-{counter}"
            trimmed = base[:max(1, budget - len(suffix))].rstrip(" .-") or base[:1]
            candidate = f"{trimmed}{suffix}"
            counter += 1
        taken.add(candidate)
    return candidate


def run_root(scope: str, stamp: str, base: Path | None = None) -> Path:
    """`<base>/<scope>/<stamp>`. The part that is never truncated."""
    root = Path(base) if base is not None else OUTPUT_DIR
    return (root / sanitise_component(scope, fallback="scope")
            / sanitise_component(stamp, fallback="run"))


def job_dir(scope: str, stamp: str, job_label: str,
            base: Path | None = None,
            taken: set[str] | None = None) -> Path:
    """The full directory for one job's deliverables. Does not create it."""
    parent = run_root(scope, stamp, base)
    return parent / job_component(job_label, parent, taken)


def job_label(job: dict) -> str:
    """`Title - Company`, or whichever half exists.

    Company is included because the same title from three employers is the
    normal case in a scope, and three directories called "Data Engineer" tell
    the user nothing.
    """
    title = str((job or {}).get("title") or "").strip()
    company = str((job or {}).get("company") or "").strip()
    if title and company:
        return f"{title} - {company}"
    return title or company or FALLBACK


def ensure_dir(path: Path) -> Path:
    """Create the directory and return it. The only function here that writes."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
