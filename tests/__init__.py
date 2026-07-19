"""Offline test suites for jobbuddy.

    py -m unittest discover -s tests -t .

Every suite here runs with no network, no API key and no cost -- except
test_deepseek.py, which is a live smoke test and is excluded from CI.

Writable paths are redirected to a temp directory before any suite imports the
package. The pipeline reaches `company_registry` several layers down, so no
test named it and the suite spent an unknown length of time writing test
fixtures into the real `config/companies.json` -- visible only as a dirty
working tree that was easy to mistake for someone's edit.
"""

import os
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="jobbuddy-tests-"))
os.environ.setdefault("JB_COMPANY_REGISTRY", str(_SANDBOX / "companies.json"))
