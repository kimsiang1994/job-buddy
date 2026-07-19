"""Offline test suites for jobbuddy.

    py -m unittest discover -s tests -t .

Every suite here runs with no network, no API key and no cost -- except
test_deepseek.py, which is a live smoke test and is excluded from CI.
"""
