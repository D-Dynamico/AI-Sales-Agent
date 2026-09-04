"""Bump this by hand whenever a SYSTEM_PROMPT in agents/ changes.

Scores in eval/history.csv are only comparable across runs that share a version,
so this is what lets you say "v3 raised grounded-claim rate from 61% to 88%".
"""

PROMPT_VERSION = "v2"
