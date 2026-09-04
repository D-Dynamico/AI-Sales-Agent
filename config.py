"""Shared runtime config.

Model ids live here because Groq retires models on short notice — llama-3.1-8b-instant,
which this project was built on, is already gone. Overridable via .env so a swap needs
no code change.
"""

import os

PIPELINE_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Judging is one call per lead, so a stronger model is affordable here and makes the
# score worth more than the drafter grading its own homework.
JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
