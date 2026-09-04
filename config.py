"""Shared runtime config.

Model ids live here because Groq retires models on short notice — llama-3.1-8b-instant,
which this project was built on, is already gone. Overridable via .env so a swap needs
no code change.
"""

import os

# gpt-oss-120b over the 20b: it scores better on the eval AND costs fewer tokens per draft
# (1125 vs 1536), because the 20b spends ~89% of its output on reasoning it discards. Every
# model here shares one 8000 TPM ceiling, so tokens-per-draft is the real throughput limit.
# qwen3.8-27b is the throughput alternative — 478 tokens/draft, ~3x more leads per minute,
# but weaker personalisation (judge 2.25 vs 3.00). Swap via GROQ_MODEL if volume matters more.
PIPELINE_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Judging is one call per lead, so a stronger model is affordable here and makes the
# score worth more than the drafter grading its own homework.
JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
