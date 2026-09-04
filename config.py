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

# The critic decides whether an email is allowed to reach a real person, so it gets the
# strongest model available and temperature 0 — a flaky gate is worse than no gate.
CRITIC_MODEL = os.getenv("GROQ_CRITIC_MODEL", "openai/gpt-oss-120b")

# Drafts allowed per lead before it goes to a human: the first draft plus one rewrite.
MAX_DRAFT_ATTEMPTS = int(os.getenv("MAX_DRAFT_ATTEMPTS", "2"))

# Research is cached per company, so a batch of leads at one firm costs one pass. Set
# USE_RESEARCH_CACHE=0 to force fresh lookups (e.g. when evaluating enrichment changes —
# a cache hit would otherwise serve results from the previous prompt version).
USE_RESEARCH_CACHE = os.getenv("USE_RESEARCH_CACHE", "1") != "0"
ENRICHMENT_TTL_SECONDS = int(os.getenv("ENRICHMENT_TTL_SECONDS", str(7 * 24 * 60 * 60)))

# Minimum ICP score for a lead to be worth spending tokens on. Deliberately permissive by
# default: it drops clear mismatches (interns, missing company) rather than making fine
# judgement calls, because a wrongly skipped lead is invisible unless you go looking.
ICP_MIN_SCORE = float(os.getenv("ICP_MIN_SCORE", "0.3"))
