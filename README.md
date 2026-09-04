# AI Sales Outreach Agent

An autonomous AI agent that researches B2B leads in real-time, drafts hyper-personalised
cold emails, and logs everything — with zero human input per lead.

Built with Python, LangChain, GPT-OSS on Groq, and Tavily Search.

🔗 **Live Demo**: [ai-sales-agent.streamlit.app](https://ai-sales-agent-3texcwappcokvpr7afudyav.streamlit.app/)

---

## What it does

1. **Researches each lead** — searches the web for recent company news, funding, product launches, and role-specific context (last 30 days only)
2. **Drafts a personalised email** — uses an LLM on Groq to write a <120-word cold email grounded in that research
3. **Logs everything** — saves status, draft preview, and timestamps to a CSV
4. **Sends via SendGrid or SMTP** — with rate limiting and resume support

## Architecture

```
leads.csv
    │
    ▼
ICP filter        ──→  Rule-based, no tokens. Poor-fit leads skipped before spending
    │
    ▼
Planner           ──→  Picks 3 lookup angles from the role (funding, hiring, launches...)
    │
    ▼
Enrichment Agent  ──→  Those angles searched in parallel via Tavily (last 30 days)
    │                  Cached per company — 10 leads at one firm = 1 research pass
    │                  GPT-OSS 120B structures them into trigger_events + confidence
    │                  Returns: research brief + raw source snippets
    ▼
Email Drafter     ──→  GPT-OSS 120B via Groq, structured output
    │                  Returns: EmailDraft(subject, body) — no regex parsing
    ▼
Critic (gate)     ──→  Extracts every claim, traces each to a source quote
    │                  Fails → rewrite with the critique → re-check
    │                  Still failing after 2 attempts → NEEDS_REVIEW, never sent
    ▼
Sender            ──→  SendGrid (primary) / SMTP (fallback)
    │
    ▼
Logger            ──→  output_log.csv
```

---

## Stack

| Layer | Tool |
|---|---|
| LLM | GPT-OSS 120B via Groq (model id in `config.py`, overridable via `GROQ_MODEL`) |
| Orchestration | LangChain |
| Web search | Tavily Search (30-day recency filter) |
| Email delivery | SendGrid / Gmail SMTP |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## Project structure

```
sales-agent/
├── app.py               ← Streamlit UI
├── main.py              ← CLI runner
├── config.py            ← Model ids, budgets, thresholds (env-overridable)
├── persona.toml         ← Who is sending, what they sell, who they sell to
├── persona.py           ← Loads persona.toml (env vars win)
├── requirements.txt
├── .env.example
├── agents/
│   ├── enricher.py      ← Plans, searches in parallel, structures + caches research
│   ├── planner.py       ← Picks lookup angles per role; parallel execution + budget
│   ├── icp.py           ← Rule-based fit scoring, runs before any tokens are spent
│   ├── cache.py         ← Per-company research cache (7-day TTL)
│   ├── drafter.py       ← LLM email drafting + rewrite-on-critique chain
│   ├── critic.py        ← Grounding gate: claim extraction + source verification
│   ├── errors.py        ← User-safe error messages; redacts ids/keys from logs
│   ├── pipeline.py      ← draft → critique → rewrite loop (shared by CLI, UI, evals)
│   ├── structured.py    ← Structured output + JSON fallback, shared by all callers
│   └── schemas.py       ← EmailDraft / FactualClaim / CriticVerdict / ResearchResult
├── tools/
│   ├── sender.py        ← SendGrid / SMTP sender
│   └── logger.py        ← CSV logger with duplicate detection
├── eval/
│   ├── runner.py        ← Runs the golden set, scores every draft
│   ├── checks.py        ← Deterministic checks (words, subject, clichés, CTA)
│   ├── llm_judge.py     ← LLM judge for personalisation quality
│   ├── report.py        ← Aggregation + history.csv
│   ├── prompt_version.py← Hand-bumped version label for score comparison
│   └── history.csv      ← One row per eval run
└── data/
    ├── leads.csv        ← Input leads
    ├── golden_leads.csv ← 25-lead fixed set for evals
    └── output_log.csv   ← Auto-created on first run
```

---

## Setup

```bash
git clone https://github.com/yourusername/ai-sales-agent.git
cd ai-sales-agent
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly_...
```

Then make it yours — edit `persona.toml`:

```toml
[sender]
name = "Your Name"
company = "Your Company"
from_email = "you@yourcompany.com"

[offering]
description = "what you do for the recipient, in one concrete line"
```

That single file drives the drafter's prompt, the email headers, the Streamlit title and
the ICP keyword lists. No prompt editing required, and every value can be overridden by an
environment variable (`SENDER_NAME`, `SENDER_COMPANY`, `SENDER_OFFERING`, `FROM_EMAIL`) for
deployments.

---

## Usage

### Streamlit UI
```bash
streamlit run app.py
```

### CLI
```bash
# Dry run — prints emails, sends nothing
python main.py --dry-run

# Real send — 60s delay between emails
python main.py

# Resume after interruption
python main.py --resume

# Custom delay between sends
python main.py --delay 90
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

194 tests, no network calls — every LLM and search client is mocked, so the suite is free,
offline and runs in about a second. That's deliberate: the live models sit behind a 200k
tokens-per-day cap, and a suite that called them would compete with the eval harness for the
same budget. Tests and evals answer different questions — tests verify logic, evals score
output quality.

### Evals

Every prompt change should be measured, not eyeballed. The harness runs a fixed 25-lead
golden set through enrichment and drafting (never sends), scores each draft, and appends a
row to `eval/history.csv` tagged with the current `PROMPT_VERSION`.

```bash
python -m eval.runner              # full golden set
python -m eval.runner --limit 5    # quick pass while iterating
python -m eval.runner --no-judge   # deterministic checks only, no LLM judge cost
```

Scored on: word count ≤120, subject line present, no banned phrases, exactly one CTA, an LLM
judge rating personalisation specificity 1-5, and — the headline number — **grounded claim
rate**: what fraction of all factual claims across all drafts trace back to a real source.
Bump `PROMPT_VERSION` in `eval/prompt_version.py` whenever you edit a system prompt, so
history rows stay comparable; `pipeline_model` and `judge_model` are recorded automatically.

---

## Input format

`data/leads.csv`:
```
name,role,company,email,website
Priya Sharma,Head of Sales,Razorpay,priya@razorpay.com,razorpay.com
Arjun Mehta,VP Marketing,Zepto,arjun@zepto.team,zeptonow.com
```

## Output format

`data/output_log.csv`:
```
timestamp,name,role,company,email,status,draft_preview
2026-03-31T10:23:01Z,Priya Sharma,Head of Sales,Razorpay,...,SENT,"Subject: ..."
```

Statuses: `SENT` · `DRY_RUN` · `NEEDS_REVIEW` · `ICP_SKIPPED` · `ENRICH_ERROR` · `DRAFT_ERROR` · `SEND_ERROR`

`NEEDS_REVIEW` rows carry the critic's objections in the `critique` column, so the review
queue can be worked straight from the CSV. They are never sent.

---

## Key design decisions

- **Nothing sends unless its claims trace to a source** — the critic extracts every factual
  claim about the recipient and requires a verbatim supporting quote from the search results.
  Claims are re-verified in Python, so the gate can't be talked into passing; `passed` is
  computed from evidence, never asserted by the model. Two attempts, then a human looks at it
- **The critic checks drafts against sources, not against the research brief** — so it catches
  errors the *enricher* introduced too, not just the drafter's. It has already caught a bad
  unit conversion (Rs 12,626 crore restated as "Rs 12.6 billion", off by 10x)
- **Provider errors are never shown raw** — Groq's rate-limit message carries the org id,
  the quota and a billing upsell. Viewers get "Model capacity for today is used up. Try
  again in about 18m." while logs keep the redacted technical detail
- **Sender identity is config, not prompt text** — company, name, offering and ICP keywords
  live in `persona.toml`. Changing who the emails come from used to mean editing the system
  prompt, which is the one place an edit silently changes model behaviour
- **Structured output everywhere, with a plain-JSON fallback** — the drafter returns
  `EmailDraft(subject, body)` instead of a blob a regex has to unpick, so a model that
  forgets the header can't silently turn the first line of the email into its subject. The
  fallback exists because one live call produced four perfect research events and then wrote
  `"confidence": 0. nine`, losing all of it to a single malformed token
- **Research is cached per company, not per lead** — ten contacts at one firm cost one
  research pass. Not just tidiness: the models sit behind a 200k tokens-per-day cap
- **The ICP filter and lookup planner are rule-based, not LLM calls** — both exist to spend
  tokens better, so paying a model call to run them would defeat the point. Rules are also
  inspectable, which matters when the decision is "never contact this person"
- **`days=30` on Tavily** — filters search results to last 30 days so research is always current, not cached old news
- **Two-step pipeline** — enrichment and drafting are separate so each can fail/retry independently
- **Rate limiting** — 60s default between sends to avoid spam filters
- **Modular** — swap Groq → OpenAI, Tavily → SerpAPI, SendGrid → SMTP with one-line changes
- **GPT-OSS 120B over the 20B** — every Groq model here shares one 8000 TPM ceiling, so
  tokens-per-draft sets throughput, not the rate limit. The 20B spends ~89% of its output on
  reasoning it discards (1536 tokens/draft vs the 120B's 1125), making the *larger* model both
  cheaper and better here. `qwen3.8-27b` is the volume option — 478 tokens/draft and ~3x the
  throughput, but weaker personalisation; set `GROQ_MODEL` to switch, no code change needed

---