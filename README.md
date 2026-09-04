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
Enrichment Agent  ──→  Tavily Search (last 30 days)
    │                  GPT-OSS 20B via Groq
    │                  Returns: 3-5 sentence research brief
    ▼
Email Drafter     ──→  GPT-OSS 20B via Groq
    │                  Returns: subject line + <120 word email
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
| LLM | GPT-OSS 20B via Groq (model id set in `config.py`, overridable via `GROQ_MODEL`) |
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
├── config.py            ← Model ids (env-overridable)
├── requirements.txt
├── .env.example
├── agents/
│   ├── enricher.py      ← LangChain + Tavily research agent
│   └── drafter.py       ← LLM email drafting chain
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

### Evals

Every prompt change should be measured, not eyeballed. The harness runs a fixed 25-lead
golden set through enrichment and drafting (never sends), scores each draft, and appends a
row to `eval/history.csv` tagged with the current `PROMPT_VERSION`.

```bash
python -m eval.runner              # full golden set
python -m eval.runner --limit 5    # quick pass while iterating
python -m eval.runner --no-judge   # deterministic checks only, no LLM judge cost
```

Scored on: word count ≤120, subject line present, no banned phrases, exactly one CTA, plus
an LLM judge rating personalisation specificity 1-5. Bump `PROMPT_VERSION` in
`eval/prompt_version.py` whenever you edit a system prompt, so history rows stay comparable.

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

Statuses: `SENT` · `DRY_RUN` · `ENRICH_ERROR` · `DRAFT_ERROR` · `SEND_ERROR`

---

## Key design decisions

- **`days=30` on Tavily** — filters search results to last 30 days so research is always current, not cached old news
- **Two-step pipeline** — enrichment and drafting are separate so each can fail/retry independently
- **Rate limiting** — 60s default between sends to avoid spam filters
- **Modular** — swap Groq → OpenAI, Tavily → SerpAPI, SendGrid → SMTP with one-line changes

---