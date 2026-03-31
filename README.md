# ⚗️ AI Sales Outreach Agent

An autonomous AI agent that researches B2B leads in real-time, drafts hyper-personalised
cold emails, and logs everything — with zero human input per lead.

Built with Python, LangChain, LLaMA 3.1, and Tavily Search.

🔗 **Live Demo**: [ai-sales-agent.streamlit.app](https://ai-sales-agent-3texcwappcokvpr7afudyav.streamlit.app/)

---

## What it does

1. **Researches each lead** — searches the web for recent company news, funding, product launches, and role-specific context (last 30 days only)
2. **Drafts a personalised email** — uses LLaMA 3.1 to write a <120-word cold email grounded in that research
3. **Logs everything** — saves status, draft preview, and timestamps to a CSV
4. **Sends via SendGrid or SMTP** — with rate limiting and resume support

## Architecture

```
leads.csv
    │
    ▼
Enrichment Agent  ──→  Tavily Search (last 30 days)
    │                  LLaMA 3.1 8B via Groq
    │                  Returns: 3-5 sentence research brief
    ▼
Email Drafter     ──→  LLaMA 3.1 8B via Groq
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
| LLM | LLaMA 3.1 8B Instant via Groq |
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
├── requirements.txt
├── .env.example
├── agents/
│   ├── enricher.py      ← LangChain + Tavily research agent
│   └── drafter.py       ← LLM email drafting chain
├── tools/
│   ├── sender.py        ← SendGrid / SMTP sender
│   └── logger.py        ← CSV logger with duplicate detection
└── data/
    ├── leads.csv        ← Input leads
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
- **`--dry-run` flag** — safe to test without sending a single real email
- **`--resume` flag** — idempotent, safe to re-run after crashes without double-sending
- **Rate limiting** — 60s default between sends to avoid spam filters
- **Modular** — swap Groq → OpenAI, Tavily → SerpAPI, SendGrid → SMTP with one-line changes

---