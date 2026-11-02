import csv
import html
import io
import os
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st

# Keys come from Streamlit Cloud secrets when deployed, and from .env when run locally.
from dotenv import load_dotenv

load_dotenv()
for _key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
    try:
        os.environ[_key] = st.secrets[_key]
    except Exception:
        pass  # no secrets.toml locally — the .env value stands

from agents.enricher import enrich_lead
from agents.errors import log_detail, user_message
from agents.pipeline import draft_with_gate
from persona import PERSONA


def _build_id() -> str:
    """Short commit hash, so it's visible at a glance whether a deploy has picked up a fix.

    Read from .git directly rather than by running git: the subprocess was intermittently
    timing out under Streamlit and reporting "unknown"."""
    from pathlib import Path
    try:
        git = Path(__file__).resolve().parent / ".git"
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head.split(" ", 1)[1]
        ref_file = git / ref
        if ref_file.exists():
            return ref_file.read_text().strip()[:7]
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line[:7]
    except Exception:
        pass
    return "unknown"


BUILD = _build_id()
LEAD_COLUMNS = ["name", "role", "company", "email", "website"]
REQUIRED_COLUMNS = ("name", "role", "company", "email")
CSV_FIELDS = ["name", "role", "company", "email", "status", "subject",
              "claims_total", "claims_grounded", "critique", "draft"]

SAMPLE = """name,role,company,email,website
Priya Sharma,Head of Sales,Razorpay,priya@razorpay.com,razorpay.com
Arjun Mehta,VP Marketing,Zepto,arjun@zepto.team,zeptonow.com
Neha Gupta,Co-founder,Sarvam AI,neha@sarvam.ai,sarvam.ai"""

st.set_page_config(page_title=f"Outreach · {PERSONA.company}", page_icon="✉️",
                   layout="wide")

# ── Style: framed sheet ───────────────────────────────────────────────────────
# The sidebar and Streamlit's top bar share one tinted frame (an alder green-grey, from
# "Aldermere"); the work sits on a white sheet inset 8px, 16px corners, a 1px edge, no
# shadow. Colour is for status and for the one data hue (teal) on the claim trace.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {
  --frame: #eff3f1; --surface: #ffffff; --raised: #f5f7f6; --active: #e6ebe8;
  --border: #e3e8e5; --rule: #eef1ef; --control: #cdd3cf;
  --text: #131416; --muted: #5a5f5c;
  --accent: #1b6c80; --positive: #1e7b4f; --negative: #c4242a; --warning: #92400e;
}
html, body, [class*="st-"], .stMarkdown, button, input, textarea, [data-testid="stWidgetLabel"] {
  font-family: "IBM Plex Sans", Inter, system-ui, sans-serif !important;
}
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material"] {
  font-family: "Material Symbols Rounded" !important;
}

/* The frame: app ground, Streamlit's header and the sidebar all one tint, no rules. */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--frame); }
[data-testid="stHeader"] { background: var(--frame); }
[data-testid="stSidebar"] { background: var(--frame); border-right: none; }
[data-testid="stSidebar"] > div { background: var(--frame); }

/* The sheet. Its top margin clears Streamlit's own header bar, which is what cut the
   title off before. */
[data-testid="stMainBlockContainer"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  margin: 3.75rem 8px 8px 0; padding: 28px 32px 48px !important; max-width: none;
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { margin: 3.5rem 0 0 0; border-radius: 0;
      border-width: 1px 0 0; padding: 20px 16px 40px !important; }
}

/* Title row */
.crumb { color: var(--muted); font-size: 14px; margin-bottom: 4px; }
.crumb b { color: var(--text); font-weight: 600; }
.title-row { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px;
             flex-wrap: wrap; padding-bottom: 18px; border-bottom: 1px solid var(--rule);
             margin-bottom: 24px; }
.title-row h1 { font-size: 22px; font-weight: 600; margin: 0; padding: 0; line-height: 1.3;
                letter-spacing: -0.01em; }
.title-row p { color: var(--muted); font-size: 14px; margin: 4px 0 0; max-width: 70ch; }
.stamp { color: var(--muted); font-size: 13px; white-space: nowrap; }

/* Section headings inside the sheet */
.sec-h { font-size: 16px; font-weight: 600; margin: 0; }
.sec-sub { color: var(--muted); font-size: 13px; margin: 2px 0 0; }

/* Sidebar */
.mark { display: flex; align-items: center; gap: 10px; margin: 4px 0 20px; }
.mark-box { width: 28px; height: 28px; border-radius: 7px; background: var(--accent); color: #fff;
            display: grid; place-items: center; font-weight: 600; font-size: 14px; }
.mark-name { font-weight: 600; font-size: 15px; }
.side-label { color: var(--muted); font-size: 12px; margin: 0 0 2px; }
.side-value { font-size: 14px; margin: 0 0 14px; }
.side-note { color: var(--muted); font-size: 13px; line-height: 1.5; }

/* Lead cards */
.lead-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.lead-name { font-weight: 600; font-size: 15px; }
.lead-meta { color: var(--muted); font-size: 14px; }
.lead-facts { color: var(--muted); font-size: 13px; margin: 2px 0 14px;
              font-variant-numeric: tabular-nums; }
.pill { flex: none; white-space: nowrap; font-size: 12px; font-weight: 500;
        padding: 2px 9px; border-radius: 999px; }
.pill.ok   { background: #dcfae5; color: #04250e; }
.pill.held { background: #fffbeb; color: #92400e; box-shadow: inset 0 0 0 1px #fde68a; }
.pill.err  { background: #ffedea; color: #831c1e; }
.subject { font-weight: 600; font-size: 15px; margin: 0 0 8px; }
.notice { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; border-radius: 6px;
          padding: 10px 12px; font-size: 13.5px; margin: 0 0 12px; line-height: 1.5; }
.failure { background: #ffedea; color: #831c1e; border-radius: 6px; padding: 10px 12px;
           font-size: 14px; margin: 8px 0; }

/* The claim trace: one mark per factual claim. Teal = traced to a source. */
.trace-h { font-size: 13px; font-weight: 600; margin: 0 0 6px; }
.trace-h span { color: var(--muted); font-weight: 400; }
.trace { display: flex; gap: 4px; margin: 0 0 12px; }
.trace i { width: 22px; height: 8px; border-radius: 2px; background: var(--accent); }
.trace i.miss { background: transparent; box-shadow: inset 0 0 0 1.5px var(--negative); }
.claim { font-size: 13px; line-height: 1.45; margin: 0 0 8px; padding-left: 12px;
         border-left: 2px solid var(--accent); }
.claim.miss { border-left-color: var(--negative); }
.claim q { display: block; color: var(--muted); margin-top: 2px; }
.claim .tag { color: var(--negative); font-weight: 500; }

.empty { border: 1px dashed var(--control); border-radius: 8px; padding: 22px 20px;
         color: var(--muted); font-size: 14px; }
.empty b { color: var(--text); display: block; font-size: 15px; margin-bottom: 4px; font-weight: 600; }
.finding { font-size: 14px; margin: 4px 0 16px; }

/* Email body: st.code for its copy button, set in the text face so it reads as an email. */
[data-testid="stCode"] pre, [data-testid="stCode"] code {
  font-family: "IBM Plex Sans", system-ui, sans-serif !important; font-size: 14.5px !important;
  line-height: 1.65 !important; white-space: pre-wrap !important;
  background: var(--raised) !important; color: var(--text) !important;
}
[data-testid="stCode"] pre { padding: 14px 16px !important; border-radius: 6px; }

/* Bordered containers are the cards: one edge colour, 8px corners, no shadow. */
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border) !important;
                                               border-radius: 8px !important; }
.foot { color: var(--muted); font-size: 12px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

E = html.escape  # every lead field is user input and goes into HTML below


# ── Sidebar (the frame) ───────────────────────────────────────────────────────
def _key_shape(name: str, prefix: str) -> str:
    raw = os.environ.get(name, "")
    if not raw:
        return f"{name}: MISSING"
    problems = []
    if raw != raw.strip():
        problems.append("leading/trailing whitespace")
    stripped = raw.strip()
    if stripped[:1] in "\"'" or stripped[-1:] in "\"'":
        problems.append("wrapped in quotes")
    if not stripped.strip("\"'").startswith(prefix):
        problems.append(f"does not start with '{prefix}'")
    status = "; ".join(problems) if problems else "looks well-formed"
    return f"{name}: present, {len(raw)} chars, {status}"


with st.sidebar:
    st.markdown(f"""
<div class="mark"><div class="mark-box">{E(PERSONA.company[:1])}</div>
<span class="mark-name">{E(PERSONA.company)}</span></div>
<div class="side-label">Sending as</div>
<div class="side-value">{E(PERSONA.name)}</div>
<div class="side-label">How drafts are checked</div>
<div class="side-note">Each company is researched live. The email is written from that research,
and every factual claim in it is matched against the sources. A draft with a claim that can't be
matched is rewritten once, then held for you.</div>
""", unsafe_allow_html=True)
    st.write("")
    with st.expander("Diagnostics"):
        if st.button("Test web search and model"):
            st.text(_key_shape("TAVILY_API_KEY", "tvly-"))
            st.text(_key_shape("GROQ_API_KEY", "gsk_"))
            try:
                from langchain_tavily import TavilySearch
                raw = TavilySearch(max_results=3, topic="news", search_depth="advanced").invoke(
                    "Razorpay funding round recent news")
                if isinstance(raw, dict) and raw.get("error"):
                    err = raw["error"]
                    st.error(f"Tavily returned an error: "
                             f"{log_detail(err) if isinstance(err, Exception) else err}")
                else:
                    n = len(raw.get("results", [])) if isinstance(raw, dict) else 0
                    st.success(f"Tavily OK: {n} result(s)")
            except Exception as e:
                st.error(f"Tavily raised: {log_detail(e)}")
            try:
                from langchain_groq import ChatGroq
                from config import PIPELINE_MODEL
                ChatGroq(model=PIPELINE_MODEL, max_retries=0, timeout=30).invoke("Reply with: ok")
                st.success(f"Groq OK ({PIPELINE_MODEL})")
            except Exception as e:
                st.error(f"Groq raised: {log_detail(e)}")
    st.caption(f"GPT-OSS 120B on Groq · Tavily search · build {BUILD}")


# ── Leads: data ───────────────────────────────────────────────────────────────
def df_from_csv(text: str) -> tuple[pd.DataFrame | None, str]:
    try:
        rows = list(csv.DictReader(io.StringIO(text.strip())))
    except Exception as e:
        return None, f"Couldn't read that as CSV: {e}"
    if not rows:
        return None, "No leads found. Add a header row and at least one lead."
    normalised = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
                  for r in rows]
    missing = [c for c in REQUIRED_COLUMNS if c not in normalised[0]]
    if missing:
        return None, f"Missing column{'s' if len(missing) > 1 else ''}: {', '.join(missing)}."
    clean = [{c: r.get(c, "") for c in LEAD_COLUMNS} for r in normalised]
    return pd.DataFrame(clean, columns=LEAD_COLUMNS), ""


def leads_from_df(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    leads, problems = [], []
    for i, row in enumerate(df.fillna("").to_dict("records"), start=1):
        lead = {c: str(row.get(c, "") or "").strip() for c in LEAD_COLUMNS}
        if not any(lead.values()):
            continue  # a blank row added in the editor
        missing = [c for c in REQUIRED_COLUMNS if not lead[c]]
        if missing:
            problems.append(f"Row {i} is missing {', '.join(missing)}.")
        leads.append(lead)
    return leads, problems


if "leads_df" not in st.session_state:
    st.session_state.leads_df, _ = df_from_csv(SAMPLE)
    st.session_state.editor_version = 0


# ── Results: rendering ────────────────────────────────────────────────────────
STATUS = {
    "APPROVED": ("ok", "Ready to send"),
    "NEEDS_REVIEW": ("held", "Held for review"),
    "ENRICH_ERROR": ("err", "Research failed"),
    "DRAFT_ERROR": ("err", "Drafting failed"),
}
FILTER_OF = {"APPROVED": "Ready", "NEEDS_REVIEW": "Held",
             "ENRICH_ERROR": "Failed", "DRAFT_ERROR": "Failed"}


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def claim_trace_html(claims: list[dict]) -> str:
    traced = sum(c["grounded"] for c in claims)
    if not claims:
        return ('<div class="trace-h">Claim trace <span>· no specific claims to check'
                '</span></div>')
    marks = "".join("<i></i>" if c["grounded"] else '<i class="miss"></i>' for c in claims)
    items = []
    for c in claims:
        tag = "" if c["grounded"] else '<span class="tag">Not in sources · </span>'
        # Source quotes often arrive already wrapped in quote marks; strip them so the
        # page doesn't show doubled ones.
        raw_quote = (c.get("quote") or "").strip().strip("\"'“”").strip()
        quote = f"<q>“{E(raw_quote[:220])}”</q>" if raw_quote else ""
        cls = "claim" if c["grounded"] else "claim miss"
        items.append(f'<div class="{cls}">{tag}{E(c["text"])}{quote}</div>')
    return (f'<div class="trace-h">Claim trace <span>· {traced} of {len(claims)} traced to a '
            f'source</span></div><div class="trace">{marks}</div>{"".join(items)}')


def render_lead(r: dict) -> None:
    kind, label = STATUS.get(r["status"], ("err", r["status"]))
    with st.container(border=True):
        st.markdown(f"""
<div class="lead-head">
  <div><span class="lead-name">{E(r['name'])}</span>
       <span class="lead-meta"> · {E(r['role'])} at {E(r['company'])}</span></div>
  <span class="pill {kind}">{label}</span>
</div>""", unsafe_allow_html=True)

        if r.get("error"):
            st.markdown(f'<div class="lead-facts">{E(r["email"])}</div>'
                        f'<div class="failure">{E(r["error"])}</div>', unsafe_allow_html=True)
            if r.get("error_detail"):
                with st.expander("Technical detail"):
                    st.code(r["error_detail"], language=None, wrap_lines=True)
            return

        facts = [E(r["email"]), plural(len(r.get("sources", [])), "source"),
                 plural(r["attempts"], "attempt")]
        st.markdown(f'<div class="lead-facts">{" · ".join(facts)}</div>',
                    unsafe_allow_html=True)

        email_col, trace_col = st.columns([3, 2], gap="large")
        with email_col:
            if r["status"] == "NEEDS_REVIEW" and r.get("critique"):
                reasons = [ln.strip().lstrip("-•* ").strip()
                           for ln in r["critique"].splitlines() if ln.strip()]
                st.markdown('<div class="notice"><b>Why it was held</b><br>'
                            + "<br>".join(E(x) for x in reasons if x) + "</div>",
                            unsafe_allow_html=True)
            st.markdown(f'<div class="subject">{E(r["subject"])}</div>', unsafe_allow_html=True)
            st.code(r["draft"], language=None, wrap_lines=True)

        with trace_col:
            st.markdown(claim_trace_html(r.get("claims", [])), unsafe_allow_html=True)
            with st.expander(f"Research brief · {plural(len(r.get('sources', [])), 'source')}"):
                st.write(r.get("research", ""))
                for src in r.get("sources", []):
                    st.caption(src[:400])


def counts(results: list[dict]) -> dict:
    c = {"All": len(results), "Ready": 0, "Held": 0, "Failed": 0}
    for r in results:
        c[FILTER_OF.get(r["status"], "Failed")] += 1
    return c


def finding(c: dict) -> str:
    if not c["All"]:
        return ""
    if c["Ready"] == c["All"]:
        return f"All {c['All']} drafts are ready to send. Every claim traced to a source."
    parts = [f"{c['Ready']} of {c['All']} ready to send"]
    if c["Held"]:
        parts.append(f"{c['Held']} held because a claim couldn't be traced to the research")
    if c["Failed"]:
        parts.append(f"{c['Failed']} failed. The reason is on each card")
    return ". ".join(parts) + "."


def results_csv(results: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return out.getvalue()


# ── Pipeline ──────────────────────────────────────────────────────────────────
def process(lead: dict) -> dict:
    base = {k: lead.get(k, "") for k in ("name", "role", "company", "email")}
    try:
        research, sources = enrich_lead(lead)
    except Exception as e:
        print(f"[enrich] {lead['company']}: {log_detail(e)}", file=sys.stderr)
        return {**base, "status": "ENRICH_ERROR", "draft": "",
                "error": user_message(e, "researching this company"),
                "error_detail": log_detail(e)}
    try:
        gated = draft_with_gate(lead, research, sources)
    except Exception as e:
        print(f"[draft] {lead['name']}: {log_detail(e)}", file=sys.stderr)
        return {**base, "status": "DRAFT_ERROR", "draft": "",
                "error": user_message(e, "drafting this email"),
                "error_detail": log_detail(e)}

    draft, verdict = gated.draft, gated.verdict
    return {
        **base, "status": gated.status, "subject": draft.subject, "draft": draft.body,
        "claims_total": len(verdict.claims), "claims_grounded": len(verdict.grounded_claims),
        "critique": verdict.critique, "attempts": gated.attempts,
        "research": research, "sources": sources,
        "claims": [{"text": c.text, "grounded": c.grounded, "quote": c.source_quote}
                   for c in verdict.claims],
    }


# ── Title row ─────────────────────────────────────────────────────────────────
last = st.session_state.get("last_run")
stamp = (f"Last run {last['at']} · {plural(last['n'], 'lead')} · {last['took']}"
         if last else "No runs yet")
st.markdown(f"""
<div class="crumb">{E(PERSONA.company)} / <b>Outreach</b></div>
<div class="title-row">
  <div><h1>Research and draft cold emails</h1>
  <p>Add leads, run the agent, and send the drafts marked ready. Each one is written from live
  research, and every claim in it is checked against the sources.</p></div>
  <div class="stamp">{stamp}</div>
</div>""", unsafe_allow_html=True)


# ── Leads ─────────────────────────────────────────────────────────────────────
head_l, head_r = st.columns([3, 1], vertical_alignment="bottom")
with head_l:
    st.markdown('<div class="sec-h">Leads</div><div class="sec-sub">One row per person. Name, '
                'role, company and email are required; website is optional. Add a row at the '
                'bottom of the table, or paste a CSV below.</div>', unsafe_allow_html=True)

edited = st.data_editor(
    st.session_state.leads_df,
    key=f"leads_editor_{st.session_state.editor_version}",
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("Name"),
        "role": st.column_config.TextColumn("Role"),
        "company": st.column_config.TextColumn("Company"),
        "email": st.column_config.TextColumn("Email"),
        "website": st.column_config.TextColumn("Website"),
    },
)
leads, problems = leads_from_df(edited)

with head_r:
    run_btn = st.button(f"Research and draft {plural(len(leads), 'lead')}" if leads
                        else "Research and draft", type="primary",
                        use_container_width=True, disabled=bool(problems) or not leads)

if problems:
    st.markdown('<div class="failure">' + "<br>".join(E(p) for p in problems) + "</div>",
                unsafe_allow_html=True)
else:
    st.caption(f"About {max(1, round(len(leads) * 0.6))} min to research and draft "
               f"{'this lead' if len(leads) == 1 else 'these leads'}.")

with st.expander("Paste a CSV instead"):
    pasted = st.text_area(
        "CSV with a header row", height=140,
        placeholder="name,role,company,email,website\n"
                    "Priya Sharma,Head of Sales,Razorpay,priya@razorpay.com,razorpay.com")
    if st.button("Replace the table with this CSV", disabled=not pasted.strip()):
        df, problem = df_from_csv(pasted)
        if problem:
            st.markdown(f'<div class="failure">{E(problem)}</div>', unsafe_allow_html=True)
        else:
            st.session_state.leads_df = df
            st.session_state.editor_version += 1
            st.rerun()

st.write("")


# ── Drafts ────────────────────────────────────────────────────────────────────
if run_btn and leads:
    started = time.time()
    st.session_state.results = []
    st.markdown('<div class="sec-h">Drafts</div>', unsafe_allow_html=True)
    status_line = st.empty()
    progress = st.progress(0.0)
    for i, lead in enumerate(leads):
        status_line.markdown(f'<div class="finding">Researching {E(lead["company"])} and '
                             f'drafting for {E(lead["name"])} · lead {i + 1} of '
                             f'{len(leads)}</div>', unsafe_allow_html=True)
        result = process(lead)
        st.session_state.results.append(result)
        progress.progress((i + 1) / len(leads))
        render_lead(result)
        if i < len(leads) - 1:
            time.sleep(1)
    took = int(time.time() - started)
    st.session_state.last_run = {"at": datetime.now().strftime("%H:%M"), "n": len(leads),
                                 "took": f"{took // 60} min {took % 60:02d} s"}
    st.session_state.pop("draft_filter", None)
    st.rerun()  # redraw from the stored results, with the filter

results = st.session_state.get("results") or []
c = counts(results)

dh_l, dh_r = st.columns([3, 1], vertical_alignment="bottom")
with dh_l:
    st.markdown(f'<div class="sec-h">Drafts</div>'
                f'<div class="sec-sub">{finding(c) or "Drafts appear here after a run."}</div>',
                unsafe_allow_html=True)
with dh_r:
    if results:
        st.download_button("Download CSV", data=results_csv(results),
                           file_name="outreach_results.csv", mime="text/csv",
                           use_container_width=True)
st.write("")

if not results:
    st.markdown("""
<div class="empty"><b>No drafts yet</b>
Run the agent on the leads above. Each lead shows up here with its status, the email ready to
copy, the research it was written from, and a trace of which claims matched a source.</div>""",
                unsafe_allow_html=True)
else:
    choice = st.segmented_control(
        "Show", options=["All", "Ready", "Held", "Failed"],
        format_func=lambda o: f"{o} · {c[o]}", default="All", key="draft_filter",
        label_visibility="collapsed",
    ) or "All"
    shown = [r for r in results if choice == "All" or FILTER_OF.get(r["status"]) == choice]
    if not shown:
        st.markdown(f'<div class="empty">No {choice.lower()} drafts in this run.</div>',
                    unsafe_allow_html=True)
    for r in shown:
        render_lead(r)
    if c["Held"]:
        st.markdown('<div class="foot">Held drafts are in the CSV with the reason they were '
                    'held. Don’t send them as they are.</div>', unsafe_allow_html=True)
