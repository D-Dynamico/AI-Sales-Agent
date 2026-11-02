import csv
import html
import io
import os
import sys
import time

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
REQUIRED_COLUMNS = ("name", "role", "company", "email")
CSV_FIELDS = ["name", "role", "company", "email", "status", "subject",
              "claims_total", "claims_grounded", "critique", "draft"]

SAMPLE = """name,role,company,email,website
Priya Sharma,Head of Sales,Razorpay,priya@razorpay.com,razorpay.com
Arjun Mehta,VP Marketing,Zepto,arjun@zepto.team,zeptonow.com
Neha Gupta,Co-founder,Sarvam AI,neha@sarvam.ai,sarvam.ai"""

st.set_page_config(
    page_title=f"Outreach · {PERSONA.company}",
    page_icon="✉️",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
# One grotesk, near-black ink on a warm off-white ground, white panels split by 1px
# lines. Colour is reserved for status: green verified, amber held, red failed.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, textarea {
    font-family: "Instrument Sans", system-ui, sans-serif !important;
}
/* Streamlit's icon font must survive the override above. */
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material"] {
    font-family: "Material Symbols Rounded" !important;
}
.block-container { padding-top: 2.25rem; padding-bottom: 4rem; max-width: 1240px; }
h1, h2, h3 { letter-spacing: -0.01em; }

/* Title row */
.brand { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
.brand-mark {
    width: 28px; height: 28px; border-radius: 6px; background: #1d1f20; color: #fff;
    display: grid; place-items: center; font-weight: 700; font-size: 15px;
}
.brand-name { font-weight: 600; font-size: 15px; }
.brand-scope { color: #5f6368; font-size: 14px; }
.title-row h1 { font-size: 26px; font-weight: 700; margin: 0; padding: 0; line-height: 1.2; }
.title-row p { color: #5f6368; font-size: 15px; margin: 6px 0 0; max-width: 62ch; }

/* Panel headings */
.panel-h { font-size: 16px; font-weight: 600; margin: 0 0 2px; }
.panel-sub { color: #5f6368; font-size: 13px; margin: 0 0 12px; }

/* Summary strip: one container split by hairlines, label over value */
.summary { display: grid; grid-template-columns: repeat(3, 1fr); background: #fff;
           border: 1px solid #e6e6e3; border-radius: 8px; margin-bottom: 8px; }
.summary > div { padding: 14px 18px; }
.summary > div + div { border-left: 1px solid #e6e6e3; }
.summary .label { color: #5f6368; font-size: 13px; }
.summary .value { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums;
                  line-height: 1.25; }
.summary .value.ok { color: #247b50; }
.summary .value.held { color: #92400e; }
.summary .value.err { color: #c4242a; }
.finding { font-size: 14px; color: #141414; margin: 10px 0 18px; }

/* Lead header */
.lead-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.lead-name { font-weight: 600; font-size: 16px; }
.lead-meta { color: #5f6368; font-size: 14px; }
.pill { flex: none; white-space: nowrap; font-size: 12px; font-weight: 500;
        padding: 2px 9px; border-radius: 999px; }
.pill.ok   { background: #dcfae5; color: #04250e; }
.pill.held { background: #fffbeb; color: #92400e; box-shadow: inset 0 0 0 1px #fde68a; }
.pill.err  { background: #ffedea; color: #831c1e; }
.lead-facts { color: #5f6368; font-size: 13px; margin: 2px 0 10px; }

.subject { font-weight: 600; font-size: 15px; margin: 4px 0 6px; }
.notice { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; border-radius: 6px;
          padding: 10px 12px; font-size: 13.5px; margin: 0 0 10px; }
.failure { background: #ffedea; color: #831c1e; border-radius: 6px; padding: 10px 12px;
           font-size: 14px; margin: 4px 0 8px; }
.empty { border: 1px dashed #d4d4d0; border-radius: 8px; padding: 28px 24px; color: #5f6368;
         font-size: 14px; background: #fff; }
.empty strong { color: #141414; display: block; font-size: 15px; margin-bottom: 4px; }

/* The email body uses st.code for its copy button, set in the text face so it reads as
   an email rather than a code block. */
[data-testid="stCode"] pre, [data-testid="stCode"] code {
    font-family: "Instrument Sans", system-ui, sans-serif !important;
    font-size: 14.5px !important; line-height: 1.6 !important; white-space: pre-wrap !important;
    background: #fafaf8 !important; color: #141414 !important;
}
[data-testid="stCode"] pre { padding: 14px 16px !important; border-radius: 6px; }

/* Leads are CSV: one lead per line reads far better than wrapped mid-address. */
[data-testid="stTextArea"] textarea { white-space: pre !important; overflow-x: auto !important;
    overflow-wrap: normal !important; font-size: 13.5px !important; line-height: 1.7 !important; }

.foot { color: #6b6b6b; font-size: 12px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

E = html.escape  # every lead field is user input and goes into HTML below


# ── Sidebar: who is sending, and diagnostics ──────────────────────────────────
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
    st.markdown(f"**Sending as**  \n{E(PERSONA.name)}, {E(PERSONA.company)}")
    st.caption(PERSONA.offering[:1].upper() + PERSONA.offering[1:] + ".")
    st.divider()
    st.markdown("**How a draft is checked**")
    st.caption(
        "Each company is researched live with Tavily. The email is drafted from that "
        "research, then every factual claim is checked against the sources. Drafts with "
        "untraceable claims are rewritten once, then held for you to review."
    )
    st.divider()
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


# ── Title row ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="brand">
  <div class="brand-mark">{E(PERSONA.company[:1])}</div>
  <span class="brand-name">{E(PERSONA.company)}</span>
  <span class="brand-scope">· Outreach</span>
</div>
<div class="title-row">
  <h1>Research and draft cold emails</h1>
  <p>Paste a lead list. Each company is researched live, an email is drafted from what was
  found, and every claim in it is checked against the sources before it's marked ready.</p>
</div>
""", unsafe_allow_html=True)
st.write("")


# ── Rendering ─────────────────────────────────────────────────────────────────
STATUS_PILL = {
    "APPROVED": ("ok", "Ready to send"),
    "NEEDS_REVIEW": ("held", "Held for review"),
    "ENRICH_ERROR": ("err", "Research failed"),
    "DRAFT_ERROR": ("err", "Drafting failed"),
}


def render_lead(r: dict) -> None:
    kind, label = STATUS_PILL.get(r["status"], ("err", r["status"]))
    with st.container(border=True):
        st.markdown(f"""
<div class="lead-head">
  <div><span class="lead-name">{E(r.get('name', ''))}</span>
       <span class="lead-meta"> · {E(r.get('role', ''))} at {E(r.get('company', ''))}</span></div>
  <span class="pill {kind}">{label}</span>
</div>""", unsafe_allow_html=True)

        if r.get("error"):
            st.markdown(f'<div class="failure">{E(r["error"])}</div>', unsafe_allow_html=True)
            if r.get("error_detail"):
                with st.expander("Technical detail"):
                    st.code(r["error_detail"], language=None)
            return

        n_src = len(r.get("sources", []))
        facts = [f"{E(r.get('email', ''))}", f"{n_src} source{'s' if n_src != 1 else ''}"]
        if r.get("claims_total"):
            facts.append(f"{r['claims_grounded']} of {r['claims_total']} claims traced")
        facts.append(f"{r['attempts']} attempt{'s' if r['attempts'] != 1 else ''}")
        st.markdown(f'<div class="lead-facts">{" · ".join(facts)}</div>',
                    unsafe_allow_html=True)

        if r["status"] == "NEEDS_REVIEW" and r.get("critique"):
            # The critic writes a bullet list; one reason per line reads better than dashes.
            reasons = [line.strip().lstrip("-•* ").strip()
                       for line in r["critique"].splitlines() if line.strip()]
            body = "<br>".join(E(x) for x in reasons if x)
            st.markdown(f'<div class="notice"><strong>Why it was held</strong><br>{body}</div>',
                        unsafe_allow_html=True)

        st.markdown(f'<div class="subject">{E(r.get("subject", ""))}</div>',
                    unsafe_allow_html=True)
        st.code(r.get("draft", ""), language=None, wrap_lines=True)

        c1, c2 = st.columns(2)
        with c1:
            with st.expander(f"Research brief · {n_src} source{'s' if n_src != 1 else ''}"):
                st.write(r.get("research", ""))
                for src in r.get("sources", []):
                    st.caption(src[:400])
        with c2:
            claims = r.get("claims", [])
            with st.expander(f"Claim check · {r.get('claims_grounded', 0)} of "
                             f"{len(claims)} traced"):
                if not claims:
                    st.caption("The draft makes no specific factual claims to check.")
                for c in claims:
                    st.markdown(f"{'Traced' if c['grounded'] else '**Not found in sources**'}"
                                f" — {c['text']}")
                    if c.get("quote"):
                        st.caption(f"“{c['quote'][:300]}”")


def render_summary(results: list[dict], total: int) -> None:
    ready = sum(r["status"] == "APPROVED" for r in results)
    held = sum(r["status"] == "NEEDS_REVIEW" for r in results)
    failed = len(results) - ready - held
    st.markdown(f"""
<div class="summary">
  <div><div class="label">Ready</div><div class="value {'ok' if ready else ''}">{ready}</div></div>
  <div><div class="label">Held</div><div class="value {'held' if held else ''}">{held}</div></div>
  <div><div class="label">Failed</div><div class="value {'err' if failed else ''}">{failed}</div></div>
</div>""", unsafe_allow_html=True)

    if len(results) < total:
        finding = f"Working through lead {len(results) + 1} of {total}…"
    elif held or failed:
        parts = []
        if held:
            parts.append(f"{held} held because some claims couldn't be traced to the research")
        if failed:
            parts.append(f"{failed} failed — see the reason on each")
        finding = f"{ready} of {total} ready to send. " + "; ".join(parts) + "."
    else:
        finding = f"All {total} drafts are ready to send — every claim traced to a source."
    st.markdown(f'<div class="finding">{finding}</div>', unsafe_allow_html=True)


def results_csv(results: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return out.getvalue()


# ── Pipeline ──────────────────────────────────────────────────────────────────
def process(lead: dict) -> dict:
    name = lead.get("name", "").strip()
    company = lead.get("company", "").strip()
    base = {k: (lead.get(k) or "").strip() for k in ("name", "role", "company", "email")}

    try:
        research, sources = enrich_lead(lead)
    except Exception as e:
        print(f"[enrich] {company}: {log_detail(e)}", file=sys.stderr)
        return {**base, "status": "ENRICH_ERROR", "draft": "",
                "error": user_message(e, "researching this company"),
                "error_detail": log_detail(e)}

    try:
        gated = draft_with_gate(lead, research, sources)
    except Exception as e:
        print(f"[draft] {name}: {log_detail(e)}", file=sys.stderr)
        return {**base, "status": "DRAFT_ERROR", "draft": "",
                "error": user_message(e, "drafting this email"),
                "error_detail": log_detail(e)}

    draft, verdict = gated.draft, gated.verdict
    return {
        **base,
        "status": gated.status,
        "subject": draft.subject,
        "draft": draft.body,
        "claims_total": len(verdict.claims),
        "claims_grounded": len(verdict.grounded_claims),
        "critique": verdict.critique,
        "attempts": gated.attempts,
        "research": research,
        "sources": sources,
        "claims": [{"text": c.text, "grounded": c.grounded, "quote": c.source_quote}
                   for c in verdict.claims],
    }


def parse_leads(text: str) -> tuple[list[dict], str]:
    try:
        leads = list(csv.DictReader(io.StringIO(text.strip())))
    except Exception as e:
        return [], f"Couldn't read that as CSV: {e}"
    if not leads:
        return [], "No leads found. Add a header row and at least one lead."
    missing = [c for c in REQUIRED_COLUMNS if c not in (leads[0].keys() or [])]
    if missing:
        return [], f"Missing column{'s' if len(missing) > 1 else ''}: {', '.join(missing)}."
    return leads, ""


# ── Layout: leads on the left, results on the right ───────────────────────────
left, right = st.columns([5, 8], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="panel-h">Leads</div>'
                    '<div class="panel-sub">CSV with a header row: name, role, company, '
                    'email, and optionally website.</div>', unsafe_allow_html=True)
        csv_input = st.text_area("Leads CSV", value=SAMPLE, height=220,
                                 label_visibility="collapsed")
        leads, problem = parse_leads(csv_input)
        if problem:
            st.markdown(f'<div class="failure">{E(problem)}</div>', unsafe_allow_html=True)
        else:
            st.caption(f"{len(leads)} lead{'s' if len(leads) != 1 else ''} · about "
                       f"{max(1, round(len(leads) * 0.6))} min to research and draft")
        run_btn = st.button("Research and draft", type="primary", use_container_width=True,
                            disabled=bool(problem))

with right:
    if run_btn and leads:
        st.session_state.results = []
        st.session_state.total = len(leads)
        summary_slot = st.empty()
        progress = st.progress(0.0)
        for i, lead in enumerate(leads):
            with summary_slot.container():
                render_summary(st.session_state.results, len(leads))
            with st.spinner(f"Researching {lead.get('company', '').strip() or 'company'} "
                            f"and drafting for {lead.get('name', '').strip() or 'lead'}…"):
                result = process(lead)
            st.session_state.results.append(result)
            progress.progress((i + 1) / len(leads))
            render_lead(result)
            if i < len(leads) - 1:
                time.sleep(1)
        progress.empty()
        with summary_slot.container():
            render_summary(st.session_state.results, len(leads))

    elif st.session_state.get("results"):
        # Stored, so downloading the CSV (which reruns the script) doesn't wipe the results.
        render_summary(st.session_state.results, st.session_state.total)
        for r in st.session_state.results:
            render_lead(r)

    else:
        st.markdown("""
<div class="empty">
  <strong>No drafts yet</strong>
  Run the agent on your lead list. Each lead appears here with its status, the email
  ready to copy, the research it was written from, and which claims were traced to a source.
</div>""", unsafe_allow_html=True)

    results = st.session_state.get("results")
    if results and len(results) == st.session_state.get("total"):
        st.download_button("Download results CSV", data=results_csv(results),
                           file_name="outreach_results.csv", mime="text/csv",
                           use_container_width=True)
        st.markdown('<div class="foot">Held drafts are included with the reason they were '
                    'held. Don’t send them as-is.</div>', unsafe_allow_html=True)
