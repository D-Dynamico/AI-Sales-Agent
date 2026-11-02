import streamlit as st
import os
import csv
import io
import time

# Load secrets from Streamlit Cloud
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

from agents.enricher import enrich_lead
import sys

from agents.errors import log_detail, user_message
from agents.pipeline import draft_with_gate
from persona import PERSONA

def _build_id() -> str:
    """Short commit hash, so it's visible at a glance whether a deploy has picked up a fix."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


BUILD = _build_id()

# Page config
st.set_page_config(
    page_title=f"AI Sales Agent · {PERSONA.company}",
    page_icon="⚗️",
    layout="centered",
)

# Styles
st.markdown("""
<style>
    .main { max-width: 760px; }
    .lead-card {
        background: rgba(108, 99, 255, 0.08);
        border-left: 3px solid #6c63ff;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: inherit;
    }
    .lead-card strong { color: inherit; }
    .email-box {
        background: rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        margin-top: 0.5rem;
        color: inherit;
    }
    .status-ok  { color: #4caf50; font-weight: 600; }
    .status-err { color: #f44336; font-weight: 600; }
    .status-warn { color: #ff9800; font-weight: 600; }
    .critique-box {
        background: rgba(255, 152, 0, 0.10);
        border-left: 3px solid #ff9800;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        white-space: pre-wrap;
        margin: 0.5rem 0;
        color: inherit;
    }
    .badge {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        color: #9c8fff;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Diagnostics — checks the deployed environment directly. Never shows key values, only
# their shape, and every error goes through the same redaction as the logs.
def _key_shape(name: str, prefix: str) -> str:
    raw = os.environ.get(name, "")
    if not raw:
        return f"{name}: MISSING"
    problems = []
    if raw != raw.strip():
        problems.append("leading/trailing whitespace")
    if raw.strip()[:1] in "\"'" or raw.strip()[-1:] in "\"'":
        problems.append("wrapped in quotes")
    if not raw.strip().strip("\"'").startswith(prefix):
        problems.append(f"does not start with '{prefix}'")
    status = "; ".join(problems) if problems else "looks well-formed"
    return f"{name}: present, {len(raw)} chars, {status}"


with st.sidebar.expander("Diagnostics"):
    st.caption(f"build {BUILD}")
    if st.button("Test web search and model"):
        st.text(_key_shape("TAVILY_API_KEY", "tvly-"))
        st.text(_key_shape("GROQ_API_KEY", "gsk_"))
        try:
            from importlib.metadata import version
            st.text(f"langchain-tavily {version('langchain-tavily')}")
        except Exception:
            pass

        try:
            from langchain_tavily import TavilySearch
            raw = TavilySearch(max_results=3, topic="news", search_depth="advanced").invoke(
                "Razorpay funding round recent news")
            if isinstance(raw, dict) and raw.get("error"):
                st.error(f"Tavily returned an error: {log_detail(raw['error']) if isinstance(raw['error'], Exception) else raw['error']}")
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

# Header
st.markdown("## AI Sales Outreach Agent")
st.markdown(
    "Paste your leads, hit **Run**. The agent researches each company in real-time "
    "and drafts a hyper-personalised cold email automatically."
)
st.divider()

# Sample data
SAMPLE = """name,role,company,email,website
Priya Sharma,Head of Sales,Razorpay,priya@razorpay.com,razorpay.com
Arjun Mehta,VP Marketing,Zepto,arjun@zepto.team,zeptonow.com
Neha Gupta,Co-founder,Sarvam AI,neha@sarvam.ai,sarvam.ai"""

# Input
st.markdown("### Lead List")
st.caption("CSV format: `name, role, company, email, website`")

csv_input = st.text_area(
    label="Paste leads here",
    value=SAMPLE,
    height=160,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 3])
with col1:
    run_btn = st.button("Run Agent", type="primary", use_container_width=True)
with col2:
    st.caption("Powered by GPT-OSS 120B · Groq · Tavily Search · grounding-checked"
               f" · build {BUILD}")

st.divider()

# Pipeline
if run_btn:
    try:
        reader = csv.DictReader(io.StringIO(csv_input.strip()))
        leads = list(reader)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()

    if not leads:
        st.warning("No leads found. Check your CSV format.")
        st.stop()

    required = {"name", "role", "company", "email"}
    if not required.issubset(set(leads[0].keys())):
        st.error(f"CSV must have columns: {', '.join(required)}")
        st.stop()

    st.markdown(f"### Running pipeline for {len(leads)} lead(s)")

    results = []

    for i, lead in enumerate(leads):
        name = lead.get("name", "Unknown").strip()
        company = lead.get("company", "").strip()
        email = lead.get("email", "").strip()

        with st.container():
            st.markdown(
                f'<div class="lead-card">'
                f'<span class="badge">Lead {i+1}</span> '
                f'<strong>{name}</strong> · {lead.get("role", "")} @ {company}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Step 1: Enrich
            with st.spinner(f"Researching {company}..."):
                try:
                    research, sources = enrich_lead(lead)
                    st.markdown(
                        f'<p class="status-ok">Research complete — {len(sources)} source(s)</p>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("View research brief"):
                        st.write(research)
                        if sources:
                            st.caption("Sources the critic checks claims against:")
                            for src in sources:
                                st.text(src[:500])
                except Exception as e:
                    st.markdown(
                        f'<p class="status-err">{user_message(e, "researching this company")}</p>',
                        unsafe_allow_html=True,
                    )
                    # Redacted (no keys, org ids or billing links), so safe on a public page —
                    # and without it a deployed failure is undiagnosable from the UI alone.
                    with st.expander("Technical detail"):
                        st.code(log_detail(e))
                    print(f"[enrich] {company}: {log_detail(e)}", file=sys.stderr)
                    results.append({**lead, "status": "ENRICH_ERROR", "draft": ""})
                    continue

            # Step 2: Draft, then gate on grounding before it counts as usable
            with st.spinner(f"Drafting and fact-checking email for {name}..."):
                try:
                    gated = draft_with_gate(lead, research, sources)
                    draft, verdict = gated.draft, gated.verdict
                    n_claims = len(verdict.claims)
                    n_grounded = len(verdict.grounded_claims)

                    if gated.approved:
                        st.markdown(
                            f'<p class="status-ok">Email drafted and verified — '
                            f'{n_grounded}/{n_claims} claims traced to sources '
                            f'({gated.attempts} attempt(s))</p>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<p class="status-warn">Held for review — '
                            f'{n_grounded}/{n_claims} claims traced to sources '
                            f'after {gated.attempts} attempt(s)</p>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="critique-box">{verdict.critique}</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div class="email-box">{draft.render()}</div>',
                        unsafe_allow_html=True,
                    )
                    if verdict.claims:
                        with st.expander("Claim-by-claim grounding"):
                            for c in verdict.claims:
                                mark = "grounded" if c.grounded else "UNSUPPORTED"
                                st.markdown(f"**[{mark}]** {c.text}")
                                if c.source_quote:
                                    st.caption(f"source: {c.source_quote[:300]}")

                    results.append({
                        **lead,
                        "status": gated.status,
                        "subject": draft.subject,
                        "draft": draft.body,
                        "claims_total": n_claims,
                        "claims_grounded": n_grounded,
                        "critique": verdict.critique,
                    })
                except Exception as e:
                    st.markdown(
                        f'<p class="status-err">{user_message(e, "drafting this email")}</p>',
                        unsafe_allow_html=True,
                    )
                    results.append({**lead, "status": "DRAFT_ERROR", "draft": ""})
                    continue

        if i < len(leads) - 1:
            time.sleep(1)

    # Summary + Download
    st.divider()
    approved = sum(1 for r in results if r["status"] == "APPROVED")
    review = sum(1 for r in results if r["status"] == "NEEDS_REVIEW")
    errors = len(results) - approved - review

    st.markdown(f"### Done — {approved} verified, {review} held for review, {errors} failed")
    if review:
        st.caption(
            "Held drafts made claims that couldn't be traced back to the research. "
            "They're included in the CSV with the critique, but shouldn't be sent as-is."
        )

    if results:
        out = io.StringIO()
        fields = ["name", "role", "company", "email", "status", "subject",
                  "claims_total", "claims_grounded", "critique", "draft"]
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

        st.download_button(
            label="Download results CSV",
            data=out.getvalue(),
            file_name="outreach_results.csv",
            mime="text/csv",
            use_container_width=True,
        )