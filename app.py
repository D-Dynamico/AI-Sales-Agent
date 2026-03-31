import streamlit as st
import os
import csv
import io
import time

# Load secrets from Streamlit Cloud
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

from agents.enricher import enrich_lead
from agents.drafter import draft_email

# Page config
st.set_page_config(
    page_title="AI Sales Agent · Alchemyst AI",
    layout="centered",
)

# Styles
st.markdown("""
<style>
    .main { max-width: 760px; }
    .lead-card {
        background: #f8f9fa;
        border-left: 3px solid #6c63ff;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .email-box {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        margin-top: 0.5rem;
    }
    .status-ok  { color: #2e7d32; font-weight: 600; }
    .status-err { color: #c62828; font-weight: 600; }
    .badge {
        display: inline-block;
        background: #ede7f6;
        color: #4527a0;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("## AI Sales Outreach Agent")
st.markdown(
    "Paste your leads and run the agent. The system researches each company "
    "and drafts a personalized cold email automatically."
)
st.divider()

# Sample data
SAMPLE = """name,role,company,email,website
Priya Sharma,Head of Sales,Razorpay,priya@razorpay.com,razorpay.com
Arjun Mehta,VP Marketing,Zepto,arjun@zepto.team,zeptonow.com
Neha Gupta,Co-founder,Sarvam AI,neha@sarvam.ai,sarvam.ai"""

# Input
st.markdown("### Lead List")
st.caption("CSV format: name, role, company, email, website")

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
    st.caption("Powered by LLaMA 3.1 · Groq · Tavily Search")

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

        with st.container():
            st.markdown(
                f'<div class="lead-card">'
                f'<span class="badge">Lead {i+1}</span> '
                f'<strong>{name}</strong> · {lead.get("role", "")} @ {company}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Enrich
            with st.spinner(f"Researching {company}..."):
                try:
                    research = enrich_lead(lead)
                    st.markdown(
                        '<p class="status-ok">Research complete</p>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("View research brief"):
                        st.write(research)
                except Exception as e:
                    st.markdown(
                        f'<p class="status-err">Research failed: {e}</p>',
                        unsafe_allow_html=True,
                    )
                    results.append({**lead, "status": "ENRICH_ERROR", "draft": ""})
                    continue

            # Draft
            with st.spinner(f"Drafting email for {name}..."):
                try:
                    draft = draft_email(lead, research)
                    st.markdown(
                        '<p class="status-ok">Email drafted</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="email-box">{draft}</div>',
                        unsafe_allow_html=True,
                    )
                    results.append({**lead, "status": "DRAFTED", "draft": draft})
                except Exception as e:
                    st.markdown(
                        f'<p class="status-err">Draft failed: {e}</p>',
                        unsafe_allow_html=True,
                    )
                    results.append({**lead, "status": "DRAFT_ERROR", "draft": ""})
                    continue

        if i < len(leads) - 1:
            time.sleep(1)

    # Summary + Download
    st.divider()
    sent = sum(1 for r in results if r["status"] == "DRAFTED")
    errors = len(results) - sent

    st.markdown(f"### Completed — {sent} drafted, {errors} failed")

    if results:
        out = io.StringIO()
        fields = ["name", "role", "company", "email", "status", "draft"]
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