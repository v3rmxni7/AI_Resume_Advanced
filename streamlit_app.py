# streamlit_app.py
"""
Streamlit app for upgraded AI Resume Analyzer (single-stack).
Place this file at repo root and the utils package beside it.
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import tempfile
import json
from utils import parser, embeddings, scorer

# UI config
st.set_page_config(page_title="AI Resume Analyzer", layout="wide", page_icon="🤖")

st.title("🤖 AI Resume Analyzer — Upgraded")
st.caption("Hybrid semantic + keyword matching · Improved parsing · Explainable suggestions")
st.divider()

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size for embeddings", 256, 1024, 512, step=64)
    use_role_classification = st.checkbox("Predict job role (zero-shot)", value=True)
    st.markdown("**Notes:** The app may download models on first run. This may take time and memory.")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    resume_file = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("Paste Job Description", height=300, placeholder="Paste the job description here...")

with col2:
    st.markdown("### Quick Tips")
    st.write("- Add a clear JD with responsibilities and required skills for best results.")
    st.write("- If parsing fails for a PDF, try uploading a DOCX or TXT version.")
    st.markdown("---")
    st.markdown("### Example actions")
    st.write("• Download a sample resume or copy JD text to test.")

analyze_btn = st.button("Analyze")

if analyze_btn:
    if not resume_file:
        st.warning("Please upload a resume.")
        st.stop()
    if not jd_text or jd_text.strip() == "":
        st.warning("Please paste a job description.")
        st.stop()

    # save uploaded file to temp
    tdir = tempfile.mkdtemp()
    path = os.path.join(tdir, resume_file.name)
    with open(path, "wb") as f:
        f.write(resume_file.read())

    with st.spinner("Parsing resume..."):
        # build skill bank list for parser
        skill_bank_map = scorer.load_skill_bank()
        # flatten bank for parser keywords
        flat_skills = [s for subs in skill_bank_map.values() for s in subs]
        parsed = parser.parse_resume(path, skill_bank=flat_skills)

    resume_text = parsed.get("raw_text", "")
    if not resume_text:
        st.error("Failed to extract resume text. Try another format.")
        st.stop()

    with st.spinner("Computing semantic similarity..."):
        # set chunking config by temporarily overriding chunk function behavior if needed
        embeddings.chunk_text  # ensure available
        sim_val = embeddings.compute_semantic_similarity(resume_text, jd_text)

    with st.spinner("Computing skill metrics..."):
        skill_metrics = scorer.compute_skill_metrics(resume_text, jd_text, skill_bank_map)

    with st.spinner("Combining scores..."):
        hybrid = scorer.compute_hybrid_similarity(resume_text, jd_text, embed_sim_func=embeddings.compute_semantic_similarity)
        final = scorer.compute_final_score(hybrid["combined_similarity"], skill_metrics, weight_semantic=0.65)

    # Results panel
    st.divider()
    st.subheader("📊 Overall Fit")
    c1, c2, c3 = st.columns(3)
    c1.metric("Semantic Similarity", f"{final['semantic_similarity']}%")
    c2.metric("Skill Match Score", f"{final['skill_score']}%")
    c3.metric("Final Fit Score", f"{final['final_fit_score']}%")

    st.progress(int(min(100, final['final_fit_score'])))

    # Role prediction (optional)
    if use_role_classification:
        with st.expander("Predicted Roles"):
            try:
                role_out = embeddings.predict_job_role(resume_text)
                if role_out:
                    for lbl, score in zip(role_out.get("labels", [])[:5], role_out.get("scores", [])[:5]):
                        st.write(f"- **{lbl}** — {round(score*100,2)}%")
                else:
                    st.write("Role classifier not available or failed to run.")
            except Exception as e:
                st.write("Role prediction failed.", e)

    # Category-wise insights
    st.subheader("🧩 Category-wise Skill Insights")
    for cat, details in skill_metrics["categories"].items():
        st.markdown(f"**{cat.capitalize()} Skills — Coverage: {details['coverage']}%**")
        cols = st.columns(2)
        with cols[0]:
            st.write("Matched:", details["matched"] or "None")
        with cols[1]:
            st.write("Missing:", details["missing"] or "None")
        st.divider()

    # AI-based suggestions or rule-based fallback
    st.subheader("🧠 Suggestions & Improvements")
    suggestions = scorer.generate_suggestions(parsed, skill_metrics, top_k=5)
    for s in suggestions:
        st.info(s)

    # Raw parsed resume preview (truncated)
    with st.expander("Parsed Resume (truncated preview)"):
        st.json({
            "name": parsed.get("name"),
            "email": parsed.get("email"),
            "phone": parsed.get("phone"),
            "links": parsed.get("links"),
            "skills": parsed.get("skills")[:50],
            "education": parsed.get("education")[:10],
            "experience": parsed.get("experience")[:10]
        })

    # allow download of results
    result_obj = {
        "final": final,
        "hybrid": hybrid,
        "skills": skill_metrics,
        "parsed": parsed
    }
    st.download_button("Download JSON Report", data=json.dumps(result_obj, indent=2), file_name="analysis_report.json")