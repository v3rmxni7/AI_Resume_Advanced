import streamlit as st
import os
from utils.parser import parse_resume
from utils.embeddings import compute_semantic_similarity
from utils.scorer import compute_skill_score, compute_final_score

# Streamlit configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# 🎨 PAGE HEADER
# -------------------------------
st.title("🤖 AI Resume Analyzer")
st.caption("Advanced Resume–Job Matching using Transformer Embeddings + Skill Intelligence")
st.divider()

# -------------------------------
# 📄 INPUTS
# -------------------------------
col1, col2 = st.columns([0.6, 0.4])

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
with col2:
    job_desc = st.text_area("💼 Paste Job Description", height=200)

# -------------------------------
# 🚀 ANALYSIS PIPELINE
# -------------------------------
if resume_file and job_desc:
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", resume_file.name)

    with open(path, "wb") as f:
        f.write(resume_file.read())

    with st.spinner("Extracting resume text..."):
        parsed_resume = parse_resume(path)

        # Ensure raw_text exists
        resume_text = parsed_resume.get("raw_text", "")
        if not resume_text:
            st.error("Failed to extract text from resume.")
            st.stop()

    with st.spinner("Computing semantic similarity..."):
        similarity = compute_semantic_similarity(resume_text, job_desc)

    with st.spinner("Analyzing skill alignment..."):
        skill_metrics = compute_skill_score(resume_text, job_desc)

    result = compute_final_score(similarity, skill_metrics)

    st.divider()
    st.subheader("📊 Overall Fit Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Semantic Similarity", f"{result['semantic_similarity']}%")
    c2.metric("Skill Match Score", f"{result['skill_score']}%")
    c3.metric("Final Fit Score", f"{result['final_fit_score']}%")

    st.progress(int(result["final_fit_score"]))

    # -------------------------------
    # 🔍 CATEGORY INSIGHTS
    # -------------------------------
    st.subheader("🧩 Category-wise Skill Insights")
    for category, details in skill_metrics["categories"].items():
        st.markdown(f"### • {category.capitalize()} Skills")
        st.write(f"**Coverage:** {details['coverage']}%")
        col_a, col_b = st.columns(2)
        with col_a:
            st.success(
                f"✅ Matched ({len(details['matched'])}): " + 
                ", ".join(details['matched']) if details['matched'] else "None"
            )
        with col_b:
            st.error(
                f"❌ Missing ({len(details['missing'])}): " + 
                ", ".join(details['missing']) if details['missing'] else "None"
            )
        st.divider()

    # -------------------------------
    # 🔮 AI Summary
    # -------------------------------
    st.subheader("🧠 AI Summary")
    score = result["final_fit_score"]
    if score > 80:
        st.success("Excellent match! Your resume aligns strongly with the job requirements. ✅")
    elif score > 60:
        st.warning("Moderate match — consider adding missing domain or soft skills.")
    else:
        st.error("Low match — revise resume with targeted technical and domain skills.")
