# utils/scorer.py
"""
Improved scoring module:
- Hybrid skill extraction (keyword + fuzzy)
- TF-IDF keyword similarity + embedding similarity
- Final scoring is explainable and returns breakdown
"""

from typing import Dict, List
import numpy as np
import re

# TF-IDF & cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# fuzzy string match (optional high-quality)
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# default skill bank (extendable)
def load_skill_bank() -> Dict[str, List[str]]:
    return {
        "technical": [
            "python", "java", "c++", "sql", "tensorflow", "pytorch", "nlp", "bert",
            "machine learning", "deep learning", "data analysis", "computer vision",
            "transformers", "statistics", "scikit-learn", "huggingface", "streamlit",
            "docker", "aws", "azure", "git"
        ],
        "soft": ["communication", "leadership", "problem solving", "teamwork", "adaptability"],
        "domain": ["finance", "healthcare", "retail", "education", "marketing", "ai", "cloud"]
    }


# -------------------------------
# Skill extraction + fuzzy helper
# -------------------------------
def _fuzzy_contains(text: str, skill: str, threshold: int = 80) -> bool:
    text_l = text.lower()
    if skill.lower() in text_l:
        return True
    if fuzz:
        score = fuzz.partial_ratio(skill.lower(), text_l)
        return score >= threshold
    # fallback: crude substring check
    return skill.lower() in text_l


def extract_skills_from_text(text: str, skill_bank: List[str]) -> List[str]:
    found = []
    for s in skill_bank:
        if _fuzzy_contains(text, s):
            found.append(s)
    return sorted(set(found))


# -------------------------------
# Hybrid scoring
# -------------------------------
def compute_skill_metrics(resume_text: str, jd_text: str, skill_bank_map: Dict[str, List[str]] = None) -> Dict:
    if skill_bank_map is None:
        skill_bank_map = load_skill_bank()

    categories = {}
    resume_skills = []
    jd_skills = []

    for cat, skills in skill_bank_map.items():
        r_found = extract_skills_from_text(resume_text, skills)
        j_found = extract_skills_from_text(jd_text, skills)

        matched = list(set(r_found) & set(j_found))
        missing = list(set(j_found) - set(r_found))
        coverage = (len(matched) / max(1, len(j_found))) * 100 if j_found else 0.0

        categories[cat] = {
            "resume_skills": sorted(r_found),
            "jd_skills": sorted(j_found),
            "matched": sorted(matched),
            "missing": sorted(missing),
            "coverage": round(coverage, 2)
        }

        resume_skills.extend(r_found)
        jd_skills.extend(j_found)

    overall_coverage = float(np.mean([v["coverage"] for v in categories.values()])) if categories else 0.0

    return {
        "categories": categories,
        "overall_coverage": round(overall_coverage, 2),
        "resume_skills": sorted(set(resume_skills)),
        "jd_skills": sorted(set(jd_skills))
    }


def compute_hybrid_similarity(resume_text: str, jd_text: str, embed_sim_func=None) -> Dict:
    """
    Combine TF-IDF keyword similarity with embedding semantic similarity.
    embed_sim_func should be a callable that returns float in [0,1]
    """
    texts = [resume_text, jd_text]
    try:
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        tfidf_m = tfidf.fit_transform(texts)
        keyword_sim = float(cosine_similarity(tfidf_m[0:1], tfidf_m[1:2])[0][0])
    except Exception:
        keyword_sim = 0.0

    embed_sim = 0.0
    if embed_sim_func:
        try:
            embed_sim = float(embed_sim_func(resume_text, jd_text))
        except Exception:
            embed_sim = 0.0

    # blend - tunable weights
    w_embed = 0.6
    w_keyword = 0.4
    combined = w_embed * embed_sim + w_keyword * keyword_sim
    combined = max(0.0, min(1.0, combined))
    return {
        "keyword_similarity": round(keyword_sim, 4),
        "embedding_similarity": round(embed_sim, 4),
        "combined_similarity": round(combined, 4)
    }


# -------------------------------
# final scoring and explainability
# -------------------------------
def compute_final_score(similarity_combined: float, skill_metrics: Dict, weight_semantic: float = 0.65) -> Dict:
    """
    Returns:
    - semantic_similarity (0-100)
    - skill_score (0-100)
    - final_fit_score (0-100)
    """
    semantic_pct = round(similarity_combined * 100, 2)
    skill_pct = skill_metrics.get("overall_coverage", 0.0)

    final = round(weight_semantic * semantic_pct + (1 - weight_semantic) * skill_pct, 2)

    return {
        "semantic_similarity": semantic_pct,
        "skill_score": round(skill_pct, 2),
        "final_fit_score": final
    }


def generate_suggestions(parsed_resume: Dict, skill_metrics: Dict, top_k: int = 3) -> List[str]:
    """
    Rule-based feedback generator (used if no LLM available).
    Produces short, actionable suggestions.
    """
    suggestions = []
    # missing top domain/technical skills
    missing = []
    for cat, details in skill_metrics["categories"].items():
        if details["missing"]:
            missing.extend(details["missing"])
    missing = sorted(set(missing))
    if missing:
        suggestions.append(f"Add or emphasize missing skills: {', '.join(missing[:7])}.")

    # experience
    exp = parsed_resume.get("experience", [])
    if not exp or len(exp) < 2:
        suggestions.append("Expand experience section with specific project details and achievements (metrics preferred).")

    # education / links
    if not parsed_resume.get("links"):
        suggestions.append("Add links to GitHub/LinkedIn/Portfolio to demonstrate projects and code.")
    # soft skills if low
    soft_cov = skill_metrics["categories"].get("soft", {}).get("coverage", 0)
    if soft_cov < 50:
        suggestions.append("Include 2–3 brief bullet points highlighting leadership / teamwork examples.")

    if not suggestions:
        suggestions = ["Resume looks reasonably aligned — consider adding quantified achievements and keywords from the JD."]
    return suggestions[:top_k]
