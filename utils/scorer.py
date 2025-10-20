import re
import numpy as np
from collections import defaultdict
from typing import List, Dict

# -------------------------------
# 🧩 SKILL EXTRACTION UTILS
# -------------------------------

def extract_skills(text: str, skill_list: List[str]) -> List[str]:
    """
    Extract matching skills from text using regex-based fuzzy matching.
    """
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))


def load_skill_bank() -> Dict[str, List[str]]:
    """
    Define or load categorized skill lists.
    (Can later be loaded from an external JSON or DB.)
    """
    return {
        "technical": [
            "python", "java", "sql", "tensorflow", "pytorch", "nlp", "bert", "ml",
            "machine learning", "deep learning", "data analysis", "computer vision",
            "transformers", "statistics", "scikit-learn", "huggingface", "streamlit"
        ],
        "soft": ["communication", "leadership", "problem solving", "teamwork", "adaptability"],
        "domain": ["finance", "healthcare", "retail", "education", "marketing", "ai", "cloud"]
    }


# -------------------------------
# 🧮 SCORE COMPUTATION
# -------------------------------

def compute_skill_score(resume_text: str, jd_text: str) -> Dict:
    """
    Compare extracted skills from resume vs job description.
    Returns skill overlap, missing skills, and category-wise coverage.
    """
    skill_bank = load_skill_bank()
    results = {}

    resume_skills = []
    jd_skills = []

    for cat, skills in skill_bank.items():
        r_skills = extract_skills(resume_text, skills)
        j_skills = extract_skills(jd_text, skills)

        overlap = list(set(r_skills) & set(j_skills))
        missing = list(set(j_skills) - set(r_skills))
        coverage = (len(overlap) / len(j_skills)) * 100 if j_skills else 0

        results[cat] = {
            "resume_skills": r_skills,
            "jd_skills": j_skills,
            "matched": overlap,
            "missing": missing,
            "coverage": round(coverage, 2)
        }

        resume_skills.extend(r_skills)
        jd_skills.extend(j_skills)

    overall_coverage = np.mean([v["coverage"] for v in results.values()])

    return {
        "categories": results,
        "overall_coverage": round(overall_coverage, 2),
        "resume_skills": list(set(resume_skills)),
        "jd_skills": list(set(jd_skills))
    }


def compute_final_score(similarity: float, skill_metrics: Dict) -> Dict:
    """
    Combine semantic similarity and skill overlap for a final explainable score.
    """
    semantic_weight = 0.65
    skill_weight = 0.35

    skill_score = skill_metrics["overall_coverage"]
    final_score = round((semantic_weight * similarity * 100) + (skill_weight * skill_score), 2)

    return {
        "semantic_similarity": round(similarity * 100, 2),
        "skill_score": round(skill_score, 2),
        "final_fit_score": final_score
    }
