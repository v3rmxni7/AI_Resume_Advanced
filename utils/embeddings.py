import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load a high-performance transformer model
# all-mpnet-base-v2 provides richer semantic representation than MiniLM
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# -------------------------------
# 🧠 TEXT CLEANING
# -------------------------------

def clean_text(text: str) -> str:
    """
    Cleans resume or JD text by removing special characters,
    excessive whitespace, and irrelevant formatting.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()\-\s]", "", text)
    return text.strip().lower()


# -------------------------------
# 🔍 SECTION-BASED EMBEDDINGS
# -------------------------------

def segment_text(text: str, chunk_size: int = 512, overlap: int = 50):
    """
    Split long documents (resumes or job descriptions)
    into overlapping chunks for better embedding coverage.
    """
    words = text.split()
    segments = []
    for i in range(0, len(words), chunk_size - overlap):
        segment = " ".join(words[i:i + chunk_size])
        segments.append(segment)
        if i + chunk_size >= len(words):
            break
    return segments


# -------------------------------
# 🧩 EMBEDDING FUNCTIONS
# -------------------------------

def get_embedding(text: str) -> np.ndarray:
    """
    Get sentence-level embedding for a given text.
    """
    if not text or len(text.strip()) == 0:
        return np.zeros(model.get_sentence_embedding_dimension())
    text = clean_text(text)
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def get_document_embedding(text: str) -> np.ndarray:
    """
    Get an aggregated (mean-pooled) embedding for an entire document.
    It handles long resumes by chunking and averaging embeddings.
    """
    text = clean_text(text)
    chunks = segment_text(text)
    embeddings = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
    doc_embedding = torch.mean(embeddings, dim=0)
    return doc_embedding.cpu().numpy()


# -------------------------------
# 📊 SIMILARITY FUNCTIONS
# -------------------------------

def compute_semantic_similarity(resume_text: str, jd_text: str) -> float:
    """
    Computes semantic similarity between a resume and a job description.
    Combines both global and section-based comparison.
    """
    # Get embeddings
    resume_emb = get_document_embedding(resume_text)
    jd_emb = get_document_embedding(jd_text)

    # Global similarity
    global_sim = cosine_similarity([resume_emb], [jd_emb])[0][0]

    # Section-wise comparison for fine-grained alignment
    resume_sections = segment_text(resume_text, chunk_size=256)
    jd_sections = segment_text(jd_text, chunk_size=256)

    section_sims = []
    for r in resume_sections:
        r_emb = get_embedding(r)
        sims = [cosine_similarity([r_emb], [get_embedding(j)])[0][0] for j in jd_sections]
        section_sims.append(max(sims))  # best match for that resume section

    section_score = np.mean(section_sims) if section_sims else 0

    # Weighted hybrid similarity
    final_similarity = 0.7 * global_sim + 0.3 * section_score

    return round(float(final_similarity), 4)


# -------------------------------
# ⚡ EXAMPLE TEST
# -------------------------------
if __name__ == "__main__":
    resume_text = """Experienced data scientist skilled in Python, NLP, and ML pipelines using BERT and TensorFlow."""
    jd_text = """Looking for a Machine Learning Engineer with experience in NLP, deep learning, and Python frameworks."""

    sim = compute_semantic_similarity(resume_text, jd_text)
    print(f"Semantic Similarity: {sim * 100:.2f}%")
