# utils/embeddings.py
"""
Upgraded embeddings & semantic utilities.
- Uses a stronger embedding model (instructor-xl / bge if available)
- Chunking + mean pooling for long docs
- Optional zero-shot role classifier (Bart MNLI) for auto-tagging
"""

from typing import List, Tuple
import re
import numpy as np

# Model imports (try sentence-transformers then fallback)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

# transformer pipeline for zero-shot classification (optional)
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# choose model names (user can swap)
EMBED_MODEL_NAMES = [
    "hkunlp/instructor-xl",           # high quality instruction tuned
    "BAAI/bge-large-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2"
]

# lazy load model
_embedding_model = None
_role_classifier = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Install via pip.")
    for name in EMBED_MODEL_NAMES:
        try:
            _embedding_model = SentenceTransformer(name)
            return _embedding_model
        except Exception:
            continue
    # If none loaded, raise
    raise RuntimeError("Failed to load any embedding model from EMBED_MODEL_NAMES.")


def _get_role_classifier():
    global _role_classifier
    if _role_classifier is not None:
        return _role_classifier
    if pipeline is None:
        return None
    try:
        _role_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
        return _role_classifier
    except Exception:
        return None


# -------------------------------
# Text cleaning & chunking
# -------------------------------
def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\r\n|\r|\n", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words = clean_text(text).split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# -------------------------------
# Embedding helpers
# -------------------------------
def get_sentence_embedding(text: str) -> np.ndarray:
    """
    Return a normalized numpy embedding vector for short text.
    """
    model = _get_embedding_model()
    if not text or text.strip() == "":
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=float)
    cleaned = clean_text(text)
    emb = model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
    return emb


def get_document_embedding(text: str, chunk_size: int = 512) -> np.ndarray:
    """
    For long documents: chunk, embed, mean-pool (GPU-friendly if model supports tensors).
    """
    model = _get_embedding_model()
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=int(chunk_size*0.12))
    if not chunks:
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=float)
    try:
        # prefer batch encode
        embs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        pooled = np.mean(embs, axis=0)
    except Exception:
        # fallback: encode per chunk
        embs = [model.encode(c, convert_to_numpy=True, normalize_embeddings=True) for c in chunks]
        pooled = np.mean(embs, axis=0)
    # normalize pooled vector
    norm = np.linalg.norm(pooled)
    if norm == 0:
        return pooled
    return pooled / norm


# -------------------------------
# Semantic similarity utilities
# -------------------------------
from sklearn.metrics.pairwise import cosine_similarity

def semantic_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.atleast_2d(vec_a)
    b = np.atleast_2d(vec_b)
    sim = cosine_similarity(a, b)[0][0]
    return float(np.clip(sim, -1.0, 1.0))


def compute_semantic_similarity(resume_text: str, jd_text: str) -> float:
    """
    Hybrid strategy: global doc embedding + section-level max-match
    Returns a float in [0,1]
    """
    res_doc = get_document_embedding(resume_text)
    jd_doc = get_document_embedding(jd_text)
    global_sim = semantic_similarity(res_doc, jd_doc)

    # section-wise
    res_chunks = chunk_text(resume_text, chunk_size=300)
    jd_chunks = chunk_text(jd_text, chunk_size=300)

    section_sims = []
    if res_chunks and jd_chunks:
        jd_embs = [get_sentence_embedding(j) for j in jd_chunks]
        for r in res_chunks:
            r_emb = get_sentence_embedding(r)
            sims = [semantic_similarity(r_emb, j_emb) for j_emb in jd_embs]
            section_sims.append(max(sims) if sims else 0.0)
        section_score = float(np.mean(section_sims))
    else:
        section_score = 0.0

    # weighted blend (tunable)
    final = 0.7 * global_sim + 0.3 * section_score
    # clip to [0,1]
    final = max(0.0, min(1.0, final))
    return round(final, 4)


# -------------------------------
# Role classification (optional)
# -------------------------------
def predict_job_role(resume_text: str, candidate_labels: List[str] = None) -> dict:
    """
    Optional zero-shot classification to infer probable role/domain.
    Returns dict like {"labels": [...], "scores":[...]}
    If transformers not installed or fails, returns empty dict.
    """
    classifier = _get_role_classifier()
    if classifier is None:
        return {}
    if not candidate_labels:
        candidate_labels = [
            "Data Scientist", "Machine Learning Engineer", "Software Engineer",
            "Data Engineer", "Product Manager", "Business Analyst", "DevOps", "Marketing"
        ]
    out = classifier(resume_text[:2000], candidate_labels)
    return out
