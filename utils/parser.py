# utils/parser.py
"""
Upgraded resume parser:
- Robust PDF extraction using PyMuPDF (fitz) or pdfminer fallback
- DOCX extraction via docx2txt
- NER using spaCy; attempts to clean & structure fields
- Returns structured dictionary ready for scoring pipeline
"""

import os
import re
from typing import Dict, List
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfminer.high_level as pdfminer_hl
except Exception:
    pdfminer_hl = None

try:
    import docx2txt
except Exception:
    docx2txt = None

# spaCy for NER
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    try:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

# -------------------------------
# TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(path: str) -> str:
    path = str(path)
    if fitz:
        try:
            doc = fitz.open(path)
            text = []
            for page in doc:
                text.append(page.get_text())
            return "\n".join(text)
        except Exception:
            pass
    if pdfminer_hl:
        try:
            return pdfminer_hl.extract_text(path) or ""
        except Exception:
            pass
    # fallback
    with open(path, "rb") as f:
        try:
            raw = f.read().decode("utf-8", errors="ignore")
            return raw
        except Exception:
            return ""


def extract_text_from_docx(path: str) -> str:
    if docx2txt:
        try:
            return docx2txt.process(path) or ""
        except Exception:
            pass
    # fallback: try python-docx
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def extract_text_from_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def parse_file(file_path: str) -> str:
    file_path = str(file_path)
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_path)
    elif ext in ("txt", "text"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


# -------------------------------
# TEXT CLEANING & FIELD EXTRACTION
# -------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\r\n|\r", "\n", text)
    t = re.sub(r"\n\s+\n", "\n", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# regex utilities
_email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_phone_re = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5})")
_link_re = re.compile(r"(https?://\S+|www\.\S+|linkedin\.com/[^\s,]+|github\.com/[^\s,]+)")

def extract_email(text: str):
    m = _email_re.search(text)
    return m.group(0) if m else None

def extract_phone(text: str):
    m = _phone_re.search(text)
    return m.group(0) if m else None

def extract_links(text: str):
    return list(set(_link_re.findall(text)))


def extract_skills_keyword(text: str, skill_bank: List[str]) -> List[str]:
    text_l = text.lower()
    found = []
    for s in skill_bank:
        if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_l):
            found.append(s)
    return sorted(set(found))


# spaCy-based entities
def extract_entities(text: str) -> Dict[str, List[str]]:
    if nlp is None:
        return {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}
    doc = nlp(text[:20000])  # limit length for spaCy speed
    ent = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}
    for e in doc.ents:
        if e.label_ in ent:
            ent[e.label_].append(e.text)
    # dedupe while preserving order
    for k in ent:
        seen = set()
        uniq = []
        for x in ent[k]:
            if x not in seen:
                uniq.append(x); seen.add(x)
        ent[k] = uniq
    return ent


# education & experience heuristics
_EDU_KEYWORDS = ["bachelor", "master", "b.tech", "b.e", "m.tech", "m.sc", "bsc", "msc", "mba", "phd", "diploma"]
_EXP_KEYWORDS = ["experience", "worked", "intern", "project", "role", "responsible", "contributed", "years"]

def extract_education(text: str) -> List[str]:
    sentences = re.split(r"[.\n]", text)
    edu = [s.strip() for s in sentences if any(k in s.lower() for k in _EDU_KEYWORDS)]
    return edu[:10]


def extract_experience(text: str) -> List[str]:
    sentences = re.split(r"[.\n]", text)
    exp = [s.strip() for s in sentences if any(k in s.lower() for k in _EXP_KEYWORDS)]
    return exp[:20]


# main parser
def parse_resume(file_path: str, skill_bank: List[str] = None) -> dict:
    raw = parse_file(file_path)
    cleaned = clean_text(raw)
    entities = extract_entities(cleaned)
    skills = extract_skills_keyword(cleaned, skill_bank or [])

    return {
        "name": entities["PERSON"][0] if entities["PERSON"] else None,
        "email": extract_email(cleaned),
        "phone": extract_phone(cleaned),
        "links": extract_links(cleaned),
        "skills": skills,
        "education": extract_education(cleaned),
        "experience": extract_experience(cleaned),
        "organizations": entities.get("ORG", []),
        "locations": entities.get("GPE", []),
        "raw_text": cleaned
    }
