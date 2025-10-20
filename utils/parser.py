"""
parser.py
Advanced Resume Parser for AI Resume Analyzer
----------------------------------------------
Extracts structured information from resumes using NLP, regex, and rule-based logic.
"""

import re
import os
import pdfminer.high_level
import docx2txt
import spacy

# Load spaCy model for NER (you can use 'en_core_web_trf' for transformer version)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# =============== TEXT EXTRACTION ===============

def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF with fallback handling."""
    try:
        return pdfminer.high_level.extract_text(path)
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        return ""


def extract_text_from_docx(path: str) -> str:
    """Extract text from DOCX."""
    try:
        return docx2txt.process(path)
    except Exception as e:
        print(f"[ERROR] DOCX extraction failed: {e}")
        return ""


def extract_text_from_txt(path: str) -> str:
    """Read plain text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] TXT extraction failed: {e}")
        return ""


def parse_file(file_path: str) -> str:
    """Unified interface to extract text from any supported file."""
    file_path = file_path.lower()
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
    return clean_text(text)


# =============== TEXT CLEANING ===============

def clean_text(text: str) -> str:
    """Basic text cleaning and normalization."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)  # collapse spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII chars
    return text.strip()


# =============== REGEX UTILITIES ===============

def extract_email(text: str):
    match = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)
    return match.group(0) if match else None


def extract_phone(text: str):
    match = re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{3,5}\)?[\s-]?\d{3,5}[\s-]?\d{3,5}', text)
    return match.group(0) if match else None


def extract_links(text: str):
    return re.findall(r'(https?://[^\s]+)', text)


def extract_skills(text: str):
    """Basic skill extraction via keyword matching (can be upgraded with ML)."""
    # Common technical skills (can be extended dynamically)
    tech_skills = [
        "python", "java", "c++", "sql", "r", "excel", "power bi", "tableau",
        "tensorflow", "pytorch", "keras", "scikit-learn", "aws", "azure",
        "nlp", "deep learning", "machine learning", "data analysis",
        "react", "javascript", "html", "css", "docker", "git", "flask"
    ]
    text_lower = text.lower()
    found = [skill for skill in tech_skills if skill in text_lower]
    return sorted(list(set(found)))


# =============== NLP-BASED EXTRACTION ===============

def extract_entities(text: str):
    """Use spaCy NER to identify entities."""
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities


def extract_education(text: str):
    """Rule-based extraction for education details."""
    edu_keywords = [
        "bachelor", "master", "b.tech", "b.e", "m.tech", "phd",
        "bsc", "msc", "mba", "mca", "diploma"
    ]
    edu_lines = [line for line in text.split('.') if any(k in line.lower() for k in edu_keywords)]
    return [clean_text(e) for e in edu_lines]


def extract_experience(text: str):
    """Extract experience-related sentences."""
    exp_keywords = ["experience", "worked at", "intern", "project", "role", "company"]
    exp_lines = [line for line in text.split('.') if any(k in line.lower() for k in exp_keywords)]
    return [clean_text(e) for e in exp_lines]


# =============== MAIN INTERFACE ===============

def parse_resume(file_path: str) -> dict:
    """
    Master resume parser — combines all extraction methods
    and returns structured output.
    """
    raw_text = parse_file(file_path)
    entities = extract_entities(raw_text)

    return {
        "name": entities["PERSON"][0] if entities["PERSON"] else None,
        "email": extract_email(raw_text),
        "phone": extract_phone(raw_text),
        "links": extract_links(raw_text),
        "skills": extract_skills(raw_text),
        "education": extract_education(raw_text),
        "experience": extract_experience(raw_text),
        "organizations": list(set(entities["ORG"])),
        "locations": list(set(entities["GPE"])),
        "raw_text": raw_text[:3000]  # truncate for lightweight downstream use
    }
