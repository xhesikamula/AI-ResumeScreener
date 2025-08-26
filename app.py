# streamlit run app.py

import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from pdf2image import convert_from_bytes
import pytesseract
import time
import os
import re

# ------------------ NLTK Setup ------------------
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("punkt_tab", download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
stop_words = set(stopwords.words("english"))

# ------------------ Load Model ------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Helper Functions ------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text.strip()

def extract_text_from_pdf_fallback(uploaded_file):
    text = extract_text_from_pdf(uploaded_file)
    if not text:
        images = convert_from_bytes(uploaded_file.read())
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
    return text

def extract_keywords(text, top_n=15):
    r = Rake()
    r.extract_keywords_from_text(text)
    key_phrases = r.get_ranked_phrases()[:top_n]
    return set(key_phrases)

def clean_and_tokenize(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    return set(tokens)

def keyword_match(job_text, resume_text):
    job_keywords = extract_keywords(job_text)
    resume_words = clean_and_tokenize(resume_text)
    matched = {kw for kw in job_keywords if any(word in resume_words for word in kw.split())}
    missing = job_keywords - matched
    return matched, missing, job_keywords

def compute_keyword_coverage(matched, job_keywords):
    if not job_keywords:
        return 0
    return round((len(matched) / len(job_keywords)) * 100, 2)

def compute_final_score(job_text, resume_text, matched, job_keywords):
    with st.spinner("Computing embeddings..."):
        embeddings = model.encode([job_text, resume_text])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100
    keyword_score = compute_keyword_coverage(matched, job_keywords)
    final_score = (0.6 * sim_score) + (0.4 * keyword_score)
    return round(final_score, 2), round(sim_score, 2), keyword_score

def show_keywords(title, words, color):
    st.subheader(title)
    if words:
        chips = " ".join(
            [f"<span style='background:{color}; padding:6px 12px; border-radius:20px; margin:4px; display:inline-block; font-size:0.9rem;'>{w}</span>" for w in words]
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.write("None found")

def highlight_keywords_in_resume(resume_text, matched_keywords):
    st.subheader("📄 Keywords in Context")
    resume_sentences = [s.strip() for s in resume_text.split('.') if s.strip()]
    for sentence in resume_sentences:
        sentence_lower = sentence.lower()
        kws_in_sentence = [kw for kw in matched_keywords if any(word in sentence_lower for word in kw.lower().split())]
        if kws_in_sentence:
            highlighted = sentence
            for kw in kws_in_sentence:
                highlighted = re.sub(f"(?i)({re.escape(kw)})", r"**\1**", highlighted)
            st.write(f"✓ {highlighted}")

# ------------------ Streamlit Custom Styling ------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Streamlit UI ------------------
st.markdown("<h1 class='main-title'>🚀 AI Resume Screener</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload your resume & job description to see how well you match!</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    job_description = st.text_area("📄 Paste Job Description", height=300)

with col2:
    resume_file = st.file_uploader("📂 Upload Resume (PDF only)", type=["pdf"])

if st.button("Analyze") and job_description and resume_file:
    with st.spinner("Extracting resume text..."):
        resume_text = extract_text_from_pdf_fallback(resume_file)
        time.sleep(0.5)

    matched, missing, job_keywords = keyword_match(job_description, resume_text)
    final_score, sim_score, keyword_score = compute_final_score(job_description, resume_text, matched, job_keywords)

    # Score Cards
    st.markdown(
        f"""
        <div style="display:flex; gap:20px; justify-content:center; margin-top:20px; flex-wrap:wrap;">
            <div style="flex:1; min-width:200px; padding:20px; border-radius:15px; background:#EEF2FF; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h3 style="color:#4338CA;">Final Score</h3>
                <p style="font-size:2rem; font-weight:bold; color:#111827;">{final_score}%</p>
            </div>
            <div style="flex:1; min-width:200px; padding:20px; border-radius:15px; background:#ECFDF5; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h3 style="color:#047857;">Semantic Similarity</h3>
                <p style="font-size:2rem; font-weight:bold; color:#111827;">{sim_score}%</p>
            </div>
            <div style="flex:1; min-width:200px; padding:20px; border-radius:15px; background:#FEF3C7; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h3 style="color:#B45309;">Keyword Coverage</h3>
                <p style="font-size:2rem; font-weight:bold; color:#111827;">{keyword_score}%</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_keywords("✅ Skills Matched", matched, "#DCFCE7")
    show_keywords("❌ Skills Missing", missing, "#FEE2E2")

    highlight_keywords_in_resume(resume_text, matched)
