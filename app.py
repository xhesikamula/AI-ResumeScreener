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

# ------------------ NLTK Setup ------------------
nltk.download("punkt")
nltk.download("stopwords")
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

def show_keywords(title, words, icon):
    with st.expander(title):
        if words:
            st.write(" ".join([f"{icon} {w}" for w in list(words)[:50]]))
        else:
            st.write("None found")

def highlight_keywords_in_resume(resume_text, matched_keywords):
    st.subheader("📄 Keywords in Context")
    resume_sentences = [s.strip() for s in resume_text.split('.') if s.strip()]
    for sentence in resume_sentences:
        highlighted = sentence
        for kw in matched_keywords:
            if kw.lower() in sentence.lower():
                highlighted = highlighted.replace(kw, f"**{kw}**")
        if any(kw.lower() in sentence.lower() for kw in matched_keywords):
            st.write(f"✓ {highlighted}")

# ------------------ Streamlit UI ------------------

st.title("🧑‍💻 AI Resume Screener")
st.write("Upload a job description and your resume to check the match score!")

job_description = st.text_area("Paste Job Description here:")
resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

if st.button("Analyze") and job_description and resume_file:
    with st.spinner("Extracting resume text..."):
        resume_text = extract_text_from_pdf_fallback(resume_file)
        time.sleep(0.5)

    matched, missing, job_keywords = keyword_match(job_description, resume_text)
    final_score, sim_score, keyword_score = compute_final_score(
        job_description, resume_text, matched, job_keywords
    )

    st.subheader("✓ Overall Match Score")
    st.metric("Final Score", f"{final_score}%")
    st.write(f"🔹 Semantic Similarity: {sim_score}%")
    st.write(f"🔹 Keyword Coverage: {keyword_score}%")

    show_keywords("🎯 Skills Matched", matched, "✓")
    show_keywords("❗ Skills Missing", missing, "✖")

    highlight_keywords_in_resume(resume_text, matched)
