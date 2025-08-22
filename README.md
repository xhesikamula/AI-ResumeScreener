# 🧑‍💻 AI Resume Screener

A lightweight NLP tool to match candidate resumes to job descriptions. Built with Streamlit and Sentence-BERT embeddings, it helps candidates see how well their skills align with job requirements.

---

## 🚀 Features

- Upload a **job description** (text) and a **resume** (PDF).  
- Compute **semantic similarity** between resume and job description using Sentence-BERT.  
- Extract **skills matched** and **skills missing** using keyword extraction.  
- Highlight **matched keywords in context** within the resume.  
- Clean, interactive Streamlit interface.

---

## 📂 Project Structure
resume_screener/

│── app.py # Main Streamlit app

│── requirements.txt # Python dependencies

│── sample_resume.pdf # Example resume for testing

│── sample_job.txt # Example job description for testing

│── README.md # Project description

---

## 🛠 Tech Stack

- **Python**  
- **Streamlit** (interactive UI)  
- **Sentence-BERT** (`all-MiniLM-L6-v2`) for semantic similarity  
- **RAKE + NLTK** for keyword extraction  
- **PyPDF2** & **pytesseract** for PDF parsing and OCR fallback  

---

## 💻 Installation & Running Locally

1. Clone the repository:

```bash
git clone <this repo>
cd resume_screener

---

2. Install dependencies:
pip install -r requirements.txt

---

3. Run the Streamlit app:
streamlit run app.py

