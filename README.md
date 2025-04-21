# 📄 Resume–Job Matcher

An intelligent tool built with NLP and deep learning to compare resumes with job descriptions and generate insights, scores, and personalized feedback.

---

### 🚀 Features

- Upload job descriptions and resumes (supports multi-resume)
- Sentence-level similarity scoring using `MiniLM-L6-v2`
- Keyword extraction and match analysis
- Visual dashboards (bar and pie charts)
- Downloadable PDF report
- Streamlit UI

---

### 🧪 Technologies Used

- Python
- Streamlit
- scikit-learn
- SentenceTransformers
- matplotlib, pandas
- FPDF

---

### 📁 File Structure

├── app.py # Streamlit app ├── main.py # Main logic (optional entrypoint) ├── utils.py # Helper functions (text extraction, similarity, etc.) ├── job_description.txt # Sample JD file ├── resume.pdf # Sample Resume └── README.md # Project overview