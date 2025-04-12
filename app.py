import streamlit as st
from utils import (
    extract_text_from_pdf, clean_text, extract_keywords,
    find_similar_sentences, extract_entities, group_keywords
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from fpdf import FPDF
import re
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📄 Resume–Job Description Matcher")
st.write("Upload a resume and a job description to get insights and recommendations.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (TXT or PDF)", type=["txt", "pdf"])

def sanitize(text):
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
        "\xa0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return re.sub(r"[^\x00-\xFF]", "?", text)

if resume_file and jd_file:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = jd_file.read().decode("utf-8") if jd_file.name.endswith(".txt") else extract_text_from_pdf(jd_file)

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    match_score = cosine_similarity([model.encode(jd_clean)], [model.encode(resume_clean)])[0][0] * 100
    st.metric("🎯 Match Score", f"{match_score:.2f}%")

    jd_keywords = set([kw.lower().strip() for kw in extract_keywords(jd_clean)])
    resume_words = set([word.lower().strip() for word in resume_clean.split()])
    missing_keywords = jd_keywords - resume_words
    grouped_keywords = group_keywords(missing_keywords)

    st.subheader("🔍 Missing Keywords by Category")
    for category, items in grouped_keywords.items():
        if items:
            st.markdown(f"**{category}:**")
            st.write(", ".join(sorted(set(items))))

    st.subheader("🧬 Entity Comparison")
    jd_entities = extract_entities(jd_text)
    resume_entities = extract_entities(resume_text)
    both = jd_entities & resume_entities
    missing = jd_entities - resume_entities

    st.markdown("**✅ Found in Both:**")
    st.write(", ".join(sorted(both)) if both else "None")

    st.markdown("**❌ Missing from Resume:**")
    st.write(", ".join(sorted(missing)) if missing else "None")

    st.subheader("📌 Top Sentence Matches")
    matches = find_similar_sentences(jd_text, resume_text, model, threshold=0.5)
    matches.sort(key=lambda x: x[2], reverse=True)
    top_matches = matches[:5]

    for jd_sent, res_sent, score in top_matches:
        st.markdown(f"**JD:** {jd_sent.strip()}")
        st.markdown(f"**Resume:** {res_sent.strip()}")
        st.markdown(f"**Score:** {score}")
        st.markdown("---")

    def generate_pdf(score, grouped, both, missing, matches):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def section_header(title):
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 10, sanitize(title), ln=True)
            pdf.ln(3)

        def body_text(text):
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 10, sanitize(text))
            pdf.ln(1)

        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 10, sanitize("Resume–JD Matching Report"), ln=True, align='C')
        pdf.ln(10)

        section_header(f"🎯 Match Score: {score:.2f}%")

        section_header("🔍 Missing Keywords by Category")
        for category, items in grouped.items():
            if items:
                body_text(f"{category}: {', '.join(items)}")

        section_header("🧬 Entities Found in Both")
        body_text(", ".join(both) if both else "None")

        section_header("❌ Entities Missing from Resume")
        body_text(", ".join(missing) if missing else "None")

        section_header("📌 Top Sentence Matches")
        for jd_sent, res_sent, s in matches[:5]:
            body_text(f"JD: {jd_sent.strip()}\nResume: {res_sent.strip()}\nScore: {s}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            return tmp.name

    if st.button("📄 Download PDF Report"):
        pdf_path = generate_pdf(match_score, grouped_keywords, both, missing, top_matches)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="resume_match_report.pdf")

# ========== NEW MULTI-RESUME SECTION ==========

st.subheader("📥 Upload Multiple Resumes (PDF)")
resume_files = st.file_uploader("Upload resumes", type=["pdf"], accept_multiple_files=True)

if resume_files and jd_file:
    jd_text_multi = jd_file.read().decode("utf-8") if jd_file.name.endswith(".txt") else extract_text_from_pdf(jd_file)
    jd_clean_multi = clean_text(jd_text_multi)
    jd_embedding_multi = model.encode([jd_clean_multi])[0]

    results = []

    for file in resume_files:
        resume_text = extract_text_from_pdf(file)
        resume_clean = clean_text(resume_text)
        resume_embedding = model.encode([resume_clean])[0]

        similarity = cosine_similarity([jd_embedding_multi], [resume_embedding])[0][0]
        score = round(similarity * 100, 2)

        jd_keywords = set(extract_keywords(jd_clean_multi))
        resume_words = set(resume_clean.split())
        missing_keywords = jd_keywords - resume_words

        results.append({
            "Resume": file.name,
            "Match Score (%)": score,
            "Missing Keywords": len(missing_keywords)
        })

    results_df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)

    st.subheader("📊 Resume Match Ranking")
    st.dataframe(results_df.reset_index(drop=True))
