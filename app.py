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
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📄 Resume–Job Description Matcher")
st.write("Upload one or more resumes and a job description to compare their alignment.")

resume_files = st.file_uploader("Upload Resume(s) (PDF)", type=["pdf"], accept_multiple_files=True)
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

if resume_files and jd_file:
    jd_text = jd_file.read().decode("utf-8") if jd_file.name.endswith(".txt") else extract_text_from_pdf(jd_file)
    jd_clean = clean_text(jd_text)

    results = []
    for resume_file in resume_files:
        resume_text = extract_text_from_pdf(resume_file)
        resume_clean = clean_text(resume_text)

        match_score = cosine_similarity([model.encode(jd_clean)], [model.encode(resume_clean)])[0][0] * 100

        results.append({
            "filename": resume_file.name,
            "text": resume_text,
            "cleaned": resume_clean,
            "score": round(match_score, 2)
        })

    results.sort(key=lambda x: x['score'], reverse=True)

    st.subheader("🏆 Resume Match Rankings")
    for idx, res in enumerate(results):
        st.markdown(f"**#{idx+1}: {res['filename']} — {res['score']}%**")

    labels = [res['filename'] for res in results]
    scores = [res['score'] for res in results]

    fig, ax = plt.subplots()
    ax.barh(labels, scores, color='lightblue')
    ax.set_xlabel("Match Score (%)")
    ax.set_title("Resume Match Comparison")
    st.pyplot(fig)

    selected_resume = st.selectbox("Select a resume to analyze in detail:", labels)
    chosen = next(res for res in results if res['filename'] == selected_resume)

    resume_text = chosen['text']
    resume_clean = chosen['cleaned']
    match_score = chosen['score']

    jd_keywords = set([kw.lower().strip() for kw in extract_keywords(jd_clean)])
    resume_words = set([word.lower().strip() for word in resume_clean.split()])
    missing_keywords = jd_keywords - resume_words
    matched_keywords = jd_keywords & resume_words
    grouped_keywords = group_keywords(missing_keywords)

    st.metric("🎯 Match Score", f"{match_score:.2f}%")

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
    jd_filtered = re.findall(r"Responsibilities:(.*?)Requirements:", jd_text, re.DOTALL)
    jd_filtered += re.findall(r"Requirements:(.*?)Nice to Have:", jd_text, re.DOTALL)
    jd_lines = "\n".join(jd_filtered).splitlines()
    jd_sentences = [line.strip() for line in jd_lines if line.strip()]

    matches = find_similar_sentences(jd_sentences, resume_text, model, threshold=0.5, pre_tokenized=True)
    matches.sort(key=lambda x: x[2], reverse=True)
    top_matches = matches[:5]

    for jd_sent, res_sent, score in top_matches:
        st.markdown(f"**JD:** {jd_sent.strip()}")
        st.markdown(f"**Resume:** {res_sent.strip()}")
        st.markdown(f"**Score:** {score}")
        st.markdown("---")

    st.subheader("🧠 Explainability Insights")
    action_verbs = ["developed", "led", "implemented", "managed", "created", "built", "designed", "analyzed", "collaborated"]
    resume_action_verbs = [word for word in resume_clean.split() if word in action_verbs]

    st.markdown("**✅ Matched Skills:**")
    st.write(", ".join(sorted(matched_keywords)) if matched_keywords else "None")

    st.markdown("**❌ Missing Skills from JD:**")
    st.write(", ".join(sorted(missing_keywords)) if missing_keywords else "None")

    st.markdown("**🗣️ Action Verbs Used in Resume:**")
    st.write(", ".join(sorted(set(resume_action_verbs))) if resume_action_verbs else "None")

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

        section_header(f"Match Score: {score:.2f}%")
        section_header("Missing Keywords by Category")
        for category, items in grouped.items():
            if items:
                body_text(f"{category}: {', '.join(items)}")

        section_header("Entities Found in Both")
        body_text(", ".join(both) if both else "None")

        section_header("Entities Missing from Resume")
        body_text(", ".join(missing) if missing else "None")

        section_header("Top Sentence Matches")
        for jd_sent, res_sent, s in matches[:5]:
            body_text(f"JD: {jd_sent.strip()}\nResume: {res_sent.strip()}\nScore: {s}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            return tmp.name

    if st.button("📄 Download PDF Report"):
        pdf_path = generate_pdf(match_score, grouped_keywords, both, missing, top_matches)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name="resume_match_report.pdf")

    st.subheader("📊 Visual Dashboard")
    if top_matches:
        labels = [f"Match {i+1}" for i in range(len(top_matches))]
        scores = [m[2] for m in top_matches]

        fig, ax = plt.subplots()
        ax.bar(labels, scores, color='skyblue')
        ax.set_ylim([0, 1.05])
        ax.set_ylabel("Similarity Score")
        ax.set_title("Top 5 JD-Resume Sentence Matches")
        for i, v in enumerate(scores):
            ax.text(i, v + 0.02, str(v), ha='center')
        st.pyplot(fig)

    pie_labels = ['Matched Keywords', 'Missing Keywords']
    pie_values = [len(matched_keywords), len(missing_keywords)]

    if sum(pie_values) > 0:
        fig2, ax2 = plt.subplots()
        ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
        ax2.axis('equal')
        st.pyplot(fig2)
    else:
        st.info("📉 Not enough keyword data to display pie chart.")
