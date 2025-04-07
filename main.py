from utils import extract_text_from_pdf, clean_text, extract_keywords, find_similar_sentences
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_job_description(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == "__main__":
    # --- Load job description and resume ---
    jd_path = "job_description.txt"
    job_description_raw = load_job_description(jd_path)

    resume_path = "resume.pdf"
    resume_text_raw = extract_text_from_pdf(resume_path)

    # --- Clean for cosine match & keyword check ---
    job_description_clean = clean_text(job_description_raw)
    resume_text_clean = clean_text(resume_text_raw)

    # --- Match Score ---
    jd_embedding = model.encode([job_description_clean])[0]
    resume_embedding = model.encode([resume_text_clean])[0]
    similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
    match_percentage = round(similarity * 100, 2)
    print(f"\nMatch Score: {match_percentage}%")

    # --- Skill Gap Analysis ---
    jd_keywords = set(extract_keywords(job_description_clean))
    resume_words = set(resume_text_clean.split())
    missing_keywords = jd_keywords - resume_words
    print("\nMissing Keywords in Resume:")
    print(missing_keywords)

    # --- Extract only JD responsibilities & requirements (Fix 1) ---
    jd_filtered = re.findall(r"Responsibilities:(.*?)Requirements:", job_description_raw, re.DOTALL)
    jd_filtered += re.findall(r"Requirements:(.*)", job_description_raw, re.DOTALL)
    jd_filtered_text = "\n".join(jd_filtered).strip()

    # --- Sentence-level matching (Fix 2, 3, 4, 5) ---
    similar_sentences = find_similar_sentences(jd_filtered_text, resume_text_raw, model, threshold=0.5)

    # Sort and get top 5
    similar_sentences.sort(key=lambda x: x[2], reverse=True)
    top_matches = similar_sentences[:5]

    # Pretty-print top matches
    print("\n🔍 Top 5 Similar Sentences Between JD and Resume:\n")
    for jd_sent, res_sent, score in top_matches:
        print(f"🟢 JD ➤ {jd_sent.strip()}\n🟣 Resume ➤ {res_sent.strip()}\n🔢 Score: {score}")
        print("-" * 50)

    # Summary
    total_matches = len(similar_sentences)
    highest_score = max([score for _, _, score in similar_sentences]) if total_matches else 0
    avg_score = round(sum([score for _, _, score in similar_sentences]) / total_matches, 2) if total_matches else 0

    print(f"\n📊 Summary:")
    print(f"🧠 Total JD Sentences Matched: {total_matches}")
    print(f"📈 Highest Similarity: {highest_score}")
    print(f"📉 Average Match Score: {avg_score}")
