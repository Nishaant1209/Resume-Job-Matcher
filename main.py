from utils import extract_text_from_pdf, clean_text, extract_keywords, find_similar_sentences
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_entities
from utils import group_keywords
import re
import matplotlib.pyplot as plt

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

    # --- Grouped Skill Gap Analysis ---
    jd_keywords = set(extract_keywords(job_description_clean))
    resume_words = set(resume_text_clean.split())
    missing_keywords = jd_keywords - resume_words

    # Group and display missing keywords by category
    grouped_keywords = group_keywords(missing_keywords)

    print("\n🧠 Grouped Missing Keywords in Resume:")
    for category, items in grouped_keywords.items():
        if items:
            print(f"\n🔹 {category}:")
            for kw in sorted(set(items)):
                print(f"  - {kw}")


    #Entity extraction and comparison 
    jd_entites = extract_entities(job_description_raw)
    resume_entities = extract_entities(resume_text_raw)

    common_entites = jd_entites & resume_entities
    missing_entities = jd_entites - resume_entities

    print("\n Entities Found in Both JD and Resume:")
    for ent in sorted(common_entites):
        print(f" - {ent}")

    print("\n Entities Present in JD but Missing in Resume:")
    for ent in sorted(missing_entities):
        print(f" - {ent}")

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


    def plot_top_matches(matches):
        labels = [f"Match{i+1}" for i in range(len(matches))]
        scores = [score for _, _, score in matches]

        plt.figure(figsize=(8,5))
        bars = plt.bar(labels, scores)
        plt.ylim(0,1.05)
        plt.xlabel("Top JD-Resume Sentence Matches")
        plt.ylabel("Similarity Score")
        plt.title("Top 5 Sentence Similarity Scores")

        for bar, score in zip(bars,scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 , str(score), ha='center', fontsize =10)

        plt.tight_layout()
        plt.show()
    
    plot_top_matches(top_matches)

    
    # Summary
    total_matches = len(similar_sentences)
    highest_score = max([score for _, _, score in similar_sentences]) if total_matches else 0
    avg_score = round(sum([score for _, _, score in similar_sentences]) / total_matches, 2) if total_matches else 0

    print(f"\n📊 Summary:")
    print(f"🧠 Total JD Sentences Matched: {total_matches}")
    print(f"📈 Highest Similarity: {highest_score}")
    print(f"📉 Average Match Score: {avg_score}")
