from utils import extract_text_from_pdf, clean_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_job_description(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        jd_text = file.read()
    return jd_text

if __name__ == "__main__":
    jd_path = "job_description.txt"
    job_description = load_job_description(jd_path)
    print("Job Description Loaded:\n")
    print(job_description)

    resume_path = "resume.pdf"
    resume_text = extract_text_from_pdf(resume_path)
    print("\n Resume Text Extracted:\n")
    print(resume_text)

if __name__ == "__main__":
    jd_path = "job_description.txt"
    job_description = clean_text(load_job_description(jd_path))

    resume_path = "resume.pdf"
    resume_text = clean_text(extract_text_from_pdf(resume_path))

    jd_embedding = model.encode([job_description])[0]
    resume_embedding = model.encode([resume_text])[0]

    similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
    match_percentage = round(similarity * 100, 2)

    print(f"\n Match Score: {match_percentage}%")
