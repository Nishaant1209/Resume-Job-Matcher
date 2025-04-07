import fitz
import spacy
import string
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer, util 
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def split_sentences(text):
    return sent_tokenize(text)


def clean_sentence(sentence):
    return sentence.translate(str.maketrans('','',string.punctuation)).strip().lower()


def extract_text_from_pdf(pdf_path):
    text = " "
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def clean_text(text):
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.text not in stop_words and token.text not in string.punctuation and token.is_alpha:
            tokens.append(token.lemma_)
    
    return " ".join(tokens)


def extract_keywords(text, max_words=2):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()
    filtered_keywords = [kw.lower() for kw in keywords if len(kw.split()) <= max_words]
    return filtered_keywords


def find_similar_sentences(jd_text, resume_text, model, threshold=0.5):
    jd_sentences = sent_tokenize(jd_text)
    resume_sentences = sent_tokenize(resume_text)

    jd_sentences_cleaned = [clean_sentence(s) for s in jd_sentences]
    resume_sentences_cleaned = [clean_sentence(s) for s in resume_sentences]

    jd_embeddings = model.encode(jd_sentences_cleaned)
    resume_embeddings = model.encode(resume_sentences_cleaned)

    similar_pairs = []

    for i, jd_emb in enumerate(jd_embeddings):
        for j, res_emb in enumerate(resume_embeddings):
            sim_score = util.cos_sim(jd_emb, res_emb)
            if sim_score >= threshold:
                similar_pairs.append((jd_sentences[i], resume_sentences[j], round(float(sim_score), 2)))

    # Remove duplicates
    unique_pairs = []
    seen = set()
    for jd_sent, res_sent, score in similar_pairs:
        key = (jd_sent[:40], res_sent[:40])
        if key not in seen:
            seen.add(key)
            unique_pairs.append((jd_sent, res_sent, score))

    return unique_pairs
