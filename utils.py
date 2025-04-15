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

# Define keyword categories (add more as needed)
KEYWORD_CATEGORIES = {
    'technical_skills': {'python', 'sql', 'data structures', 'matlab', 'tensorflow', 'scikit-learn'},
    'tools_platforms': {'power bi', 'tableau', 'github', 'mysql', 'firebase', 'autocad', 'figma'},
    'domain_knowledge': {'machine learning', 'statistical analysis', 'data modeling', 'data science'},
    'soft_skills': {'communication', 'problem-solving', 'collaboration', 'teamwork'},
}


def group_keywords(missing_keywords):
    grouped = {
        'Technical Skills': [],
        'Tools & Platforms': [],
        'Domain Knowledge': [],
        'Soft Skills': [],
        'Uncategorized': []
    }

    for word in missing_keywords:
        word = word.lower()
        if word in KEYWORD_CATEGORIES['technical_skills']:
            grouped['Technical Skills'].append(word)
        elif word in KEYWORD_CATEGORIES['tools_platforms']:
            grouped['Tools & Platforms'].append(word)
        elif word in KEYWORD_CATEGORIES['domain_knowledge']:
            grouped['Domain Knowledge'].append(word)
        elif word in KEYWORD_CATEGORIES['soft_skills']:
            grouped['Soft Skills'].append(word)
        else:
            grouped['Uncategorized'].append(word)

    return grouped


def split_sentences(text):
    return sent_tokenize(text)


def clean_sentence(sentence):
    return sentence.translate(str.maketrans('','',string.punctuation)).strip().lower()


def extract_text_from_pdf(pdf_path_or_file):
    import fitz

    text = ""

    if hasattr(pdf_path_or_file, 'read'):
        doc = fitz.open(stream=pdf_path_or_file.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_path_or_file)

    for page in doc:
        text += page.get_text()
    doc.close()

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


def extract_entities(text):
    doc = nlp(text)
    entities = set()

    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'NORP', 'GPE', 'EVENT', 'WORK_OF_ART']:
            cleaned_ent = ent.text.lower().replace('\n', ' ').replace('-', ' ').strip()
            entities.add(cleaned_ent)
    
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3 and chunk.text.lower() not in stop_words:
            cleaned_chunk = chunk.text.lower().replace('\n', ' ').replace('-', ' ').strip()
            entities.add(cleaned_chunk)

    return entities

def find_similar_sentences(jd_text_or_list, resume_text, model, threshold=0.5, pre_tokenized=False):
    if pre_tokenized:
        jd_sentences = jd_text_or_list
    else:
        jd_sentences = sent_tokenize(jd_text_or_list)

    resume_sentences = sent_tokenize(resume_text)

    jd_sentences_cleaned = [s.strip().lower() for s in jd_sentences if s.strip()]
    resume_sentences_cleaned = [s.strip().lower() for s in resume_sentences if s.strip()]

    jd_embeddings = model.encode(jd_sentences_cleaned)
    resume_embeddings = model.encode(resume_sentences_cleaned)

    similar_pairs = []
    for i, jd_emb in enumerate(jd_embeddings):
        for j, res_emb in enumerate(resume_embeddings):
            sim_score = util.cos_sim(jd_emb, res_emb)
            if sim_score >= threshold:
                similar_pairs.append((jd_sentences[i], resume_sentences[j], round(float(sim_score), 2)))

    return similar_pairs


