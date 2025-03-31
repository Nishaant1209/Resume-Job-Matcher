import fitz
import spacy
import string
import nltk
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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