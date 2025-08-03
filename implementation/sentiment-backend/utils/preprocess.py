import re
import emoji
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def clean_text(text):
    text = emoji.replace_emoji(text, replace='')  # remove emojis
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # remove non-alphabetic characters
    text = text.lower()                           # convert to lowercase
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text

def split_into_sentences(text):
    # Split first while punctuation is intact
    sentences = sent_tokenize(text)
    # Then clean each sentence
    return [clean_text(sentence) for sentence in sentences if sentence.strip()]
