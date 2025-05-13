import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Lowercasing and removing punctuation
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenization and removing stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_dataset(df):
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df
