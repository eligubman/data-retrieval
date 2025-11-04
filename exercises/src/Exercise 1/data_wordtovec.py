import os
import re
import numpy as np
from gensim.models import Word2Vec
from pathlib import Path
from joblib import dump
import pandas as pd
from nltk.corpus import stopwords

LEMM_DIR = Path("lemmatized_data")

try:
    STOPWORDS = set(stopwords.words("english"))
except:
    import nltk
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

def clean_tokens(tokens):
    return [re.sub(r'[^a-zA-Z]+', '', t) for t in tokens if re.sub(r'[^a-zA-Z]+', '', t)]

documents = []
file_names = []

for file in sorted(LEMM_DIR.glob("*.txt")):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().lower()
        tokens = text.split()
        tokens = clean_tokens(tokens)
        documents.append(tokens)
        file_names.append(file.name)

print("Loaded", len(documents), "documents.")

model = Word2Vec(
    sentences=documents,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    sg=1
)

print("Training done for normal Word2Vec.")

def document_vector(tokens, mdl):
    valid = [mdl.wv[t] for t in tokens if t in mdl.wv]
    if len(valid) == 0:
        return np.zeros(mdl.vector_size)
    return np.mean(valid, axis=0)


document_vectors = np.array([document_vector(doc, model) for doc in documents])
print("Document embedding shape:", document_vectors.shape)

os.makedirs("word_to_vec", exist_ok=True)
dump(model, "word_to_vec/word2vec_model.joblib")
np.save("word_to_vec/document_vectors_word2vec.npy", document_vectors)
pd.DataFrame({"file": file_names}).to_csv("word_to_vec/file_map_word2vec.csv", index=False)

print(" Saved Word2Vec model and vectors.")

print("\nSample document vectors (first 3 rows):")
print(document_vectors[:3])
print("\nFile map sample:")
print(pd.read_csv("word_to_vec/file_map_word2vec.csv").head())

print("\n=== Creating vectors WITHOUT stop words ===")

documents_no_stop = []
for doc in documents:
    tokens = [t for t in doc if t not in STOPWORDS]
    documents_no_stop.append(tokens)

model_no_stop = Word2Vec(
    sentences=documents_no_stop,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    sg=1
)

document_vectors_no_stop = np.array([document_vector(doc, model_no_stop) for doc in documents_no_stop])

print("Document embedding shape (no stop words):", document_vectors_no_stop.shape)

dump(model_no_stop, "word_to_vec/word2vec_model_no_stop.joblib")
np.save("word_to_vec/document_vectors_word2vec_no_stop.npy", document_vectors_no_stop)
pd.DataFrame({"file": file_names}).to_csv("word_to_vec/file_map_word2vec_no_stop.csv", index=False)

print("Saved Word2Vec WITHOUT stop words.")

print("\nSample no-stop vectors (first 3):")
print(document_vectors_no_stop[:3])
