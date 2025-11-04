import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

DATA_FOLDER = "lemmatized_data/"

files = glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
documents = []
file_names = []

for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        documents.append(fp.read())
        file_names.append(os.path.basename(f))

vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=5,      
    max_df=0.9, 
)

tfidf_matrix = vectorizer.fit_transform(documents)

print("Shape:", tfidf_matrix.shape)

save_npz("tf_idf_vectors/tfidf_sparse_matrix.npz", tfidf_matrix)

df = pd.DataFrame({
    "file": file_names,
    "index": range(len(file_names))
})
df.to_csv("tf_idf_vectors/file_map.csv", index=False)
