import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import math
from pathlib import Path

# -------------------------------------------------
#     Settings — choose which corpus to process
# -------------------------------------------------

LABEL = "lemmatized_data"      # or "cleaned_text"

BASE_DIR = Path(__file__).resolve().parent
TFIDF_DIR = BASE_DIR / f"metrics/tf_idf_vectors/{LABEL}"

MATRIX_PATH = TFIDF_DIR / "tfidf_sparse_matrix.npz"
FEATURES_PATH = TFIDF_DIR / "feature_names.csv"

OUTPUT_CSV = TFIDF_DIR / f"information_gain_{LABEL}.csv"


# -------------------------------------------------
#       Load TF-IDF data + feature names
# -------------------------------------------------

print("Loading TF-IDF matrix...")
X = load_npz(MATRIX_PATH)        # Sparse (docs × terms)

print("Loading feature names...")
feature_names = pd.read_csv(FEATURES_PATH)["term"].tolist()

num_docs, num_terms = X.shape
print(f"Loaded: {num_docs} documents × {num_terms} terms")


# -------------------------------------------------
#         Convert TF-IDF to presence/absence
# -------------------------------------------------

print("Converting to binary matrix (presence)...")
binary_matrix = (X > 0).astype(int)

# Number of docs containing each term
presence_count = np.array(binary_matrix.sum(axis=0)).flatten()  
absence_count = num_docs - presence_count 


# -------------------------------------------------
#       Unsupervised Information Gain
# -------------------------------------------------
# Explanation:
# Since we have no labels, we measure the entropy of:
#     H(term_present) + H(term_absent)
# This is equivalent to measuring how “informative” the term is
# based on how unevenly it is distributed across documents.
# -------------------------------------------------

def entropy(p):
    if p <= 0:
        return 0.0
    return -p * math.log2(p)


print("Computing Information Gain for each word...")

IG_values = []

for N1 in presence_count:
    p1 = N1 / num_docs    # probability term appears in a document
    p0 = 1 - p1           # probability term does NOT appear
    
    IG = entropy(p1) + entropy(p0)
    IG_values.append(IG)

IG_values = np.array(IG_values)


# -------------------------------------------------
#         Save results to CSV
# -------------------------------------------------

output_df = pd.DataFrame({
    "term": feature_names,
    "presence": presence_count,
    "absence": absence_count,
    "IG": IG_values
})

output_df = output_df.sort_values("IG", ascending=False)
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved IG results → {OUTPUT_CSV}\n")


# -------------------------------------------------
#        Show sample output (Top 20)
# -------------------------------------------------

print("Top 20 words by Information Gain:\n")
print(output_df.head(20))
