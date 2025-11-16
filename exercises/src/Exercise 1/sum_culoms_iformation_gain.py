import pandas as pd
from scipy.sparse import load_npz
from pathlib import Path

# === קלט ===
BASE_DIR = Path(__file__).resolve().parent
LABEL = "lemmatized_data"   # או cleaned_text
TFIDF_DIR = BASE_DIR / "metrics/tf_idf_vectors" / LABEL

matrix_path = TFIDF_DIR / "tfidf_sparse_matrix.npz"
feature_path = TFIDF_DIR / "feature_names.csv"
output_csv = TFIDF_DIR / "tfidf_column_sums.csv"

# === טעינת matrix ו-feature names ===
print("Loading TF-IDF matrix...")
tfidf_matrix = load_npz(matrix_path)

print("Loading vocabulary (feature names)...")
feature_df = pd.read_csv(feature_path)
terms = feature_df["term"].tolist()

# === סכום כל עמודה (כל מילה) ===
# sparse.sum(axis=0) → מחזיר מטריצת 1×N
column_sums = tfidf_matrix.sum(axis=0).A1  # A1 הופך ל-numpy vector

# === סכום כולל של כל המטריצה ===
total_sum = column_sums.sum()

# === טבלה מסודרת ===
result = pd.DataFrame({
    "word": terms,
    "tfidf_sum": column_sums,
    "relative_importance": column_sums / total_sum
})

# === שמירה ===
result.to_csv(output_csv, index=False)

print("✔ Done!")
print(f"Saved → {output_csv}")
print("Total TF-IDF sum:", total_sum)
