import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack, coo_matrix
from sklearn.cluster import KMeans, DBSCAN ,HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import hdbscan
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import umap
import matplotlib.patches as mpatches
import seaborn as sns

# --- helper: load vocab (robust to header/no-header) ---
def load_vocab_csv(path):
    df = pd.read_csv(path, header=None)
    # take first column as list of tokens
    return df.iloc[:, 0].astype(str).tolist()

# --- helper: reindex sparse matrix columns to a new union vocabulary ---
def reindex_matrix_to_union(mat, old_vocab, union_index, new_vocab_size):
    """
    mat: scipy sparse (CSR/CSC) with shape (n_docs, len(old_vocab))
    old_vocab: list where index -> token (length == mat.shape[1])
    union_index: dict token -> new column index in union vocab
    new_vocab_size: int size of the union vocab
    returns: csr_matrix with shape (n_docs, new_vocab_size)
    """
    coo = mat.tocoo()
    # map old column indices -> tokens -> new column indices
    # faster to use list lookup
    old_cols = coo.col
    # get token for each old column
    tokens = [old_vocab[c] for c in old_cols]
    new_cols = [union_index[t] for t in tokens]
    new_coo = coo_matrix((coo.data, (coo.row, new_cols)), shape=(mat.shape[0], new_vocab_size))
    return new_coo.tocsr()

# ============================
#    LOAD MATRICES + VOCABS
# ============================
uk_mat = load_npz("matrices/lemmatized_uk_bm25.npz")
us_mat = load_npz("matrices/lemmatized_us_bm25.npz")

uk_docs = pd.read_csv("matrices/lemmatized_uk_documents.csv")
us_docs = pd.read_csv("matrices/lemmatized_us_documents.csv")

uk_vocab = load_vocab_csv("matrices/lemmatized_uk_vocab.csv")
us_vocab = load_vocab_csv("matrices/lemmatized_us_vocab.csv")

print("uk_mat.shape:", uk_mat.shape, "us_mat.shape:", us_mat.shape)
print("uk_vocab:", len(uk_vocab), "us_vocab:", len(us_vocab))

# ============================
#  BUILD UNION VOCAB (uk-first)
# ============================
union_vocab = list(uk_vocab)  # keep UK ordering first
union_set = set(union_vocab)
for w in us_vocab:
    if w not in union_set:
        union_set.add(w)
        union_vocab.append(w)

union_index = {w: i for i, w in enumerate(union_vocab)}
print("Union vocab size:", len(union_vocab))

# ============================
#  REINDEX BOTH MATS TO UNION
# ============================
uk_reindexed = reindex_matrix_to_union(uk_mat, uk_vocab, union_index, len(union_vocab))
us_reindexed = reindex_matrix_to_union(us_mat, us_vocab, union_index, len(union_vocab))

print("uk_reindexed.shape:", uk_reindexed.shape, "us_reindexed.shape:", us_reindexed.shape)

# now safe to vstack
X = vstack([uk_reindexed, us_reindexed])
docs = pd.concat([uk_docs.assign(source=0), us_docs.assign(source=1)], ignore_index=True)
labels = docs["source"].values

print("Loaded combined matrix:", X.shape)

# ============================
#  (the rest of your original script)
#  -- keep your evaluate_clustering and clustering code
# ============================

def evaluate_clustering(true_labels, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    mapping = {}

    for c in unique_clusters:
        if c == -1:
            mapping[c] = -1
            continue

        mask = cluster_labels == c
        if mask.sum() == 0:
            mapping[c] = 0
        else:
            mapping[c] = np.bincount(true_labels[mask]).argmax()

    predicted = np.array([mapping[c] for c in cluster_labels])

    return {
        "precision": precision_score(true_labels, predicted, zero_division=0),
        "recall": recall_score(true_labels, predicted, zero_division=0),
        "f1": f1_score(true_labels, predicted, zero_division=0),
        "accuracy": accuracy_score(true_labels, predicted)
    }

results = {}
cluster_outputs = {}

# KMeans
print("Running KMeans...")
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
results["KMeans"] = evaluate_clustering(labels, kmeans_labels)
cluster_outputs["KMeans"] = kmeans_labels

# DBSCAN (same as you had)
print("Estimating eps for DBSCAN...")
dist = cosine_distances(X[:500])
eps_guess = np.percentile(dist, 2)
print("DBSCAN eps guess:", eps_guess)

db = DBSCAN(eps=eps_guess, min_samples=5, metric="cosine")
db_labels = db.fit_predict(X)
results["DBSCAN"] = evaluate_clustering(labels, db_labels)
cluster_outputs["DBSCAN"] = db_labels

# HDBSCAN
print("Running HDBSCAN...")
hdb = hdbscan.HDBSCAN(metric="cosine", min_cluster_size=10)
hdb_labels = hdb.fit_predict(X)
results["HDBSCAN"] = evaluate_clustering(labels, hdb_labels)
cluster_outputs["HDBSCAN"] = hdb_labels

# GMM via PCA (CAUTION: dense!)
print("Running GMM via PCA reduction...")
X_dense = X.toarray()   # beware memory if union vocab is very large
pca = PCA(n_components=200)
X_pca = pca.fit_transform(X_dense)

gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
gmm_labels = gmm.fit_predict(X_pca)
results["GMM"] = evaluate_clustering(labels, gmm_labels)
cluster_outputs["GMM"] = gmm_labels

# Results & UMAP (unchanged)
print("\n========== RESULTS ==========\n")
for k, v in results.items():
    print(k, v)

print("Running UMAP...")
um = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
embedding = um.fit_transform(X)

# ------------------------------
# PLOT WITH SEABORN & SAVE
# ------------------------------
import os
os.makedirs("images", exist_ok=True)

# Create a DataFrame for easier plotting with Seaborn
plot_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
plot_df["True Label"] = ["USA" if l == 1 else "UK" for l in labels]

# 1. Plot True Labels
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=plot_df, x="UMAP1", y="UMAP2", hue="True Label", palette="coolwarm", s=15, alpha=0.7
)
plt.title("UMAP – True Labels", fontsize=14)
plt.savefig("images/lemmatized_umap_true_labels.png", dpi=300, bbox_inches="tight")
print("Saved images/lemmatized_umap_true_labels.png")
plt.close()

# 2. Plot Cluster Assignments
for method, cl in cluster_outputs.items():
    plot_df["Cluster"] = cl
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df, x="UMAP1", y="UMAP2", hue="Cluster", palette="tab20", s=15, alpha=0.7, legend="full"
    )
    plt.title(f"UMAP – Clusters from {method}", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"images/lemmatized_umap_{method}.png", dpi=300, bbox_inches="tight")
    print(f"Saved images/lemmatized_umap_{method}.png")
    plt.close()