import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack, coo_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    old_cols = coo.col
    tokens = [old_vocab[c] for c in old_cols]
    new_cols = [union_index[t] for t in tokens]
    new_coo = coo_matrix((coo.data, (coo.row, new_cols)), shape=(mat.shape[0], new_vocab_size))
    return new_coo.tocsr()

def load_data():
    print(" Loading data...")
    uk_mat = load_npz("matrices/lemmatized_uk_bm25.npz")
    us_mat = load_npz("matrices/lemmatized_us_bm25.npz")
    uk_docs = pd.read_csv("matrices/lemmatized_uk_documents.csv")
    us_docs = pd.read_csv("matrices/lemmatized_us_documents.csv")
    uk_vocab = load_vocab_csv("matrices/lemmatized_uk_vocab.csv")
    us_vocab = load_vocab_csv("matrices/lemmatized_us_vocab.csv")

    # Build Union Vocab
    union_vocab = list(uk_vocab)
    union_set = set(union_vocab)
    for w in us_vocab:
        if w not in union_set:
            union_set.add(w)
            union_vocab.append(w)
    union_index = {w: i for i, w in enumerate(union_vocab)}

    # Reindex
    uk_reindexed = reindex_matrix_to_union(uk_mat, uk_vocab, union_index, len(union_vocab))
    us_reindexed = reindex_matrix_to_union(us_mat, us_vocab, union_index, len(union_vocab))

    # Stack
    X = vstack([uk_reindexed, us_reindexed])
    docs = pd.concat([uk_docs.assign(source=0), us_docs.assign(source=1)], ignore_index=True)
    y = docs["source"].values
    
    print(f" Data loaded. Shape: {X.shape}")
    return X, y

def run_classification():
    X, y = load_data()

    # Define classifiers
    # Note: For SVM on sparse text data, Linear kernel is standard and faster.
    # For ANN, we use MLPClassifier.
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    # Define metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }

    # 10-Fold Cross Validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results_data = []

    print("\n Starting 10-Fold Cross-Validation...")
    
    for name, clf in classifiers.items():
        print(f"   Running {name}...")
        # n_jobs=-1 uses all processors
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        res = {
            "Model": name,
            "Accuracy": np.mean(scores['test_accuracy']),
            "Precision": np.mean(scores['test_precision']),
            "Recall": np.mean(scores['test_recall']),
            "F1 Score": np.mean(scores['test_f1'])
        }
        results_data.append(res)

    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    print("\n Classification Results (Average of 10 Folds):")
    print(results_df.to_string(index=False))
    
    # Save to CSV
    if not os.path.exists("classification"):
        os.makedirs("classification")
    results_df.to_csv("classification/lemmatized_classification_results.csv", index=False)
    print("\n Saved results to 'classification/lemmatized_classification_results.csv'")

if __name__ == "__main__":
    run_classification()
