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
    return df.iloc[:, 0].astype(str).tolist()

# --- helper: reindex sparse matrix columns to a new union vocabulary ---
def reindex_matrix_to_union(mat, old_vocab, union_index, new_vocab_size):
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

    union_vocab = list(uk_vocab)
    union_set = set(union_vocab)
    for w in us_vocab:
        if w not in union_set:
            union_set.add(w)
            union_vocab.append(w)
    union_index = {w: i for i, w in enumerate(union_vocab)}

    uk_reindexed = reindex_matrix_to_union(uk_mat, uk_vocab, union_index, len(union_vocab))
    us_reindexed = reindex_matrix_to_union(us_mat, us_vocab, union_index, len(union_vocab))

    X = vstack([uk_reindexed, us_reindexed])
    docs = pd.concat([uk_docs.assign(source=0), us_docs.assign(source=1)], ignore_index=True)
    y = docs["source"].values

    print(f" Data loaded. Shape: {X.shape}")
    return X, y, union_vocab

# --- extract top 20 features for each model ---
def extract_top_features(model, model_name, vocab):
    feature_importances = None

    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        idx = np.argsort(np.abs(coef))[::-1][:20]
        feature_importances = [(vocab[i], coef[i]) for i in idx]

    elif hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        idx = np.argsort(fi)[::-1][:20]
        feature_importances = [(vocab[i], fi[i]) for i in idx]

    else:
        return None

    return feature_importances


def run_classification():
    X, y, vocab = load_data()

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results_data = []
    feature_results = {}

    print("\n Starting 10-Fold Cross-Validation...")

    for name, clf in classifiers.items():
        print(f"   Running {name}...")
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=True)

        res = {
            "Model": name,
            "Accuracy": np.mean(scores['test_accuracy']),
            "Precision": np.mean(scores['test_precision']),
            "Recall": np.mean(scores['test_recall']),
            "F1 Score": np.mean(scores['test_f1'])
        }
        results_data.append(res)

        last_estimator = scores['estimator'][-1]
        top_features = extract_top_features(last_estimator, name, vocab)
        feature_results[name] = top_features

    results_df = pd.DataFrame(results_data)
    print("\n Classification Results (Average of 10 Folds):")
    print(results_df.to_string(index=False))

    if not os.path.exists("classification"):
        os.makedirs("classification")
    results_df.to_csv("classification/lemmatized_classification_results.csv", index=False)

    feat_path = "classification/top_20_features.txt"
    with open(feat_path, "w", encoding="utf8") as f:
        for model_name, features in feature_results.items():
            f.write(f"\n=== {model_name} ===\n")
            if features is None:
                f.write("(Model does not support feature importance)\n")
            else:
                for token, val in features:
                    f.write(f"{token}: {val}\n")

    print(f"\n Saved feature importances to {feat_path}")


if __name__ == "__main__":
    run_classification()
