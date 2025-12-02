import os
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from collections import Counter


def load_clean_texts(folder):
    texts = []
    filenames = []

    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read().split())   # list of tokens
        filenames.append(filename)

    return texts, filenames


def build_bm25_matrix(input_folder, output_prefix, k1=1.5, b=0.75):
    print(f"📂 Loading cleaned files from: {input_folder}")
    docs, filenames = load_clean_texts(input_folder)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    N = len(docs)
    print(f"📄 Loaded {N} documents.")

    # build vocabulary
    vocab = {}
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    V = len(vocab)
    print(f"🔡 Vocabulary size: {V}")

    # document lengths
    doc_lens = np.array([len(doc) for doc in docs])
    avgdl = doc_lens.mean()

    # document frequency for each term
    df = np.zeros(V, dtype=int)
    for doc in docs:
        for w in set(doc):
            df[vocab[w]] += 1

    # idf according to BM25 formula
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1)

    # build sparse matrix data
    data = []
    rows = []
    cols = []

    print("⚙️ Building BM25 matrix...")

    for i, doc in enumerate(docs):
        counts = Counter(doc)
        dl = doc_lens[i]

        for word, tf in counts.items():
            j = vocab[word]

            # BM25 weight
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score = idf[j] * (tf * (k1 + 1)) / denom

            rows.append(i)
            cols.append(j)
            data.append(score)

    X = csr_matrix((data, (rows, cols)), shape=(N, V))

    print("📏 Matrix shape:", X.shape)

    # save outputs
    save_npz(f"{output_prefix}_bm25.npz", X)
    pd.Series(list(vocab.keys())).to_csv(f"{output_prefix}_vocab.csv", index=False)
    pd.Series(filenames).to_csv(f"{output_prefix}_documents.csv", index=False)

    print(f"✔ Saved matrix to: {output_prefix}_bm25.npz")
    print(f"✔ Saved vocab to:   {output_prefix}_vocab.csv")
    print(f"✔ Saved docs to:    {output_prefix}_documents.csv")


# ---- MAIN ----
if __name__ == "__main__":
    build_bm25_matrix("data/cleaned_us", "matrices/us")
    build_bm25_matrix("data/cleaned_uk", "matrices/uk")

    print("\n🎉 Done! BM25 matrices created.")
