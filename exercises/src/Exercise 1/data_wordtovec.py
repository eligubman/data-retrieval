from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from joblib import dump
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIRS = {
    "lemmatized_data": BASE_DIR / "data/lemmatized_data",
    "cleaned_text": BASE_DIR / "data/cleaned_text",
}
OUTPUT_ROOT = BASE_DIR / "metrics/word_to_vec"

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk

    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.lower().split():
        cleaned = re.sub(r"[^a-zA-Z]+", "", raw)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def load_documents(source_dir: Path) -> tuple[list[list[str]], list[str]]:
    documents: list[list[str]] = []
    file_names: list[str] = []

    for file_path in sorted(source_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        tokens = tokenize(text)
        if not tokens:
            continue
        documents.append(tokens)
        file_names.append(file_path.name)

    return documents, file_names


def document_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_corpus(label: str, source_dir: Path) -> bool:
    if not source_dir.exists():
        print(f"Skipping {label}: directory not found ({source_dir})")
        return False

    documents, file_names = load_documents(source_dir)
    if not documents:
        print(f"Skipping {label}: no usable documents found in {source_dir}")
        return False

    print(f"Training Word2Vec for {label} on {len(documents)} documents")

    workers = max(1, (os.cpu_count() or 1) - 1)
    model = Word2Vec(
        sentences=documents,
        vector_size=300,
        window=5,
        min_count=2,
        workers=workers,
        sg=1,
    )

    doc_vectors = np.array([document_vector(doc, model) for doc in documents])

    output_dir = OUTPUT_ROOT / label
    ensure_output_dir(output_dir)

    dump(model, output_dir / "word2vec_model.joblib")
    np.save(output_dir / "document_vectors_word2vec.npy", doc_vectors)
    pd.DataFrame({"file": file_names}).to_csv(output_dir / "file_map_word2vec.csv", index=False)

    print(f"Saved Word2Vec outputs for {label} -> {output_dir}")

    docs_no_stop = [[token for token in doc if token not in STOPWORDS] for doc in documents]
    effective_docs = [doc for doc in docs_no_stop if doc]

    if not effective_docs:
        print(f"Skipping no-stop-word model for {label}: all tokens removed")
        return True

    model_no_stop = Word2Vec(
        sentences=effective_docs,
        vector_size=300,
        window=5,
        min_count=2,
        workers=workers,
        sg=1,
    )

    doc_vectors_no_stop = np.array([document_vector(doc, model_no_stop) for doc in docs_no_stop])

    dump(model_no_stop, output_dir / "word2vec_model_no_stop.joblib")
    np.save(output_dir / "document_vectors_word2vec_no_stop.npy", doc_vectors_no_stop)
    pd.DataFrame({"file": file_names}).to_csv(output_dir / "file_map_word2vec_no_stop.csv", index=False)

    print(f"Saved no-stop-word Word2Vec outputs for {label} -> {output_dir}")
    return True


def main() -> None:
    ensure_output_dir(OUTPUT_ROOT)
    processed = False

    for label, directory in SOURCE_DIRS.items():
        if process_corpus(label, directory):
            processed = True

    if not processed:
        raise SystemExit(
            "No corpora were processed. Ensure cleaned_text/ or lemmatized_data/ contain .txt files."
        )


if __name__ == "__main__":
    main()
