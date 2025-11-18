from __future__ import annotations

from collections import Counter
import re
from pathlib import Path

import pandas as pd
from joblib import dump
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix, save_npz

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIRS = {
    "lemmatized_data": BASE_DIR / "data/lemmatized_data",
    "cleaned_text": BASE_DIR / "data/cleaned_text",
}
OUTPUT_ROOT = BASE_DIR / "metrics/tf_idf_vectors"

WORD_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def load_documents(source_dir: Path) -> tuple[list[list[str]], list[str]]:
    documents: list[list[str]] = []
    file_names: list[str] = []

    for file_path in sorted(source_dir.glob("*.txt")):
        tokens = tokenize(file_path.read_text(encoding="utf-8"))
        if not tokens:
            continue
        documents.append(tokens)
        file_names.append(file_path.name)

    return documents, file_names


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def vectorize_corpus(label: str, source_dir: Path) -> bool:
    if not source_dir.exists():
        print(f"Skipping {label}: directory not found ({source_dir})")
        return False

    documents, file_names = load_documents(source_dir)
    if not documents:
        print(f"Skipping {label}: no text files found in {source_dir}")
        return False

    bm25 = BM25Okapi(documents)

    vocabulary = sorted(bm25.idf.keys())
    vocab_index = {term: idx for idx, term in enumerate(vocabulary)}

    if not vocabulary:
        print(f"Skipping {label}: empty vocabulary")
        return False

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []

    for doc_idx, tokens in enumerate(documents):
        freq_counter = Counter(tokens)
        doc_len = len(tokens)
        if doc_len == 0:
            continue

        length_norm = bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)

        for term, freq in freq_counter.items():
            idf = bm25.idf.get(term)
            if idf is None:
                continue

            denominator = freq + length_norm
            score = idf * freq * (bm25.k1 + 1) / denominator

            row_idx.append(doc_idx)
            col_idx.append(vocab_index[term])
            data.append(score)

    bm25_matrix = csr_matrix((data, (row_idx, col_idx)), shape=(len(documents), len(vocabulary)))

    print(f"BM25 for {label} -> shape {bm25_matrix.shape}")

    output_dir = OUTPUT_ROOT / label
    ensure_output_dir(output_dir)

    save_npz(output_dir / "tfidf_sparse_matrix.npz", bm25_matrix)
    dump(bm25, output_dir / "bm25_model.joblib")

    pd.DataFrame({"term": vocabulary}).to_csv(output_dir / "feature_names.csv", index=False)

    df = pd.DataFrame({
        "file": file_names,
        "index": range(len(file_names)),
    })
    df.to_csv(output_dir / "file_map.csv", index=False)

    return True


def main() -> None:
    ensure_output_dir(OUTPUT_ROOT)
    processed = False

    for label, directory in SOURCE_DIRS.items():
        if vectorize_corpus(label, directory):
            processed = True

    if not processed:
        raise SystemExit(
            "No corpora were processed. Ensure cleaned_text/ or lemmatized_data/ contain .txt files."
        )


if __name__ == "__main__":
    main()
