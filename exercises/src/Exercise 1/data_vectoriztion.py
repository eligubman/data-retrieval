from pathlib import Path

import pandas as pd
from joblib import dump
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIRS = {
    "lemmatized_data": BASE_DIR / "data/lemmatized_data",
    "cleaned_text": BASE_DIR / "data/cleaned_text",
}
OUTPUT_ROOT = BASE_DIR / "metrics/tf_idf_vectors"


def load_documents(source_dir: Path) -> tuple[list[str], list[str]]:
    documents: list[str] = []
    file_names: list[str] = []

    for file_path in sorted(source_dir.glob("*.txt")):
        documents.append(file_path.read_text(encoding="utf-8"))
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

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.9,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:
        print(f"Skipping {label}: {exc}")
        return False

    print(f"TF-IDF for {label} -> shape {tfidf_matrix.shape}")

    output_dir = OUTPUT_ROOT / label
    ensure_output_dir(output_dir)

    save_npz(output_dir / "tfidf_sparse_matrix.npz", tfidf_matrix)
    dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")

    feature_names = vectorizer.get_feature_names_out()
    pd.DataFrame({"term": feature_names}).to_csv(output_dir / "feature_names.csv", index=False)

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
