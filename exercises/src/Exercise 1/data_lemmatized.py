import os
import platform
import sys
from pathlib import Path
import spacy
import re

IS_APPLE = platform.system() == "Darwin"

if IS_APPLE:
    try:
        import thinc_apple_ops  # type: ignore  # MPS acceleration for spaCy
    except ImportError:
        print(
            "Apple hardware detected. For faster lemmatization install spaCy's Apple extras:"
            " uv add 'spacy[apple]'",
            file=sys.stderr,
        )

GPU_ENABLED = spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

BASE_DIR = Path(__file__).resolve().parent
CLEANED_TEXT_DIR = BASE_DIR / "data/cleaned_text"
LEMMATIZED_DIR = BASE_DIR / "data/lemmatized_data"
BATCH_SIZE = 32

def extract_text_from_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def prepare_text(source: Path) -> str:
    text = extract_text_from_file(source)

    # שלב 2: הוצאת מילים בלבד
    words = re.findall(r"[A-Za-z']+", text)
    return " ".join(words)

def lemmatize_documents(sources: list[Path], batch_size: int, n_process: int) -> None:
    prepared = (prepare_text(source) for source in sources)
    for source, doc in zip(sources, nlp.pipe(prepared, batch_size=batch_size, n_process=n_process)):
        lemmas = " ".join(token.lemma_ for token in doc)
        destination = LEMMATIZED_DIR / (source.stem + "_lemma.txt")
        destination.write_text(lemmas, encoding="utf-8")
        print(f"Lemmatized {source.name}")

def main():
    sources = sorted(CLEANED_TEXT_DIR.glob("*.txt"))
    if not sources:
        print("No text files found.")
        return

    LEMMATIZED_DIR.mkdir(parents=True, exist_ok=True)
    n_process = 1 if GPU_ENABLED else max(1, min(4, (os.cpu_count() or 1) - 1))

    # Batch the documents through spaCy to amortize pipeline overhead.
    lemmatize_documents(sources, batch_size=BATCH_SIZE, n_process=n_process)

if __name__ == "__main__":
    main()
