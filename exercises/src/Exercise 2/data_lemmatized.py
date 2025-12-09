import os
import platform
import sys
from pathlib import Path
import spacy
import re

IS_APPLE = platform.system() == "Darwin"

if IS_APPLE:
    import importlib.util
    if importlib.util.find_spec("thinc_apple_ops") is None:
        print(
            "Apple hardware detected. For faster lemmatization install spaCy's Apple extras:"
            " uv add 'spacy[apple]'",
            file=sys.stderr,
        )

GPU_ENABLED = spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

BASE_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 32

def extract_text_from_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def prepare_text(source: Path) -> str:
    text = extract_text_from_file(source)

    # שלב 2: הוצאת מילים בלבד
    words = re.findall(r"[A-Za-z']+", text)
    return " ".join(words)

def lemmatize_documents(sources: list[Path], output_dir: Path, batch_size: int, n_process: int) -> None:
    # Filter out already processed files
    to_process = []
    for source in sources:
        destination = output_dir / (source.stem + "_lemma.txt")
        if not destination.exists():
            to_process.append(source)
    
    if not to_process:
        print(f"All files in {output_dir.name} are already lemmatized.")
        return

    print(f"Processing {len(to_process)} files in {output_dir.name}...")

    prepared = (prepare_text(source) for source in to_process)
    for source, doc in zip(to_process, nlp.pipe(prepared, batch_size=batch_size, n_process=n_process)):
        lemmas = " ".join(token.lemma_ for token in doc)
        destination = output_dir / (source.stem + "_lemma.txt")
        destination.write_text(lemmas, encoding="utf-8")
        print(f"Lemmatized {source.name}")

def main():
    # Define input/output pairs
    tasks = [
        ("data/cleaned_uk", "data/lemmatized_uk"),
        ("data/cleaned_us", "data/lemmatized_us")
    ]

    n_process = 1 if GPU_ENABLED else max(1, min(4, (os.cpu_count() or 1) - 1))

    for input_path_str, output_path_str in tasks:
        input_dir = BASE_DIR / input_path_str
        output_dir = BASE_DIR / output_path_str

        if not input_dir.exists():
            print(f"Directory not found: {input_dir}")
            continue

        sources = sorted(input_dir.glob("*.txt"))
        if not sources:
            print(f"No text files found in {input_dir}.")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        lemmatize_documents(sources, output_dir, batch_size=BATCH_SIZE, n_process=n_process)

if __name__ == "__main__":
    main()
