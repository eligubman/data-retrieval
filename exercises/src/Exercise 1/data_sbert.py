from __future__ import annotations

import json
import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent

_input_override = os.environ.get("SBERT_INPUT_DIR")
if _input_override:
    raw_candidate = Path(_input_override)
    RAW_DIR = raw_candidate if raw_candidate.is_absolute() else (BASE_DIR / raw_candidate)
else:
    RAW_DIR = BASE_DIR / "data/raw_data"

if not RAW_DIR.exists():
    raise SystemExit(f"Input directory does not exist: {RAW_DIR}")

OUTPUT_DIR = BASE_DIR / "metrics/sbert_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _extract_text(file_path: Path) -> str:
    try:
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(file_path, parser=parser)
        text_segments: list[str] = []
        for segment in tree.getroot().itertext():
            cleaned = re.sub(r"\s+", " ", segment.strip())
            if cleaned:
                text_segments.append(cleaned)
        return " ".join(text_segments)
    except ET.ParseError:
        with file_path.open("r", encoding="utf-8", errors="ignore") as stream:
            return stream.read()


def _load_documents(raw_dir: Path) -> tuple[list[str], list[str]]:
    documents: list[str] = []
    file_names: list[str] = []

    for source in sorted(raw_dir.glob("*.xml")):
        text = _extract_text(source).strip()
        if len(text) < 10:
            continue
        documents.append(text)
        file_names.append(source.name)

    return documents, file_names


def main() -> None:
    documents, file_names = _load_documents(RAW_DIR)
    if not documents:
        raise SystemExit(f"No XML documents with content found in {RAW_DIR}")

    model_name = os.environ.get("SBERT_MODEL", "sentence-transformers/all-mpnet-base-v2")

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - surface clear failure to caller
        raise SystemExit(f"Failed to load SBERT model '{model_name}': {exc}") from exc

    print(f"Using model '{model_name}'")
    print(f"Encoding {len(documents)} documents from {RAW_DIR}...")

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
    )
    embeddings_array = np.asarray(embeddings, dtype=np.float32)

    np.save(OUTPUT_DIR / "document_embeddings_sbert.npy", embeddings_array)
    pd.DataFrame({"file": file_names}).to_csv(OUTPUT_DIR / "file_map_sbert.csv", index=False)
    model.save(str(OUTPUT_DIR / "sbert_model"))

    metadata = {
        "model": model_name,
        "documents": len(documents),
        "source_dir": str(RAW_DIR),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved SBERT outputs:")
    print(" - document_embeddings_sbert.npy")
    print(" - file_map_sbert.csv")
    print(" - sbert_model/")
    print(" - metadata.json")


if __name__ == "__main__":
    main()
