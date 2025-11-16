import os
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from huggingface_hub import login
login()

BASE_DIR = Path(__file__).resolve().parent
PREFERRED_INPUTS = [
    BASE_DIR / "data/cleaned_text",
    BASE_DIR / "data/cleaned_data",
    BASE_DIR / "data/raw_data",
]

input_override = os.environ.get("SIMCSE_INPUT_DIR")
if input_override:
    RAW_DIR = Path(input_override)
else:
    RAW_DIR = next((path for path in PREFERRED_INPUTS if path.exists()), PREFERRED_INPUTS[-1])

if not RAW_DIR.exists():
    raise SystemExit(f"Input directory does not exist: {RAW_DIR}")

OUTPUT_DIR = BASE_DIR / "metrics/simcse_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

documents = []
file_names = []

def extract_text_from_xml(file_path):
    try:
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(file_path, parser=parser)
        root = tree.getroot()
        text_parts = []
        for elem in root.iter():
            if elem.text:
                text = elem.text.strip()
                text = re.sub(r'\s+', ' ', text)
                if text:
                    text_parts.append(text)
        return " ".join(text_parts)
    except ET.ParseError:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

print(f"Loading documents from {RAW_DIR}...")

for file in sorted(RAW_DIR.glob("*")):
    if file.is_file() and file.suffix.lower() in {".xml", ".txt"}:
        text = extract_text_from_xml(file)
        text = text.strip()
        if len(text) < 10:
            continue 
        documents.append(text)
        file_names.append(file.name)

print("Loaded", len(documents), "documents")

if not documents:
    raise SystemExit(f"No documents found in {RAW_DIR}")

model_name = os.environ.get("SIMCSE_MODEL", "princeton-nlp/sup-simcse-bert-base-uncased")
fallback_model = os.environ.get("SIMCSE_FALLBACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

model_name_in_use = model_name

try:
    model = SentenceTransformer(model_name_in_use)
except Exception as exc:
    print(
        f"Failed to load '{model_name_in_use}': {exc}\nFalling back to '{fallback_model}'.",
        file=sys.stderr,
    )
    model_name_in_use = fallback_model
    try:
        model = SentenceTransformer(model_name_in_use)
    except Exception as fallback_exc:
        raise SystemExit(
            f"Failed to load fallback model '{fallback_model}': {fallback_exc}"
        ) from fallback_exc

print(f"Using model '{model_name_in_use}'")

print("Encoding documents...")
document_embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True
)

document_embeddings = np.array(document_embeddings)
print("Embedding shape:", document_embeddings.shape)

np.save(OUTPUT_DIR / "document_embeddings_simcse.npy", document_embeddings)
pd.DataFrame({"file": file_names}).to_csv(OUTPUT_DIR / "file_map_simcse.csv", index=False)
model.save(str(OUTPUT_DIR / "simcse_model"))

print("Saved all outputs:")
print(" - document_embeddings_simcse.npy")
print(" - file_map_simcse.csv")
print(" - simcse_model/")
