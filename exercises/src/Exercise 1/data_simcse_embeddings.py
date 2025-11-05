import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("raw_data")  # כאן שמים את קבצי המקור
OUTPUT_DIR = Path("simcse_output")
OUTPUT_DIR.mkdir(exist_ok=True)

documents = []
file_names = []

def extract_text_from_xml(file_path):
    """חילוץ טקסט מכל תגית שיש לה תוכן"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text_parts = []
        for elem in root.iter():
            if elem.text:
                text = elem.text.strip()
                # מסירים יותר מדי רווחים
                text = re.sub(r'\s+', ' ', text)
                if text:
                    text_parts.append(text)
        return " ".join(text_parts)
    except ET.ParseError:
        # אם הקובץ לא XML תקין, נקרא כטקסט רגיל
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

print("Loading raw documents...")

for file in sorted(RAW_DIR.glob("*")):
    if file.is_file() and file.suffix.lower() in {".xml", ".txt"}:
        text = extract_text_from_xml(file)
        text = text.strip()
        if len(text) < 10:
            continue  # מסמכים ריקים
        documents.append(text)
        file_names.append(file.name)

print("Loaded", len(documents), "documents")

# טוענים מודל SimCSE ( ללא פיקוח - Unsupervised )
model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")

print("Encoding documents...")
document_embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True
)

document_embeddings = np.array(document_embeddings)
print("Embedding shape:", document_embeddings.shape)

# שמירת תוצאות
np.save(OUTPUT_DIR / "document_embeddings_simcse.npy", document_embeddings)
pd.DataFrame({"file": file_names}).to_csv(OUTPUT_DIR / "file_map_simcse.csv", index=False)
model.save(str(OUTPUT_DIR / "simcse_model"))

print(" Saved all outputs:")
print(" - document_embeddings_simcse.npy")
print(" - file_map_simcse.csv")
print(" - simcse_model/")
