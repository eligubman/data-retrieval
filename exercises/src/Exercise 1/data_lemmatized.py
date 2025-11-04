from pathlib import Path
import xml.etree.ElementTree as ET
import spacy
import re

nlp = spacy.load("en_core_web_sm")

CLEANED_DATA_DIR = Path("cleaned_data")
LEMMATIZED_DIR = Path("lemmatized_data")

def extract_text_from_xml(path: Path) -> str:
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(path, parser=parser)
    root = tree.getroot()

    parts = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            parts.append(elem.tail.strip())
    return " ".join(parts)

def lemmatize_file(source: Path, destination: Path):
    # שלב 1: חילוץ טקסט אמיתי מה־XML
    text = extract_text_from_xml(source)

    # שלב 2: הוצאת מילים בלבד
    words = re.findall(r"[A-Za-z']+", text)

    # שלב 3: Lemmatization
    doc = nlp(" ".join(words))
    lemmas = [token.lemma_ for token in doc]

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(" ".join(lemmas), encoding="utf-8")

def main():
    for source in sorted(CLEANED_DATA_DIR.glob("*.xml")):
        dest = LEMMATIZED_DIR / (source.stem + "_lemma.txt")
        lemmatize_file(source, dest)
        print(f"Lemmatized {source.name}")

if __name__ == "__main__":
    main()
