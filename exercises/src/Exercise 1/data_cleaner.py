from __future__ import annotations

import re
from pathlib import Path
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
CLEANED_DATA_DIR = BASE_DIR / "cleaned_data"

_TOKEN_PATTERN = re.compile(
    r"\r\n|\r|\n"
    r"|[\w]+(?:['’ʼ׳״][\w]+)*"
    r"|[^\s]",
    re.UNICODE,
)


def _clean_segment(segment: str) -> str:
    if not segment or segment.isspace():
        return segment

    leading_ws_match = re.match(r"^\s*", segment)
    trailing_ws_match = re.search(r"\s*$", segment)
    leading_ws = leading_ws_match.group(0) if leading_ws_match else ""
    trailing_ws = trailing_ws_match.group(0) if trailing_ws_match else ""
    core_start = len(leading_ws)
    core_end = len(segment) - len(trailing_ws)
    if core_start >= core_end:
        return segment

    core = segment[core_start:core_end]
    tokens = _TOKEN_PATTERN.findall(core)
    if not tokens:
        return segment

    parts: list[str] = []
    for token in tokens:
        if token in {"\r\n", "\r"}:
            token = "\n"
        if token == "\n":
            while parts and parts[-1] == " ":
                parts.pop()
            parts.append("\n")
            continue
        if parts and parts[-1] not in {" ", "\n"}:
            parts.append(" ")
        parts.append(token)

    while parts and parts[-1] == " ":
        parts.pop()

    cleaned_core = "".join(parts)
    return f"{leading_ws}{cleaned_core}{trailing_ws}"


def _process_file(source: Path, destination: Path) -> None:
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(source, parser=parser)
    root = tree.getroot()

    for element in root.iter():
        if element.text and not element.text.isspace():
            element.text = _clean_segment(element.text)
        if element.tail and not element.tail.isspace():
            element.tail = _clean_segment(element.tail)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if not RAW_DATA_DIR.exists():
        raise SystemExit(f"Missing raw data directory: {RAW_DATA_DIR}")

    files = sorted(RAW_DATA_DIR.glob("*.xml"))
    if not files:
        print("No XML files found to clean.")
        return

    for source_path in files:
        target_path = CLEANED_DATA_DIR / source_path.name
        _process_file(source_path, target_path)
        print(f"Cleaned {source_path.name}")


if __name__ == "__main__":
    main()
