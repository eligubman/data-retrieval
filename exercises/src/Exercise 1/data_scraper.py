from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE_URL = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"
START_FILE = "debates2023-06-28d.xml"
DATA_DIR = Path(__file__).resolve().parent / "data/raw_data"


def fetch_available_files() -> list[str]:
    response = requests.get(BASE_URL, timeout=30)
    response.raise_for_status()
    candidates = re.findall(r'href="(debates[^"]+\.xml)"', response.text)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def download_file(filename: str) -> None:
    target_path = DATA_DIR / filename
    if target_path.exists():
        print(f"Skipping {filename}; already present")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    download_url = urljoin(BASE_URL, filename)
    with requests.get(download_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    output.write(chunk)
    print(f"Downloaded {filename}")


def main() -> None:
    available_files = fetch_available_files()
    if START_FILE not in available_files:
        raise SystemExit(f"Start file {START_FILE} not found in remote index")

    start_index = available_files.index(START_FILE)
    for filename in available_files[start_index:]:
        download_file(filename)


if __name__ == "__main__":
    main()
