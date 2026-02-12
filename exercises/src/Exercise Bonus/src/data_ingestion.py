from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from config import CONFIG


OUTER_TO_CHANNEL = {
    "BBC_News.zip": "uk_media",
    "UK_british_debates_text_files_normalize.zip": "uk_parliament",
    "NBC_News_Complete.zip": "us_media",
    "US_congressional_speeches_Text_Files.zip": "us_congress",
}


@dataclass
class LoadedCorpora:
    docs: pd.DataFrame
    overlap: Dict[str, tuple[datetime, datetime]]


def _parse_media_date(filename: str) -> datetime | None:
    match = re.search(r"([A-Za-z]{3})_(\d{2})_([A-Za-z]{3})_(\d{4})", filename)
    if not match:
        return None
    _, day, month, year = match.groups()
    month_num = datetime.strptime(month, "%b").month
    return datetime(int(year), month_num, int(day))


def _parse_iso_date(value: str) -> datetime | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", value)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d")


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_us_sections(raw_text: str) -> List[str]:
    chunks: List[str] = []
    parts = raw_text.split("=" * 80)
    for part in parts:
        cleaned = _clean_text(part)
        if cleaned:
            chunks.append(cleaned)
    return chunks


def _iter_outer_archives(files_zip: Path) -> Iterable[tuple[str, zipfile.ZipFile]]:
    with zipfile.ZipFile(files_zip) as outer:
        for member in outer.namelist():
            if not member.endswith(".zip") or "__MACOSX" in member:
                continue
            raw = outer.read(member)
            inner_name = Path(member).name
            inner = zipfile.ZipFile(io.BytesIO(raw))
            yield inner_name, inner


def load_documents(files_zip: Path | None = None) -> pd.DataFrame:
    files_zip = files_zip or CONFIG.paths.files_zip
    rows: List[dict] = []

    for inner_name, inner_zip in _iter_outer_archives(files_zip):
        if inner_name not in OUTER_TO_CHANNEL:
            continue
        channel = OUTER_TO_CHANNEL[inner_name]
        spec = CONFIG.corpora[channel]

        for member in inner_zip.namelist():
            if not member.lower().endswith(".txt"):
                continue

            raw_text = inner_zip.read(member).decode("utf-8", errors="ignore")

            if channel in {"uk_media", "us_media"}:
                doc_date = _parse_media_date(Path(member).name)
                if not doc_date:
                    continue
                text = _clean_text(raw_text)
                if not text:
                    continue
                rows.append(
                    {
                        "doc_id": f"{channel}:{Path(member).name}",
                        "channel": channel,
                        "country": spec.country,
                        "arena": spec.arena,
                        "date": doc_date,
                        "filename": Path(member).name,
                        "text": text,
                    }
                )
                continue

            if channel == "uk_parliament":
                doc_date = _parse_iso_date(member)
                if not doc_date:
                    continue
                text = _clean_text(raw_text)
                if not text:
                    continue
                rows.append(
                    {
                        "doc_id": f"{channel}:{Path(member).name}",
                        "channel": channel,
                        "country": spec.country,
                        "arena": spec.arena,
                        "date": doc_date,
                        "filename": Path(member).name,
                        "text": text,
                    }
                )
                continue

            if channel == "us_congress":
                doc_date = _parse_iso_date(member)
                if not doc_date:
                    continue
                sections = _extract_us_sections(raw_text)
                for idx, section in enumerate(sections):
                    rows.append(
                        {
                            "doc_id": f"{channel}:{Path(member).name}:{idx}",
                            "channel": channel,
                            "country": spec.country,
                            "arena": spec.arena,
                            "date": doc_date,
                            "filename": Path(member).name,
                            "text": section,
                        }
                    )

    docs = pd.DataFrame(rows)
    if docs.empty:
        raise RuntimeError(f"No documents loaded from {files_zip}")

    docs = docs.sort_values(["country", "channel", "date", "doc_id"]).reset_index(drop=True)
    return docs


def compute_country_overlap(docs: pd.DataFrame) -> Dict[str, tuple[datetime, datetime]]:
    overlap: Dict[str, tuple[datetime, datetime]] = {}
    for country, channels in CONFIG.country_channels.items():
        subset = docs[docs["country"] == country]
        channel_bounds = []
        for channel in channels:
            channel_docs = subset[subset["channel"] == channel]
            if channel_docs.empty:
                continue
            channel_bounds.append((channel_docs["date"].min(), channel_docs["date"].max()))

        if len(channel_bounds) != 2:
            continue
        start = max(bound[0] for bound in channel_bounds)
        end = min(bound[1] for bound in channel_bounds)
        overlap[country] = (start, end)

    return overlap


def restrict_to_overlap(docs: pd.DataFrame, overlap: Dict[str, tuple[datetime, datetime]]) -> pd.DataFrame:
    kept = []
    for country, (start, end) in overlap.items():
        country_docs = docs[(docs["country"] == country) & (docs["date"] >= start) & (docs["date"] <= end)]
        kept.append(country_docs)
    if not kept:
        return docs.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True).sort_values(["country", "channel", "date"])


def save_clean_corpora(docs: pd.DataFrame) -> None:
    CONFIG.paths.results_tables.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(CONFIG.paths.results_tables / "clean_documents.parquet", index=False)
    docs.to_csv(CONFIG.paths.results_tables / "clean_documents.csv", index=False)


def build_loaded_corpora(files_zip: Path | None = None) -> LoadedCorpora:
    docs = load_documents(files_zip=files_zip)
    overlap = compute_country_overlap(docs)
    aligned_docs = restrict_to_overlap(docs, overlap)
    save_clean_corpora(aligned_docs)
    return LoadedCorpora(docs=aligned_docs, overlap=overlap)
