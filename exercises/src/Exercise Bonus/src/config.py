from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class StageConfig:
    topic_count: int = 20
    window_days: int = 14
    step_days: int = 7
    max_lag: int = 3
    decimal_places: int = 3
    random_state: int = 42


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data: Path = root / "data"
    src: Path = root / "src"
    results: Path = root / "results"
    results_excel: Path = results / "excel"
    results_graphs: Path = results / "graphs"
    results_tables: Path = results / "tables"
    cache: Path = results / "cache"
    files_zip: Path = root / "files.zip"


@dataclass(frozen=True)
class CorpusSpec:
    channel: str
    country: str
    arena: str


@dataclass(frozen=True)
class BonusConfig:
    stage: StageConfig = StageConfig()
    paths: Paths = Paths()
    corpora: Dict[str, CorpusSpec] = field(
        default_factory=lambda: {
            "uk_parliament": CorpusSpec(
                channel="uk_parliament", country="UK", arena="politics"
            ),
            "uk_media": CorpusSpec(channel="uk_media", country="UK", arena="media"),
            "us_congress": CorpusSpec(
                channel="us_congress", country="US", arena="politics"
            ),
            "us_media": CorpusSpec(channel="us_media", country="US", arena="media"),
        }
    )

    @property
    def country_channels(self) -> Dict[str, List[str]]:
        return {
            "UK": ["uk_parliament", "uk_media"],
            "US": ["us_congress", "us_media"],
        }


CONFIG = BonusConfig()
