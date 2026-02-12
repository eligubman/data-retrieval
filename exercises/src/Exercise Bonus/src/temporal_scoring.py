from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from config import CONFIG
from io_utils import normalize_distribution
from time_windows import TimeWindow, filter_window, generate_windows
from topic_discovery import load_topic_model, load_topic_table


@dataclass
class ChannelScores:
    channel: str
    country: str
    windows: List[TimeWindow]
    scores: pd.DataFrame


def _topic_rank_map(country: str) -> Dict[int, int]:
    table = load_topic_table(country)
    return {int(row["topic_id"]): int(row["topic_rank"]) for _, row in table.iterrows()}


def _empty_row(time_label: str) -> dict:
    row: dict[str, float | str] = {"time_label": time_label}
    for i in range(1, CONFIG.stage.topic_count + 1):
        row[f"topic_{i}"] = 0.0
    return row


def _score_docs_with_bertopic(country: str, texts: List[str]) -> List[float]:
    if not texts:
        return normalize_distribution([0.0] * CONFIG.stage.topic_count)

    model = load_topic_model(country)
    rank_map = _topic_rank_map(country)
    topics, _ = model.transform(texts)

    values = [0.0] * CONFIG.stage.topic_count

    for topic_id in topics:
        if topic_id in rank_map:
            rank = rank_map[topic_id]
            values[rank - 1] += 1.0

    return normalize_distribution(values)


def compute_topic_model_scores(docs: pd.DataFrame, overlap: Dict[str, tuple[datetime, datetime]]) -> Dict[str, ChannelScores]:
    outputs: Dict[str, ChannelScores] = {}

    for country, (start, end) in overlap.items():
        windows = generate_windows(start, end)
        country_docs = docs[docs["country"] == country]

        for channel in CONFIG.country_channels[country]:
            channel_docs = country_docs[country_docs["channel"] == channel]
            rows = []
            for window in windows:
                win_docs = filter_window(channel_docs, window)
                values = _score_docs_with_bertopic(country, win_docs["text"].tolist())
                row = _empty_row(window.label)
                for i, value in enumerate(values, start=1):
                    row[f"topic_{i}"] = value
                rows.append(row)

            scores = pd.DataFrame(rows)
            outputs[channel] = ChannelScores(
                channel=channel,
                country=country,
                windows=windows,
                scores=scores,
            )
    return outputs


def top_dominant_topics(scores: pd.DataFrame, top_n: int = 5) -> List[int]:
    means = [scores[f"topic_{i}"].mean() for i in range(1, CONFIG.stage.topic_count + 1)]
    top = np.argsort(means)[::-1][:top_n]
    return [int(i + 1) for i in top]
