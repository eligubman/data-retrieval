from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from config import CONFIG


@dataclass
class TopicInfluenceRow:
    country: str
    topic: int
    best_lag: int
    best_correlation: float
    dtw_distance: float
    edit_similarity: float
    direction: str
    politics_amplitude: float
    media_amplitude: float
    amplitude_gap: float


def _lag_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    if lag > 0:
        a_slice, b_slice = a[:-lag], b[lag:]
    elif lag < 0:
        a_slice, b_slice = a[-lag:], b[:lag]
    else:
        a_slice, b_slice = a, b
    if len(a_slice) < 3 or len(b_slice) < 3:
        return 0.0
    if np.std(a_slice) == 0 or np.std(b_slice) == 0:
        return 0.0
    return float(np.corrcoef(a_slice, b_slice)[0, 1])


def _best_lag_corr(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    best = (0, -2.0)
    for lag in range(-CONFIG.stage.max_lag, CONFIG.stage.max_lag + 1):
        corr = _lag_corr(a, b, lag)
        if corr > best[1]:
            best = (lag, corr)
    return best


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _real_edit_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_str = " ".join(f"{int(round(v * 1000)):04d}" for v in a)
    b_str = " ".join(f"{int(round(v * 1000)):04d}" for v in b)
    dist = Levenshtein.distance(a_str, b_str)
    max_len = max(len(a_str), len(b_str), 1)
    return 1.0 - (dist / max_len)


def _direction(lag: int, corr: float) -> str:
    if corr < 0.15:
        return "weak/no clear influence"
    if lag > 0:
        return "politics leads media"
    if lag < 0:
        return "media leads politics"
    return "synchronous / bidirectional"


def analyze_country(
    country: str,
    politics_scores: pd.DataFrame,
    media_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[TopicInfluenceRow] = []
    for topic in range(1, CONFIG.stage.topic_count + 1):
        p = politics_scores[f"topic_{topic}"].to_numpy(dtype=float)
        m = media_scores[f"topic_{topic}"].to_numpy(dtype=float)

        lag, corr = _best_lag_corr(p, m)
        dtw = _dtw_distance(p, m)
        edit = _real_edit_similarity(p, m)
        p_amp = float(np.max(p) - np.min(p))
        m_amp = float(np.max(m) - np.min(m))
        row = TopicInfluenceRow(
            country=country,
            topic=topic,
            best_lag=lag,
            best_correlation=corr,
            dtw_distance=dtw,
            edit_similarity=edit,
            direction=_direction(lag, corr),
            politics_amplitude=p_amp,
            media_amplitude=m_amp,
            amplitude_gap=abs(p_amp - m_amp),
        )
        rows.append(row)
    return pd.DataFrame([vars(r) for r in rows])


def plot_country_topics(
    country: str,
    politics_scores: pd.DataFrame,
    media_scores: pd.DataFrame,
    selected_topics: List[int],
    prefix: str,
) -> List[Path]:
    paths: List[Path] = []
    x = range(1, len(politics_scores) + 1)
    for topic in selected_topics:
        plt.figure(figsize=(12, 4))
        plt.plot(x, politics_scores[f"topic_{topic}"], label="politics", linewidth=2)
        plt.plot(x, media_scores[f"topic_{topic}"], label="media", linewidth=2)
        plt.title(f"{country} topic {topic}: politics vs media")
        plt.xlabel("Time point")
        plt.ylabel("Dominance")
        plt.legend()
        plt.tight_layout()
        out = CONFIG.paths.results_graphs / f"{prefix}_{country.lower()}_topic_{topic}.png"
        plt.savefig(out)
        plt.close()
        paths.append(out)
    return paths
