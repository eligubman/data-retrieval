from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CONFIG
from io_utils import normalize_distribution
from time_windows import filter_window
from topic_discovery import load_topic_table


def _cache_path(channel: str) -> Path:
    return CONFIG.paths.cache / f"rag_scores_{channel}.json"


def _load_cache(channel: str) -> Dict[str, List[float]]:
    path = _cache_path(channel)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_cache(channel: str, cache: Dict[str, List[float]]) -> None:
    with _cache_path(channel).open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)


def _build_topics_block(country: str) -> str:
    table = load_topic_table(country).sort_values("topic_rank")
    lines = []
    for _, row in table.iterrows():
        lines.append(
            f"topic {int(row['topic_rank'])}: {row['label']} | keywords: {row['keywords']}"
        )
    return "\n".join(lines)


def _load_topic_texts(country: str) -> Dict[int, str]:
    table = load_topic_table(country).sort_values("topic_rank")
    topic_texts: Dict[int, str] = {}
    for _, row in table.iterrows():
        rank = int(row["topic_rank"])
        label = str(row["label"])
        keywords = str(row["keywords"])
        topic_texts[rank] = f"{label}. {keywords}"
    return topic_texts


def _build_context(window_docs: pd.DataFrame, top_k: int = 20) -> str:
    if window_docs.empty:
        return "No documents in this time window."
    docs = window_docs.copy()
    docs["length"] = docs["text"].str.len()
    docs = docs.sort_values(["date", "length"], ascending=[False, False]).head(top_k)
    snippets = []
    for _, row in docs.iterrows():
        snippets.append(
            f"[{row['date'].date()}] {row['filename']} :: {str(row['text'])[:1200]}"
        )
    return "\n\n".join(snippets)


def _parse_scores(raw: str) -> List[float]:
    pairs = re.findall(r"topic\s*(\d+)\s*[-:]\s*([0-9]+(?:\.[0-9]+)?)%", raw, flags=re.IGNORECASE)
    values = [0.0] * CONFIG.stage.topic_count
    for topic_idx, pct in pairs:
        idx = int(topic_idx)
        if 1 <= idx <= CONFIG.stage.topic_count:
            values[idx - 1] = float(pct) / 100.0
    return normalize_distribution(values)


class RAGScorer:
    backend_name = "base"

    def score_window(
        self,
        country: str,
        channel: str,
        time_label: str,
        window_docs: pd.DataFrame,
        use_graph_context: str | None = None,
    ) -> List[float]:
        raise NotImplementedError


class OpenRouterRAGScorer(RAGScorer):
    backend_name = "openrouter"

    def __init__(self, model: str = "google/gemini-2.0-flash-exp:free"):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing in environment")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model

    def _call(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    extra_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Bonus_Exercise_RAG_Scoring",
                    },
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover
                if "429" in str(exc) and attempt < 2:
                    time.sleep(20)
                    continue
                raise
        return ""

    def score_window(
        self,
        country: str,
        channel: str,
        time_label: str,
        window_docs: pd.DataFrame,
        use_graph_context: str | None = None,
    ) -> List[float]:
        cache = _load_cache(channel)
        cache_key = f"{time_label}|graph={bool(use_graph_context)}"
        if cache_key in cache:
            return cache[cache_key]

        topics_block = _build_topics_block(country)
        docs_block = _build_context(window_docs)
        graph_block = f"\nGRAPH CONTEXT:\n{use_graph_context}\n" if use_graph_context else ""

        prompt = (
            "You are scoring topic dominance in one time window.\n"
            "Return only one line in this exact format with all topics: "
            "topic 1 - x.xxx%, topic 2 - x.xxx%, ... topic 20 - x.xxx%\n"
            "No explanation. Keep all topics in ascending order.\n\n"
            f"CHANNEL: {channel}\n"
            f"TIME WINDOW: {time_label}\n\n"
            f"TOPICS:\n{topics_block}\n\n"
            f"TEXT CONTEXT:\n{docs_block}\n"
            f"{graph_block}\n"
        )
        raw = self._call(prompt)
        values = _parse_scores(raw)
        cache[cache_key] = values
        _save_cache(channel, cache)
        return values


class LocalRAGScorer(RAGScorer):
    backend_name = "local"

    def _score_docs(
        self,
        topic_texts: Dict[int, str],
        docs: List[str],
        graph_context: str | None,
    ) -> List[float]:
        if not docs:
            return normalize_distribution([0.0] * CONFIG.stage.topic_count)

        corpus = [topic_texts[i] for i in range(1, CONFIG.stage.topic_count + 1)] + docs
        if graph_context:
            corpus.append(graph_context)

        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        matrix = vectorizer.fit_transform(corpus)

        topic_mat = matrix[: CONFIG.stage.topic_count]
        doc_mat = matrix[CONFIG.stage.topic_count : CONFIG.stage.topic_count + len(docs)]

        sim = cosine_similarity(topic_mat, doc_mat)
        topic_scores = np.maximum(0.0, sim).sum(axis=1)

        if graph_context:
            graph_vec = matrix[-1]
            graph_sim = cosine_similarity(topic_mat, graph_vec)
            topic_scores = topic_scores + (0.2 * np.maximum(0.0, graph_sim.reshape(-1)))

        return normalize_distribution(topic_scores.tolist())

    def score_window(
        self,
        country: str,
        channel: str,
        time_label: str,
        window_docs: pd.DataFrame,
        use_graph_context: str | None = None,
    ) -> List[float]:
        cache = _load_cache(channel)
        cache_key = f"{time_label}|graph={bool(use_graph_context)}|backend=local"
        if cache_key in cache:
            return cache[cache_key]

        topic_texts = _load_topic_texts(country)
        docs = []
        if not window_docs.empty:
            ranked_docs = window_docs.copy()
            ranked_docs["length"] = ranked_docs["text"].str.len()
            ranked_docs = ranked_docs.sort_values(["date", "length"], ascending=[False, False]).head(40)
            docs = [str(t)[:2000] for t in ranked_docs["text"].tolist()]

        values = self._score_docs(topic_texts, docs, use_graph_context)
        cache[cache_key] = values
        _save_cache(channel, cache)
        return values


def build_scorer(backend: str | None = None) -> RAGScorer:
    backend = (backend or os.getenv("RAG_BACKEND", "local")).strip().lower()
    if backend == "openrouter":
        return OpenRouterRAGScorer()
    return LocalRAGScorer()


def compute_rag_scores(
    docs: pd.DataFrame,
    temporal_scores: Dict[str, Any],
    graph_context: Dict[str, Dict[str, str]] | None = None,
    backend: str | None = None,
) -> Dict[str, pd.DataFrame]:
    scorer = build_scorer(backend=backend)
    outputs: Dict[str, pd.DataFrame] = {}
    graph_context = graph_context or {}

    for channel, ch_scores in temporal_scores.items():
        channel_docs = docs[docs["channel"] == channel]
        rows = []
        for window in ch_scores.windows:
            win_docs = filter_window(channel_docs, window)
            gctx = graph_context.get(channel, {}).get(window.label)
            values = scorer.score_window(
                country=ch_scores.country,
                channel=channel,
                time_label=window.label,
                window_docs=win_docs,
                use_graph_context=gctx,
            )
            row = {"time_label": window.label}
            for idx, value in enumerate(values, start=1):
                row[f"topic_{idx}"] = value
            rows.append(row)
        outputs[channel] = pd.DataFrame(rows)

    return outputs
