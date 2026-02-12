from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from config import CONFIG
from io_utils import ensure_result_dirs

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "BERTopic dependencies are missing. Install requirements in Exercise Bonus/requirements.txt"
    ) from exc


def _build_topic_model() -> BERTopic:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return BERTopic(
        embedding_model=embedding_model,
        language="english",
        calculate_probabilities=True,
        min_topic_size=20,
        verbose=True,
    )


def _topic_label(words: List[Tuple[str, float]], max_terms: int = 4) -> str:
    top = [w for w, _ in words[:max_terms] if w]
    return " / ".join(top) if top else "misc"


def fit_country_topics(docs: pd.DataFrame, country: str) -> tuple[BERTopic, pd.DataFrame, pd.DataFrame]:
    country_docs = docs[docs["country"] == country].copy()
    if country_docs.empty:
        raise ValueError(f"No docs for country {country}")

    texts = country_docs["text"].tolist()
    model = _build_topic_model()
    topics, _ = model.fit_transform(texts)

    # Enforce 20 topics for assignment requirement
    if len(set(t for t in topics if t != -1)) != CONFIG.stage.topic_count:
        model = model.reduce_topics(texts, nr_topics=CONFIG.stage.topic_count)

    topic_info = model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1].head(CONFIG.stage.topic_count).copy()
    topic_info = topic_info.reset_index(drop=True)
    topic_info["topic_rank"] = range(1, len(topic_info) + 1)

    rows = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        words = model.get_topic(topic_id) or []
        rows.append(
            {
                "country": country,
                "topic_rank": int(row["topic_rank"]),
                "topic_id": topic_id,
                "label": _topic_label(words),
                "keywords": ", ".join([w for w, _ in words[:10]]),
                "count": int(row.get("Count", 0)),
            }
        )

    topic_table = pd.DataFrame(rows)
    return model, topic_info, topic_table


def save_topic_artifacts(
    country: str,
    model: BERTopic,
    topic_info: pd.DataFrame,
    topic_table: pd.DataFrame,
) -> Dict[str, Path]:
    ensure_result_dirs()
    cache_dir = CONFIG.paths.cache
    graphs_dir = CONFIG.paths.results_graphs
    tables_dir = CONFIG.paths.results_tables

    model_path = cache_dir / f"bertopic_{country.lower()}.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(model, fh)

    info_path = tables_dir / f"topic_info_{country.lower()}.csv"
    table_path = tables_dir / f"topics_{country.lower()}.csv"
    topic_info.to_csv(info_path, index=False)
    topic_table.to_csv(table_path, index=False)

    plt.figure(figsize=(12, 5))
    top = topic_table.sort_values("count", ascending=False)
    plt.bar([f"T{t}" for t in top["topic_rank"]], top["count"], color="#2B6CB0")
    plt.title(f"{country} Topic Frequency Overview")
    plt.xlabel("Topic")
    plt.ylabel("Document count")
    plt.tight_layout()
    plot_path = graphs_dir / f"stage_a_topic_frequency_{country.lower()}.png"
    plt.savefig(plot_path)
    plt.close()

    return {
        "model": model_path,
        "info": info_path,
        "table": table_path,
        "plot": plot_path,
    }


def save_topics_catalog_excel(uk_topics: pd.DataFrame, us_topics: pd.DataFrame) -> Path:
    ensure_result_dirs()
    out_path = CONFIG.paths.results_excel / "topics_catalog.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        uk_topics.to_excel(writer, sheet_name="UK_topics", index=False)
        us_topics.to_excel(writer, sheet_name="US_topics", index=False)
    return out_path


def load_topic_model(country: str) -> BERTopic:
    model_path = CONFIG.paths.cache / f"bertopic_{country.lower()}.pkl"
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def load_topic_table(country: str) -> pd.DataFrame:
    return pd.read_csv(CONFIG.paths.results_tables / f"topics_{country.lower()}.csv")
