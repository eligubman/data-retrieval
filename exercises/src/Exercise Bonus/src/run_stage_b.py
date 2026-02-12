from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import CONFIG
from data_ingestion import build_loaded_corpora
from io_utils import ensure_result_dirs, export_channel_scores_to_template
from rag_scoring import compute_rag_scores
from temporal_scoring import compute_topic_model_scores, top_dominant_topics


def _save_score_tables(prefix: str, scores: dict[str, pd.DataFrame]) -> None:
    for channel, df in scores.items():
        out = CONFIG.paths.results_tables / f"{prefix}_{channel}.csv"
        df.to_csv(out, index=False)


def _plot_top_topics_for_country(
    country: str,
    topic_scores: dict[str, pd.DataFrame],
    output_prefix: str,
) -> None:
    politics_channel, media_channel = CONFIG.country_channels[country]
    politics = topic_scores[politics_channel]
    media = topic_scores[media_channel]

    top_topics = top_dominant_topics((politics.drop(columns=["time_label"]) + media.drop(columns=["time_label"])) / 2.0)
    x = range(1, len(politics) + 1)

    for topic_idx in top_topics:
        plt.figure(figsize=(12, 4))
        plt.plot(x, politics[f"topic_{topic_idx}"], label=politics_channel, linewidth=2)
        plt.plot(x, media[f"topic_{topic_idx}"], label=media_channel, linewidth=2)
        plt.title(f"{country} topic {topic_idx} dominance over time")
        plt.xlabel("Time point")
        plt.ylabel("Dominance score (0..1)")
        plt.legend()
        plt.tight_layout()
        out = CONFIG.paths.results_graphs / f"{output_prefix}_{country.lower()}_topic_{topic_idx}.png"
        plt.savefig(out)
        plt.close()


def main() -> None:
    ensure_result_dirs()
    loaded = build_loaded_corpora()

    topic_outputs = compute_topic_model_scores(loaded.docs, loaded.overlap)
    topic_score_tables = {channel: ch.scores for channel, ch in topic_outputs.items()}
    _save_score_tables("topic_model", topic_score_tables)

    for channel, df in topic_score_tables.items():
        export_channel_scores_to_template(channel, df, method="topic_model")

    rag_outputs = compute_rag_scores(loaded.docs, topic_outputs)
    _save_score_tables("rag", rag_outputs)
    for channel, df in rag_outputs.items():
        export_channel_scores_to_template(channel, df, method="rag")

    for country in CONFIG.country_channels:
        _plot_top_topics_for_country(country, topic_score_tables, output_prefix="stage_b_topic_model")
        _plot_top_topics_for_country(country, rag_outputs, output_prefix="stage_b_rag")

    print("Stage B completed")
    print("8 excel files saved under", CONFIG.paths.results_excel)


if __name__ == "__main__":
    main()
