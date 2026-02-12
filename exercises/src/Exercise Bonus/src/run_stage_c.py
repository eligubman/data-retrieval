from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from config import CONFIG
from data_ingestion import build_loaded_corpora
from io_utils import ensure_result_dirs, export_channel_scores_to_template
from knowledge_graph import (
    build_graph_context_by_window,
    build_topic_graphs,
    select_top3_topics_by_correlation,
)
from rag_scoring import compute_rag_scores
from temporal_scoring import compute_topic_model_scores


def _draw_graph(graph: nx.MultiDiGraph, output_name: str) -> None:
    plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(graph, seed=42, k=1.2)
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="#90CDF4")
    nx.draw_networkx_labels(graph, pos, font_size=8)
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.5)
    plt.title(output_name)
    plt.axis("off")
    out = CONFIG.paths.results_graphs / f"stage_c_{output_name}.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _compare_tables(
    baseline: dict[str, pd.DataFrame],
    graphrag: dict[str, pd.DataFrame],
    selected_topics: dict[str, list[int]],
) -> pd.DataFrame:
    rows = []
    for country, channels in CONFIG.country_channels.items():
        topics = selected_topics[country]
        for channel in channels:
            base = baseline[channel]
            graph = graphrag[channel]
            if base.empty:
                base = graph.copy()
            for _, row in base.iterrows():
                label = row["time_label"]
                graph_row = graph[graph["time_label"] == label].iloc[0]
                for topic in topics:
                    key = f"topic_{topic}"
                    rows.append(
                        {
                            "country": country,
                            "channel": channel,
                            "time_label": label,
                            "topic": topic,
                            "rag_score": float(row[key]),
                            "graphrag_score": float(graph_row[key]),
                            "delta": float(graph_row[key]) - float(row[key]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_result_dirs()
    loaded = build_loaded_corpora()

    influence_path = CONFIG.paths.results_tables / "stage_d_rag_influence_summary.csv"
    if not influence_path.exists():
        raise FileNotFoundError(
            "Run Stage D with --method rag before Stage C to pick top correlated topics"
        )

    summary = pd.read_csv(influence_path)
    selected_topics = select_top3_topics_by_correlation(summary)
    bundle = build_topic_graphs(loaded.docs, selected_topics)

    for (channel, topic_rank), graph in bundle.graphs.items():
        if graph.number_of_edges() > 0:
            _draw_graph(graph, output_name=f"{channel}_topic_{topic_rank}")

    graph_context = build_graph_context_by_window(loaded.docs, loaded.overlap, bundle)
    temporal = compute_topic_model_scores(loaded.docs, loaded.overlap)

    baseline = {}
    for channel in temporal:
        path = CONFIG.paths.results_tables / f"rag_{channel}.csv"
        if path.exists():
            baseline[channel] = pd.read_csv(path)
        else:
            baseline[channel] = pd.DataFrame()

    graph_scores = compute_rag_scores(loaded.docs, temporal, graph_context=graph_context)
    for channel, df in graph_scores.items():
        out = CONFIG.paths.results_tables / f"graphrag_{channel}.csv"
        df.to_csv(out, index=False)
        export_channel_scores_to_template(channel, df, method="graphrag")

    comparison = _compare_tables(baseline, graph_scores, selected_topics)
    comparison_out = CONFIG.paths.results_tables / "stage_c_rag_vs_graphrag.csv"
    comparison.to_csv(comparison_out, index=False)

    selected_rows = []
    for country, topics in selected_topics.items():
        for topic in topics:
            selected_rows.append({"country": country, "topic": topic})
    pd.DataFrame(selected_rows).to_csv(CONFIG.paths.results_tables / "stage_c_selected_topics.csv", index=False)

    print("Stage C completed")
    print("Selected topics:", selected_topics)
    print("Comparison table:", comparison_out)


if __name__ == "__main__":
    main()
