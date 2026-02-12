from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

from config import CONFIG
from temporal_scoring import compute_topic_model_scores
from time_windows import filter_window
from topic_discovery import load_topic_model, load_topic_table


RELATION_PATTERNS = {
    "supports": r"\b(support|back|approve)\b",
    "opposes": r"\b(oppose|reject|criticize|condemn)\b",
    "calls_for": r"\b(call for|urge|demand|ask for)\b",
    "votes": r"\b(vote|voted|ballot)\b",
    "mentions": r"\b(said|stated|announced|according to)\b",
}


@dataclass
class TopicGraphBundle:
    graphs: Dict[Tuple[str, int], nx.MultiDiGraph]
    selected_topics: Dict[str, List[int]]


def _simple_entities(text: str, max_entities: int = 8) -> List[str]:
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    counts = Counter(candidates)
    return [e for e, _ in counts.most_common(max_entities)]


def _detect_relation(text: str) -> str:
    low = text.lower()
    for rel, pattern in RELATION_PATTERNS.items():
        if re.search(pattern, low):
            return rel
    return "mentions"


def _topic_rank_map(country: str) -> Dict[int, int]:
    table = load_topic_table(country)
    return {int(row["topic_id"]): int(row["topic_rank"]) for _, row in table.iterrows()}


def _assign_topic_ranks(country_docs: pd.DataFrame, country: str) -> pd.DataFrame:
    model = load_topic_model(country)
    rank_map = _topic_rank_map(country)
    topics, _ = model.transform(country_docs["text"].tolist())
    assigned = []
    for topic_id in topics:
        assigned.append(rank_map.get(topic_id, None))
    out = country_docs.copy()
    out["topic_rank"] = assigned
    return out.dropna(subset=["topic_rank"]).copy()


def select_top3_topics_by_correlation(summary_df: pd.DataFrame) -> Dict[str, List[int]]:
    selected: Dict[str, List[int]] = {}
    for country in ["UK", "US"]:
        country_df = summary_df[summary_df["country"] == country]
        top = country_df.sort_values("best_correlation", ascending=False).head(3)
        selected[country] = top["topic"].astype(int).tolist()
    return selected


def build_topic_graphs(docs: pd.DataFrame, selected_topics: Dict[str, List[int]]) -> TopicGraphBundle:
    graphs: Dict[Tuple[str, int], nx.MultiDiGraph] = {}

    for country in ["UK", "US"]:
        country_docs = docs[docs["country"] == country]
        assigned = _assign_topic_ranks(country_docs, country)

        for topic_rank in selected_topics[country]:
            topic_docs = assigned[assigned["topic_rank"] == topic_rank]
            for channel in topic_docs["channel"].unique():
                graph = nx.MultiDiGraph(country=country, channel=channel, topic=topic_rank)
                ch_docs = topic_docs[topic_docs["channel"] == channel]
                for _, row in ch_docs.iterrows():
                    entities = _simple_entities(str(row["text"]))
                    relation = _detect_relation(str(row["text"]))
                    for entity in entities:
                        graph.add_node(entity, type="entity")
                    if len(entities) >= 2:
                        src = entities[0]
                        for dst in entities[1:]:
                            graph.add_edge(
                                src,
                                dst,
                                relation=relation,
                                date=str(pd.to_datetime(row["date"]).date()),
                                evidence=str(row["text"])[:240],
                            )
                graphs[(channel, topic_rank)] = graph

    return TopicGraphBundle(graphs=graphs, selected_topics=selected_topics)


def build_graph_context_by_window(
    docs: pd.DataFrame,
    overlap: Dict[str, tuple[datetime, datetime]],
    bundle: TopicGraphBundle,
) -> Dict[str, Dict[str, str]]:
    temporal = compute_topic_model_scores(docs, overlap)
    contexts: Dict[str, Dict[str, str]] = defaultdict(dict)

    for channel, ch_scores in temporal.items():
        country = ch_scores.country
        channel_docs = docs[docs["channel"] == channel]
        selected = bundle.selected_topics[country]

        for window in ch_scores.windows:
            win_docs = filter_window(channel_docs, window)
            if win_docs.empty:
                contexts[channel][window.label] = "No graph evidence for this window."
                continue

            topic_lines = []
            for topic_rank in selected:
                graph = bundle.graphs.get((channel, topic_rank))
                if not graph or graph.number_of_edges() == 0:
                    continue

                edge_counter = Counter()
                node_counter = Counter()
                for u, v, data in graph.edges(data=True):
                    relation = data.get("relation", "mentions")
                    edge_counter[f"{u} -[{relation}]-> {v}"] += 1
                    node_counter[u] += 1
                    node_counter[v] += 1

                top_edges = "; ".join([e for e, _ in edge_counter.most_common(3)])
                top_nodes = ", ".join([n for n, _ in node_counter.most_common(5)])
                topic_lines.append(
                    f"topic {topic_rank}: entities [{top_nodes}] | relations [{top_edges}]"
                )

            contexts[channel][window.label] = "\n".join(topic_lines) if topic_lines else "No graph evidence."

    return contexts
