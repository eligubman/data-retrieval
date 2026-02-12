from __future__ import annotations

import argparse

import pandas as pd

from config import CONFIG
from influence_analysis import analyze_country, plot_country_topics
from io_utils import ensure_result_dirs


def _load_scores(method: str, channel: str) -> pd.DataFrame:
    path = CONFIG.paths.results_tables / f"{method}_{channel}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="rag", choices=["rag", "topic_model"], help="Score source")
    args = parser.parse_args()

    ensure_result_dirs()
    all_rows = []

    for country, (politics_channel, media_channel) in CONFIG.country_channels.items():
        politics = _load_scores(args.method, politics_channel)
        media = _load_scores(args.method, media_channel)

        df = analyze_country(country, politics, media)
        all_rows.append(df)

        selected = (
            df.sort_values(["best_correlation", "edit_similarity"], ascending=False)
            .head(10)["topic"]
            .astype(int)
            .tolist()
        )
        plot_country_topics(country, politics, media, selected_topics=selected, prefix=f"stage_d_{args.method}")

    summary = pd.concat(all_rows, ignore_index=True)
    out_path = CONFIG.paths.results_tables / f"stage_d_{args.method}_influence_summary.csv"
    summary.to_csv(out_path, index=False)
    print("Stage D completed. Summary:", out_path)


if __name__ == "__main__":
    main()
