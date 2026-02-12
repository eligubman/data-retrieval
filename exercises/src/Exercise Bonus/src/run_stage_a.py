from __future__ import annotations

from data_ingestion import build_loaded_corpora
from io_utils import ensure_result_dirs
from topic_discovery import (
    fit_country_topics,
    save_topic_artifacts,
    save_topics_catalog_excel,
)


def main() -> None:
    ensure_result_dirs()
    loaded = build_loaded_corpora()

    uk_model, uk_info, uk_topics = fit_country_topics(loaded.docs, "UK")
    us_model, us_info, us_topics = fit_country_topics(loaded.docs, "US")

    uk_paths = save_topic_artifacts("UK", uk_model, uk_info, uk_topics)
    us_paths = save_topic_artifacts("US", us_model, us_info, us_topics)
    catalog = save_topics_catalog_excel(uk_topics, us_topics)

    print("Stage A completed")
    print("UK artifacts:", uk_paths)
    print("US artifacts:", us_paths)
    print("Topics catalog:", catalog)
    print("Overlap windows:", loaded.overlap)


if __name__ == "__main__":
    main()
