# Exercise Bonus: Political-Media Influence Dynamics

End-to-end pipeline for the bonus assignment question: **"Who leads whom?"** between politics and media in the UK and US.

## What This Repository Runs

- Stage A: topic discovery with BERTopic (20 topics per country).
- Stage B: temporal dominance scoring (Topic Modeling method + RAG method).
- Stage C: Knowledge Graph + GraphRAG comparison.
- Stage D: influence direction analysis (lagged correlation, DTW, numeric edit similarity).

The pipeline reads the original shared dataset from `files.zip` (nested archives + Excel templates).

## Prerequisites

- Python 3.11+
- `uv` installed
- Local mode works out of the box (no external API needed)
- Optional: `OPENROUTER_API_KEY` only if you want remote LLM scoring

Optional (only for remote backend):

```bash
export OPENROUTER_API_KEY="your_key_here"
```

Backend selection (default is local LangChain BM25):

```bash
export RAG_BACKEND=langchain_bm25  # default, no API, uses LangChain BM25
# or
export RAG_BACKEND=local           # tf-idf fallback, no API
# or
export RAG_BACKEND=openrouter # uses OPENROUTER_API_KEY
```

## Quick Start (Recommended)

```bash
cd "Exercise Bonus"
uv sync
uv run python src/run_all.py
```

## Run Stage by Stage

Run in this exact order (Stage C depends on Stage D RAG outputs):

```bash
cd "Exercise Bonus"
uv run python src/run_stage_a.py
uv run python src/run_stage_b.py
uv run python src/run_stage_d.py --method rag
uv run python src/run_stage_c.py
uv run python src/run_stage_d.py --method topic_model
```

## Expected Outputs

### Required numeric files

- `results/excel/uk_parliament_topic_model.xlsx`
- `results/excel/uk_media_topic_model.xlsx`
- `results/excel/us_congress_topic_model.xlsx`
- `results/excel/us_media_topic_model.xlsx`
- `results/excel/uk_parliament_rag.xlsx`
- `results/excel/uk_media_rag.xlsx`
- `results/excel/us_congress_rag.xlsx`
- `results/excel/us_media_rag.xlsx`

### Additional analysis files

- `results/excel/topics_catalog.xlsx` (topic name + BERTopic keywords)
- `results/excel/*_graphrag.xlsx`
- `results/tables/stage_d_rag_influence_summary.csv`
- `results/tables/stage_d_topic_model_influence_summary.csv`
- `results/tables/stage_c_rag_vs_graphrag.csv`
- `results/graphs/*.png`

## Scoring Conventions

- Each time point has 20 topic scores in range `0..1`.
- Scores are normalized so each row sums to `1.000`.
- Sliding window is 14 days with 7-day step.
- Lag search range is `-3..+3`.

## What To Do After Running

1. Verify all 8 required Excel files were created under `results/excel`.
2. Verify per-row sum in those files is exactly `1.000` (within rounding tolerance).
3. Use `results/graphs` + `results/tables` to write the final report.
4. Fill/report using `REPORT_TEMPLATE.md`.

## Report Writing Checklist

- Describe corpora + overlap time ranges used.
- Explain methodology (BERTopic, RAG, GraphRAG, lag/DTW/edit metrics).
- Present 10 topics per country with direction + lag + interpretation.
- Include amplitude and amplitude-gap interpretation.
- Compare UK vs US patterns and discuss why differences may occur.
- Add limitations/biases and concrete improvement ideas.

## Assignment Reference

See `../BONUS_ASSIGNMENT.md` for the official full instructions.
