## Exercise 4 — Temporal RAG (Retrieval-Augmented Generation)

**Authors:** Eli Gubman - 213662364 , Efraim Elgrabli - 212451074

## Project Overview

Standard RAG systems are "time blind" (like exercie 3): they treat older and newer documents equally, which can cause temporal hallucinations (citing outdated facts as if they are current). This project extends a static RAG pipeline to be time-aware - a Temporal RAG - by adding timestamp extraction, temporal indexing, and time-aware retrieval strategies.

### Goals
- Extract and store time metadata alongside text/vector data.
- Implement retrieval strategies that respect time (hard filters and time-decay scoring).
- Add evolutionary inference to detect how discussions change across time windows.

## What changed (high level)
- `data_loader.py` - Extracts timestamps (from filenames and metadata), normalizes to ISO 8601, and indexes tuples `(text, timestamp)`.
- `chunker.py` - Ensures timestamps are propagated to every text chunk during splitting.
- `retriever.py` - Adds `HardFilter` (strict year-based filtering), `SoftDecay` (time-decay weighting combined with similarity), and `retrieve_evolutionary()` (splits results into early/late buckets).
- `rag.py` - Uses a local LLM via Ollama (`llama3`) and adds `evolutionary_answer()` to analyze differences between two time-period contexts.
- `main.py` - Orchestrates indexing, retrieval, histogram generation, and evaluation across question types.

## Methodology

1. Data Engineering (Temporal Indexing)

- **Date Extraction:** Implemented Regex logic in data_loader.py to extract dates from filenames (e.g., debates2025-10-31.txt $\rightarrow$ 2025-10-31) and internal document headers.
- **Normalization:** Converted all extracted dates to the standard ISO 8601 format (YYYY-MM-DD) to ensure consistent chronological sorting.
- **Metadata Propagation:** Updated the chunking logic (chunker.py) to ensure that the timestamp metadata extracted at the document level is propagated down to every individual text chunk.
- **Storage:** Each record in the vector store is a tuple of (Text_Content, Vector_Embedding, Timestamp), allowing for hybrid filtering during the retrieval phase.

2. Retrieval Methods (Sparse vs. Dense)

- **Sparse Retrieval (BM25):** mplemented using rank_bm25.
Relies on probabilistic keyword matching.
Use Case: Highly effective for precise queries containing specific entities (e.g., names, budget figures, specific bills).
- **Dense Retrieval (Semantic Search):** Implemented using sentence-transformers (Model: all-MiniLM-L6-v2).
Generates 384-dimensional embeddings to calculate Cosine Similarity.
Use Case: Captures semantic meaning and nuance, which is critical for identifying changes in rhetoric (e.g., "crisis" vs. "challenge") even when specific keywords change over the years.

3. Temporal Algorithms

- **Hard Filter:** Strict year-based (or date-based) filtering for point-in-time queries (e.g., "what was true in 2024?").
- **Soft Decay (Rational Decay):** Combine similarity with a time-decay term to favor more recent documents without fully discarding older but highly relevant sources. The scoring formula used is:

$$
\\mathrm{Score} = (1 - \\alpha) \\cdot \\mathrm{Sim} + \\alpha \\cdot \\frac{1}{1 + \\Delta t \\cdot \\lambda}
$$

- **Variables:**
	- $\\Delta t$ — time difference in years between the query date and the document date.
	- $\\alpha$ — balance factor (example default: `0.3`).
	- $\\lambda$ — decay rate (example default: `0.5`).

4. Evolutionary Inference Pipeline

- **Dual Retrieval:** Sort relevant chunks by timestamp and split into an Early bucket (e.g., 2023) and a Late bucket (e.g., 2025).
- **Top-K Extraction:** Retrieve top `K` chunks (default `K=5`) from each bucket independently.
- **Synthesis:** Provide both time-sliced contexts to the local LLM (via `ollama` / `llama3`) with an explicit compare-and-contrast prompt to generate an "evolutionary" answer.

These methods are implemented across the repository files listed below.

## Evaluation & Example Results

We evaluated four question types with comparative examples (Baseline = time-blind retrieval, Temporal RAG = our methods):

- Category A - Point-in-Time: Hard Filter finds the authoritative document for year X and avoids older projections.
- Category B - Recency: Soft Decay favors recent authoritative sources over semantically strong but older documents.
- Category C - Role Changes: Soft Decay helps surface newer documents that reflect personnel changes.
- Category D - Evolutionary: Dual retrieval (Early vs Late) allows the model to describe how discussion topics or emphasis shifted over time.

For detailed CSV results see `qeries.txt` and the generated histogram image `temporal_distribution_histogram.png`.

## Files
- [`data_loader.py`](exercises/src/Exercise%204/data_loader.py) - data extraction and timestamp normalization
- [`chunker.py`](exercises/src/Exercise%204/chunker.py) - chunk splitting with metadata propagation
- [`retriever.py`](exercises/src/Exercise%204/retriever.py) - HardFilter / SoftDecay / evolutionary retrieval
- [`rag.py`](exercises/src/Exercise%204/rag.py) - LLM prompting and evolutionary analysis
- [`main.py`](exercises/src/Exercise%204/main.py) - orchestrator for indexing, retrieval, and evaluation


## How to run (local dev)

1. Start Ollama (Docker):

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama run llama3
```

2. Install Python dependencies (example using `pip` - adapt to your environment):

```bash
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python main.py
```

This will index the data, generate the temporal histogram, and run the evaluation across question types. Output files include `qeries.txt` and `temporal_distribution_histogram.png`.
