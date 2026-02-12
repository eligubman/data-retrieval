# Bonus Exercise Report Template

## 1) Corpora and Time Coverage
- UK politics: Hansard corpus details and date range.
- UK media: BBC corpus details and date range.
- US politics: Congressional Record corpus details and date range.
- US media: NBC corpus details and date range.
- Final overlap ranges used for fair comparison.

## 2) Methods
### Topic Modeling (BERTopic)
- Why 20 topics.
- Key parameters.
- Topic labeling approach.

### RAG Scoring
- Window size and step.
- Retrieval strategy and prompt engineering.
- Output format and normalization checks.

### GraphRAG
- Selected high-correlation topics and why.
- Entity and relation extraction logic.
- How graph context was injected to the same prompt.

### Temporal Comparison
- Lagged correlation method (`-3..+3`).
- Dynamic Time Warping (DTW).
- Numeric adaptation of edit distance.

## 3) Results
### Stage A
- Topic quality check with frequency/overview figures.
- Topic catalog table reference.

### Stage B
- Five dominant-topic trend charts per country.
- Observed amplitude and amplitude-gap differences between media/politics.

### Stage C
- Three selected topics and rationale.
- KG visualizations.
- RAG vs GraphRAG comparison table and interpretation.

### Stage D
- 10 topics in UK: direction, lag, metrics, interpretation.
- 10 topics in US: direction, lag, metrics, interpretation.
- Cross-country comparison: where directionality differs and why.

## 4) Discussion and Critical Review
- Main answer to: "Who leads whom?"
- Topics with stable lead vs changing lead.
- Potential causes for reversals (events, elections, conflict spikes).
- Limitations, bias sources, and suggested improvements.

## 5) Submission Checklist
- [ ] 8 required Excel files (4 channels x 2 methods).
- [ ] Topic catalog Excel (topic name + BERTopic words).
- [ ] Correlation/Lag summary tables.
- [ ] Graph and trend visualizations.
- [ ] Full code and runnable scripts.
