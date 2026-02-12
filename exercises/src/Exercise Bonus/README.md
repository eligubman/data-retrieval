# Exercise Bonus: Political-Media Influence Dynamics

**Authors:** Ephraim Elgrabli & Elihu Gubman  
**Due:** Sunday 9:00 AM  
**Points:** Up to 8 bonus points

## Research Question

**"Who leads whom?"** - Does the political system dictate the media agenda, or does media influence political discourse?

## Implementation Plan

See `../BONUS_ASSIGNMENT.md` for full assignment details.

### Data Structure

```
Exercise Bonus/
├── data/              # Raw data (from Google Drive)
│   ├── uk_parliament/
│   ├── uk_media/
│   ├── us_congress/
│   └── us_media/
├── src/               # Source code
├── results/           # Outputs
│   ├── excel/        # 8 Excel files + topics file
│   ├── graphs/       # Visualizations
│   └── tables/       # Analysis tables
├── notebooks/         # Jupyter notebooks
└── README.md         # This file
```

## Running the Analysis

```bash
# From exercises/src/
cd "Exercise Bonus"

# Install dependencies
python -m pip install -r requirements.txt

# Run full pipeline
python src/run_all.py
```

Or run stages separately:

```bash
python src/run_stage_a.py
python src/run_stage_b.py
python src/run_stage_d.py --method rag
python src/run_stage_c.py
python src/run_stage_d.py --method topic_model
```

## Deliverables Produced

- `results/excel/*_topic_model.xlsx` - 4 channel files for Method 1
- `results/excel/*_rag.xlsx` - 4 channel files for Method 2
- `results/excel/*_graphrag.xlsx` - GraphRAG extension outputs
- `results/excel/topics_catalog.xlsx` - topic name + BERTopic keywords
- `results/tables/stage_d_*_influence_summary.csv` - lag/correlation/DTW/edit-distance
- `results/tables/stage_c_rag_vs_graphrag.csv` - RAG vs GraphRAG comparison table
- `results/graphs/` - topic frequency, time-series plots, and graph visualizations

## Important Output Conventions

- Every time window has 20 topic scores in `0..1`.
- Topic scores are normalized to sum to `1.000` per row.
- Sliding windows are 14 days with a 7-day step.
- Lag analysis checks offsets from `-3` to `+3`.

## Notes

- Reusing utilities from previous exercises where applicable
- Following same code quality standards as Exercises 1-4
- All commits will be atomic and descriptive
