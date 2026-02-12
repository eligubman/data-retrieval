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
# From exercises/ directory
cd "Exercise Bonus"

# Stage A: Topic Discovery
python src/stage_a_discover_topics.py

# Stage B: Temporal Dynamics
python src/stage_b_dynamics.py

# Stage C: Knowledge Graph
python src/stage_c_knowledge_graph.py

# Stage D: Influence Analysis
python src/stage_d_influence_analysis.py
```

## Notes

- Reusing utilities from previous exercises where applicable
- Following same code quality standards as Exercises 1-4
- All commits will be atomic and descriptive
