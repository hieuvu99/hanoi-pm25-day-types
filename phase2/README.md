# Phase 2 — analysis and data mining (notebooks in Vietnamese)

Run from this folder; data are read from `../phase1/data/clean/` (override with `HANOI_AQ_DIR`).

- `03_eda.ipynb` — EDA (analysis). Figures/tables `reports/01_*`–`10_*`.
- `04_clustering.ipynb` — day-shape clustering (mining). Outputs `data/day_clusters.csv|parquet`, `data/shape_centroids.csv`, `reports/11_*`–`19_*`.
- `05_association_rules.ipynb` — FP-Growth rules within season (mining). Outputs `data/rules_global.csv`, `data/rules_by_season.csv`, `reports/20_*`–`27_*`.
- `06_anomaly.ipynb` — anomaly detection (mining). Outputs `data/day_anomaly.csv`, `reports/30_*`–`37_*`.

Seasons: winter Dec–Feb, spring Mar–Apr, summer May–Sep, autumn Oct–Nov.
