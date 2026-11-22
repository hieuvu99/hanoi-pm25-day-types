# Phase 1 — data collection and cleaning (notebooks in Vietnamese)

- `01_thu_thap_lam_sach.ipynb`: download PM2.5 (OpenAQ archive) and weather (Open-Meteo), convert to Asia/Ho_Chi_Minh, shift OpenAQ end-of-hour labels by −1 h, flag/clean values. Outputs `data/clean/hanoi_hourly_clean.parquet` and `reports/missing_*`, `timezone_check.png`.
- `02_bien_bo_sung.ipynb`: NASA FIRMS fire counts (3 regions), Tết, lockdowns; builds `hanoi_hourly_enriched.parquet` and the daily table `hanoi_daily.parquet|csv` (with 24-h curves `pm25_h00..h23`).
- `data/raw_openaq_2539_7441.tar.gz`, `data/raw_openmeteo.tar.gz`: raw cache; extract into `data/raw/` to re-run without downloading.

Cleaning rules: negative → NaN; > 1000 → NaN; days with ≥ 6 h of raw PM2.5 ≤ 3 µg/m³ → those hours NaN (`flag_suspect_zero`); spikes/flatlines only flagged; a day is complete with ≥ 18/24 valid hours; gaps ≤ 3 h linearly interpolated into `pm25_filled`.
