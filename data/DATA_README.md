# Data Directory Structure

This document tracks all data files in the project for the NK Coercive Diplomacy Reddit analysis.

## Analysis Period

- **Full Period**: 2017.01 - 2019.12
- **Period 1 (Pre-Singapore)**: 2017.01 - 2018.05
- **Period 2 (Singapore-Hanoi)**: 2018.06 - 2019.02
- **Period 3 (Post-Hanoi)**: 2019.03 - 2019.12

---

## Final Analysis Files (USE THESE)

| File | Group | Period | Posts | Sentiment | Framing |
|------|-------|--------|-------|-----------|---------|
| `data/final/nk_final.csv` | NK | 2017.01-2019.12 | ~12K | ✅ | ✅ |
| `data/final/china_final.csv` | China | 2017.01-2019.12 | ~7K | ✅ | ✅ |
| `data/final/iran_final.csv` | Iran | 2017.01-2019.12 | ~5K | ✅ | ✅ |
| `data/final/russia_final.csv` | Russia | 2017.01-2019.12 | ~9K | ✅ | ✅ |

---

## Source Data Files

### NK Data (`data/nk/`)

| File | Description | Period |
|------|-------------|--------|
| `nk_posts_merged.csv` | Original collected posts | 2017.01-2019.06 |
| `nk_posts_hanoi_extended.csv` | Extended collection | 2019.07-2019.12 |

### Control Groups (`data/control/`)

| File | Description | Period |
|------|-------------|--------|
| `china_posts_full.csv` | Full China posts | 2017.01-2019.06 |
| `china_posts_hanoi_extended.csv` | Extended China | 2019.07-2019.12 |
| `iran_posts_full.csv` | Full Iran posts | 2017.01-2019.06 |
| `iran_posts_hanoi_extended.csv` | Extended Iran | 2019.07-2019.12 |
| `russia_posts_full.csv` | Full Russia posts | 2017.01-2019.06 |
| `russia_posts_hanoi_extended.csv` | Extended Russia | 2019.07-2019.12 |

---

## Analysis Columns in Final Files

| Column | Description |
|--------|-------------|
| `id` | Reddit post ID |
| `title` | Post title |
| `selftext` | Post body |
| `subreddit` | Source subreddit |
| `created_utc` | Unix timestamp |
| `datetime` | Parsed datetime |
| `month` | YYYY-MM format |
| `period` | P1_PreSingapore / P2_SingaporeHanoi / P3_PostHanoi |
| `topic` | nk / china / iran / russia |
| `sentiment_score` | RoBERTa sentiment (-1 to +1) |
| `sentiment_label` | negative / neutral / positive |
| `frame` | THREAT / DIPLOMACY / NEUTRAL / ECONOMIC / HUMANITARIAN |
| `frame_score` | Numeric frame (-2 to +2) |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/create_final_dataset.py` | Creates unified final datasets |
| `scripts/sentiment_3period_did_analysis.py` | 3-period DID on sentiment |
| `scripts/hanoi_3period_did_analysis.py` | 3-period DID on framing |

---

*Last updated: 2025-12-13*
