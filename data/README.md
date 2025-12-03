# Data Documentation

## Overview

This directory contains data for the NK Coercive Diplomacy Reddit Analysis project.

---

## Data Structure

```
data/
├── sample/                              # Sample data (included in repo)
│   ├── posts_period1_sample.csv         # 50 posts from Tension Period
│   └── posts_period2_sample.csv         # 50 posts from Diplomacy Period
│
└── results/                             # Analysis results (JSON)
    ├── sentiment_comparison_results.json
    ├── framing_analysis_results.json
    ├── keyword_analysis_results.json
    ├── openai_framing_results.json
    ├── its_analysis_results.json
    ├── graphrag_comparison_results.json
    └── analysis_summary.json
```

---

## Full Dataset Download

The complete dataset is available on Google Drive:

🔗 **[Download Full Dataset](https://drive.google.com/drive/folders/1v0MlJEjHp5kODXL5jRflJvJLSPpAcFWZ?usp=sharing)**

### Full Dataset Contents

| File | Description | Size |
|------|-------------|------|
| `posts_period1_tension.csv` | All posts from Tension Period | 380 posts |
| `posts_period2_diplomacy.csv` | All posts from Diplomacy Period | 326 posts |
| `comments_period1_tension.csv` | Comments from Tension Period | 5,123 comments |
| `comments_period2_diplomacy.csv` | Comments from Diplomacy Period | 4,337 comments |

---

## Data Schema

### Posts CSV

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Reddit post ID |
| `title` | string | Post title |
| `selftext` | string | Post body text |
| `author` | string | Username |
| `subreddit` | string | Subreddit name |
| `score` | int | Upvotes - Downvotes |
| `num_comments` | int | Number of comments |
| `created_utc` | int | Unix timestamp |
| `permalink` | string | Reddit URL path |

### Comments CSV

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Comment ID |
| `body` | string | Comment text |
| `author` | string | Username |
| `parent_id` | string | Parent post/comment ID |
| `link_id` | string | Parent post ID |
| `score` | int | Upvotes - Downvotes |
| `created_utc` | int | Unix timestamp |

---

## Analysis Periods

### Period 1: Tension (2017.01.01 - 2018.02.28)

**Key Events:**
- Trump Inauguration (2017.01.20)
- "Fire and Fury" Speech (2017.08.08)
- 6th Nuclear Test (2017.09.03)
- Hwasong-15 ICBM Launch (2017.11.29)

**Data:**
- Posts: 380
- Comments: 5,123

### Period 2: Diplomacy (2018.06.01 - 2019.06.30)

**Key Events:**
- Singapore Summit (2018.06.12)
- Hanoi Summit (2019.02.27-28)
- Panmunjom Meeting (2019.06.30)

**Data:**
- Posts: 326
- Comments: 4,337

### Intervention Point

**Date**: March 8, 2018

Trump accepts Kim Jong Un's invitation for summit meeting.

---

## Data Collection

### Source
Reddit data collected via **Arctic Shift API** (https://arctic-shift.photon-reddit.com/)

### Search Terms
- "North Korea"
- "DPRK"
- "Kim Jong Un"
- "Korean Peninsula"
- "US Korea"

### Subreddits
- r/worldnews
- r/geopolitics
- r/politics
- r/northkorea
- r/korea
- r/news
- r/AskAnAmerican

---

## Results Files

### sentiment_comparison_results.json

Contains BERT sentiment analysis comparing two periods:
- Period means, std, n
- t-test and Mann-Whitney results
- Cohen's d effect size

### openai_framing_results.json

Contains GPT-4o-mini framing classification:
- Frame distribution per period
- Chi-square test results
- Sample classifications with reasoning

### its_analysis_results.json

Contains Interrupted Time Series regression:
- Model coefficients (β₀, β₁, β₂, β₃)
- p-values and significance
- Model fit statistics

### graphrag_comparison_results.json

Contains knowledge graph comparison:
- Entity statistics
- Relationship types
- Network metrics

---

## Reproducibility

To reproduce the analysis with full data:

1. Download full dataset from Google Drive
2. Place files in `data/full/` directory
3. Update paths in `src/config.py`
4. Run `python src/run_analysis.py`

---

## License

Data is provided for research purposes only. Reddit content is subject to Reddit's Terms of Service.
