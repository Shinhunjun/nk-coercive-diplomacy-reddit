# Data Version Control

Last Updated: 2025-12-13

## Source Data Files (Use These for Analysis)

### Primary Analysis Files (2017.01 - 2019.06)

| Group | File | Rows | Size | Modified |
|-------|------|------|------|----------|
| **NK** | `data/nk/nk_posts_merged.csv` | 9,007 | 3.8MB | 2025-12-09 |
| **China** | `data/control/china_posts_merged.csv` | 5,179 | 1.6MB | 2025-12-09 |
| **Iran** | `data/control/iran_posts_merged.csv` | 4,279 | 1.5MB | 2025-12-09 |
| **Russia** | `data/control/russia_posts_merged.csv` | 7,988 | 2.9MB | 2025-12-09 |

### Extended Period (2019.07 - 2019.12)

| Group | File | Rows | Size | Modified |
|-------|------|------|------|----------|
| NK | `data/nk/nk_posts_hanoi_extended.csv` | 1,652 | 0.6MB | 2025-12-12 |
| China | `data/control/china_posts_hanoi_extended.csv` | 2,845 | 1.2MB | 2025-12-12 |
| Iran | `data/control/iran_posts_hanoi_extended.csv` | 1,704 | 0.6MB | 2025-12-12 |
| Russia | `data/control/russia_posts_hanoi_extended.csv` | 2,273 | 0.8MB | 2025-12-12 |

---

## Sentiment Data (RoBERTa)

| Group | File | Rows | Modified |
|-------|------|------|----------|
| NK | `data/sentiment/nk_posts_sentiment.csv` | 11,887 | 2025-12-09 |
| China | `data/sentiment/china_posts_sentiment.csv` | 6,663 | 2025-12-09 |
| Iran | `data/sentiment/iran_posts_sentiment.csv` | 5,219 | 2025-12-09 |
| Russia | `data/sentiment/russia_posts_sentiment.csv` | 9,152 | 2025-12-09 |

---

## Framing Data (GPT-4o-mini)

| Group | File | Rows | Modified |
|-------|------|------|----------|
| NK | `data/framing/nk_posts_framed.csv` | 10,448 | 2025-12-09 |
| China | `data/framing/china_posts_framed.csv` | 5,921 | 2025-12-09 |
| Iran | `data/framing/iran_posts_framed.csv` | 4,749 | 2025-12-09 |
| Russia | `data/framing/russia_posts_framed.csv` | 8,570 | 2025-12-09 |

---

## Analysis Periods

- **P1_PreSingapore**: 2017-01 to 2018-05
- **P2_SingaporeHanoi**: 2018-06 to 2019-02
- **P3_PostHanoi**: 2019-03 to 2019-12

---

## Data Pipeline

```
1. Raw collection -> *_posts.csv, *_posts_full.csv
2. Merged/balanced -> *_posts_merged.csv (USE THIS)
3. Extended period -> *_posts_hanoi_extended.csv
4. Sentiment -> data/sentiment/*_posts_sentiment.csv
5. Framing -> data/framing/*_posts_framed.csv
6. Final -> data/final/*_final.csv (combined with sentiment + framing)
```

---

## Notes

- `_merged.csv`: Balanced monthly distribution, primary source
- `_full.csv`: Complete unfiltered data
- `_balanced.csv`: Subset with equal pre/post counts
- `_roberta.csv`: Subset with RoBERTa sentiment applied
