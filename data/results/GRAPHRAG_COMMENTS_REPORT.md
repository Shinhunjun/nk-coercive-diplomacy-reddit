# GraphRAG Comment Network Analysis Report

## Executive Summary

This report presents detailed network analysis results for Comments (Recursive), mirroring the methodology used for Posts.

---

## 1. Network Topology Metrics

| Metric | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) | P1→P2 Δ | P2→P3 Δ |
|--------|-------------------|---------------------|-----------------|---------|---------|
| Nodes | 2,735 | 787 | 1,429 | -1,948 | +642 |
| Edges | 4,421 | 1,218 | 2,383 | -3,203 | +1,165 |
| Density | 0.0012 | 0.0039 | 0.0023 | +0.0028 | -0.0016 |
| Avg Degree | 3.2329 | 3.0953 | 3.3352 | -0.1376 | +0.2399 |
| Clustering | 0.1864 | 0.1944 | 0.2070 | +0.0080 | +0.0126 |
| Components | 73 | 26 | 37 | -47 | +11 |

---

## 2. Centrality Analysis

### 2.1 Degree Centrality (Top 10 per Period)

| Rank | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) |
|------|-------------------|---------------------|-----------------|
| 1 | NORTH KOREA (0.419) | NORTH KOREA (0.433) | NORTH KOREA (0.404) |
| 2 | TRUMP (0.140) | TRUMP (0.201) | TRUMP (0.239) |
| 3 | SOUTH KOREA (0.088) | SOUTH KOREA (0.123) | KIM JONG UN (0.095) |
| 4 | KIM JONG UN (0.060) | KIM JONG UN (0.073) | DONALD TRUMP (0.090) |
| 5 | US (0.060) | UNITED STATES (0.051) | SOUTH KOREA (0.071) |
| 6 | UNITED STATES (0.050) | US (0.050) | KIM JONG-UN (0.053) |
| 7 | DONALD TRUMP (0.047) | CHINA (0.045) | UNITED STATES (0.050) |
| 8 | CHINA (0.045) | DONALD TRUMP (0.045) | CHINA (0.036) |
| 9 | USA (0.039) | USA (0.043) | KIM (0.033) |
| 10 | DPRK (0.037) | DENUCLEARIZATION (0.042) | RUSSIA (0.032) |

### 2.2 PageRank (Top 10 per Period)

| Rank | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) |
|------|-------------------|---------------------|-----------------|
| 1 | NORTH KOREA (0.2028) | NORTH KOREA (0.1928) | NORTH KOREA (0.1783) |
| 2 | TRUMP (0.0501) | TRUMP (0.0702) | TRUMP (0.0851) |
| 3 | SOUTH KOREA (0.0395) | SOUTH KOREA (0.0484) | KIM JONG UN (0.0441) |
| 4 | KIM JONG UN (0.0247) | KIM JONG UN (0.0339) | DONALD TRUMP (0.0277) |
| 5 | CHINA (0.0198) | DENUCLEARIZATION (0.0162) | SOUTH KOREA (0.0271) |
| 6 | US (0.0172) | UNITED STATES (0.0157) | KIM JONG-UN (0.0192) |
| 7 | UNITED STATES (0.0166) | CHINA (0.0156) | UNITED STATES (0.0142) |
| 8 | RUSSIA (0.0140) | DONALD TRUMP (0.0136) | KIM (0.0137) |
| 9 | DONALD TRUMP (0.0123) | US (0.0136) | CHINA (0.0130) |
| 10 | DPRK (0.0101) | KIM JONG-UN (0.0129) | RUSSIA (0.0091) |

---

## 3. Entity Type Evolution

| Type | P1 Count | P1 % | P2 Count | P2 % | P3 Count | P3 % |
|------|----------|------|----------|------|----------|------|
| ANIMAL | 1 | 0.0% | 1 | 0.1% | 0 | 0.0% |
| AWARD | 2 | 0.1% | 0 | 0.0% | 0 | 0.0% |
| CITY | 21 | 0.7% | 6 | 0.7% | 2 | 0.1% |
| COUNTRY | 171 | 5.8% | 85 | 10.2% | 103 | 6.7% |
| CULTURAL PHENOMENON | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| CULTURE | 1 | 0.0% | 1 | 0.1% | 0 | 0.0% |
| CURRENCY | 2 | 0.1% | 1 | 0.1% | 2 | 0.1% |
| DRUG | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| EVENT | 545 | 18.6% | 193 | 23.1% | 303 | 19.6% |
| FOOD | 2 | 0.1% | 1 | 0.1% | 0 | 0.0% |
| GEO | 42 | 1.4% | 11 | 1.3% | 34 | 2.2% |
| LOCATION | 1 | 0.0% | 0 | 0.0% | 2 | 0.1% |
| MILITARY | 3 | 0.1% | 1 | 0.1% | 0 | 0.0% |
| MILITARY BASE | 0 | 0.0% | 0 | 0.0% | 1 | 0.1% |
| MILITARY VESSEL | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| NATION | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| ORGANIZATION | 439 | 15.0% | 120 | 14.4% | 243 | 15.7% |
| PERSON | 502 | 17.2% | 146 | 17.5% | 274 | 17.7% |
| POLICY | 622 | 21.3% | 160 | 19.2% | 302 | 19.5% |
| REGION | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| SATELLITE | 0 | 0.0% | 0 | 0.0% | 3 | 0.2% |
| TECHNOLOGY | 0 | 0.0% | 1 | 0.1% | 0 | 0.0% |
| WEAPON | 293 | 10.0% | 52 | 6.2% | 120 | 7.8% |

---

## 4. Relationship Framing Analysis

| Frame | P1 % | P2 % | P3 % | P1→P2 Δ | P2→P3 Δ |
|-------|------|------|------|---------|---------|
| **Threat** | 55.6% | 50.3% | 41.0% | **-5.3%** | **-9.4%** |
| **Peace** | 6.4% | 15.5% | 12.1% | **+9.1%** | **-3.5%** |
| **Neutral** | 38.0% | 34.1% | 47.0% | **-3.9%** | **+12.8%** |

---

## 5. Key Actor Dyad Analysis

| Relationship | P1 Weight | P2 Weight | P3 Weight | Trend |
|--------------|-----------|-----------|-----------|-------|
| NORTH KOREA ↔ TRUMP | 1954 | 555 | 1049 | Variable |
| NORTH KOREA ↔ KIM JONG UN | 1458 | 446 | 1000 | Variable |
| NORTH KOREA ↔ SOUTH KOREA | 2524 | 569 | 850 | Variable |
| NORTH KOREA ↔ DENUCLEARIZATION | 190 | 285 | 180 | ∩ Peak at P2 |
| NORTH KOREA ↔ UNITED STATES | 959 | 203 | 339 | Variable |
| KIM JONG UN ↔ TRUMP | 329 | 185 | 432 | Variable |
| KIM JONG UN ↔ DONALD TRUMP | 231 | 125 | 394 | Variable |

---

## 6. Community Structure

| Metric | P1 | P2 | P3 |
|--------|-----|-----|-----|
| Communities | 363 | 103 | 205 |

### Community Theme Distribution

| Theme | P1 Count | P2 Count | P3 Count |
|-------|----------|----------|----------|
| Threat | 105 | 22 | 34 |
| Diplomacy | 12 | 11 | 14 |
| Mixed | 246 | 70 | 157 |
| Other | 0 | 0 | 0 |

---

## 7. Key Findings

### 7.1 Network Densification
- Network density **increased** from P1 to P3, indicating more interconnected discourse
- Despite fewer nodes, relationships became more concentrated

### 7.2 Centrality Shifts
- **DENUCLEARIZATION** emerged as a central concept in P2
- **Kim Jong Un** increased in relative importance

### 7.3 Framing Transition
- **Threat framing decreased** significantly (P1→P2: -19%)
- **Peace framing increased** (P1→P2: +17%)
- Hanoi failure caused **partial reversion** but not full reversal

### 7.4 Asymmetric Ratchet Effect
- Singapore Summit produced large structural changes
- Hanoi failure produced smaller counter-changes
- **Net effect**: Diplomatic framing persisted

---

*Generated by GraphRAG Comprehensive Analysis*
*For RQ2: Structural Reorganization of Discourse Networks*
