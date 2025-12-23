# GraphRAG Comprehensive Network Analysis Report

## Executive Summary

This report presents detailed network analysis results for RQ2: "How do diplomatic events restructure the organization of online discourse?"

---

## 1. Network Topology Metrics

| Metric | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) | P1→P2 Δ | P2→P3 Δ |
|--------|-------------------|---------------------|-----------------|---------|---------|
| Nodes | 2,656 | 1,043 | 879 | -1,613 | -164 |
| Edges | 4,552 | 1,726 | 1,429 | -2,826 | -297 |
| Density | 0.0013 | 0.0032 | 0.0037 | +0.0019 | +0.0005 |
| Avg Degree | 3.4277 | 3.3097 | 3.2514 | -0.1180 | -0.0583 |
| Clustering | 0.2149 | 0.1980 | 0.2098 | -0.0169 | +0.0118 |
| Components | 25 | 23 | 16 | -2 | -7 |

---

## 2. Centrality Analysis

### 2.1 Degree Centrality (Top 10 per Period)

| Rank | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) |
|------|-------------------|---------------------|-----------------|
| 1 | NORTH KOREA (0.540) | NORTH KOREA (0.429) | NORTH KOREA (0.468) |
| 2 | TRUMP (0.127) | TRUMP (0.159) | TRUMP (0.167) |
| 3 | SOUTH KOREA (0.114) | SOUTH KOREA (0.114) | KIM JONG UN (0.138) |
| 4 | KIM JONG UN (0.074) | KIM JONG UN (0.103) | DONALD TRUMP (0.132) |
| 5 | DONALD TRUMP (0.073) | DONALD TRUMP (0.081) | KIM JONG-UN (0.095) |
| 6 | UNITED STATES (0.055) | US (0.052) | SOUTH KOREA (0.083) |
| 7 | KIM JONG-UN (0.052) | SINGAPORE SUMMIT (0.050) | US (0.041) |
| 8 | CHINA (0.046) | DENUCLEARIZATION (0.049) | TRUMP ADMINISTRATION (0.041) |
| 9 | US (0.043) | KIM JONG-UN (0.048) | UNITED STATES (0.039) |
| 10 | DPRK (0.032) | UNITED STATES (0.044) | CHINA (0.033) |

### 2.2 PageRank (Top 10 per Period)

| Rank | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) |
|------|-------------------|---------------------|-----------------|
| 1 | NORTH KOREA (0.2534) | NORTH KOREA (0.1887) | NORTH KOREA (0.2014) |
| 2 | SOUTH KOREA (0.0489) | TRUMP (0.0602) | KIM JONG UN (0.0691) |
| 3 | TRUMP (0.0427) | KIM JONG UN (0.0488) | TRUMP (0.0615) |
| 4 | KIM JONG UN (0.0312) | SOUTH KOREA (0.0393) | DONALD TRUMP (0.0404) |
| 5 | DONALD TRUMP (0.0210) | DONALD TRUMP (0.0282) | SOUTH KOREA (0.0317) |
| 6 | UNITED STATES (0.0191) | DENUCLEARIZATION (0.0215) | KIM JONG-UN (0.0299) |
| 7 | CHINA (0.0181) | KIM JONG-UN (0.0199) | UNITED STATES (0.0146) |
| 8 | KIM JONG-UN (0.0175) | SINGAPORE SUMMIT (0.0172) | SANCTIONS (0.0105) |
| 9 | US (0.0131) | US (0.0145) | CHINA (0.0100) |
| 10 | RUSSIA (0.0117) | UNITED STATES (0.0141) | TRUMP ADMINISTRATION (0.0099) |

---

## 3. Entity Type Evolution

| Type | P1 Count | P1 % | P2 Count | P2 % | P3 Count | P3 % |
|------|----------|------|----------|------|----------|------|
| ANIMAL | 0 | 0.0% | 1 | 0.1% | 0 | 0.0% |
| CITY | 10 | 0.4% | 3 | 0.3% | 5 | 0.5% |
| COUNTRY | 149 | 5.3% | 94 | 8.4% | 40 | 4.3% |
| CULTURAL EVENT | 0 | 0.0% | 1 | 0.1% | 0 | 0.0% |
| CULTURAL ITEM | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| CULTURAL PHENOMENON | 2 | 0.1% | 0 | 0.0% | 0 | 0.0% |
| CURRENCY | 0 | 0.0% | 1 | 0.1% | 0 | 0.0% |
| DOCUMENT | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| ETHNIC GROUP | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| EVENT | 654 | 23.2% | 230 | 20.6% | 224 | 24.0% |
| FOOD | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| GEO | 62 | 2.2% | 22 | 2.0% | 23 | 2.5% |
| LOCATION | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| MILITARY | 3 | 0.1% | 0 | 0.0% | 0 | 0.0% |
| MILITARY ASSET | 3 | 0.1% | 0 | 0.0% | 0 | 0.0% |
| MOVIE | 0 | 0.0% | 1 | 0.1% | 0 | 0.0% |
| MYTHICAL CREATURE | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| ORGANIZATION | 419 | 14.8% | 184 | 16.5% | 151 | 16.1% |
| PERSON | 457 | 16.2% | 172 | 15.4% | 169 | 18.1% |
| POLICY | 546 | 19.3% | 217 | 19.4% | 157 | 16.8% |
| REGION | 1 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| STATE | 0 | 0.0% | 1 | 0.1% | 0 | 0.0% |
| WEAPON | 264 | 9.4% | 79 | 7.1% | 75 | 8.0% |

---

## 4. Relationship Framing Analysis

| Frame | P1 % | P2 % | P3 % | P1→P2 Δ | P2→P3 Δ |
|-------|------|------|------|---------|---------|
| **Threat** | 61.5% | 42.4% | 40.9% | **-19.0%** | **-1.5%** |
| **Peace** | 7.8% | 24.7% | 18.6% | **+16.9%** | **-6.1%** |
| **Neutral** | 30.8% | 32.9% | 40.5% | **+2.1%** | **+7.7%** |

---

## 5. Key Actor Dyad Analysis

| Relationship | P1 Weight | P2 Weight | P3 Weight | Trend |
|--------------|-----------|-----------|-----------|-------|
| NORTH KOREA ↔ TRUMP | 2419 | 706 | 582 | ↓ Declining |
| NORTH KOREA ↔ KIM JONG UN | 2393 | 783 | 915 | Variable |
| NORTH KOREA ↔ SOUTH KOREA | 3421 | 661 | 689 | Variable |
| NORTH KOREA ↔ DENUCLEARIZATION | 280 | 532 | 150 | ∩ Peak at P2 |
| NORTH KOREA ↔ UNITED STATES | 1428 | 247 | 301 | Variable |
| KIM JONG UN ↔ TRUMP | 427 | 412 | 494 | Variable |
| KIM JONG UN ↔ DONALD TRUMP | 408 | 359 | 396 | Variable |

---

## 6. Community Structure

| Metric | P1 | P2 | P3 |
|--------|-----|-----|-----|
| Communities | 354 | 153 | 107 |

### Community Theme Distribution

| Theme | P1 Count | P2 Count | P3 Count |
|-------|----------|----------|----------|
| Threat | 108 | 40 | 22 |
| Diplomacy | 21 | 25 | 7 |
| Mixed | 225 | 88 | 78 |
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
