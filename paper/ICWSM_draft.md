# Summit Diplomacy and Social Media Framing

# A Causal Analysis of U.S.-North Korea Summits on Reddit Public Opinion

**Target Venue**: ICWSM 2025  
**Authors**: Jun Sin, [Co-authors TBD]  
**Affiliation**: University of Texas at Austin  

---

## Abstract

High-stakes diplomatic summits can dramatically shift public perception of foreign adversaries, yet the causal mechanisms underlying these shifts remain poorly understood in online discourse. This study examines how the 2018 Singapore Summit and the 2019 Hanoi Summit failure affected public opinion framing of North Korea on Reddit. Using a novel combination of LLM-based framing classification validated by domain experts (military officers) and a Difference-in-Differences design with multiple control countries (China, Iran, Russia), we analyze over 100,000 Reddit posts from 2017-2019. Our findings reveal [KEY FINDINGS TO BE ADDED]. This work contributes to understanding the "rally-round-the-flag" effect in digital spaces and demonstrates a rigorous methodology for causal inference in computational social science.

---

## 1. Introduction

The historic 2018 Singapore Summit between U.S. President Donald Trump and North Korean leader Kim Jong Un marked the first-ever meeting between sitting leaders of the two nations. This diplomatic breakthrough, followed by the collapsed Hanoi Summit in February 2019, provides a unique natural experiment to study how high-stakes international events shape online public opinion.

### Research Questions

**RQ1**: How do high-stakes diplomatic summits affect public opinion framing on social media, and do failed negotiations lead to symmetric reversals of opinion gains?

**RQ2**: Can social media discourse predict or reflect the "rally-round-the-flag" effect during diplomatic engagement cycles?

### Contributions

1. **Methodological**: We introduce a validated pipeline combining LLM-based framing analysis with human expert annotation (military officers) for domain-specific content classification.

2. **Empirical**: We provide causal evidence of diplomatic events' impact on online discourse using Difference-in-Differences with multiple control groups.

3. **Theoretical**: We extend the "rally-round-the-flag" literature to social media contexts, examining asymmetry between diplomatic successes and failures.

---

## 2. Related Work

### 2.1 Rally-Round-the-Flag Effect

The rally-round-the-flag effect describes the phenomenon where public approval of national leaders increases during international crises or major foreign policy events (Mueller, 1970). While extensively studied in traditional polling contexts, its manifestation in social media discourse remains underexplored.

### 2.2 Media Framing

Framing theory posits that how information is presented significantly influences audience interpretation (Entman, 1993). In international relations coverage, common frames include threat, diplomacy, economic, and humanitarian perspectives (Iyengar, 1991).

### 2.3 Social Media and Public Opinion

Reddit, as a major discussion platform, has been increasingly used to study public opinion dynamics (Baumgartner et al., 2020). Unlike Twitter, Reddit's structure enables longer-form discussion and topical organization through subreddits.

### 2.4 LLMs for Content Analysis

Recent work has demonstrated the effectiveness of large language models for text classification tasks previously requiring human annotation (Gilardi et al., 2023). However, validation against domain experts remains critical for specialized content.

---

## 3. Data

### 3.1 Data Collection

We collected Reddit posts from January 2017 to December 2019 using the Arctic Shift API, targeting major news and politics subreddits:

- r/worldnews
- r/politics  
- r/news
- r/geopolitics
- r/NeutralPolitics

### 3.2 Treatment and Control Groups

| Group | Topic | Posts | Role |
|-------|-------|-------|------|
| **Treatment** | North Korea | ~12,000 | Primary analysis |
| **Control 1** | China | ~13,000 | Trade war confounder |
| **Control 2** | Iran | ~8,000 | Nuclear deal comparison |
| **Control 3** | Russia | ~16,000 | Investigation confounder |

### 3.3 Key Events Timeline

| Date | Event | Expected Effect |
|------|-------|-----------------|
| 2018-03-08 | Summit Announcement | Initial positive shift |
| 2018-06-12 | Singapore Summit | Peak diplomatic framing |
| 2019-02-27-28 | Hanoi Summit (Failed) | Potential reversal |

### 3.4 Analysis Periods

- **Period 1**: Pre-Singapore (2017.01 - 2018.05)
- **Period 2**: Singapore-Hanoi (2018.06 - 2019.02)  
- **Period 3**: Post-Hanoi (2019.03 - 2019.12)

---

## 4. Method

### 4.1 Human Annotation Benchmark

To validate our automated classification approach, we developed a human annotation benchmark:

**Annotators**: Two military officers with expertise in North Korean affairs and international security.

**Sample**: [N] posts stratified across:

- All four time periods
- All five framing categories
- Both high and low engagement posts

**Annotation Protocol**:

1. Independent classification by each annotator
2. Disagreement resolution through discussion
3. Final gold-standard labels

**Inter-rater Reliability**: Cohen's Kappa = [TO BE CALCULATED]

### 4.2 Framing Categories

| Frame | Description | Scale Score |
|-------|-------------|-------------|
| THREAT | Military danger, nuclear weapons, war | -2 |
| ECONOMIC | Sanctions, trade, economic impact | -1 |
| NEUTRAL | Factual information, no clear framing | 0 |
| HUMANITARIAN | Human rights, refugees, citizens | +1 |
| DIPLOMACY | Negotiations, peace, cooperation | +2 |

### 4.3 LLM Classification

**Model**: GPT-4o-mini with structured JSON output

**Prompt Design**:

```
Classify this Reddit post about international politics into ONE of these frames:
- THREAT: Focus on military danger, nuclear weapons, missiles, war
- DIPLOMACY: Focus on negotiations, talks, peace, cooperation
- NEUTRAL: Factual information without clear framing
- ECONOMIC: Focus on sanctions, trade, economic aspects
- HUMANITARIAN: Focus on human rights, refugees, citizens
```

**Validation Metrics**:

- Overall Accuracy vs. Human Benchmark
- Per-category Precision, Recall, F1
- Confusion Matrix Analysis

### 4.4 Causal Inference: Difference-in-Differences

We employ a Difference-in-Differences (DID) design to estimate the causal effect of diplomatic events on framing:

$$\text{Framing}_{it} = \alpha + \beta_1 \text{Treatment}_i + \beta_2 \text{Post}_t + \beta_3 (\text{Treatment}_i \times \text{Post}_t) + \epsilon_{it}$$

Where:

- $\beta_3$ represents the DID estimator (causal effect)
- Treatment = North Korea posts
- Control = China/Iran/Russia posts
- Post = After intervention date

**Assumption Validation**:

- Parallel trends test in pre-intervention period
- Placebo tests with alternative intervention dates

### 4.5 Sentiment Analysis

As a complementary measure to cognitive framing, we employ sentiment analysis to capture emotional tone:

**Model**: RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)

**Scale**: Continuous score from -1 (Negative) to +1 (Positive)

**Rationale**: While framing captures *how* topics are discussed (threat vs. diplomacy frame), sentiment captures the *emotional valence* of discussion. Together, they provide a more complete picture of public opinion dynamics.

---

## 5. Results

### 5.1 Human Annotation Results

[TO BE ADDED: Inter-rater reliability, distribution of labels]

### 5.2 LLM Validation

[TO BE ADDED: Agreement metrics with human benchmark]

### 5.3 RQ1: Summit Effects on Sentiment

#### Period Definition

Consistent with the framing analysis, we apply clean periods excluding transition months:

- **P1**: 2017.01 - 2018.02 (Pre-Announcement)
- **P2**: 2018.06 - 2019.01 (Singapore-Hanoi)
- **P3**: 2019.03 - 2019.12 (Post-Hanoi)
- *Excluded*: 2018.03-05, 2019.02 (Transition periods)

#### Parallel Trends Validation (Monthly Aggregated)

| Comparison | Control | Pre-period | P-value | Satisfied |
|------------|---------|------------|---------|-----------|
| P1→P2 | **China** | P1 | **0.99** | **✓** |
| P1→P2 | **Iran** | P1 | **0.83** | **✓** |
| P1→P2 | **Russia** | P1 | **0.20** | **✓** |
| P2→P3 | **China** | P2 | **0.69** | **✓** |
| P2→P3 | **Iran** | P2 | **0.75** | **✓** |
| P2→P3 | **Russia** | P2 | **0.88** | **✓** |

All parallel trends tests pass (p > 0.05), validating the DID design for all control groups.

#### Singapore Summit Effect (P1→P2)

| Control | DID Estimate | P-value | Parallel Trends |
|---------|--------------|---------|-----------------|
| **China** | **+0.21** | <0.0001 | **✓ Satisfied** |
| **Iran** | **+0.10** | <0.0001 | **✓ Satisfied** |
| **Russia** | **+0.14** | <0.0001 | **✓ Satisfied** |

**Interpretation**: The Singapore Summit led to a statistically significant positive shift in sentiment (+0.10 to +0.21) toward North Korea across all control group comparisons.

#### Hanoi Summit Effect (P2→P3)

| Control | DID Estimate | P-value | Parallel Trends |
|---------|--------------|---------|-----------------|
| **China** | **-0.11** | <0.0001 | **✓ Satisfied** |
| **Iran** | **-0.06** | 0.001 | **✓ Satisfied** |
| **Russia** | **-0.12** | <0.0001 | **✓ Satisfied** |

**Interpretation**: The failed Hanoi Summit led to a statistically significant negative shift in sentiment (-0.06 to -0.12), partially reversing the gains from the Singapore Summit.

### 5.4 RQ2: Rally-Round-the-Flag Patterns

The data reveals an asymmetric rally effect:

1. **Summit Success Effect**: Singapore Summit produced a sentiment improvement of +0.10 to +0.21
2. **Summit Failure Effect**: Hanoi Summit produced a sentiment decline of -0.06 to -0.12
3. **Net Effect**: The positive gains slightly outweigh the negative reversal

This suggests that while diplomatic failures cause sentiment reversals, the gains from successful summits are not fully erased—indicating a partial "ratchet effect" in public opinion.

### 5.5 RQ1: Summit Effects on Framing

#### Period Definition and Methodology

To avoid contamination from anticipation effects, we define analysis periods excluding transition months:

| Period | Date Range | Rationale |
|--------|------------|-----------|
| **P1: Pre-Announcement** | 2017.01 - 2018.02 | Before Trump-Kim summit announcement (2018.03.08) |
| *Transition* | 2018.03 - 2018.05 | Anticipation period (excluded) |
| **P2: Singapore-Hanoi** | 2018.06 - 2019.01 | Post-Singapore, Pre-Hanoi meeting month |
| *Transition* | 2019.02 | Hanoi Summit month (excluded) |
| **P3: Post-Hanoi** | 2019.03 - 2019.12 | Post-summit collapse period |

This approach ensures clean pre-treatment periods for parallel trends testing.

#### Parallel Trends Validation (Monthly Aggregated)

| Comparison | Control | Pre-period | P-value | Satisfied |
|------------|---------|------------|---------|-----------|
| P1→P2 | **China** | P1 | **0.12** | **✓** |
| P1→P2 | **Iran** | P1 | **0.77** | **✓** |
| P1→P2 | Russia | P1 | 0.03 | ✗ |
| P2→P3 | **China** | P2 | **0.41** | **✓** |
| P2→P3 | **Iran** | P2 | **0.79** | **✓** |
| P2→P3 | **Russia** | P2 | **0.51** | **✓** |

With clean periods, China and Iran satisfy parallel trends for P1→P2. All controls satisfy for P2→P3.

#### Singapore Summit Effect on Framing (P1→P2)

| Control | DID Estimate | P-value | Parallel Trends |
|---------|--------------|---------|-----------------|
| **China** | **+1.28** | **<0.0001** | **✓ Satisfied** |
| **Iran** | **+0.85** | **<0.0001** | **✓ Satisfied** |
| Russia | +1.04 | <0.0001 | ✗ Violated |

**Primary result**: Using China and Iran as controls (both satisfying parallel trends), the Singapore Summit produced a DID estimate of **+0.85 to +1.28** (p<0.0001), indicating a substantial shift toward diplomatic framing.

#### Hanoi Summit Effect on Framing (P2→P3)

| Control | DID Estimate | P-value | Parallel Trends |
|---------|--------------|---------|-----------------|
| **China** | **-0.88** | **<0.0001** | **✓ Satisfied** |
| **Iran** | **-0.30** | **0.003** | **✓ Satisfied** |
| **Russia** | **-0.83** | **<0.0001** | **✓ Satisfied** |

**Interpretation**: The failed Hanoi Summit led to a significant reversal in framing (DID = -0.30 to -0.88), with NK-related discussions returning toward more threat-oriented framing across all control group comparisons.

### 5.6 Visualizations

#### Figure 1: Research Timeline and Key Events

- X-axis: Time (2017.01 - 2019.12)
- Annotations: Summit Announcement, Singapore Summit, Hanoi Summit
- Period shading: Pre-Singapore (gray), Singapore-Hanoi (green), Post-Hanoi (red)

#### Figure 2: Monthly Framing Score Trends

- Line plot: NK vs Control groups (China, Iran, Russia) over time
- Y-axis: Mean framing score (-2 to +2)
- Vertical lines: Key event markers
- Confidence intervals shown

#### Figure 3: Framing Distribution by Period (NK)

- Stacked bar chart or grouped bar chart
- 3 periods × 5 frame categories
- Shows shift from THREAT-dominant to DIPLOMACY-increase

#### Figure 4: DID Visualization

- 2×2 panel: Treatment (NK) vs Control, Pre vs Post
- Shows parallel trends assumption and treatment effect
- Error bars for confidence intervals

#### Figure 5: Frame Category Heatmap

- X-axis: Months (2017.01 - 2019.12)
- Y-axis: Frame categories (THREAT, DIPLOMACY, NEUTRAL, ECONOMIC, HUMANITARIAN)
- Color intensity: Proportion of posts in each category

#### Figure 6: LLM vs Human Agreement (Confusion Matrix)

- 5×5 confusion matrix
- Human labels (rows) vs LLM predictions (columns)
- Per-category accuracy highlighted

#### Figure 7: Sentiment vs Framing Correlation

- Scatter plot: X = Framing Score, Y = Sentiment Score
- Color by period
- Correlation coefficient and trend line

#### Figure 8: Monthly Sentiment Trends

- Line plot: NK vs Control groups
- Y-axis: Mean sentiment (-1 to +1)
- Comparison with framing trends (Figure 2)

---

## 6. Discussion

### 6.1 Asymmetry of Diplomatic Effects

[Discussion of whether positive effects from successful summits are reversed by diplomatic failures]

### 6.2 Persistence of Opinion Shifts

[Analysis of how long framing changes persist after events]

### 6.3 Limitations

1. **Platform Specificity**: Results may not generalize beyond Reddit
2. **English-only**: Limited to English-language discourse
3. **Observational Design**: Despite DID, unmeasured confounders possible

### 6.4 Future Work

- Extension to Twitter/X and other platforms
- Real-time monitoring applications
- Cross-cultural comparison (Korean-language forums)

---

## 7. Conclusion

This study provides causal evidence that high-stakes diplomatic summits significantly affect public opinion framing on social media. The Singapore Summit produced a large positive shift in North Korea framing, while the Hanoi Summit failure [RESULTS TBD]. Our validated LLM+DID methodology offers a rigorous framework for studying the digital manifestations of international events.

---

## References

[TO BE ADDED]

---

## Appendix

### A. Full Prompt for LLM Classification

### B. Human Annotation Guidelines

### C. Additional Statistical Results

### D. Robustness Checks
