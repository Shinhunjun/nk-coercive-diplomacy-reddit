# The Impact of North Korea's Coercive Diplomacy Strategy on U.S. Online Public Opinion: A Reddit Discourse Analysis

---

**Jun Sin**

Research Collaboration with University of Texas at Austin (Professor Mohit Singhal)

---

## Abstract

This study empirically analyzes the impact of North Korea's coercive diplomacy strategy on U.S. online public opinion. Using 706 posts and 9,460 comments collected from Reddit between January 2017 and June 2019, we compared discourse changes between the tension period (January 2017 - February 2018) and the diplomacy period (June 2018 - June 2019). Our methodology included BERT sentiment analysis, GPT-4o-mini-based framing classification, Interrupted Time Series (ITS) analysis, and GraphRAG-based knowledge graph analysis.

Key findings include: (1) sentiment toward North Korea significantly improved during the diplomacy period compared to the tension period (p < 0.001); (2) the threat frame decreased from 70.0% to 40.7% while the diplomacy frame increased from 8.7% to 31.3% (χ² = 33.17, p < 0.001); and (3) the summit announcement in March 2018 produced an immediate positive effect on public opinion (β₂ = +0.293, p = 0.044). These findings suggest that North Korea's "threat-then-negotiate" strategy had a tangible impact on U.S. public opinion, providing empirical support for coercive diplomacy theory.

**Keywords**: Coercive Diplomacy, North Korea, Public Opinion, Reddit, Sentiment Analysis, Interrupted Time Series Analysis

---

## 1. Introduction

### 1.1 Research Background

The Korean Peninsula experienced dramatic changes between 2017 and 2018. In 2017, North Korea conducted its sixth nuclear test and launched the Hwasong-15 intercontinental ballistic missile (ICBM). President Trump responded with threats of "fire and fury," escalating tensions to unprecedented levels. However, following the 2018 PyeongChang Winter Olympics, inter-Korean relations began to thaw, culminating in the historic Singapore Summit between the U.S. and North Korea in June 2018.

This dramatic shift exemplifies North Korea's "coercive diplomacy" strategy. Coercive diplomacy refers to the use of military threats to compel an adversary to change its behavior (George, 1991; Schelling, 1966). North Korea demonstrated its nuclear and missile capabilities to project an existential threat to the United States, then pivoted to offering denuclearization negotiations in exchange for sanctions relief and regime guarantees.

While existing research has focused primarily on policymaker responses and media coverage, empirical studies on general public opinion remain limited. Given that social media-generated public opinion can influence policy decisions, understanding these dynamics carries both academic and policy significance.

### 1.2 Research Purpose and Questions

This study aims to quantitatively analyze the impact of North Korea's coercive diplomacy strategy on U.S. public opinion. Specifically, we address the following research questions:

**RQ1**: Did U.S. public sentiment toward North Korea change during the transition from the tension period to the diplomacy period?

**RQ2**: Did the framing of North Korea-related discourse shift from "threat" to "diplomacy"?

**RQ3**: Was the summit announcement the direct cause of opinion change, or were other factors responsible?

**RQ4**: How did the cognitive structure (knowledge graph) of North Korea perceptions differ between periods?

### 1.3 Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews literature on coercive diplomacy theory and social media opinion analysis. Section 3 describes data collection and analytical methods. Section 4 presents results. Section 5 discusses implications and limitations. Section 6 concludes.

---

## 2. Theoretical Background

### 2.1 Coercive Diplomacy Theory

Coercive diplomacy employs threats of force to induce behavioral change in an adversary (George, 1991). Unlike traditional deterrence, which aims to maintain the status quo, coercive diplomacy actively seeks to alter the opponent's policies (Schelling, 1966).

North Korea's strategy toward the United States exemplifies this pattern. First, nuclear and missile demonstrations establish credible threats. Second, offers of negotiation create opportunities for tension reduction. Third, at the negotiating table, North Korea pursues concrete benefits such as sanctions relief and security guarantees. The 2017-2018 period represents a classic case of this strategy, with maximum tension followed by a rapid pivot to summitry.

### 2.2 Public Opinion and Foreign Policy

In democracies, public opinion plays a significant role in foreign policy decisions (Holsti, 1992; Page & Shapiro, 1983). In the United States, presidential approval ratings on foreign policy affect policy sustainability. While U.S. public opinion toward North Korea has traditionally been negative (Gallup, 2018), the summits raised questions about potential shifts.

### 2.3 Social Media-Based Opinion Analysis

Social media offers a complementary data source to traditional polling (Jungherr, 2015). Reddit is particularly valuable given its high proportion of U.S. users (approximately 50%) and anonymous posting, which encourages candid expression. While prior research has focused on Twitter and Facebook, Reddit's long-form posts and threaded discussions are well-suited for analyzing complex foreign policy issues.

---

## 3. Methodology

### 3.1 Data Collection

#### 3.1.1 Data Source

We collected North Korea-related posts and comments from Reddit. Reddit ranks among the top 10 U.S. websites by traffic, with over 500 million monthly active users. Target subreddits included:

| Subreddit | Description | Subscribers |
|-----------|-------------|-------------|
| r/worldnews | International news | 32M+ |
| r/geopolitics | International politics analysis | 600K+ |
| r/politics | U.S. politics | 8M+ |
| r/northkorea | North Korea-focused | 50K+ |
| r/korea | Korea general | 200K+ |
| r/news | General news | 25M+ |
| r/AskAnAmerican | Q&A forum | 300K+ |

#### 3.1.2 Collection Period and Scale

Data was collected via the Arctic Shift API for January 2017 through June 2019. Search terms included 'North Korea,' 'DPRK,' 'Kim Jong Un,' 'Korean Peninsula,' and 'US Korea.'

| Category | Posts | Comments |
|----------|-------|----------|
| Tension Period (2017.01-2018.02) | 380 | 5,123 |
| Diplomacy Period (2018.06-2019.06) | 326 | 4,337 |
| **Total** | **706** | **9,460** |

### 3.2 Research Design

We employed a quasi-experimental design comparing two periods:

**Period 1 - Tension Period**: January 2017 - February 2018
- Trump inauguration (January 20, 2017)
- "Fire and Fury" speech (August 8, 2017)
- North Korea's 6th nuclear test (September 3, 2017)
- Hwasong-15 ICBM launch (November 29, 2017)

**Period 2 - Diplomacy Period**: June 2018 - June 2019
- Singapore Summit (June 12, 2018)
- Hanoi Summit (February 27-28, 2019)
- Panmunjom meeting (June 30, 2019)

**Intervention Point**: March 8, 2018 - Trump's acceptance of summit invitation

### 3.3 Analytical Methods

#### 3.3.1 BERT Sentiment Analysis (H1)

We employed a BERT-based model for sentiment analysis, specifically using Hugging Face's `nlptown/bert-base-multilingual-uncased-sentiment` model. Sentiment scores were normalized to a scale of -1 (very negative) to +1 (very positive).

Statistical significance was tested using independent samples t-tests and Mann-Whitney U tests. Effect size was measured using Cohen's d.

#### 3.3.2 LLM-Based Framing Classification (H2)

OpenAI's GPT-4o-mini was used to classify post framing. Categories included:

| Frame | Definition |
|-------|------------|
| **THREAT** | Portrays North Korea as military threat |
| **DIPLOMACY** | Emphasizes negotiation, dialogue, cooperation |
| **NEUTRAL** | Neutral information delivery |
| **ECONOMIC** | Emphasizes economic aspects |
| **HUMANITARIAN** | Humanitarian perspective |

We randomly sampled 150 posts per period (300 total) for classification. Distribution differences were tested using chi-square (χ²) tests.

#### 3.3.3 Interrupted Time Series Analysis (H3)

To test whether the summit announcement caused opinion change, we conducted Interrupted Time Series (ITS) analysis. ITS compares time series data before and after a specific intervention to estimate causal effects (Bernal et al., 2017).

The model specification was:

$$Y_t = \beta_0 + \beta_1 T + \beta_2 X_t + \beta_3 (T \times X_t) + \epsilon_t$$

Where:
- $Y_t$: Monthly average sentiment score
- $T$: Time variable (months)
- $X_t$: Intervention variable (0 before, 1 after)
- $\beta_1$: Pre-intervention slope
- $\beta_2$: Level change at intervention
- $\beta_3$: Slope change post-intervention

#### 3.3.4 Knowledge Graph Analysis (H4)

Microsoft GraphRAG was used to construct knowledge graphs for each period. GraphRAG employs LLMs to automatically extract entities and relationships from text, using community detection to identify thematic structures.

We compared the two periods' knowledge graphs on:
1. Changes in key entities
2. Changes in relationship types (threat/cooperation)
3. Changes in Kim Jong Un's network connections

---

## 4. Results

### 4.1 H1: Sentiment Change

> **Hypothesis**: Sentiment toward North Korea will be more positive (or less negative) during the diplomacy period compared to the tension period.

#### 4.1.1 Descriptive Statistics

| Data Type | Tension Period Mean | Diplomacy Period Mean | Change |
|-----------|--------------------|-----------------------|--------|
| Posts | -0.475 | -0.245 | **+0.230** |
| Comments | -0.546 | -0.472 | **+0.074** |

Mean sentiment was negative in both periods, indicating persistently negative views of North Korea. However, negativity significantly decreased during the diplomacy period.

#### 4.1.2 Statistical Tests

| Data Type | t-test p-value | Mann-Whitney p-value | Cohen's d |
|-----------|----------------|---------------------|-----------|
| Posts | p = 0.0005*** | p = 0.0004*** | d = 0.26 |
| Comments | - | p < 0.001*** | d = 0.09 |

*Note: *** p < 0.001*

Sentiment differences between periods were statistically significant for both posts and comments (p < 0.001). Cohen's d values indicated small-to-medium effects for posts (0.26) and small effects for comments (0.09).

#### 4.1.3 Interpretation

**H1 is supported.** U.S. public opinion toward North Korea improved significantly during the diplomacy period. However, mean sentiment remained negative (-0.245), indicating that fundamental distrust persisted. This suggests that diplomatic gestures can improve opinion to some degree, but long-standing distrust is not easily overcome.

---

### 4.2 H2: Framing Change

> **Hypothesis**: 'Threat/war' frames will decrease while 'diplomacy/cooperation' frames will increase.

#### 4.2.1 LLM-Based Framing Classification Results

Using OpenAI GPT-4o-mini, we classified 150 posts per period (300 total):

| Frame Type | Tension (n=150) | Diplomacy (n=150) | Change |
|------------|-----------------|-------------------|--------|
| **THREAT** | 70.0% (105) | 40.7% (61) | **-29.3%p** |
| **DIPLOMACY** | 8.7% (13) | 31.3% (47) | **+22.7%p** |
| NEUTRAL | 16.7% (25) | 20.7% (31) | +4.0%p |
| ECONOMIC | 2.0% (3) | 4.7% (7) | +2.7%p |
| HUMANITARIAN | 2.7% (4) | 2.7% (4) | 0%p |

#### 4.2.2 Statistical Test

Chi-square test results:
- χ² = 33.17
- **p = 0.000001** (p < 0.001)

The difference in framing distribution between periods was highly statistically significant.

#### 4.2.3 Keyword Frequency Analysis (Supplementary)

| Keyword Type | Tension Frequency | Diplomacy Frequency | Change |
|--------------|-------------------|---------------------|--------|
| Threat keywords | 623 | 327 | **-47.5%** |
| Peace keywords | 37 | 109 | **+194.6%** |
| Threat/Peace ratio | 16.84:1 | 3:1 | **-82.2%** |

*Threat keywords: nuclear, missile, threat, war, attack, etc.*
*Peace keywords: peace, dialogue, summit, diplomacy, negotiation, etc.*

#### 4.2.4 Interpretation

**H2 is supported.** The diplomacy period saw a dramatic framing shift, with threat frames decreasing by 29.3 percentage points and diplomacy frames increasing by 22.7 percentage points. This indicates that the summits changed not just 'feelings' about North Korea, but the very 'lens' through which North Korea was perceived.

Notably, threat framing remained highest (40.7%) even during the diplomacy period, suggesting that threat perceptions remained dominant despite diplomatic overtures.

---

### 4.3 H3: Causal Effect (ITS Analysis)

> **Hypothesis**: The summit announcement was the direct cause of opinion change.

#### 4.3.1 Model Estimation Results

| Coefficient | Estimate | Std. Error | t-statistic | p-value | Interpretation |
|-------------|----------|------------|-------------|---------|----------------|
| β₀ (Intercept) | -0.487 | 0.078 | -6.24 | <0.001 | Initial sentiment level |
| β₁ (Pre-trend) | -0.017 | 0.012 | -1.42 | 0.169 | No natural trend |
| **β₂ (Level change)** | **+0.293** | **0.136** | **2.15** | **0.044** | **Significant improvement** |
| β₃ (Slope change) | +0.014 | 0.015 | 0.93 | 0.362 | No sustained effect |

#### 4.3.2 Key Findings

1. **β₁ (Pre-intervention trend) = -0.017, not significant**
   - No natural improvement trend existed before the summit announcement.
   - This rejects the alternative hypothesis that "sentiment naturally improved over time."

2. **β₂ (Immediate change) = +0.293, p = 0.044**
   - Sentiment immediately improved by 0.293 at the summit announcement.
   - The probability of this occurring by chance is 4.4%, meeting the 95% confidence threshold.

3. **β₃ (Post-intervention slope) = +0.014, not significant**
   - No sustained improvement trend emerged after the summit.
   - The effect appeared as an immediate "jump" rather than accelerating improvement.

#### 4.3.3 Counterfactual Analysis

Difference between observed values and predicted values had there been no summit:

| Period | Actual Mean | Counterfactual Prediction | Difference (Causal Effect) |
|--------|-------------|---------------------------|---------------------------|
| 2018.06-2019.06 | -0.245 | -0.521 | **+0.276** |

Without the summit, sentiment would have remained at approximately -0.521. The difference of +0.276 represents the net effect attributable to the summit.

#### 4.3.4 Interpretation

**H3 is supported.** ITS analysis confirms that the summit announcement directly caused opinion improvement. This provides evidence of causation rather than mere correlation.

However, the non-significant β₃ suggests that the summit's effect may have been temporary. Initial optimism may have waned over time.

---

### 4.4 H4: Knowledge Graph Structure Change

> **Hypothesis**: The structure of the North Korea knowledge graph will differ qualitatively between tension and diplomacy periods.

#### 4.4.1 Entity Analysis

**Changes in Kim Jong Un's Network:**

| Tension Period Links | New Diplomacy Period Links |
|---------------------|---------------------------|
| NORTH KOREA | **TRUMP** (new) |
| MISSILE LAUNCH | **PANMUNJOM** |
| NUCLEAR PROGRAM | **JOINT STATEMENTS** |
| KIM JONG-NAM | **VLADIVOSTOK** |

During the tension period, Kim Jong Un was primarily linked to military elements (missiles, nuclear program). During the diplomacy period, links to diplomatic elements (Trump, summit locations) increased.

#### 4.4.2 Trump Entity Emergence

| Metric | Tension Period | Diplomacy Period |
|--------|----------------|------------------|
| Trump connections | 5 | **18** |
| Entity ranking | Outside top 10 | **3rd** |

President Trump emerged as a central entity during the diplomacy period, becoming central to North Korea discourse.

#### 4.4.3 Relationship Type Changes

| Relationship Type | Tension Period | Diplomacy Period |
|-------------------|----------------|------------------|
| War/Threat | 58.3% | 52.6% |
| **Peace/Diplomacy** | 5.4% | **22.0%** |

Peace/diplomacy relationships increased more than fourfold, from 5.4% to 22.0%.

#### 4.4.4 Interpretation

**H4 is supported.** Knowledge graph analysis revealed structural differences in North Korea discourse between periods. During the tension period, North Korea was conceptualized as a "nuclear-armed threat state." During the diplomacy period, it was reconceptualized as a "negotiable counterpart."

Kim Jong Un's network shift particularly illustrates core coercive diplomacy dynamics—the transition from threat source to negotiating partner.

---

## 5. Discussion

### 5.1 Summary of Key Findings

All four hypotheses were supported:

| Hypothesis | Method | Result | Significance |
|------------|--------|--------|--------------|
| H1: Sentiment improvement | BERT + t-test | **Supported** | p < 0.001 |
| H2: Framing shift | GPT-4o-mini + χ² | **Supported** | p < 0.001 |
| H3: Causal effect | ITS regression | **Supported** | p = 0.044 |
| H4: Knowledge structure change | GraphRAG | **Supported** | Qualitative |

### 5.2 Theoretical Implications

These findings empirically support **Coercive Diplomacy Theory**:

1. **Effectiveness of threat-then-negotiate strategy**: North Korea's "maximum tension followed by negotiation offers" strategy demonstrably affected U.S. public opinion. Sentiment improvement (+0.230) and framing shift (threat -29.3%p, diplomacy +22.7%p) suggest partial success.

2. **Immediate vs. sustained effects**: ITS analysis revealed significant β₂ (immediate change) but non-significant β₃ (trend change). The summit announcement produced immediate improvement but not sustained gains. Diplomatic events may have temporary rather than lasting effects.

3. **Cognitive restructuring**: Knowledge graph analysis revealed changes beyond mere sentiment shifts—the very "cognitive structure" of North Korea perceptions changed. Kim Jong Un's reconceptualization from "nuclear threat actor" to "Trump's negotiating partner" represents qualitative transformation.

### 5.3 Policy Implications

1. **Diplomacy affects public opinion**: Visible diplomatic events like summits can change public perceptions. Policymakers should consider opinion dynamics when formulating diplomatic strategies.

2. **Limited effects**: Mean sentiment remained negative (-0.245) even during the diplomacy period, and threat framing remained highest (40.7%). Decades of distrust cannot be overcome through short-term diplomacy.

3. **Sustained trust-building required**: Non-significant β₃ suggests one-time events matter less than sustained trust-building. The Hanoi Summit collapse (February 2019) may have reversed gains.

### 5.4 Limitations

1. **Sample representativeness**: Reddit users may not represent the overall U.S. population. Reddit skews male, young, and educated, introducing potential sample bias.

2. **Data collection gap**: A three-month gap exists between the tension period (ending February 2018) and diplomacy period (beginning June 2018), preventing capture of transitional dynamics.

3. **Causal inference limitations**: While ITS provides evidence for causation, complete causal inference is impossible without randomized controlled experiments. Concurrent events may not be fully controlled.

4. **LLM classification reliability**: GPT-4o-mini framing classifications were not validated against human coders. Future research should assess inter-rater reliability.

---

## 6. Conclusion

This study empirically analyzed the impact of North Korea's coercive diplomacy strategy on U.S. online public opinion. Analysis of Reddit data from 2017-2019 yielded the following findings:

First, U.S. sentiment toward North Korea improved significantly during the transition from tension to diplomacy periods (p < 0.001). Mean sentiment rose from -0.475 to -0.245, an improvement of 0.230.

Second, discourse framing shifted from 'threat' to 'diplomacy.' Threat framing decreased from 70.0% to 40.7%, while diplomacy framing increased from 8.7% to 31.3% (χ² = 33.17, p < 0.001).

Third, ITS analysis confirmed that the March 2018 summit announcement directly caused opinion improvement (β₂ = +0.293, p = 0.044). This provides evidence of causation rather than mere correlation.

Fourth, knowledge graph analysis confirmed that Kim Jong Un's network shifted from military to diplomatic elements.

These results empirically support coercive diplomacy theory, demonstrating that North Korea's "threat-then-negotiate" strategy tangibly affected U.S. public opinion. However, persistently negative sentiment and dominant threat framing indicate that short-term diplomatic gestures cannot fully overcome long-standing distrust.

Future research should employ longer time series to assess the durability of diplomatic effects and compare findings across multiple social media platforms to assess generalizability.

---

## References

Bernal, J. L., Cummins, S., & Gasparrini, A. (2017). Interrupted time series regression for the evaluation of public health interventions: a tutorial. *International Journal of Epidemiology*, 46(1), 348-355.

George, A. L. (1991). *Forceful Persuasion: Coercive Diplomacy as an Alternative to War*. United States Institute of Peace Press.

Holsti, O. R. (1992). Public opinion and foreign policy: Challenges to the Almond-Lippmann consensus. *International Studies Quarterly*, 36(4), 439-466.

Jungherr, A. (2015). *Analyzing Political Communication with Digital Trace Data*. Springer.

Page, B. I., & Shapiro, R. Y. (1983). Effects of public opinion on policy. *American Political Science Review*, 77(1), 175-190.

Schelling, T. C. (1966). *Arms and Influence*. Yale University Press.

---

## Appendix

### A. Data Processing Pipeline

```
data/raw/
├── reddit_posts_combined.json    # Raw post data
└── reddit_comments_linked.json   # Raw comment data

data/processed/
├── posts_final.csv               # Preprocessed posts
└── coercive_diplomacy/
    ├── posts_period1_tension.csv
    ├── posts_period2_diplomacy.csv
    ├── sentiment_comparison_results.json
    ├── openai_framing_results.json
    └── its_analysis_results.json
```

### B. Analysis Tools

| Analysis | Tool/Library |
|----------|--------------|
| Sentiment Analysis | BERT (Hugging Face Transformers) |
| Framing Classification | OpenAI GPT-4o-mini |
| ITS Analysis | statsmodels (Python) |
| Knowledge Graph | Microsoft GraphRAG |
| Data Collection | Arctic Shift API |

### C. Code Repository

Complete analysis code is available on GitHub:
- Repository: `reddit_US_NK`
- Key scripts:
  - `src/sentiment_analyzer.py`: BERT sentiment analysis
  - `src/misinfo_detector.py`: LLM-based classification
  - `src/data_collector.py`: Data collection

---

*Analysis completed: December 2024*
*Researcher: Jun Sin*
*Collaboration: UT Austin - Professor Mohit Singhal*
