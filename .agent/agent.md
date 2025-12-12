# NK Coercive Diplomacy Reddit Analysis - Agent Summary

## 📌 Project Overview

**Research Topic**: Impact of North Korea's Coercive Diplomacy Strategy on U.S. Online Public Opinion (Reddit)

**Research Period**:
- **Tension Period** (2017.01-2018.02): Nuclear tests, ICBM launches, "Fire and Fury"
- **Diplomacy Period** (2018.06-2019.06): Singapore/Hanoi Summits
- **Intervention Event**: 2018-03-08 (NK-US Summit Announcement)

---

## 🔬 Research Hypotheses & Results

| Hypothesis | Method | Result | Significance |
|------------|--------|--------|--------------|
| **H1**: Sentiment Improvement | RoBERTa + t-test | ✅ Supported | p < 0.001 |
| **H2**: Framing Shift | GPT-4o-mini + χ² | ✅ Supported | p < 0.001 |
| **H3**: Causal Effect | ITS/DID Regression | ✅ Supported | p = 0.005 |
| **H4**: Knowledge Structure | GraphRAG | ✅ Supported | Qualitative |

### Key Findings:
- **Sentiment Score**: -0.475 → -0.245 (+0.230 improvement)
- **THREAT Framing**: 70% → 40.7% (**-29.3%p** decrease)
- **DIPLOMACY Framing**: 8.7% → 31.3% (**+22.7%p** increase)
- **DID Analysis**: +0.08 points vs China control (Cohen's d = 1.12, **Large Effect**)

---

## 📁 Project Structure

```
nk-coercive-diplomacy-reddit/
├── src/                          # Core analysis modules
│   ├── sentiment_analysis.py     # BERT/RoBERTa sentiment analysis
│   ├── framing_analysis.py       # LLM framing classification
│   ├── openai_framing_analysis.py
│   ├── vertex_ai_framing_analysis.py
│   ├── its_analysis.py           # Interrupted Time Series
│   ├── did_analysis.py           # Difference-in-Differences
│   ├── parallel_trends_test.py   # DID assumption validation
│   ├── visualizations.py         # Figure generation
│   ├── run_analysis.py           # Main analysis script
│   └── config.py                 # Configuration
│
├── scripts/                      # Data collection & processing (50 scripts)
│   ├── collect_*.py              # Reddit data collection (Arctic Shift API)
│   ├── apply_*.py                # Apply sentiment/framing analysis
│   ├── create_*.py               # Data aggregation (monthly/weekly)
│   ├── run_*.py                  # Run DID analysis
│   └── framing_*.py              # Framing analysis per group
│
├── data/                         # Data files
│   ├── nk/                       # North Korea data
│   ├── control/                  # Control groups (Iran, Russia, China)
│   ├── sentiment/                # Sentiment analysis results
│   ├── framing/                  # Framing classification results
│   ├── processed/                # Aggregated data
│   ├── results/                  # Final analysis results (JSON)
│   └── sample/                   # Sample data for quick testing
│
├── graphrag/                     # Microsoft GraphRAG knowledge graphs
│   ├── period1/                  # Tension period graph
│   ├── period2/                  # Diplomacy period graph
│   └── settings.yaml             # GraphRAG configuration
│
├── figures/                      # Paper figures (PNG)
│   ├── fig1_research_timeline.png
│   ├── fig2_sentiment_distribution.png
│   ├── fig3_framing_shift.png
│   ├── fig4_its_analysis.png
│   ├── fig5_knowledge_graph.png
│   └── fig6_summary_dashboard.png
│
├── paper/                        # Research paper drafts
├── reports/                      # Analysis reports
├── DID_ANALYSIS_REPORT.md        # Detailed DID methodology & results
└── DATA_COLLECTION_AND_ANALYSIS_REPORT.md
```

---

## 🛠 Technology Stack

| Area | Tool/Model |
|------|------------|
| **Sentiment Analysis** | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) |
| **Framing Classification** | OpenAI GPT-4o-mini / Vertex AI |
| **Knowledge Graph** | Microsoft GraphRAG |
| **Causal Inference** | statsmodels (DID, ITS regression) |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Data Source** | Reddit via Arctic Shift API |

---

## 📊 Data Summary

| Group | Posts | Comments | Total |
|-------|-------|----------|-------|
| **NK (Treatment)** | 10,442 | 89,766 | 100,208 |
| **Iran (Control)** | 486 | 5,663 | 6,149 |
| **Russia (Control)** | 592 | 13,358 | 13,950 |
| **China (Control)** | 494 | 10,085 | 10,579 |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (for framing analysis)
export OPENAI_API_KEY="your-api-key"

# Run complete analysis
cd src
python run_analysis.py

# View pre-computed results
cat data/results/sentiment_comparison_results.json
cat data/results/its_analysis_results.json
cat data/results/did_level_change_results.json
```

---

## 📈 Key Analysis Scripts

| Script | Purpose |
|--------|---------|
| `src/run_analysis.py` | Main analysis entry point |
| `src/sentiment_analysis.py` | BERT sentiment scoring |
| `src/its_analysis.py` | Interrupted Time Series regression |
| `src/did_analysis.py` | Difference-in-Differences analysis |
| `scripts/run_did_analysis_all_controls.py` | DID with all control groups |
| `scripts/apply_openai_framing.py` | GPT-4o-mini framing classification |
| `scripts/generate_summary_report.py` | Generate analysis reports |

---

## 📝 Key Results Files

| File | Description |
|------|-------------|
| `data/results/sentiment_comparison_results.json` | H1 sentiment analysis |
| `data/results/openai_framing_results.json` | H2 framing analysis |
| `data/results/its_analysis_results.json` | H3 ITS causal analysis |
| `data/results/did_all_controls_results.json` | DID slope change |
| `data/results/did_level_change_results.json` | DID level change |
| `data/results/parallel_trends_tests.json` | DID assumption validation |

---

## 🎯 Research Contribution

1. **Robust Causal Inference**: DID with 3 control groups (Iran, Russia, China)
2. **Multi-layered Analysis**: Sentiment + Framing + Knowledge Graph
3. **Large-scale Data**: 100,000+ Reddit posts/comments
4. **Reproducibility**: All results saved as JSON, scripts well-organized

---

## 👤 Author

- **Author**: Jun Sin
- **Collaboration**: UT Austin (Professor Mohit Singhal)
- **License**: MIT
