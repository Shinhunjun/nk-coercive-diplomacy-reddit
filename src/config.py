"""
Configuration settings for the analysis
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = DATA_DIR / "results"

# Output directories
FIGURES_DIR = PROJECT_ROOT / "figures"

# GraphRAG directories
GRAPHRAG_DIR = PROJECT_ROOT / "graphrag"

# Analysis periods
PERIODS = {
    "tension": {
        "name": "Tension Period",
        "start": "2017-01-01",
        "end": "2018-02-28",
        "events": [
            "Trump Inauguration (2017.01.20)",
            "Fire and Fury Speech (2017.08.08)",
            "6th Nuclear Test (2017.09.03)",
            "Hwasong-15 ICBM Launch (2017.11.29)"
        ]
    },
    "diplomacy": {
        "name": "Diplomacy Period",
        "start": "2018-06-01",
        "end": "2019-06-30",
        "events": [
            "Singapore Summit (2018.06.12)",
            "Hanoi Summit (2019.02.27-28)",
            "Panmunjom Meeting (2019.06.30)"
        ]
    }
}

# Intervention point for ITS analysis
INTERVENTION_DATE = "2018-03-08"  # Trump accepts summit invitation

# Framing categories
FRAME_CATEGORIES = [
    "THREAT",       # Military threat framing
    "DIPLOMACY",    # Negotiation/cooperation framing
    "NEUTRAL",      # Neutral information
    "ECONOMIC",     # Economic aspects
    "HUMANITARIAN"  # Humanitarian perspective
]

# Sentiment model
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# OpenAI settings (for framing classification)
OPENAI_MODEL = "gpt-4o-mini"
