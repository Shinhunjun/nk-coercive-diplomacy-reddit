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

# Control group topics for DID analysis
CONTROL_GROUPS = {
    'iran': {
        'keywords': [
            # Core terms (already collected)
            # 'iran', 'iranian', 'tehran',
            # 'JCPOA', 'nuclear deal', 'iran nuclear', 'rouhani', 'khamenei', 'sanctions iran',

            # NEW: Leaders
            'zarif', 'soleimani', 'ayatollah',
            # NEW: Nuclear/Politics
            'iran sanctions', 'iran deal', 'iran agreement',
            # NEW: Military
            'IRGC', 'revolutionary guard', 'quds force', 'iran military',
            'iran missile',
            # NEW: Regional
            'persian gulf', 'strait of hormuz', 'iran syria'
        ],
        'nuclear_keywords': ['nuclear', 'enrichment', 'uranium'],
        'notes': 'JCPOA withdrawal May 2018 - potential confounder'
    },
    'russia': {
        'keywords': [
            # Core terms (already collected)
            # 'russia', 'russian', 'putin', 'moscow', 'kremlin',

            # NEW: Leaders
            'lavrov', 'medvedev',
            # NEW: Politics
            'russia sanctions', 'russian sanctions', 'russia election',
            'russia interference', 'russia hack', 'russian hacking',
            # NEW: Military
            'russian military', 'russia ukraine', 'russia crimea',
            'russia syria', 'russia nato',
            # NEW: Economy
            'gazprom', 'nord stream', 'russia oil'
        ],
        'investigation_keywords': ['mueller', 'investigation', 'collusion'],
        'notes': 'Mueller investigation March 2019 - potential confounder'
    },
    'china': {
        'keywords': [
            # Core terms (already collected)
            # 'china', 'chinese', 'beijing', 'xi jinping',

            # NEW: Leaders
            'li keqiang', 'wang yi',
            # NEW: Politics
            'china trade', 'one china', 'taiwan china',
            'south china sea', 'china policy',
            # NEW: Military
            'PLA', 'china military', 'china navy', 'china missile',
            # NEW: Economy
            'belt and road', 'china economy', 'china manufacturing'
        ],
        'trade_keywords': ['trade war', 'tariff', 'huawei'],
        'notes': 'Trade war March 2018 - concurrent with NK intervention'
    }
}

# DID estimation settings
DID_CONFIG = {
    'pre_period_start': '2017-01',
    'pre_period_end': '2018-02',  # Month before intervention
    'post_period_start': '2018-03',  # Intervention month
    'post_period_end': '2019-06',
    'cluster_level': 'month',  # For clustered standard errors
    'alpha': 0.05,  # Significance level
    'parallel_trends_threshold': 0.10  # p-value threshold for violation
}

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
