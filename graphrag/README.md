# GraphRAG Knowledge Graph Analysis

## Overview

This directory contains knowledge graph outputs from Microsoft GraphRAG analysis of Reddit discourse about North Korea. The analysis is aligned with the Difference-in-Differences (DID) study design documented in `DID_ANALYSIS_REPORT.md`.

---

## Analysis Period Design

The GraphRAG analysis follows the same temporal structure as the DID framing analysis:

| Period | Date Range | Duration | Description |
|--------|------------|----------|-------------|
| **Pre-Intervention** | 2017-01 ~ 2018-02 | 14 months | Maximum pressure / tension era |
| **Gap (Excluded)** | 2018-03 ~ 2018-05 | 3 months | Transition period excluded to reduce noise |
| **Post-Intervention** | 2018-06 ~ 2019-06 | 13 months | Summit diplomacy era |

**Intervention Event**: 2018-03-08 (NK-US Summit Announcement)

> **Note**: The 3-month gap period (March-May 2018) is excluded to capture long-term structural changes rather than immediate news saturation effects.

---

## Directory Structure

```
graphrag/
├── README.md              # This file
├── settings.yaml          # GraphRAG configuration
├── period1/               # Pre-Intervention (2017.01-2018.02)
│   ├── entities.parquet
│   ├── relationships.parquet
│   ├── communities.parquet
│   ├── community_reports.parquet
│   └── text_units.parquet
└── period2/               # Post-Intervention (2018.06-2019.06)
    ├── entities.parquet
    ├── relationships.parquet
    ├── communities.parquet
    ├── community_reports.parquet
    └── text_units.parquet
```

---

## What is GraphRAG?

**GraphRAG** (Graph Retrieval-Augmented Generation) is a Microsoft research project that:

1. Extracts **entities** (people, places, events) from text using LLMs
2. Identifies **relationships** between entities
3. Builds a **knowledge graph** of the corpus
4. Detects **communities** of related entities
5. Generates **summaries** at different abstraction levels

---

## Key Findings

### Kim Jong Un's Network Evolution

**Pre-Intervention Connections:**

- NORTH KOREA
- MISSILE LAUNCH
- NUCLEAR PROGRAM
- KIM JONG-NAM (assassination)
- SANCTIONS

**Post-Intervention New Connections:**

- TRUMP (new, central)
- SINGAPORE SUMMIT
- PANMUNJOM
- JOINT STATEMENT
- DENUCLEARIZATION

### Trump Entity Emergence

| Metric | Pre-Intervention | Post-Intervention |
|--------|------------------|-------------------|
| Connections | 5 | 18 |
| Entity Rank | >10 | 3rd |

### Relationship Type Changes

| Type | Pre-Intervention | Post-Intervention |
|------|------------------|-------------------|
| War/Threat | 58.3% | 52.6% |
| Peace/Diplomacy | 5.4% | **22.0%** |

---

## Output Files Description

### entities.parquet

Contains all extracted entities:

- `id`: Unique entity ID
- `name`: Entity name
- `type`: Entity type (PERSON, GEO, EVENT, ORGANIZATION)
- `description`: LLM-generated description
- `text_unit_ids`: Source text references

### relationships.parquet

Contains entity relationships:

- `source`: Source entity ID
- `target`: Target entity ID
- `description`: Relationship description
- `weight`: Relationship strength
- `type`: Relationship category

### communities.parquet

Contains detected communities:

- `id`: Community ID
- `level`: Hierarchy level (0-3)
- `title`: Community title
- `entity_ids`: Member entities

### community_reports.parquet

Contains LLM-generated community summaries:

- `community_id`: Reference to community
- `title`: Summary title
- `summary`: Full text summary
- `findings`: Key findings list

---

## How to Reproduce

### Prerequisites

```bash
pip install graphrag
```

### Step 1: Prepare Input Data

Create input text file from Reddit posts:

```python
import pandas as pd

# For Period 1
posts = pd.read_csv('data/full/posts_period1_tension.csv')
with open('graphrag/period1/input/posts.txt', 'w') as f:
    for _, row in posts.iterrows():
        f.write(f"{row['title']}\n{row['selftext']}\n\n")
```

### Step 2: Initialize GraphRAG

```bash
cd graphrag/period1
graphrag init --root .
```

### Step 3: Configure settings.yaml

Key settings:

```yaml
llm:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini

embeddings:
  api_key: ${OPENAI_API_KEY}
  model: text-embedding-3-small

chunks:
  size: 1200
  overlap: 100
```

### Step 4: Run Indexing

```bash
graphrag index --root .
```

This will:

1. Chunk the input text
2. Extract entities and relationships
3. Build the knowledge graph
4. Detect communities
5. Generate community reports

### Step 5: Query the Graph

```bash
# Global search (uses community summaries)
graphrag query --root . --method global \
    "How did perceptions of Kim Jong Un change?"

# Local search (uses entity relationships)
graphrag query --root . --method local \
    "What events involved Trump and Kim Jong Un?"
```

---

## Analysis Scripts

### Load and Analyze Entities

```python
import pandas as pd

# Load entities
entities_p1 = pd.read_parquet('period1/entities.parquet')
entities_p2 = pd.read_parquet('period2/entities.parquet')

# Compare entity types
print("Period 1 Entity Types:")
print(entities_p1['type'].value_counts())

print("\nPeriod 2 Entity Types:")
print(entities_p2['type'].value_counts())
```

### Compare Kim Jong Un's Network

```python
# Load relationships
rels_p1 = pd.read_parquet('period1/relationships.parquet')
rels_p2 = pd.read_parquet('period2/relationships.parquet')

# Find Kim Jong Un connections
kim_id = entities_p1[entities_p1['name'].str.contains('KIM JONG UN')]['id'].iloc[0]

kim_connections_p1 = rels_p1[(rels_p1['source'] == kim_id) | (rels_p1['target'] == kim_id)]
kim_connections_p2 = rels_p2[(rels_p2['source'] == kim_id) | (rels_p2['target'] == kim_id)]

print(f"Kim Jong Un connections - Tension: {len(kim_connections_p1)}")
print(f"Kim Jong Un connections - Diplomacy: {len(kim_connections_p2)}")
```

---

## Resources

- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [Research Paper](https://arxiv.org/abs/2404.16130)

---

## Notes

- GraphRAG requires OpenAI API access
- Full indexing takes ~30 minutes per period
- Output size: ~9MB per period
- Parquet files can be read with pandas or pyarrow
