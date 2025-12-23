# GraphRAG Customization for NK Coercive Diplomacy Research

This document records all customizations made to GraphRAG for this research project.

---

## Research Alignment

**Paper RQ2**: *"How do these diplomatic events restructure the organization of online discourse, as reflected in changes to discourse networks and narrative connectivity?"*

---

## 1. Entity Types

**File**: `graphrag/settings.yaml`

| Type | Description | Examples |
|------|-------------|----------|
| PERSON | Political leaders, diplomats | Kim Jong Un, Trump, Moon Jae-in |
| COUNTRY | Nations involved | North Korea, USA, South Korea |
| EVENT | Summits, meetings, tests | Singapore Summit, ICBM test |
| ORGANIZATION | Agencies, bodies | CIA, UN, IAEA |
| **WEAPON** | Military capabilities | ICBM, nuclear weapon, Hwasong-15 |
| **POLICY** | Sanctions, agreements | Denuclearization, maximum pressure |

---

## 2. Custom Prompts

### extract_graph.txt

**Purpose**: NK-specific entity extraction

**Customizations**:

- Added WEAPON and POLICY entity types
- Examples using Singapore and Hanoi summits
- Relationship strength guidelines (threat vs diplomacy)

### community_report_graph.txt  

**Purpose**: Structural analysis for RQ2

**Added Fields**:

- `NARRATIVE_CONNECTIVITY`: Hub entities, bridge narratives, cohesion level
- `CENTRAL_ACTORS`: Top connected entities with structural roles
- Structural pattern identification (hubs, bridges, isolated clusters)

---

## 3. Analysis Script Enhancements

**File**: `scripts/analyze_graphrag_communities.py`

### Network Topology Metrics (RQ2)

| Metric | Description |
|--------|-------------|
| Network Density | Edge count / max possible edges |
| Avg Degree | Mean connections per entity |
| Max Degree | Most connected entity |
| Top Entities | 5 highest-degree nodes |

**Function**: `analyze_network_topology(p1_data, p2_data)`

---

## 4. Period Definitions

Aligned with DID analysis:

| Period | Date Range | Key Event |
|--------|------------|-----------|
| P1_PreSingapore | 2017-01 ~ 2018-05 | Before Singapore Summit |
| P2_SingaporeHanoi | 2018-06 ~ 2019-02 | Singapore to Hanoi |
| P3_PostHanoi | 2019-03 ~ 2019-12 | After Hanoi |

---

## Files Modified

| File | Changes |
|------|---------|
| `settings.yaml` | Entity types: added WEAPON, POLICY |
| `prompts/extract_graph.txt` | NK examples, relationship guidelines |
| `prompts/community_report_graph.txt` | NARRATIVE_CONNECTIVITY, CENTRAL_ACTORS |
| `scripts/analyze_graphrag_communities.py` | `analyze_network_topology()` |
