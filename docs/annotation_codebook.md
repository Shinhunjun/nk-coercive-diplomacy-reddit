# Framing Classification Codebook

**Version**: 1.0  
**Last Updated**: 2025-12-20  
**Annotators**: [Hunjun Shin], [Hunbae Moon]

---

## 1. Frame Definitions

### THREAT

**Definition**: Military threat, nuclear weapons, missiles, war risk

**Inclusion criteria**:

- Posts emphasizing military danger or aggression
- Focus on weapons development or testing
- War rhetoric or threats of force
- Military exercises presented as threatening

**Exclusion criteria**:

- Military topics framed diplomatically (e.g., "talks to reduce tensions")
- Historical military events without current threat framing

---

### DIPLOMACY

**Definition**: Negotiation, dialogue, peace, cooperation

**Inclusion criteria**:

- Posts about negotiations, summits, or talks
- Emphasis on diplomatic engagement
- Peace initiatives or cooperation efforts
- Treaty discussions

**Exclusion criteria**:

- Failed diplomacy framed as threatening
- Diplomatic events mentioned only factually (may be NEUTRAL)

---

### NEUTRAL

**Definition**: Neutral information delivery

**Inclusion criteria**:

- Factual reporting without clear framing
- Announcements or statements
- Objective news updates
- No clear emotional or evaluative tone

**Exclusion criteria**:

- Posts with subtle framing (look deeper)
- Posts that seem neutral but emphasize specific aspects

---

### ECONOMIC

**Definition**: Economic sanctions, trade aspects

**Inclusion criteria**:

- Sanctions as primary focus
- Trade disputes or agreements
- Economic impacts or consequences
- Business/financial relations

**Exclusion criteria**:

- Economic impacts mentioned only as background
- Humanitarian consequences of sanctions (may be HUMANITARIAN)

---

### HUMANITARIAN

**Definition**: Human rights, refugees, civilian issues

**Inclusion criteria**:

- Human rights violations or advocacy
- Refugee or defector stories
- Civilian suffering or welfare
- Protests or civil society issues

**Exclusion criteria**:

- Military casualties (may be THREAT)
- Economic hardship without human focus (may be ECONOMIC)

---

## 2. Decision Rules for Edge Cases

### Case 1: Nuclear Negotiations

**Scenario**: Post discusses both nuclear weapons AND diplomatic talks

**Rule**:

- If emphasis is on **progress/talks** → DIPLOMACY
- If emphasis is on **weapons/danger** → THREAT
- If purely factual → NEUTRAL

**Example**: "North Korea agrees to nuclear talks" → DIPLOMACY  
**Example**: "North Korea develops new nuclear missile despite talks" → THREAT

---

### Case 2: Sanctions with Humanitarian Impact

**Scenario**: Post discusses sanctions AND civilian suffering

**Rule**:

- If focus is on **sanctions mechanism/policy** → ECONOMIC
- If focus is on **human suffering/impact** → HUMANITARIAN

**Example**: "New sanctions target North Korean economy" → ECONOMIC  
**Example**: "Sanctions leave North Korean children malnourished" → HUMANITARIAN

---

### Case 3: Failed Diplomacy

**Scenario**: Summit or talks collapse or fail

**Rule**: Still DIPLOMACY if the post is **about the diplomatic process**

**Example**: "Hanoi Summit ends with no deal" → DIPLOMACY  
**Example**: "After summit failure, North Korea threatens US" → THREAT

---

### Case 4: Multiple Frames in One Post

**Scenario**: Post contains multiple frames

**Rule**: Choose the **primary/dominant** frame based on:

1. What is emphasized in the **title**?
2. What takes up **most of the text**?
3. What is the **main point** the author is making?

---

## 3. Annotation Log (Updated During Process)

### Pilot Round (Posts 1-100)

**Date**: [TBD]  
**Initial Kappa**: [TBD]  
**Key Issues Found**:

- [Issue 1]
- [Issue 2]

**Codebook Updates**:

- [Update 1]
- [Update 2]

---

### Batch 1 (Posts 101-300)

**Date**: [TBD]  
**Kappa**: [TBD]  
**New Edge Cases**:

- [Case description]

**Codebook Updates**:

- [Update]

---

## 4. Difficult Cases Registry

| Post ID | Title | Annotator 1 | Annotator 2 | Final | Reasoning |
|---------|-------|-------------|-------------|-------|-----------|
| [ID] | [Title] | THREAT | DIPLOMACY | DIPLOMACY | Talks emphasized over weapons |
| | | | | | |

---

## 5. Inter-Rater Reliability Tracking

| Batch | N Posts | Agreement % | Cohen's Kappa | Notes |
|-------|---------|-------------|---------------|-------|
| Pilot (v1.0) | 100 | [TBD] | [TBD] | Initial attempt |
| Pilot (v1.1) | 100 | [TBD] | [TBD] | After codebook update |
| Batch 1 | 200 | [TBD] | [TBD] | |
| Batch 2 | 200 | [TBD] | [TBD] | |
| **Overall** | **1,330** | **[TBD]** | **[TBD]** | **Final IRR** |

---

## 6. Version History

### Version 1.0 (2025-12-20)

- Initial codebook based on LLM prompt definitions
- 5 frame categories defined
- Basic decision rules established

### Version 1.1 (TBD)

- [Updates after pilot round]

---

## 7. Notes for Paper Reporting

**Codebook Development Process**:

1. Initial codebook developed based on LLM classification prompt
2. Pilot annotation of 100 posts revealed [X] edge cases
3. Codebook refined with [Y] additional decision rules
4. Final inter-rater reliability: Cohen's κ = [Z]
