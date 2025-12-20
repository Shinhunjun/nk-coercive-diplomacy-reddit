# Human Annotation Benchmark: Complete Guide

**Project**: NK Coercive Diplomacy Reddit Analysis  
**Annotators**: Hunjun Shin, Hunbae Moon  
**Created**: 2025-12-20  
**Version**: 1.0

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Overview](#2-overview)
3. [Dataset](#3-dataset)
4. [Framing Categories](#4-framing-categories)
5. [Annotation Process](#5-annotation-process)
6. [Codebook & Decision Rules](#6-codebook--decision-rules)
7. [Quality Control](#7-quality-control)
8. [Timeline & Deliverables](#8-timeline--deliverables)
9. [Technical Instructions](#9-technical-instructions)

---

## 1. Project Goal

### Research Objective

This study examines how the 2018 Singapore Summit and 2019 Hanoi Summit affected public opinion framing of North Korea on Reddit using:

1. **LLM-based framing classification** (GPT-4o-mini)
2. **Human expert validation** (military officers)
3. **Difference-in-Differences causal analysis**

### Human Annotation Purpose

Create a **gold-standard benchmark** to:

- **Validate LLM accuracy** on framing classification
- **Measure inter-rater reliability** (Cohen's Kappa)
- **Report methodology** in ICWSM 2025 paper
- **Ensure scientific rigor** through human expert judgment

### Success Criteria

| Metric | Target | Importance |
|--------|--------|------------|
| Cohen's Kappa | > 0.70 | Inter-rater reliability |
| Sample Size | 1,330 posts | Statistical power |
| LLM Accuracy | Report actual | Validation metric |
| Completion Time | 2 weeks | Project timeline |

---

## 2. Overview

### What You're Annotating

**1,330 Reddit posts** about international affairs, stratified across:

- **4 countries**: North Korea (treatment), China/Iran/Russia (controls)
- **3 time periods**: Pre-Singapore, Singapore-Hanoi, Post-Hanoi
- **5 framing categories**: THREAT, DIPLOMACY, NEUTRAL, ECONOMIC, HUMANITARIAN

### Why Stratified Sampling?

Ensures representation of:

- All countries in the DID analysis
- All time periods (to capture temporal changes)
- All frames (including rare ones like HUMANITARIAN)

### Annotation Method

**Iterative process with codebook refinement**:

1. **Pilot round** (100 posts) → identify edge cases → refine codebook
2. **Main annotation** (1,230 posts) → batch monitoring → ongoing refinement
3. **Disagreement resolution** → consensus labels → final dataset

---

## 3. Dataset

### Sample Distribution

| Country | Posts | Role in Study |
|---------|-------|---------------|
| **North Korea** | 335 | Treatment group |
| **China** | 337 | Control (trade war confounder) |
| **Iran** | 337 | Control (nuclear deal comparison) |
| **Russia** | 321 | Control (investigation confounder) |
| **TOTAL** | **1,330** | |

### By Time Period

| Period | Date Range | Posts | Significance |
|--------|------------|-------|--------------|
| **P1: Pre-Singapore** | 2017-01 to 2018-05 | 512 | Baseline |
| **P2: Singapore-Hanoi** | 2018-06 to 2019-02 | 256 | Diplomatic engagement |
| **P3: Post-Hanoi** | 2019-03 to 2019-12 | 562 | Post-failure |

### By Frame (Proportional)

| Frame | Posts | % | Statistical Power |
|-------|-------|---|-------------------|
| THREAT | 489 | 36.8% | High |
| NEUTRAL | 290 | 21.8% | Good |
| DIPLOMACY | 226 | 17.0% | Good |
| ECONOMIC | 205 | 15.4% | Good |
| HUMANITARIAN | 120 | 9.0% | Adequate |

---

## 4. Framing Categories

> **Critical**: These definitions **exactly match** the LLM prompt to ensure fair comparison.

### THREAT

**Definition**: Military threat, nuclear weapons, missiles, war risk

**What to look for**:

- Military danger or aggression
- Weapons development/testing
- War rhetoric or threats
- Military exercises as threatening

**Examples**:

- ✅ "North Korea fires ballistic missile over Japan"
- ✅ "Iran threatens to restart nuclear program"
- ❌ "North Korea agrees to denuclearization talks" (DIPLOMACY)

---

### DIPLOMACY

**Definition**: Negotiation, dialogue, peace, cooperation

**What to look for**:

- Negotiations, summits, talks
- Diplomatic engagement
- Peace initiatives
- Cooperation efforts

**Examples**:

- ✅ "Trump and Kim to meet for historic summit"
- ✅ "Iran nuclear deal negotiations resume"
- ❌ "Summit ends with no deal, tensions rise" (THREAT)

---

### NEUTRAL

**Definition**: Neutral information delivery

**What to look for**:

- Factual reporting without framing
- Announcements or statements
- Objective news updates
- No clear emotional tone

**Examples**:

- ✅ "China announces new economic policy"
- ✅ "Russia holds presidential election"
- ❌ "China's aggressive new policy threatens region" (THREAT)

---

### ECONOMIC

**Definition**: Economic sanctions, trade aspects

**What to look for**:

- Sanctions as primary focus
- Trade disputes/agreements
- Economic impacts
- Business/financial relations

**Examples**:

- ✅ "New sanctions imposed on North Korea"
- ✅ "US-China trade war escalates"
- ❌ "Sanctions leave children malnourished" (HUMANITARIAN)

---

### HUMANITARIAN

**Definition**: Human rights, refugees, civilian issues

**What to look for**:

- Human rights violations/advocacy
- Refugee or defector stories
- Civilian suffering or welfare
- Protests or civil society

**Examples**:

- ✅ "North Korean defector shares escape story"
- ✅ "Human rights violations reported in Iran"
- ❌ "Military strikes kill civilians" (THREAT)

---

## 5. Annotation Process

### Phase 1: Pilot Round (Week 1)

**Goal**: Test guidelines, identify edge cases, refine codebook

#### Step 1: Independent Annotation (2-3 days)

1. Read all documentation (this guide, codebook, workflow)
2. **Independently** annotate posts 1-100 in Google Sheets
3. **Do NOT discuss** during annotation
4. Document uncertain cases in "notes" column

#### Step 2: Calculate IRR (30 min)

```bash
# Download Google Sheet as CSV
python scripts/calculate_irr.py data/sample/pilot_batch.csv "Pilot v1.0"
```

**Output**: Cohen's Kappa, confusion matrix, disagreements list

#### Step 3: Disagreement Meeting (2-3 hours)

**Agenda**:

- Review each disagreement
- Discuss reasoning (not who's right/wrong)
- Identify patterns in disagreements
- Note edge cases needing rules

**Do NOT change labels yet!**

#### Step 4: Codebook Refinement (1-2 hours)

Update `annotation_codebook.md`:

- Add decision rules for edge cases
- Add examples from actual data
- Clarify ambiguous definitions
- Update version (1.0 → 1.1)

#### Step 5: Re-annotate Pilot (1-2 days)

- Re-annotate posts 1-100 with updated codebook
- Calculate IRR again
- **Expected**: Kappa should improve

**Decision Point**:

- ✅ Kappa > 0.60 → Proceed to main annotation
- ❌ Kappa < 0.60 → Another refinement round

---

### Phase 2: Main Annotation (Weeks 2-6)

**Goal**: Annotate remaining 1,230 posts with ongoing quality monitoring

#### Batch Strategy

| Batch | Posts | Timeline |
|-------|-------|----------|
| Batch 1 | 101-300 | Day 4-5 |
| Batch 2 | 301-500 | Day 4-5 |
| Batch 3 | 501-700 | Day 6-7 |
| Batch 4 | 701-900 | Day 6-7 |
| Batch 5 | 901-1,100 | Day 8-9 |
| Batch 6 | 1,101-1,330 | Day 8-9 |

#### After Each Batch

1. Calculate IRR for that batch
2. Review major disagreements (brief check)
3. Update codebook if new edge cases appear
4. Continue to next batch

**Red Flags**:

- Kappa drops below 0.60
- Disagreement rate increasing
- New edge cases appearing frequently

---

### Phase 3: Final Resolution (Week 7)

**Goal**: Reach consensus on all disagreements

#### Step 1: Generate Disagreements List

```bash
python scripts/calculate_irr.py data/sample/human_benchmark_sample_annotation.csv "Final"
```

Output: `human_benchmark_sample_annotation_disagreements.csv`

#### Step 2: Resolution Meeting (4-6 hours)

For each disagreement:

1. Review post together
2. Discuss using codebook rules
3. Reach consensus
4. Document reasoning in notes
5. Fill `final_frame` column

#### Step 3: Final Dataset

- Verify all 1,330 posts have `final_frame`
- Save as `human_benchmark_sample_annotation_final.csv`
- Calculate final IRR (should be 1.0 after consensus)

---

### ⚠️ IMPORTANT: What to Do with Disagreements?

**Question**: After achieving Kappa > 0.70, do we only keep samples where both annotators agreed, or do we resolve disagreements?

**Answer**: **Resolve ALL disagreements through discussion. Keep all 1,330 samples.**

#### Why NOT Discard Disagreements?

| Approach | Problems |
|----------|----------|
| ❌ **Only keep agreements** | • Biases dataset toward "easy" cases<br>• Loses valuable edge cases<br>• Reduces sample size<br>• Not standard practice<br>• Artificially inflates LLM accuracy |
| ✅ **Resolve all disagreements** | • Uses full 1,330 samples<br>• Captures difficult cases<br>• Standard in NLP/social science<br>• More rigorous<br>• True LLM performance |

#### Example Scenario

If you achieve κ = 0.75 (substantial agreement):

| Outcome | Count | What to Do |
|---------|-------|------------|
| **Both agree** | ~1,000 posts (75%) | `final_frame` = agreed label |
| **Disagree** | ~330 posts (25%) | **Discuss → consensus → `final_frame`** |
| **Final dataset** | **1,330 posts** | **All have consensus labels** |

#### What Gets Reported in Paper

**Methods Section**:
> "Two annotators independently labeled all 1,330 posts. Initial inter-rater reliability was Cohen's κ = 0.75 (substantial agreement). Disagreements (n = 330, 24.8%) were resolved through discussion to produce final consensus labels used for LLM validation."

**Results Section**:

- **IRR (before consensus)**: κ = 0.75
- **LLM vs Human (consensus)**: Accuracy = X%, F1 = Y

#### Rare Exception: Irreconcilable Disagreements

If after discussion you **still cannot agree** on a post (<1% of cases):

- Document it as "ambiguous"
- Consider excluding (but this should be extremely rare)
- Report in paper: "X posts (0.X%) were excluded due to irreconcilable ambiguity"

#### Summary

**Standard practice**:

1. ✅ Calculate Kappa on independent annotations (report this)
2. ✅ Resolve ALL disagreements through discussion
3. ✅ Use all 1,330 consensus labels for LLM validation

**Do NOT**:

- ❌ Discard disagreements
- ❌ Only use agreements
- ❌ Vote without discussion

---

## 6. Codebook & Decision Rules

### Core Annotation Rules

#### Rule 1: Choose PRIMARY Frame

If multiple frames present, choose the **dominant** one based on:

1. What's emphasized in the **title**?
2. What takes up **most of the text**?
3. What's the **main point**?

#### Rule 2: Focus on FRAMING, Not Topic

- Nuclear weapons can be framed as DIPLOMACY (if about talks)
- Summits can be framed as THREAT (if emphasizing danger)

#### Rule 3: When Uncertain

- Consider author's perspective
- What emotion/reaction is intended?
- Document in notes column

#### Rule 4: NEUTRAL is Restrictive

Only use when **truly** objective with no clear framing.

---

### Edge Case Decision Rules

#### Case 1: Nuclear Negotiations

**Scenario**: Post discusses both weapons AND talks

**Decision**:

- Talks emphasized → **DIPLOMACY**
- Weapons/danger emphasized → **THREAT**
- Purely factual → **NEUTRAL**

**Examples**:

- "North Korea agrees to nuclear talks" → **DIPLOMACY**
- "North Korea develops new missile despite talks" → **THREAT**

---

#### Case 2: Sanctions + Humanitarian Impact

**Scenario**: Post discusses sanctions AND civilian suffering

**Decision**:

- Sanctions mechanism/policy focus → **ECONOMIC**
- Human suffering focus → **HUMANITARIAN**

**Examples**:

- "New sanctions target North Korean economy" → **ECONOMIC**
- "Sanctions leave children malnourished" → **HUMANITARIAN**

---

#### Case 3: Failed Diplomacy

**Scenario**: Summit/talks collapse or fail

**Decision**: Still **DIPLOMACY** if about the diplomatic process

**Examples**:

- "Hanoi Summit ends with no deal" → **DIPLOMACY**
- "After summit failure, NK threatens US" → **THREAT**

---

#### Case 4: Multiple Frames

**Scenario**: Post contains multiple frames

**Decision**: Use the 3-question test:

1. What's in the **title**? (strongest signal)
2. What's **most of the text** about?
3. What's the **main point**?

---

## 7. Quality Control

### Inter-Rater Reliability Targets

| Stage | Target Kappa | Interpretation |
|-------|--------------|----------------|
| Pilot v1.0 | 0.40-0.60 | Fair to Moderate |
| Pilot v1.1 | > 0.60 | Substantial |
| Main Batches | > 0.60 | Substantial |
| **Final** | **> 0.70** | **Substantial to Almost Perfect** |

### Kappa Interpretation

| Kappa | Interpretation |
|-------|----------------|
| < 0.00 | Poor (worse than chance) |
| 0.00-0.20 | Slight |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost Perfect |

### Quality Checkpoints

**After Pilot**:

- [ ] Kappa > 0.60
- [ ] Codebook has edge case rules
- [ ] Both annotators understand all categories
- [ ] Disagreement patterns identified

**During Main Annotation**:

- [ ] Batch Kappa stays > 0.60
- [ ] Disagreement rate stable or decreasing
- [ ] New edge cases documented
- [ ] Codebook updated as needed

**Before Final**:

- [ ] All 1,330 posts annotated
- [ ] Overall Kappa > 0.70
- [ ] Disagreements categorized
- [ ] Codebook finalized
- [ ] Consensus reached on all disagreements

---

## 8. Timeline & Deliverables

### Timeline (2 Weeks)

| Day | Phase | Activities | Hours/Person |
|-----|-------|------------|--------------|
| **Day 1-2** | Pilot | Annotate 100, calculate IRR | 5-6 |
| **Day 3** | Pilot | Discussion, codebook refinement, re-annotate | 4-5 |
| **Day 4-5** | Main | Batch 1-2 (400 posts) | 10 |
| **Day 6-7** | Main | Batch 3-4 (400 posts) | 10 |
| **Day 8-9** | Main | Batch 5-6 (430 posts) | 10 |
| **Day 10-11** | Main | Final batch + IRR check | 8 |
| **Day 12-13** | Resolution | Disagreement resolution | 8 |
| **Day 14** | Finalize | Final review, data cleanup | 3-4 |
| **Total** | | | **58-63 hours** |

**Daily commitment**: ~4-5 hours/day for 2 weeks

### Accelerated Schedule Notes

> [!TIP]
> **2-week timeline is achievable with**:
>
> - Consistent 4-5 hours daily commitment
> - Focused annotation sessions (minimize distractions)
> - Quick turnaround on batch IRR checks
> - Efficient disagreement resolution meetings

> [!WARNING]
> **To maintain quality**:
>
> - Don't rush through posts (2-3 min each minimum)
> - Take breaks every 1-2 hours
> - Document uncertain cases thoroughly
> - Keep codebook updated in real-time

### Deliverables

| File | Description | Due |
|------|-------------|-----|
| `annotation_codebook.md` (final) | Finalized codebook with all rules | Week 7 |
| `human_benchmark_sample_annotation_final.csv` | Final consensus labels | Week 7 |
| IRR reports | Kappa scores for each phase | Ongoing |
| Disagreement log | Documented resolution reasoning | Week 7 |

---

## 9. Technical Instructions

### Google Sheets Setup

**Columns**:

- `sample_id`: Sequential number (1-1,330)
- `post_id`: Reddit post ID
- `country`: NK, CHINA, IRAN, or RUSSIA
- `title`: Post title
- `text`: Post body (selftext)
- `annotator_1_frame`: Your label (Hunjun)
- `annotator_2_frame`: Your label (Hunbae)
- `final_frame`: Consensus label (after resolution)
- `notes`: Optional notes on difficult cases

**Annotation**:

- Enter one of: `THREAT`, `DIPLOMACY`, `NEUTRAL`, `ECONOMIC`, `HUMANITARIAN`
- Use exact capitalization
- Leave `final_frame` blank until resolution phase

---

### Running IRR Calculation

After each batch, download Google Sheet as CSV and run:

```bash
cd /Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit
python scripts/calculate_irr.py data/sample/[filename].csv "Batch Name"
```

**Output**:

- Cohen's Kappa score
- Agreement percentage
- Confusion matrix
- Disagreements CSV file

---

### Communication Protocol

**During Independent Annotation**:

- ❌ Do NOT discuss specific posts
- ✅ Can ask clarifying questions about codebook
- ✅ Document questions in notes column

**During Discussion**:

- ✅ Discuss reasoning openly
- ✅ Reference codebook rules
- ✅ Document new rules needed
- ✅ Focus on learning, not being right

---

## 10. For the Paper

### Methods Section Template

> "Two military officers with expertise in North Korean affairs independently annotated 1,330 Reddit posts using a structured codebook. The annotation process followed an iterative refinement approach: (1) A pilot round of 100 posts (Cohen's κ = [X]) identified edge cases, leading to codebook refinement; (2) Main annotation of 1,230 posts in batches with ongoing IRR monitoring (mean κ = [Y]); (3) Disagreement resolution through discussion achieved final consensus labels. Final inter-rater reliability was κ = [Z], indicating [substantial/almost perfect] agreement (Landis & Koch, 1977)."

### Validation Results to Report

1. **Inter-rater reliability**: Cohen's Kappa
2. **LLM vs Human accuracy**: Overall accuracy, per-category F1
3. **Confusion matrix**: LLM predictions vs human labels
4. **Systematic errors**: Where does LLM fail?

---

## Quick Reference

### Files Location

```
docs/
├── annotation_complete_guide.md (this file)
├── annotation_guidelines.md (quick reference)
├── annotation_codebook.md (living document)
└── annotation_workflow.md (detailed process)

data/sample/
├── human_benchmark_sample_annotation.csv (Google Sheet)
└── human_benchmark_sample_validation.csv (LLM labels)

scripts/
└── calculate_irr.py (IRR calculation)
```

### Key Contacts

- **Project Lead**: Hunjun Shin
- **Annotators**: Hunjun Shin, Hunbae Moon
- **Questions**: Document in Google Sheet notes or codebook

---

## Final Checklist

Before starting:

- [ ] Read this complete guide
- [ ] Read annotation guidelines
- [ ] Read codebook
- [ ] Understand all 5 frame categories
- [ ] Know how to use Google Sheet
- [ ] Know how to run IRR script

During annotation:

- [ ] Annotate independently
- [ ] Document uncertain cases
- [ ] Check IRR after each batch
- [ ] Update codebook as needed

After completion:

- [ ] All posts have final labels
- [ ] Final Kappa > 0.70
- [ ] Codebook finalized
- [ ] Ready for LLM validation

---

**Good luck! This rigorous process will produce high-quality data for your ICWSM paper.** 🎓
