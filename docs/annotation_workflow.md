# Annotation Workflow: Step-by-Step Guide

## Overview

This is an **iterative annotation process** with codebook refinement. You'll annotate in phases, calculate agreement, refine guidelines, and repeat.

---

## Phase 1: Pilot Round (First 100 Posts)

### Step 1.1: Initial Independent Annotation

**Timeline**: 2-3 days

1. Both annotators read the [annotation guidelines](annotation_guidelines.md)
2. Both annotators read the [codebook](annotation_codebook.md)
3. **Independently** annotate posts 1-100 in Google Sheets
4. **Do NOT discuss** during this phase

### Step 1.2: Calculate Initial IRR

**Timeline**: 30 minutes

```bash
# Download Google Sheet as CSV
# Run IRR calculation
python scripts/calculate_irr.py data/sample/pilot_batch.csv "Pilot v1.0"
```

**Expected output**:

- Cohen's Kappa score
- Confusion matrix
- List of disagreements

### Step 1.3: Disagreement Discussion Meeting

**Timeline**: 2-3 hours

**Agenda**:

1. Review disagreements together
2. For each disagreement, discuss:
   - Why did you choose that label?
   - What was ambiguous?
   - What rule would help decide?
3. **Document patterns** in disagreements
4. **Do NOT change labels yet**

### Step 1.4: Codebook Refinement

**Timeline**: 1-2 hours

Based on discussion, update `annotation_codebook.md`:

1. Add new **decision rules** for edge cases found
2. Add **examples from actual data**
3. Clarify **ambiguous definitions**
4. Update **version number** (v1.0 → v1.1)

### Step 1.5: Re-annotate Pilot

**Timeline**: 1-2 days

1. Both annotators **re-annotate** posts 1-100 using updated codebook
2. Calculate IRR again
3. **Expected**: Kappa should improve

```bash
python scripts/calculate_irr.py data/sample/pilot_batch_v2.csv "Pilot v1.1"
```

**Decision point**:

- If Kappa > 0.60: Proceed to main annotation
- If Kappa < 0.60: Another refinement round

---

## Phase 2: Main Annotation (Posts 101-1,330)

### Batch Annotation Strategy

Annotate in **batches of 200-300 posts**:

| Batch | Posts | Check IRR? |
|-------|-------|-----------|
| Batch 1 | 101-300 | ✓ Yes |
| Batch 2 | 301-500 | ✓ Yes |
| Batch 3 | 501-700 | ✓ Yes |
| Batch 4 | 701-900 | ✓ Yes |
| Batch 5 | 901-1,100 | ✓ Yes |
| Batch 6 | 1,101-1,330 | ✓ Yes |

### After Each Batch

1. **Calculate IRR** for that batch
2. **Review major disagreements** (don't need full meeting)
3. **Update codebook** if new edge cases appear
4. **Continue** to next batch

---

## Phase 3: Final Disagreement Resolution

### Step 3.1: Identify All Disagreements

**Timeline**: 1 hour

```bash
# Calculate final IRR on all 1,330 posts
python scripts/calculate_irr.py data/sample/human_benchmark_sample_annotation.csv "Final"
```

This generates:

- `human_benchmark_sample_annotation_disagreements.csv`

### Step 3.2: Resolution Meeting

**Timeline**: 4-6 hours (depending on disagreements)

**Process**:

1. Go through each disagreement
2. **Discuss** using codebook rules
3. **Reach consensus** on final label
4. **Document reasoning** in notes column
5. Fill in `final_frame` column in Google Sheet

### Step 3.3: Final Dataset

**Timeline**: 30 minutes

1. Download final Google Sheet
2. Verify all posts have `final_frame` filled
3. Save as `human_benchmark_sample_annotation_final.csv`

---

## Phase 4: LLM Validation Analysis

### Step 4.1: Compare LLM vs Human Labels

```bash
python scripts/analyze_llm_vs_human.py
```

This will:

- Load human consensus labels
- Load LLM labels from validation file
- Calculate accuracy, precision, recall, F1
- Generate confusion matrix
- Identify systematic errors

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Pilot Round (100 posts) | 1 week | 1 week |
| Main Annotation (1,230 posts) | 4-5 weeks | 5-6 weeks |
| Final Resolution | 1 week | 6-7 weeks |
| **Total** | **6-7 weeks** | |

**Per annotator time**: ~50-60 hours total

---

## Quality Checkpoints

### After Pilot

- [ ] Kappa > 0.60
- [ ] Codebook has edge case rules
- [ ] Both annotators understand all categories

### During Main Annotation

- [ ] Batch Kappa stays > 0.60
- [ ] Disagreement rate not increasing
- [ ] New edge cases documented

### Before Final

- [ ] All 1,330 posts annotated
- [ ] Overall Kappa > 0.70 (target)
- [ ] Disagreements categorized
- [ ] Codebook finalized

---

## Files to Track

| File | Purpose | Update Frequency |
|------|---------|------------------|
| `annotation_guidelines.md` | Quick reference | Rarely |
| `annotation_codebook.md` | Detailed rules | After each phase |
| Google Sheet | Annotation data | Daily |
| `*_disagreements.csv` | IRR tracking | After each batch |

---

## Communication Protocol

### During Independent Annotation

- ❌ Do NOT discuss specific posts
- ✓ Can ask clarifying questions about codebook
- ✓ Document questions in notes column

### During Discussion

- ✓ Discuss reasoning openly
- ✓ Reference codebook rules
- ✓ Document new rules needed

---

## For Paper Methods Section

Document in your paper:

> "Two military officers with expertise in North Korean affairs independently annotated 1,330 Reddit posts using a structured codebook. The annotation process followed an iterative refinement approach: (1) A pilot round of 100 posts (Cohen's κ = [X]) identified edge cases, leading to codebook refinement; (2) Main annotation of 1,230 posts in batches with ongoing IRR monitoring (κ = [Y]); (3) Disagreement resolution through discussion achieved final consensus labels. Final inter-rater reliability was κ = [Z], indicating [substantial/almost perfect] agreement."
