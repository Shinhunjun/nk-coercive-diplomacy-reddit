# Framing Classification Annotation Guidelines

## Overview

You are classifying Reddit posts about international affairs into **one of five framing categories**. Each post discusses one of four countries: North Korea, China, Iran, or Russia.

**Your task**: Read the post title and text, then assign the **primary frame** that best describes how the topic is presented.

---

## Framing Categories

> **Note**: These definitions match exactly what was used in the automated classification system.

### 1. THREAT

**Definition**: Military threat, nuclear weapons, missiles, war risk

**Examples**:

- "North Korea fires ballistic missile over Japan"
- "Iran threatens to restart nuclear program"  
- "Russia deploys troops near Ukrainian border"

---

### 2. DIPLOMACY

**Definition**: Negotiation, dialogue, peace, cooperation

**Examples**:

- "Trump and Kim to meet for historic summit"
- "Iran nuclear deal negotiations resume"
- "China and US agree to trade talks"

---

### 3. NEUTRAL

**Definition**: Neutral information delivery

**Examples**:

- "China announces new economic policy"
- "Russia holds presidential election"
- "Russian election results announced"

---

### 4. ECONOMIC

**Definition**: Economic sanctions, trade aspects

**Examples**:

- "New sanctions imposed on North Korea"
- "US-China trade war escalates"
- "Iran oil exports hit by sanctions"

---

### 5. HUMANITARIAN

**Definition**: Human rights, refugees, civilian issues

**Examples**:

- "North Korean defector shares story of escape"
- "Human rights violations reported in Iran"
- "Russian citizens protest government policies"

---

## Annotation Rules

### Rule 1: Choose the PRIMARY frame

If a post contains multiple frames (e.g., both threat and diplomacy), choose the **dominant** or **primary** frame based on the main focus.

### Rule 2: Focus on the FRAMING, not the topic

- A post can be about nuclear weapons but framed as DIPLOMACY if it focuses on negotiations
- A post can be about a summit but framed as THREAT if it emphasizes failures or dangers

### Rule 3: When uncertain, consider the author's perspective

- What is the post trying to emphasize?
- What emotion or reaction is it likely to evoke in readers?

### Rule 4: NEUTRAL is for truly objective posts

Only use NEUTRAL when there's no clear positive/negative/threatening framing. Pure news reporting without editorializing.

---

## Edge Cases

| Situation | Recommended Frame |
|-----------|-------------------|
| Trade war with tariffs AND diplomatic talks | Choose dominant focus |
| Nuclear negotiations (weapons + talks) | DIPLOMACY if talks emphasized, THREAT if weapons emphasized |
| Sanctions hurting citizens | ECONOMIC (primary mechanism) or HUMANITARIAN (if civilian suffering is focus) |
| Failed diplomatic summit | DIPLOMACY (still about negotiations) |
| Military exercises as show of force | THREAT |

---

## Annotation Process

1. **Read** the title carefully (main signal)
2. **Read** the full text if available
3. **Select** ONE frame from the five categories
4. If **uncertain**, make a note in the notes column
5. Work **independently** - do not discuss with other annotator until IRR calculation

---

## Quality Checklist

Before submitting:

- [ ] Every row has exactly ONE frame selected
- [ ] No rows left blank
- [ ] Difficult cases have notes
- [ ] You did NOT consult with the other annotator

---

## Contact

If you encounter posts that don't fit any category or have questions, note them in the spreadsheet for later discussion.
