# Framing Classification Annotation Guidelines

## Overview

You are classifying Reddit posts about international affairs into **one of five framing categories**. Each post discusses one of four countries: North Korea, China, Iran, or Russia.

**Your task**: Read the post title and text, then assign the **primary frame** that best describes how the topic is presented.

---

## Framing Categories

### 1. THREAT

**Focus**: Military danger, nuclear weapons, missiles, war, aggression, hostile actions

**Examples**:

- "North Korea fires ballistic missile over Japan"
- "Iran threatens to restart nuclear program"  
- "Russia deploys troops near Ukrainian border"

**Key words**: missile, nuclear, attack, threaten, war, military, weapons, invasion, provocative

---

### 2. DIPLOMACY

**Focus**: Negotiations, peace talks, diplomatic engagement, cooperation, summits, treaties

**Examples**:

- "Trump and Kim to meet for historic summit"
- "Iran nuclear deal negotiations resume"
- "China and US agree to trade talks"

**Key words**: talks, summit, negotiate, agreement, peace, diplomat, cooperation, treaty, meeting

---

### 3. NEUTRAL

**Focus**: Factual reporting without clear positive/negative framing, news updates, objective information

**Examples**:

- "South Korean President visits North Korea"
- "China announces new economic policy"
- "Russian election results announced"

**Key words**: announces, reports, states, according to, update, official statement

---

### 4. ECONOMIC

**Focus**: Sanctions, trade, economic impact, business relations, financial matters

**Examples**:

- "New sanctions imposed on North Korea"
- "US-China trade war escalates"
- "Iran oil exports hit by sanctions"

**Key words**: sanctions, trade, economy, tariffs, exports, imports, business, market, investment

---

### 5. HUMANITARIAN

**Focus**: Human rights, refugees, citizens' welfare, humanitarian issues, living conditions

**Examples**:

- "North Korean defector shares story of escape"
- "Human rights violations reported in Iran"
- "Russian citizens protest government policies"

**Key words**: human rights, refugees, citizens, people, humanitarian, prison, freedom, protest, civilians

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
| Sanctions hurting citizens | ECONOMIC (primary mechanism) or HUMANITARIAN (if citizen suffering is focus) |
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
