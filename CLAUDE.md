# Project: NK Coercive Diplomacy Reddit

## Git Rules

### paper_revision/
- This subdirectory is its own independent git repo
- Remote: `https://github.com/Shinhunjun/revision.git` (branch: `main`)
- **Always push paper_revision changes to this remote**, not the parent repo
- Paper content lives under `paper_revision/paper/` (e.g., `paper/sections/`, `paper/main.tex`)
- Commands: `cd paper_revision && git add paper/ && git commit -m "..." && git push origin main`
- This repo is synced with Overleaf via GitHub (Overleaf Standard plan)

### Parent repo (nk-coercive-diplomacy-reddit)
- Remote: separate from paper_revision
- paper_revision/ is untracked by the parent repo (nested git repo)

## Paper Revision Rules

### Marking changes in LaTeX
- Revision commands are defined in `paper/main.tex` (using xcolor + ulem, NOT the changes package which conflicts with aaai25)
- Use these commands when editing any `.tex` file in `paper_revision/paper/sections/`:
  - `\replaced{new text}{original text}` — 문장 수정 시
  - `\added{new text}` — 새 내용 추가 시
  - `\deleted{original text}` — 내용 삭제 시
- **NEVER silently delete text** — even if removing content entirely, wrap it in `\deleted{...}`
- **NEVER use raw `\textcolor{red}{...}`** for revision marking
- For final submission: remove `\replaced/\added/\deleted` wrappers manually or set commands to passthrough

### Reviewer comment annotations
- Every `\replaced`, `\added`, or `\deleted` **must** be accompanied by a `\reviewercomment{...}`
- Place `\reviewercomment{}` immediately before or after the revision command
- Format: `\reviewercomment{R# XX: reviewer's exact wording}`
- Example: `\reviewercomment{R2 m1: ``short-term but lasting'' — conceptual tension}`
- Use the reviewer's **exact wording** as much as possible (per feedback memory)

### Respond only to what reviewers explicitly asked
- **Do not invent additional alternatives, concerns, or counterarguments** beyond what reviewers raised
- If reviewer R2-M6 names "topic fatigue, agenda saturation, compositional churn" → address exactly those three, not a fourth self-generated alternative
- Self-added hedges/alternatives create a burden to rebut that the reviewer never imposed, and invite new critique
- When in doubt, map each paper change back to a specific reviewer quote — if no quote exists, reconsider whether the change is needed
- Exception: obvious errors discovered during revision (typos, compile bugs, inconsistent numbers) may be fixed without reviewer prompting

### Avoid em-dashes (`---` / `--`) in prose
- **Do not use `---` or `--` to connect clauses** in `.tex` files
- Prefer a period, comma, colon, or parenthetical instead
- Example: `yet the data show neither pattern.` (NOT `yet the data show neither pattern---a V-shape...`)
- Reason: overuse of em-dashes is a stylistic tell of AI-generated text and disrupts reading flow
- Exception: em-dash is fine in established multi-word terms (e.g., `difference-in-difference`) — but never as a clause connector

### Numerical Claims — Verify Before Writing

**NEVER write a specific number into a .tex file without first verifying it against an authoritative source.**

Authoritative sources (in order of priority):
1. **Already-reported numbers in `paper/sections/results.tex`** — use these directly, no recomputation needed
2. **Official analysis JSON files in `data/results/revision/*.json`** — cross-check period definitions match the paper's DiD periods (P1: Jan 2017–Feb 2018, P2: Jun 2018–Jan 2019, P3: Mar 2019–Dec 2019)
3. **Scripts in `scripts/`** — only if re-run with confirmed correct period definitions and complete dataset

Before inserting any statistic (%, count, p-value, etc.):
- State which source it came from
- If sources disagree, resolve the discrepancy before writing
- If data coverage is incomplete (e.g., P3 only 4 months), do NOT use that number

### Response Letter — dual-format sync (.md + .tex)

The response letter is maintained in **two parallel files**:
- `paper_revision/response_letter.md` (source of truth for editing, renders on GitHub)
- `paper_revision/response_letter.tex` (standalone LaTeX, compiles to PDF for submission)

**Rule:** Any edit to one file MUST be mirrored in the other in the same commit.

How to apply:
- When adding/modifying a reviewer comment row, update BOTH the markdown table row and the LaTeX `tabularx` row.
- When updating a Summary item (Major/Minor), update BOTH the markdown numbered list item and the LaTeX `enumerate` item.
- Numbering must stay aligned across both files.
- The LaTeX file uses `\todo{...}` in place of the markdown `*[TODO: ...]*` marker — keep these synchronized.

Verification: after editing, re-compile `response_letter.tex` (`latexmk -pdf response_letter.tex`) to catch LaTeX syntax issues introduced by the mirror edit.

### Verification (PostToolUse hook)
- `verify_changes.sh` runs automatically after every Edit on a `.tex` file
- It checks:
  1. No `\textcolor{red}` in newly added lines
  2. No silently deleted lines (must appear in `\deleted{}` or `\replaced{}{}`)
- If verification fails → fix before committing
