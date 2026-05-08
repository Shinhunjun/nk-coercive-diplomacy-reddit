"""
Revision Analysis: SUTVA Author Overlap (R2-M4)
================================================
Quantifies author overlap between North Korea (treatment) and each control
country (China, Iran, Russia) at both the post-level and comment-level.

Output:
- Post-level overlap (denominator: NK authors in v2 framing analysis sample)
- Comment-level overlap (denominator: NK comment authors in recursive sample)

This script reproduces the numbers reported in:
- response_letter (Section 1, Item 3 + Section 2, R2-M4)
- paper/sections/discussion.tex (SUTVA Limitations paragraph)

Expected output (from current data):
    Post-level:    NK ∩ Iran  4.07% (163/4009)
                   NK ∩ China 3.07% (123/4009)
                   NK ∩ Russia 4.12% (165/4009)
    Comment-level: NK ∩ Iran  18.63% (5126/27521)
                   NK ∩ China 23.40% (6440/27521)
                   NK ∩ Russia 32.55% (8959/27521)
"""

import json
import os

import pandas as pd

OUT_DIR = "data/results/revision"
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE_AUTHORS = {"[deleted]", "[removed]", "AutoModerator"}

# ── Post-level: v2 framing analysis sample ────────────────────────────────────
# NK denominator: authors appearing in nk_framing_v2.csv (the DiD analysis sample)
nk_v2 = pd.read_csv("data/results/final_framing_v2/nk_framing_v2.csv")
nk_meta = pd.read_csv(
    "data/nk/nk_posts_merged.csv", low_memory=False, usecols=["id", "author"]
).drop_duplicates(subset=["id"])
nk_posts = pd.merge(nk_v2, nk_meta, on="id", how="inner")
nk_posts = nk_posts.dropna(subset=["author"])
nk_posts = nk_posts[~nk_posts["author"].isin(EXCLUDE_AUTHORS)]
nk_post_authors = set(nk_posts["author"].unique())

post_results = {"denominator_NK_authors": len(nk_post_authors), "by_country": {}}
print("=" * 60)
print("POST-LEVEL OVERLAP (denominator: NK_v2 authors)")
print("=" * 60)
print(f"NK authors (v2 sample): {len(nk_post_authors)}")
for ctry in ["iran", "china", "russia"]:
    f = pd.read_csv(f"data/results/final_framing_v2/{ctry}_framing_v2.csv")
    m = pd.read_csv(
        f"data/control/{ctry}_posts.csv", low_memory=False, usecols=["id", "author"]
    ).drop_duplicates(subset=["id"])
    df = pd.merge(f, m, on="id", how="inner").dropna(subset=["author"])
    df = df[~df["author"].isin(EXCLUDE_AUTHORS)]
    c_authors = set(df["author"].unique())
    overlap = nk_post_authors & c_authors
    pct = 100 * len(overlap) / len(nk_post_authors)
    print(
        f"  NK ∩ {ctry.title():<7}: {len(overlap):>4d} / {len(nk_post_authors)} = {pct:.2f}%  "
        f"(country authors: {len(c_authors)})"
    )
    post_results["by_country"][ctry] = {
        "country_authors_v2": len(c_authors),
        "overlap_authors": len(overlap),
        "overlap_pct_of_NK": round(pct, 2),
    }

# ── Comment-level: recursive comment sample ───────────────────────────────────
# NK denominator: authors in nk_comments_recursive_roberta_final.csv
nk_comments = pd.read_csv(
    "data/processed/nk_comments_recursive_roberta_final.csv",
    low_memory=False,
    usecols=["author"],
).dropna()
nk_comments = nk_comments[~nk_comments["author"].isin(EXCLUDE_AUTHORS)]
nk_comment_authors = set(nk_comments["author"].unique())

comment_results = {
    "denominator_NK_comment_authors": len(nk_comment_authors),
    "by_country": {},
}
print()
print("=" * 60)
print("COMMENT-LEVEL OVERLAP (denominator: NK comment authors)")
print("=" * 60)
print(f"NK comment authors (recursive): {len(nk_comment_authors)}")
for ctry in ["iran", "china", "russia"]:
    c = pd.read_csv(
        f"data/processed/{ctry}_comments_recursive_roberta_final.csv",
        low_memory=False,
        usecols=["author"],
    ).dropna()
    c = c[~c["author"].isin(EXCLUDE_AUTHORS)]
    c_authors = set(c["author"].unique())
    overlap = nk_comment_authors & c_authors
    pct = 100 * len(overlap) / len(nk_comment_authors)
    print(
        f"  NK ∩ {ctry.title():<7}: {len(overlap):>5d} / {len(nk_comment_authors)} = {pct:.2f}%  "
        f"(country authors: {len(c_authors)})"
    )
    comment_results["by_country"][ctry] = {
        "country_comment_authors": len(c_authors),
        "overlap_authors": len(overlap),
        "overlap_pct_of_NK": round(pct, 2),
    }

# ── Save ─────────────────────────────────────────────────────────────────────
out = {
    "design": (
        "Post-level: NK_v2 authors (from final_framing_v2/nk_framing_v2.csv joined "
        "to nk_posts_merged.csv) intersected with control country v2 authors (joined "
        "to data/control/{ctry}_posts.csv). Comment-level: NK recursive comment "
        "authors intersected with control country recursive comment authors. "
        "[deleted], [removed], AutoModerator excluded from all sets."
    ),
    "post_level": post_results,
    "comment_level": comment_results,
}
out_path = os.path.join(OUT_DIR, "sutva_overlap.json")
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nSaved: {out_path}")
