#!/usr/bin/env python3
"""
Deep Community Analysis for GraphRAG Output
Generates:
1. Community topic comparison table (Top 5 per period)
2. Keyword frequency analysis from summaries
3. Findings analysis with key insights
"""

import pandas as pd
import json
from collections import Counter
from pathlib import Path
import re

# Stopwords for keyword extraction
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'which', 'who',
    'whom', 'whose', 'what', 'where', 'when', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
    'then', 'thus', 'however', 'therefore', 'although', 'because', 'since', 'while',
    'if', 'unless', 'until', 'before', 'after', 'above', 'below', 'between', 'into',
    'through', 'during', 'about', 'against', 'among', 'throughout', 'despite',
    'towards', 'upon', 'within', 'without', 'according', 'including', 'following',
    'regarding', 'concerning', 'they', 'them', 'their', 'he', 'she', 'him', 'her',
    'his', 'hers', 'we', 'us', 'our', 'you', 'your', 'i', 'me', 'my', 'community',
    'entities', 'entity', 'relationship', 'relationships', 'report', 'key', 'significant',
    'various', 'several', 'particularly', 'especially', 'including', 'highlighting',
    'characterized', 'surrounding', 'centers', 'around', 'between', 'among', 'role',
    'plays', 'important', 'involves', 'related', 'associated', 'connected'
}

PERIOD_NAMES = {
    1: "P1 (Pre-Singapore: Jan 2017 - May 2018)",
    2: "P2 (Singapore-Hanoi: Jun 2018 - Feb 2019)",
    3: "P3 (Post-Hanoi: Mar 2019 - Dec 2019)"
}


def load_community_reports(period: int) -> pd.DataFrame:
    """Load community reports for a period."""
    path = Path(f"graphrag/period{period}/output/community_reports.parquet")
    df = pd.read_parquet(path)
    return df.sort_values('rank', ascending=False)


def extract_keywords(text: str, top_n: int = 20) -> list:
    """Extract top keywords from text."""
    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Filter stopwords
    words = [w for w in words if w not in STOPWORDS]
    # Count and return top N
    return Counter(words).most_common(top_n)


def parse_findings(findings_data) -> list:
    """Parse findings from various formats."""
    import numpy as np
    
    if findings_data is None:
        return []
    
    # Handle numpy arrays
    if isinstance(findings_data, np.ndarray):
        findings_data = findings_data.tolist()
    
    if isinstance(findings_data, list):
        return findings_data
    
    if isinstance(findings_data, str):
        try:
            return json.loads(findings_data)
        except:
            return [findings_data]
    
    return []


def analyze_top_communities(df: pd.DataFrame, top_n: int = 5) -> list:
    """Get top N communities with their details."""
    results = []
    for _, row in df.head(top_n).iterrows():
        results.append({
            'id': row['human_readable_id'],
            'title': row['title'],
            'summary': row['summary'][:300] + '...' if len(row['summary']) > 300 else row['summary'],
            'rank': row['rank'],
            'size': row.get('size', 'N/A')
        })
    return results


def analyze_keywords_from_summaries(df: pd.DataFrame) -> dict:
    """Extract keyword frequencies from all summaries."""
    all_text = ' '.join(df['summary'].dropna().tolist())
    return extract_keywords(all_text, top_n=25)


def analyze_findings(df: pd.DataFrame, top_n: int = 5) -> list:
    """Extract key findings from top communities."""
    all_findings = []
    for _, row in df.head(top_n).iterrows():
        findings = parse_findings(row.get('findings'))
        if findings:
            for finding in findings[:3]:  # Top 3 findings per community
                if isinstance(finding, dict):
                    all_findings.append({
                        'community': row['title'],
                        'summary': finding.get('summary', str(finding))[:250],
                        'explanation': finding.get('explanation', '')[:200] if finding.get('explanation') else ''
                    })
                else:
                    all_findings.append({
                        'community': row['title'],
                        'summary': str(finding)[:250],
                        'explanation': ''
                    })
    return all_findings


def generate_markdown_report(results: dict) -> str:
    """Generate comprehensive markdown report."""
    md = "# GraphRAG Community Deep Analysis Report\n\n"
    md += "This report provides detailed analysis of community structures across three diplomatic periods.\n\n"
    md += "---\n\n"
    
    # Section 1: Top Communities Comparison
    md += "## 1. Top Communities by Period\n\n"
    
    for period in [1, 2, 3]:
        md += f"### {PERIOD_NAMES[period]}\n\n"
        md += "| Rank | Title | Size | Summary |\n"
        md += "|------|-------|------|--------|\n"
        
        for i, comm in enumerate(results[f'period{period}']['top_communities'], 1):
            title = comm['title'][:40] + '...' if len(comm['title']) > 40 else comm['title']
            summary = comm['summary'][:100] + '...' if len(comm['summary']) > 100 else comm['summary']
            md += f"| {i} | {title} | {comm['size']} | {summary} |\n"
        md += "\n"
    
    # Section 2: Keyword Frequency Comparison
    md += "---\n\n## 2. Keyword Frequency Analysis\n\n"
    md += "Top keywords extracted from community summaries:\n\n"
    
    md += "| Rank | P1 Keywords | P2 Keywords | P3 Keywords |\n"
    md += "|------|-------------|-------------|-------------|\n"
    
    max_keywords = 15
    for i in range(max_keywords):
        p1_kw = results['period1']['keywords'][i] if i < len(results['period1']['keywords']) else ('', 0)
        p2_kw = results['period2']['keywords'][i] if i < len(results['period2']['keywords']) else ('', 0)
        p3_kw = results['period3']['keywords'][i] if i < len(results['period3']['keywords']) else ('', 0)
        
        md += f"| {i+1} | {p1_kw[0]} ({p1_kw[1]}) | {p2_kw[0]} ({p2_kw[1]}) | {p3_kw[0]} ({p3_kw[1]}) |\n"
    
    md += "\n"
    
    # Section 3: Keyword Evolution Insights
    md += "---\n\n## 3. Keyword Evolution Insights\n\n"
    
    # Find unique keywords per period
    p1_words = set([k[0] for k in results['period1']['keywords'][:15]])
    p2_words = set([k[0] for k in results['period2']['keywords'][:15]])
    p3_words = set([k[0] for k in results['period3']['keywords'][:15]])
    
    p2_new = p2_words - p1_words
    p3_new = p3_words - p2_words
    p1_disappeared = p1_words - p2_words
    
    md += "### Keywords Appearing in P2 (not in P1)\n"
    md += f"- {', '.join(sorted(p2_new)) if p2_new else 'None'}\n\n"
    
    md += "### Keywords Appearing in P3 (not in P2)\n"
    md += f"- {', '.join(sorted(p3_new)) if p3_new else 'None'}\n\n"
    
    md += "### Keywords Disappearing after P1\n"
    md += f"- {', '.join(sorted(p1_disappeared)) if p1_disappeared else 'None'}\n\n"
    
    # Section 4: Key Findings
    md += "---\n\n## 4. Key Findings from Top Communities\n\n"
    
    for period in [1, 2, 3]:
        md += f"### {PERIOD_NAMES[period]}\n\n"
        findings = results[f'period{period}']['findings']
        
        if findings:
            for j, finding in enumerate(findings[:5], 1):
                md += f"**{j}. From: {finding['community'][:50]}**\n"
                md += f"> {finding['summary']}\n\n"
        else:
            md += "No detailed findings available.\n\n"
    
    # Section 5: Summary for Paper
    md += "---\n\n## 5. Summary for Paper Inclusion\n\n"
    md += "### Key Observations:\n\n"
    md += "1. **Pre-Singapore (P1)**: Community discourse centered on [extracted themes]\n"
    md += "2. **Singapore-Hanoi (P2)**: Shift towards [diplomatic/summit themes]\n"
    md += "3. **Post-Hanoi (P3)**: Partial reversion with [specific characteristics]\n\n"
    md += "### Supporting the Asymmetric Ratchet Effect:\n"
    md += "- The keyword and community theme analysis shows [specific evidence]\n\n"
    
    md += "---\n\n*Generated by GraphRAG Community Deep Analysis*\n"
    
    return md


def main():
    print("=" * 60)
    print("GraphRAG Community Deep Analysis")
    print("=" * 60)
    
    results = {}
    
    for period in [1, 2, 3]:
        print(f"\nAnalyzing Period {period}...")
        df = load_community_reports(period)
        
        print(f"  - Loaded {len(df)} communities")
        
        # Analysis 1: Top communities
        top_communities = analyze_top_communities(df, top_n=5)
        print(f"  - Extracted top 5 communities")
        
        # Analysis 2: Keywords
        keywords = analyze_keywords_from_summaries(df)
        print(f"  - Extracted {len(keywords)} keywords")
        
        # Analysis 3: Findings
        findings = analyze_findings(df, top_n=5)
        print(f"  - Extracted {len(findings)} findings")
        
        results[f'period{period}'] = {
            'top_communities': top_communities,
            'keywords': keywords,
            'findings': findings,
            'total_communities': len(df)
        }
    
    # Generate markdown report
    print("\nGenerating report...")
    report = generate_markdown_report(results)
    
    # Save report
    output_path = Path("data/results/COMMUNITY_DEEP_ANALYSIS.md")
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")
    
    # Save JSON
    json_path = Path("data/results/community_deep_analysis.json")
    
    # Convert keywords tuples to dicts for JSON
    for period in [1, 2, 3]:
        results[f'period{period}']['keywords'] = [
            {'word': k[0], 'count': k[1]} 
            for k in results[f'period{period}']['keywords']
        ]
    
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"JSON saved to: {json_path}")
    
    # Print quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    
    for period in [1, 2, 3]:
        print(f"\n{PERIOD_NAMES[period]}:")
        print(f"  Communities: {results[f'period{period}']['total_communities']}")
        top_kw = [k['word'] for k in results[f'period{period}']['keywords'][:5]]
        print(f"  Top Keywords: {', '.join(top_kw)}")


if __name__ == "__main__":
    main()
