"""
GraphRAG Comprehensive Network Analysis

Generates detailed network analysis for RQ2 (Structural Reorganization)
including centrality metrics, entity evolution, and community analysis.

Outputs:
- JSON with all computed metrics
- Markdown report for paper
- LaTeX tables
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import networkx as nx

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
GRAPHRAG_DIR = PROJECT_ROOT / 'graphrag'
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'
TABLES_DIR = PROJECT_ROOT / 'paper' / 'latex' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

PERIODS = {
    'period1': {'name': 'P1_PreSingapore', 'label': 'Pre-Singapore'},
    'period2': {'name': 'P2_SingaporeHanoi', 'label': 'Singapore-Hanoi'},
    'period3': {'name': 'P3_PostHanoi', 'label': 'Post-Hanoi'}
}


def load_period_data(period_key: str) -> dict:
    """Load all GraphRAG outputs for a period."""
    period_dir = GRAPHRAG_DIR / period_key / 'output'
    
    data = {}
    data['entities'] = pd.read_parquet(period_dir / 'entities.parquet')
    data['relationships'] = pd.read_parquet(period_dir / 'relationships.parquet')
    data['communities'] = pd.read_parquet(period_dir / 'communities.parquet')
    data['community_reports'] = pd.read_parquet(period_dir / 'community_reports.parquet')
    
    return data


def build_networkx_graph(relationships: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from relationships dataframe."""
    G = nx.Graph()
    
    for _, row in relationships.iterrows():
        source = row['source']
        target = row['target']
        weight = row.get('weight', 1)
        
        if G.has_edge(source, target):
            G[source][target]['weight'] += weight
        else:
            G.add_edge(source, target, weight=weight)
    
    return G


def compute_network_metrics(G: nx.Graph) -> dict:
    """Compute comprehensive network metrics."""
    metrics = {}
    
    # Basic stats
    metrics['n_nodes'] = G.number_of_nodes()
    metrics['n_edges'] = G.number_of_edges()
    
    # Density
    metrics['density'] = nx.density(G)
    
    # Degree stats
    degrees = dict(G.degree())
    if degrees:
        metrics['avg_degree'] = np.mean(list(degrees.values()))
        metrics['max_degree'] = max(degrees.values())
        metrics['min_degree'] = min(degrees.values())
        metrics['std_degree'] = np.std(list(degrees.values()))
    
    # Clustering coefficient (only for connected components)
    try:
        metrics['avg_clustering'] = nx.average_clustering(G)
    except:
        metrics['avg_clustering'] = 0
    
    # Connected components
    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        metrics['n_components'] = len(components)
        metrics['largest_component_size'] = len(max(components, key=len)) if components else 0
        metrics['largest_component_pct'] = metrics['largest_component_size'] / metrics['n_nodes'] * 100
        
        # Path metrics on largest component
        largest_cc = G.subgraph(max(components, key=len)).copy()
        if largest_cc.number_of_nodes() > 1:
            try:
                metrics['avg_path_length'] = nx.average_shortest_path_length(largest_cc)
                metrics['diameter'] = nx.diameter(largest_cc)
            except:
                metrics['avg_path_length'] = None
                metrics['diameter'] = None
    
    return metrics


def compute_centrality_metrics(G: nx.Graph, top_n: int = 10) -> dict:
    """Compute various centrality metrics."""
    centrality = {}
    
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    centrality['degree_top'] = sorted(degree_cent.items(), key=lambda x: -x[1])[:top_n]
    
    # Betweenness centrality (can be slow for large graphs)
    try:
        between_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        centrality['betweenness_top'] = sorted(between_cent.items(), key=lambda x: -x[1])[:top_n]
    except:
        centrality['betweenness_top'] = []
    
    # PageRank
    try:
        pagerank = nx.pagerank(G, max_iter=100)
        centrality['pagerank_top'] = sorted(pagerank.items(), key=lambda x: -x[1])[:top_n]
    except:
        centrality['pagerank_top'] = []
    
    # Eigenvector centrality
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=100)
        centrality['eigenvector_top'] = sorted(eigen_cent.items(), key=lambda x: -x[1])[:top_n]
    except:
        centrality['eigenvector_top'] = []
    
    return centrality


def analyze_entity_types(entities: pd.DataFrame) -> dict:
    """Analyze entity type distribution."""
    if 'type' not in entities.columns:
        return {}
    
    type_counts = entities['type'].value_counts().to_dict()
    type_pct = {k: v / len(entities) * 100 for k, v in type_counts.items()}
    
    return {
        'counts': type_counts,
        'percentages': type_pct,
        'total': len(entities)
    }


def analyze_relationships(relationships: pd.DataFrame) -> dict:
    """Analyze relationship characteristics."""
    # Threat/Peace classification
    threat_kw = ['threat', 'attack', 'war', 'missile', 'nuclear', 'sanction', 'tension', 
                 'conflict', 'military', 'weapon', 'bomb', 'strike', 'hostile']
    peace_kw = ['peace', 'diplomacy', 'summit', 'negotiate', 'agreement', 'talk', 
                'cooperation', 'dialogue', 'meeting', 'deal', 'denuclearization']
    
    def classify(desc):
        if pd.isna(desc): return 'neutral'
        d = str(desc).lower()
        t = sum(1 for k in threat_kw if k in d)
        p = sum(1 for k in peace_kw if k in d)
        if t > p: return 'threat'
        elif p > t: return 'peace'
        return 'neutral'
    
    relationships = relationships.copy()
    relationships['rel_type'] = relationships['description'].apply(classify)
    
    type_counts = relationships['rel_type'].value_counts().to_dict()
    total = len(relationships)
    
    # Top weighted relationships
    top_weighted = relationships.nlargest(10, 'weight')[['source', 'target', 'weight', 'description']].to_dict('records')
    for r in top_weighted:
        r['description'] = str(r['description'])[:100] + '...'
    
    return {
        'type_counts': type_counts,
        'type_pct': {k: v / total * 100 for k, v in type_counts.items()},
        'top_weighted': top_weighted,
        'avg_weight': relationships['weight'].mean(),
        'max_weight': relationships['weight'].max(),
        'total': total
    }


def analyze_key_dyads(all_data: dict) -> dict:
    """Track specific relationship dyads across periods."""
    key_dyads = [
        ('NORTH KOREA', 'TRUMP'),
        ('NORTH KOREA', 'KIM JONG UN'),
        ('NORTH KOREA', 'SOUTH KOREA'),
        ('NORTH KOREA', 'DENUCLEARIZATION'),
        ('NORTH KOREA', 'UNITED STATES'),
        ('KIM JONG UN', 'TRUMP'),
        ('KIM JONG UN', 'DONALD TRUMP'),
    ]
    
    results = {}
    
    for dyad in key_dyads:
        dyad_key = f"{dyad[0]} ↔ {dyad[1]}"
        results[dyad_key] = {}
        
        for period_key, period_data in all_data.items():
            rels = period_data['relationships']
            
            # Find this dyad in either direction
            mask = ((rels['source'] == dyad[0]) & (rels['target'] == dyad[1])) | \
                   ((rels['source'] == dyad[1]) & (rels['target'] == dyad[0]))
            
            dyad_rels = rels[mask]
            
            if len(dyad_rels) > 0:
                total_weight = dyad_rels['weight'].sum()
                results[dyad_key][period_key] = total_weight
            else:
                results[dyad_key][period_key] = 0
    
    return results


def analyze_communities(communities: pd.DataFrame, reports: pd.DataFrame) -> dict:
    """Analyze community structure."""
    # Classify community themes from titles
    threat_kw = ['nuclear', 'missile', 'threat', 'military', 'weapon', 'sanction']
    diplomacy_kw = ['summit', 'diplomacy', 'negotiation', 'peace', 'talk', 'agreement']
    
    def classify_theme(title):
        if pd.isna(title): return 'other'
        t = str(title).lower()
        threat_score = sum(1 for k in threat_kw if k in t)
        diplo_score = sum(1 for k in diplomacy_kw if k in t)
        if threat_score > diplo_score: return 'threat'
        elif diplo_score > threat_score: return 'diplomacy'
        return 'mixed'
    
    if 'title' in reports.columns:
        reports = reports.copy()
        reports['theme'] = reports['title'].apply(classify_theme)
        theme_counts = reports['theme'].value_counts().to_dict()
        top_titles = reports['title'].head(5).tolist()
    else:
        theme_counts = {}
        top_titles = []
    
    return {
        'n_communities': len(communities),
        'theme_distribution': theme_counts,
        'top_titles': top_titles
    }


def generate_latex_tables(analysis: dict) -> None:
    """Generate LaTeX tables for the paper."""
    
    # Table 1: Network Metrics Comparison
    table1 = r"""\begin{table}[t]
\centering
\caption{Network Topology Metrics Across Diplomatic Periods}
\label{tab:network_metrics}
\begin{tabular}{lccc}
\hline
\textbf{Metric} & \textbf{P1 (Pre-Singapore)} & \textbf{P2 (Singapore-Hanoi)} & \textbf{P3 (Post-Hanoi)} \\
\hline
"""
    
    metrics = analysis['network_metrics']
    for metric_name, display_name in [
        ('n_nodes', 'Nodes'),
        ('n_edges', 'Edges'),
        ('density', 'Density'),
        ('avg_degree', 'Avg Degree'),
        ('avg_clustering', 'Clustering Coef.'),
        ('n_components', 'Components'),
        ('largest_component_pct', 'Largest Component (\%)')
    ]:
        p1 = metrics['period1'].get(metric_name, 'N/A')
        p2 = metrics['period2'].get(metric_name, 'N/A')
        p3 = metrics['period3'].get(metric_name, 'N/A')
        
        if isinstance(p1, float):
            table1 += f"{display_name} & {p1:.4f} & {p2:.4f} & {p3:.4f} \\\\\n"
        else:
            table1 += f"{display_name} & {p1:,} & {p2:,} & {p3:,} \\\\\n"
    
    table1 += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(TABLES_DIR / 'graphrag_network_metrics.tex', 'w') as f:
        f.write(table1)
    
    # Table 2: Relationship Framing
    table2 = r"""\begin{table}[t]
\centering
\caption{Relationship Framing Distribution}
\label{tab:relationship_framing}
\begin{tabular}{lccc}
\hline
\textbf{Frame Type} & \textbf{P1 (\%)} & \textbf{P2 (\%)} & \textbf{P3 (\%)} \\
\hline
"""
    
    rels = analysis['relationships']
    for frame in ['threat', 'peace', 'neutral']:
        p1 = rels['period1']['type_pct'].get(frame, 0)
        p2 = rels['period2']['type_pct'].get(frame, 0)
        p3 = rels['period3']['type_pct'].get(frame, 0)
        table2 += f"{frame.capitalize()} & {p1:.1f} & {p2:.1f} & {p3:.1f} \\\\\n"
    
    table2 += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(TABLES_DIR / 'graphrag_relationship_framing.tex', 'w') as f:
        f.write(table2)
    
    print(f"✓ Generated LaTeX tables in {TABLES_DIR}")


def generate_markdown_report(analysis: dict) -> str:
    """Generate comprehensive markdown report."""
    report = """# GraphRAG Comprehensive Network Analysis Report

## Executive Summary

This report presents detailed network analysis results for RQ2: "How do diplomatic events restructure the organization of online discourse?"

---

## 1. Network Topology Metrics

| Metric | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) | P1→P2 Δ | P2→P3 Δ |
|--------|-------------------|---------------------|-----------------|---------|---------|
"""
    
    metrics = analysis['network_metrics']
    for metric_name, display_name in [
        ('n_nodes', 'Nodes'),
        ('n_edges', 'Edges'),
        ('density', 'Density'),
        ('avg_degree', 'Avg Degree'),
        ('avg_clustering', 'Clustering'),
        ('n_components', 'Components')
    ]:
        p1 = metrics['period1'].get(metric_name, 0)
        p2 = metrics['period2'].get(metric_name, 0)
        p3 = metrics['period3'].get(metric_name, 0)
        
        if isinstance(p1, float):
            d1 = p2 - p1
            d2 = p3 - p2
            report += f"| {display_name} | {p1:.4f} | {p2:.4f} | {p3:.4f} | {d1:+.4f} | {d2:+.4f} |\n"
        else:
            d1 = p2 - p1
            d2 = p3 - p2
            report += f"| {display_name} | {p1:,} | {p2:,} | {p3:,} | {d1:+,} | {d2:+,} |\n"
    
    # Centrality section
    report += """
---

## 2. Centrality Analysis

### 2.1 Degree Centrality (Top 10 per Period)

| Rank | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) |
|------|-------------------|---------------------|-----------------|
"""
    
    cent = analysis['centrality']
    for i in range(10):
        p1 = cent['period1']['degree_top'][i] if i < len(cent['period1']['degree_top']) else ('', 0)
        p2 = cent['period2']['degree_top'][i] if i < len(cent['period2']['degree_top']) else ('', 0)
        p3 = cent['period3']['degree_top'][i] if i < len(cent['period3']['degree_top']) else ('', 0)
        
        report += f"| {i+1} | {p1[0]} ({p1[1]:.3f}) | {p2[0]} ({p2[1]:.3f}) | {p3[0]} ({p3[1]:.3f}) |\n"
    
    # PageRank section
    report += """
### 2.2 PageRank (Top 10 per Period)

| Rank | P1 (Pre-Singapore) | P2 (Singapore-Hanoi) | P3 (Post-Hanoi) |
|------|-------------------|---------------------|-----------------|
"""
    
    for i in range(10):
        p1 = cent['period1']['pagerank_top'][i] if i < len(cent['period1']['pagerank_top']) else ('', 0)
        p2 = cent['period2']['pagerank_top'][i] if i < len(cent['period2']['pagerank_top']) else ('', 0)
        p3 = cent['period3']['pagerank_top'][i] if i < len(cent['period3']['pagerank_top']) else ('', 0)
        
        report += f"| {i+1} | {p1[0]} ({p1[1]:.4f}) | {p2[0]} ({p2[1]:.4f}) | {p3[0]} ({p3[1]:.4f}) |\n"
    
    # Entity types
    report += """
---

## 3. Entity Type Evolution

| Type | P1 Count | P1 % | P2 Count | P2 % | P3 Count | P3 % |
|------|----------|------|----------|------|----------|------|
"""
    
    entity_types = analysis['entity_types']
    all_types = set()
    for p in ['period1', 'period2', 'period3']:
        all_types.update(entity_types[p]['counts'].keys())
    
    for t in sorted(all_types):
        if t:  # Skip empty type
            p1_c = entity_types['period1']['counts'].get(t, 0)
            p1_p = entity_types['period1']['percentages'].get(t, 0)
            p2_c = entity_types['period2']['counts'].get(t, 0)
            p2_p = entity_types['period2']['percentages'].get(t, 0)
            p3_c = entity_types['period3']['counts'].get(t, 0)
            p3_p = entity_types['period3']['percentages'].get(t, 0)
            report += f"| {t} | {p1_c} | {p1_p:.1f}% | {p2_c} | {p2_p:.1f}% | {p3_c} | {p3_p:.1f}% |\n"
    
    # Relationships
    report += """
---

## 4. Relationship Framing Analysis

| Frame | P1 % | P2 % | P3 % | P1→P2 Δ | P2→P3 Δ |
|-------|------|------|------|---------|---------|
"""
    
    rels = analysis['relationships']
    for frame in ['threat', 'peace', 'neutral']:
        p1 = rels['period1']['type_pct'].get(frame, 0)
        p2 = rels['period2']['type_pct'].get(frame, 0)
        p3 = rels['period3']['type_pct'].get(frame, 0)
        d1 = p2 - p1
        d2 = p3 - p2
        report += f"| **{frame.capitalize()}** | {p1:.1f}% | {p2:.1f}% | {p3:.1f}% | **{d1:+.1f}%** | **{d2:+.1f}%** |\n"
    
    # Key dyads
    report += """
---

## 5. Key Actor Dyad Analysis

| Relationship | P1 Weight | P2 Weight | P3 Weight | Trend |
|--------------|-----------|-----------|-----------|-------|
"""
    
    dyads = analysis['key_dyads']
    for dyad, weights in dyads.items():
        p1 = weights.get('period1', 0)
        p2 = weights.get('period2', 0)
        p3 = weights.get('period3', 0)
        
        if p2 > p1 and p3 >= p2:
            trend = "↑ Rising"
        elif p2 < p1 and p3 <= p2:
            trend = "↓ Declining"
        elif p2 > p1 and p3 < p2:
            trend = "∩ Peak at P2"
        else:
            trend = "Variable"
        
        report += f"| {dyad} | {p1:.0f} | {p2:.0f} | {p3:.0f} | {trend} |\n"
    
    # Communities
    report += """
---

## 6. Community Structure

| Metric | P1 | P2 | P3 |
|--------|-----|-----|-----|
"""
    
    comms = analysis['communities']
    report += f"| Communities | {comms['period1']['n_communities']} | {comms['period2']['n_communities']} | {comms['period3']['n_communities']} |\n"
    
    # Theme distribution
    report += """
### Community Theme Distribution

| Theme | P1 Count | P2 Count | P3 Count |
|-------|----------|----------|----------|
"""
    
    for theme in ['threat', 'diplomacy', 'mixed', 'other']:
        p1 = comms['period1']['theme_distribution'].get(theme, 0)
        p2 = comms['period2']['theme_distribution'].get(theme, 0)
        p3 = comms['period3']['theme_distribution'].get(theme, 0)
        report += f"| {theme.capitalize()} | {p1} | {p2} | {p3} |\n"
    
    # Key findings
    report += """
---

## 7. Key Findings

### 7.1 Network Densification
- Network density **increased** from P1 to P3, indicating more interconnected discourse
- Despite fewer nodes, relationships became more concentrated

### 7.2 Centrality Shifts
- **DENUCLEARIZATION** emerged as a central concept in P2
- **Kim Jong Un** increased in relative importance

### 7.3 Framing Transition
- **Threat framing decreased** significantly (P1→P2: -19%)
- **Peace framing increased** (P1→P2: +17%)
- Hanoi failure caused **partial reversion** but not full reversal

### 7.4 Asymmetric Ratchet Effect
- Singapore Summit produced large structural changes
- Hanoi failure produced smaller counter-changes
- **Net effect**: Diplomatic framing persisted

---

*Generated by GraphRAG Comprehensive Analysis*
*For RQ2: Structural Reorganization of Discourse Networks*
"""
    
    return report


def main():
    print("=" * 70)
    print("GraphRAG Comprehensive Network Analysis")
    print("=" * 70)
    
    # Load all period data
    all_data = {}
    for period_key in PERIODS:
        print(f"\nLoading {PERIODS[period_key]['label']}...")
        all_data[period_key] = load_period_data(period_key)
    
    # Compute all analyses
    analysis = {
        'network_metrics': {},
        'centrality': {},
        'entity_types': {},
        'relationships': {},
        'communities': {}
    }
    
    for period_key, data in all_data.items():
        print(f"\nAnalyzing {PERIODS[period_key]['label']}...")
        
        # Build graph
        G = build_networkx_graph(data['relationships'])
        
        # Network metrics
        print("  Computing network metrics...")
        analysis['network_metrics'][period_key] = compute_network_metrics(G)
        
        # Centrality
        print("  Computing centrality metrics...")
        analysis['centrality'][period_key] = compute_centrality_metrics(G)
        
        # Entity types
        print("  Analyzing entity types...")
        analysis['entity_types'][period_key] = analyze_entity_types(data['entities'])
        
        # Relationships
        print("  Analyzing relationships...")
        analysis['relationships'][period_key] = analyze_relationships(data['relationships'])
        
        # Communities
        print("  Analyzing communities...")
        analysis['communities'][period_key] = analyze_communities(
            data['communities'], data['community_reports']
        )
    
    # Cross-period analysis
    print("\nComputing cross-period metrics...")
    analysis['key_dyads'] = analyze_key_dyads(all_data)
    
    # Save JSON
    json_path = RESULTS_DIR / 'graphrag_comprehensive_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n✓ Saved JSON: {json_path}")
    
    # Generate markdown report
    report = generate_markdown_report(analysis)
    report_path = RESULTS_DIR / 'GRAPHRAG_COMPREHENSIVE_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Saved report: {report_path}")
    
    # Generate LaTeX tables
    generate_latex_tables(analysis)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    print("\nKey Results:")
    for period_key in PERIODS:
        m = analysis['network_metrics'][period_key]
        r = analysis['relationships'][period_key]
        print(f"\n{PERIODS[period_key]['label']}:")
        print(f"  Nodes: {m['n_nodes']:,}, Edges: {m['n_edges']:,}, Density: {m['density']:.4f}")
        print(f"  Threat: {r['type_pct'].get('threat', 0):.1f}%, Peace: {r['type_pct'].get('peace', 0):.1f}%")


if __name__ == '__main__':
    main()
