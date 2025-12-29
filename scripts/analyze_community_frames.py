"""
Analyze Frame Distribution per Community

Links edge framing results with community assignments to determine
the dominant frame orientation of each community.
"""
import pandas as pd
from pathlib import Path

GRAPHRAG_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/graphrag")
RESULTS_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/data/results")

FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

def analyze_community_frames(period: str):
    """Analyze frame distribution per community for a period."""
    print(f"\n{'='*60}")
    print(f"Analyzing {period}")
    print(f"{'='*60}")
    
    # Load edge framing results
    edge_path = RESULTS_DIR / f"edge_framing_{period}.csv"
    if not edge_path.exists():
        print(f"  [WARNING] Edge framing results not found: {edge_path}")
        return None
    
    edges_df = pd.read_csv(edge_path)
    print(f"  Loaded {len(edges_df)} edge framing results")
    
    # Load community reports
    comm_path = GRAPHRAG_DIR / period / "output" / "community_reports.parquet"
    comm_df = pd.read_parquet(comm_path)
    print(f"  Loaded {len(comm_df)} communities")
    
    # Load relationships to get community assignments
    rel_path = GRAPHRAG_DIR / period / "output" / "relationships.parquet"
    rel_df = pd.read_parquet(rel_path)
    
    # Merge edge framing with edges (they should align by index or source/target)
    # edges_df already has source/target, so we can join
    
    # Strategy: For each entity in a community, find edges where it's source or target
    # Then aggregate frame distribution for those edges
    
    # Load entities with community info
    entities_path = GRAPHRAG_DIR / period / "output" / "entities.parquet"
    entities_df = pd.read_parquet(entities_path)
    
    # Community assignments are in community_reports
    # We need to understand the hierarchy - for now, use top-level communities
    top_communities = comm_df[comm_df['level'] == comm_df['level'].max()].copy()
    print(f"  Top-level communities: {len(top_communities)}")
    
    # Alternative approach: Classify communities by their summary content using our framing model
    # But for now, let's use the edge frames
    
    # Create summary statistics
    summary = {
        'period': period,
        'total_communities': len(comm_df),
        'total_edges': len(edges_df),
        'frame_distribution': edges_df['frame'].value_counts().to_dict()
    }
    
    # Print frame distribution
    print(f"\n  Overall Edge Frame Distribution:")
    for frame in FRAME_CATEGORIES:
        count = (edges_df['frame'] == frame).sum()
        pct = count / len(edges_df) * 100
        print(f"    {frame}: {count} ({pct:.1f}%)")
    
    # Dominant frame per community (simplified: use community title keywords)
    threat_keywords = ['military', 'nuclear', 'missile', 'weapon', 'war', 'threat', 'attack']
    diplo_keywords = ['diplomatic', 'negotiation', 'summit', 'talk', 'agreement', 'peace']
    
    comm_df['frame_label'] = 'NEUTRAL'
    for idx, row in comm_df.iterrows():
        title = str(row.get('title', '')).lower()
        summary = str(row.get('summary', '')).lower()
        text = title + ' ' + summary
        
        threat_score = sum(1 for kw in threat_keywords if kw in text)
        diplo_score = sum(1 for kw in diplo_keywords if kw in text)
        
        if threat_score > diplo_score:
            comm_df.at[idx, 'frame_label'] = 'THREAT'
        elif diplo_score > threat_score:
            comm_df.at[idx, 'frame_label'] = 'DIPLOMACY'
    
    # Community frame distribution
    comm_frame_dist = comm_df['frame_label'].value_counts()
    print(f"\n  Community Frame Classification (by keywords):")
    for frame, count in comm_frame_dist.items():
        pct = count / len(comm_df) * 100
        print(f"    {frame}: {count} ({pct:.1f}%)")
    
    return comm_df

# Main execution
if __name__ == "__main__":
    all_results = []
    
    for period in ["period1", "period2", "period3"]:
        result = analyze_community_frames(period)
        if result is not None:
            all_results.append(result)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
