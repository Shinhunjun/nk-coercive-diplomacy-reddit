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
    
    # Dominant frame per community based on keyword analysis
    threat_keywords = ['military', 'nuclear', 'missile', 'weapon', 'war', 'threat', 'attack', 'ballistic', 'strike', 'defense']
    diplo_keywords = ['diplomatic', 'negotiation', 'summit', 'talk', 'agreement', 'peace', 'dialogue', 'cooperation', 'meeting']
    econ_keywords = ['sanction', 'economic', 'trade', 'financial', 'bank', 'business', 'market', 'investment']
    human_keywords = ['humanitarian', 'human rights', 'refugee', 'aid', 'famine', 'prison', 'abuse', 'defector']
    
    def classify_community(row):
        title = str(row.get('title', '')).lower()
        summary = str(row.get('summary', '')).lower()
        text = title + ' ' + summary
        
        scores = {
            'THREAT': sum(1 for kw in threat_keywords if kw in text),
            'DIPLOMACY': sum(1 for kw in diplo_keywords if kw in text),
            'ECONOMIC': sum(1 for kw in econ_keywords if kw in text),
            'HUMANITARIAN': sum(1 for kw in human_keywords if kw in text)
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return 'NEUTRAL'
        return max(scores, key=scores.get)
    
    comm_df['frame_label'] = comm_df.apply(classify_community, axis=1)
    
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
