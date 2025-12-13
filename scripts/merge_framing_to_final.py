"""
Merge framing data into final datasets.
Combines existing framing (2017-2019.06) with extended framing (2019.07-2019.12).
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAME_SCALE = {"THREAT": -2, "ECONOMIC": -1, "NEUTRAL": 0, "HUMANITARIAN": 1, "DIPLOMACY": 2}


def merge_framing_to_final(group_name, final_path, existing_framing_path, extended_framing_path):
    """Merge framing data into final dataset."""
    print(f"\n{'='*60}")
    print(f"MERGING FRAMING: {group_name.upper()}")
    print(f"{'='*60}")
    
    # Load final dataset
    df_final = pd.read_csv(final_path)
    print(f"Final dataset: {len(df_final)} posts")
    
    # Load existing framing (2017-2019.06)
    df_existing = pd.read_csv(existing_framing_path)
    print(f"Existing framing: {len(df_existing)} posts")
    
    # Load extended framing (2019.07-2019.12)
    df_extended = pd.read_csv(extended_framing_path)
    print(f"Extended framing: {len(df_extended)} posts")
    
    # Combine framing data
    df_framing = pd.concat([df_existing, df_extended], ignore_index=True)
    df_framing = df_framing.drop_duplicates(subset=['id'], keep='first')
    
    # Create framing lookup
    framing_lookup = df_framing.set_index('id')[['frame', 'frame_confidence', 'frame_score']].to_dict('index')
    
    # Merge into final
    df_final['frame'] = df_final['id'].map(lambda x: framing_lookup.get(x, {}).get('frame', None))
    df_final['frame_confidence'] = df_final['id'].map(lambda x: framing_lookup.get(x, {}).get('frame_confidence', None))
    df_final['frame_score'] = df_final['id'].map(lambda x: framing_lookup.get(x, {}).get('frame_score', None))
    
    # Fill missing frame_score from frame
    mask = df_final['frame_score'].isna() & df_final['frame'].notna()
    df_final.loc[mask, 'frame_score'] = df_final.loc[mask, 'frame'].map(FRAME_SCALE)
    
    # Statistics
    matched = df_final['frame'].notna().sum()
    print(f"\nMatched: {matched}/{len(df_final)} ({matched/len(df_final)*100:.1f}%)")
    
    print(f"\nFrame distribution:")
    for frame, count in df_final['frame'].value_counts().items():
        print(f"  {frame}: {count}")
    
    # Period breakdown
    print(f"\nBy period:")
    for period in ['P1_PreSingapore', 'P2_SingaporeHanoi', 'P3_PostHanoi']:
        subset = df_final[df_final['period'] == period]
        if len(subset) > 0:
            mean_frame = subset['frame_score'].mean()
            coverage = subset['frame'].notna().sum() / len(subset) * 100
            print(f"  {period}: mean={mean_frame:.3f}, coverage={coverage:.1f}%")
    
    # Save
    df_final.to_csv(final_path, index=False)
    print(f"\n✓ Updated: {final_path}")
    
    return df_final


def main():
    print("=" * 70)
    print("MERGING FRAMING INTO FINAL DATASETS")
    print("=" * 70)
    
    datasets = [
        {
            'name': 'NK',
            'final': 'data/final/nk_final.csv',
            'existing': 'data/framing/nk_posts_framed.csv',
            'extended': 'data/framing/nk_posts_hanoi_extended_framed.csv'
        },
        {
            'name': 'China',
            'final': 'data/final/china_final.csv',
            'existing': 'data/framing/china_posts_framed.csv',
            'extended': 'data/framing/china_posts_hanoi_extended_framed.csv'
        },
        {
            'name': 'Iran',
            'final': 'data/final/iran_final.csv',
            'existing': 'data/framing/iran_posts_framed.csv',
            'extended': 'data/framing/iran_posts_hanoi_extended_framed.csv'
        },
        {
            'name': 'Russia',
            'final': 'data/final/russia_final.csv',
            'existing': 'data/framing/russia_posts_framed.csv',
            'extended': 'data/framing/russia_posts_hanoi_extended_framed.csv'
        }
    ]
    
    for ds in datasets:
        if all(os.path.exists(p) for p in [ds['final'], ds['existing'], ds['extended']]):
            merge_framing_to_final(ds['name'], ds['final'], ds['existing'], ds['extended'])
        else:
            print(f"\n❌ Missing files for {ds['name']}")
    
    print("\n" + "=" * 70)
    print("✓ Framing merge complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
