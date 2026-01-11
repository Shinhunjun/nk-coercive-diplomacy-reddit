
import pandas as pd
import os

FILES = {
    'NK': 'data/processed/nk_comments_recursive_roberta.csv',
    'China': 'data/processed/china_comments_recursive_roberta.csv',
    'Iran': 'data/processed/iran_comments_recursive_roberta.csv',
    'Russia': 'data/processed/russia_comments_recursive_roberta.csv'
}

def finalize_data():
    print("Finalizing Datasets (Removing [removed]/[deleted])...")
    print("-" * 60)
    
    total_original = 0
    total_final = 0
    
    for country, input_path in FILES.items():
        if os.path.exists(input_path):
            print(f"Processing {country}...")
            try:
                df = pd.read_csv(input_path, low_memory=False)
                count_orig = len(df)
                
                # Filter
                # Ensure body is string for comparison
                df['body'] = df['body'].astype(str)
                df_clean = df[~df['body'].isin(['[removed]', '[deleted]'])]
                
                # Also filter empty bodies if any
                df_clean = df_clean[df_clean['body'].str.strip() != '']
                df_clean = df_clean.dropna(subset=['body'])
                
                count_final = len(df_clean)
                diff = count_orig - count_final
                
                total_original += count_orig
                total_final += count_final
                
                # Save as final
                output_path = input_path.replace('.csv', '_final.csv')
                df_clean.to_csv(output_path, index=False)
                
                print(f"  [{country}]")
                print(f"    Original: {count_orig:,}")
                print(f"    Final:    {count_final:,}")
                print(f"    Removed:  {diff:,} ({diff/count_orig*100:.1f}%)")
                print(f"    Saved to: {output_path}")
                print("-" * 60)
                
            except Exception as e:
                print(f"Error processing {country}: {e}")
        else:
            print(f"File not found: {input_path}")
            
    print("\n=== TOTAL SUMMARY ===")
    print(f"Total Original: {total_original:,}")
    print(f"Total Final:    {total_final:,}")
    print(f"Total Removed:  {total_original - total_final:,}")

if __name__ == "__main__":
    finalize_data()
