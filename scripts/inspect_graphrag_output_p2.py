
import pandas as pd
import os

# Define paths
output_dir = "graphrag/P2_Recursive/output"
communities_path = os.path.join(output_dir, "community_reports.parquet")
relationships_path = os.path.join(output_dir, "relationships.parquet")
entities_path = os.path.join(output_dir, "entities.parquet")

def inspect_parquet(file_path, name, cols=None):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\n{'='*20} Inspecting {name} {'='*20}")
    try:
        df = pd.read_parquet(file_path)
        print(f"Total Rows: {len(df)}")
        if cols:
             # Filter cols if they exist
            cols_to_show = [c for c in cols if c in df.columns]
            print(df[cols_to_show].head(10).to_markdown(index=False))
        else:
            print(df.head(5).to_markdown(index=False))
    except Exception as e:
        print(f"Error reading {name}: {e}")

def main():
    # 1. Inspect Communities (What groups formed?)
    # Columns usually: title, summary, rating, explanation
    inspect_parquet(communities_path, "Community Reports", cols=["title", "summary", "rating"])

    # 2. Inspect Relationships (What edge descriptions do we have?)
    # Columns usually: source, target, description, weight
    inspect_parquet(relationships_path, "Relationships", cols=["source", "target", "description", "weight"])
    
    # 3. Inspect Entities (Just to see types)
    # Columns usually: title, type, description
    inspect_parquet(entities_path, "Entities", cols=["title", "type", "description"])

if __name__ == "__main__":
    main()
