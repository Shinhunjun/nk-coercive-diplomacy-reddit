import json
import os
import shutil
from pathlib import Path

BASE_DIR = Path('data/anonymized/results')

def merge_json_files(output_filename, patterns):
    print(f"\nCreating {output_filename}...")
    merged_data = {}
    
    files_to_delete = []
    
    for filename in os.listdir(BASE_DIR):
        if not filename.endswith('.json'):
            continue
            
        # Check if file matches any pattern
        matched = False
        for pattern in patterns:
            if pattern in filename:
                matched = True
                break
        
        if matched and filename != output_filename:
            try:
                with open(BASE_DIR / filename, 'r') as f:
                    data = json.load(f)
                    # Use filename (without extension) as key
                    key = filename.replace('.json', '')
                    merged_data[key] = data
                    files_to_delete.append(filename)
            except Exception as e:
                print(f"  Error reading {filename}: {e}")
    
    if merged_data:
        with open(BASE_DIR / output_filename, 'w') as f:
            json.dump(merged_data, f, indent=2)
        print(f"  ✅ Merged {len(merged_data)} files into {output_filename}")
        
        # Delete merged files
        for f in files_to_delete:
            os.remove(BASE_DIR / f)
        print(f"  Deleted {len(files_to_delete)} source files")
    else:
        print("  No files found to merge")

def merge_markdown_reports():
    print("\nMerging GraphRAG reports...")
    output_filename = 'GRAPHRAG_REPORTS_CONSOLIDATED.md'
    
    report_files = [f for f in os.listdir(BASE_DIR) 
                   if f.startswith('GRAPHRAG_') and f.endswith('.md') and f != output_filename]
    
    report_files.sort()
    
    if report_files:
        with open(BASE_DIR / output_filename, 'w') as outfile:
            outfile.write("# Consolidated GraphRAG Analysis Reports\n\n")
            
            for fname in report_files:
                outfile.write(f"\n\n---\n\n# Source: {fname}\n\n")
                with open(BASE_DIR / fname, 'r') as infile:
                    outfile.write(infile.read())
                
                # Delete original
                os.remove(BASE_DIR / fname)
        
        print(f"  ✅ Merged {len(report_files)} reports")
    else:
        print("  No reports found")

def cleanup_other_files():
    print("\nCleaning up remaining files...")
    
    # Files to keep
    keep = [
        'did_results_main.json',
        'graphrag_analysis_results.json',
        'framing_did_consolidated.json',
        'sentiment_did_consolidated.json',
        'GRAPHRAG_REPORTS_CONSOLIDATED.md',
        'COMMUNITY_DEEP_ANALYSIS.md'
    ]
    
    # Delete csvs in results (usually intermediate)
    # Delete subdirectories
    
    for item in os.listdir(BASE_DIR):
        path = BASE_DIR / item
        
        if item in keep:
            continue
            
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  Deleted folder: {item}")
        elif item.endswith('.csv') or item.endswith('.txt'):
            os.remove(path)
            print(f"  Deleted file: {item}")
        elif item.endswith('.json') and item not in keep:
            # Check if it should have been merged but wasn't (e.g. unique names)
            pass

def main():
    if not BASE_DIR.exists():
        print("Results folder not found")
        return

    # 1. DID Results
    merge_json_files('did_results_main.json', 
        ['did_', 'parallel_trends', 'binary_did', 'hanoi_3period'])
        
    # 2. GraphRAG & Community
    merge_json_files('graphrag_analysis_results.json', 
        ['graphrag_', 'community_', 'keyword_'])
        
    # 3. Framing Analysis
    merge_json_files('framing_did_consolidated.json',
        ['framing_did_', 'framing_analysis', 'openai_framing', 'gemini_framing'])
        
    # 4. Sentiment Analysis
    merge_json_files('sentiment_did_consolidated.json',
        ['sentiment_did_', 'sentiment_comparison'])

    # 5. Markdown Reports
    merge_markdown_reports()
    
    # 6. Cleanup
    cleanup_other_files()

if __name__ == "__main__":
    main()
