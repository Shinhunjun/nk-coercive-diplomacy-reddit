
import os
import subprocess
import sys

def main():
    # Ensure API key is present
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Define the graphrag root directory for P3
    root_dir = "./graphrag/P3_Recursive"
    
    # Construct the command
    # We use the python module syntax to run graphrag to ensure we use the installed package
    cmd = [
        "graphrag", "index",
        "--root", root_dir
    ]

    print(f"Starting GraphRAG indexing for {root_dir}...")
    
    try:
        # Run the command, passing the current environment (which includes the API key)
        subprocess.run(cmd, check=True, env=os.environ, cwd="/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit")
        print("GraphRAG indexing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"GraphRAG indexing failed with return code {e.returncode}.")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
