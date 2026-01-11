
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load root .env
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / '.env')

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment or .env file.")
        sys.exit(1)
        
    print(f"Loaded API Key: {api_key[:5]}...{api_key[-4:]}")
    
    # Set GRAPHRAG_API_KEY
    env = os.environ.copy()
    env["GRAPHRAG_API_KEY"] = api_key
    
    # Command
    cmd = ["graphrag", "index", "--root", "./graphrag/P2_Recursive"]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running GraphRAG: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
