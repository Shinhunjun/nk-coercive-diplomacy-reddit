
import os

def main():
    api_key = None
    # Read from root .env file
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break
    except Exception as e:
        print(f"Error reading root .env: {e}")
        return

    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        return

    print(f"Found API KEY: {api_key[:5]}...{api_key[-5:]}")

    # Write to P1 .env
    with open("graphrag/P1_Recursive/.env", "w") as f:
        f.write(f"GRAPHRAG_API_KEY={api_key}\n")
    print("Wrote P1 .env")

    # Write to P3 .env
    with open("graphrag/P3_Recursive/.env", "w") as f:
        f.write(f"GRAPHRAG_API_KEY={api_key}\n")
    print("Wrote P3 .env")

if __name__ == "__main__":
    main()
