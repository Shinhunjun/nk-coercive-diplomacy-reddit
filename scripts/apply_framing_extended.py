"""
Apply OpenAI GPT-4o-mini framing analysis to extended data (July-Dec 2019)
"""

import pandas as pd
import os
import sys
import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load .env file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Frame categories
FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

# Framing scale for numeric analysis
FRAME_SCALE = {
    "THREAT": -2,
    "ECONOMIC": -1, 
    "NEUTRAL": 0,
    "HUMANITARIAN": 1,
    "DIPLOMACY": 2
}


class ExtendedFramingAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    def classify_post(self, title: str, body: str = "") -> dict:
        """Classify a single post into a framing category."""
        text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"

        prompt = f"""Classify this Reddit post about international politics into ONE of these frames:
- THREAT: Focus on military danger, nuclear weapons, missiles, war
- DIPLOMACY: Focus on negotiations, talks, peace, cooperation
- NEUTRAL: Factual information without clear framing
- ECONOMIC: Focus on sanctions, trade, economic aspects
- HUMANITARIAN: Focus on human rights, refugees, citizens

Post:
{text}

You must respond with valid JSON only:
{{"frame": "CATEGORY", "confidence": 0.8, "reason": "brief explanation"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            # Validate frame is in allowed categories
            if result.get('frame') not in FRAME_CATEGORIES:
                result['frame'] = 'NEUTRAL'
            return result
        except json.JSONDecodeError as e:
            # Try to extract frame from text if JSON fails
            content = response.choices[0].message.content if 'response' in dir() else ""
            for frame in FRAME_CATEGORIES:
                if frame in content.upper():
                    return {"frame": frame, "confidence": 0.6, "reason": f"Extracted from: {content[:50]}"}
            return {"frame": "NEUTRAL", "confidence": 0.5, "reason": f"JSON Error: {str(e)}"}
        except Exception as e:
            return {"frame": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}

    def analyze_dataframe(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """Apply framing analysis to DataFrame."""
        results = []
        
        print(f"Analyzing {len(df)} posts...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Framing"):
            title = str(row.get('title', ''))
            body = str(row.get('selftext', '')) if pd.notna(row.get('selftext')) else ''
            
            result = self.classify_post(title, body)
            results.append({
                'frame': result.get('frame', 'NEUTRAL'),
                'frame_confidence': result.get('confidence', 0.5),
                'frame_reason': result.get('reason', '')
            })
        
        df = df.copy()
        df['frame'] = [r['frame'] for r in results]
        df['frame_confidence'] = [r['frame_confidence'] for r in results]
        df['frame_reason'] = [r['frame_reason'] for r in results]
        df['frame_score'] = df['frame'].map(FRAME_SCALE)
        
        return df


def main():
    print("=" * 60)
    print("EXTENDED DATA FRAMING ANALYSIS (July-Dec 2019)")
    print("=" * 60)
    
    analyzer = ExtendedFramingAnalyzer()
    
    # Files to process
    files = [
        ('data/nk/nk_posts_hanoi_extended.csv', 'data/framing/nk_posts_hanoi_extended_framed.csv'),
        ('data/control/china_posts_hanoi_extended.csv', 'data/framing/china_posts_hanoi_extended_framed.csv'),
        ('data/control/iran_posts_hanoi_extended.csv', 'data/framing/iran_posts_hanoi_extended_framed.csv'),
        ('data/control/russia_posts_hanoi_extended.csv', 'data/framing/russia_posts_hanoi_extended_framed.csv'),
    ]
    
    for input_path, output_path in files:
        print(f"\n--- Processing {input_path} ---")
        
        if not os.path.exists(input_path):
            print(f"  File not found: {input_path}")
            continue
            
        df = pd.read_csv(input_path)
        print(f"  Loaded {len(df)} posts")
        
        # Apply framing analysis
        df_framed = analyzer.analyze_dataframe(df)
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_framed.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
        
        # Print distribution
        print(f"\n  Frame distribution:")
        for frame, count in df_framed['frame'].value_counts().items():
            pct = count / len(df_framed) * 100
            print(f"    {frame}: {count} ({pct:.1f}%)")
    
    print("\n✓ Framing analysis complete!")


if __name__ == '__main__':
    main()
