"""Run framing analysis for Russia only"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_posts_framing import PostsFramingAnalyzer
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    api_key = os.getenv("OPENAI_API_KEY")
    analyzer = PostsFramingAnalyzer(api_key)
    analyzer.analyze_topic('russia', 'data/control/russia_posts_merged.csv')
