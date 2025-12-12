"""
OpenAI GPT-4 Based Framing Analysis for Reddit Posts

This module classifies posts into framing categories using OpenAI GPT-4.
Categories: THREAT, DIPLOMACY, NEUTRAL, ECONOMIC, HUMANITARIAN

Uses OpenAI API with structured JSON output.
"""

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
import time
from openai import OpenAI

from config import FRAME_CATEGORIES


class OpenAIFramingAnalyzer:
    """OpenAI GPT-4 based framing classifier for North Korea-related posts."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI framing analyzer.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
            model: OpenAI model to use (default: gpt-4o-mini - best cost/performance)
        """
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env variable or pass api_key.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model
        self.categories = FRAME_CATEGORIES

    def classify_post(self, title: str, body: str = "") -> dict:
        """
        Classify a single post into a framing category.

        Args:
            title: Post title
            body: Post body text (optional)

        Returns:
            Dictionary with frame, confidence, and reason
        """
        text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"

        prompt = f"""이 Reddit 게시글을 다음 5가지 프레임 중 하나로 분류하세요:
- THREAT: 군사적 위협, 핵무기, 미사일, 전쟁 위험 강조
- DIPLOMACY: 협상, 대화, 평화, 협력 가능성 강조
- NEUTRAL: 중립적 정보 전달
- ECONOMIC: 경제 제재, 무역 측면 강조
- HUMANITARIAN: 인권, 난민, 북한 주민 문제 강조

게시글:
{text}

JSON 형식으로 응답:
{{"frame": "카테고리", "confidence": 0.0-1.0, "reason": "간단한 설명"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a political science researcher analyzing media framing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            # Extract JSON from response
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)

            # Validate frame
            if result.get('frame') not in self.categories:
                result['frame'] = 'NEUTRAL'
                result['confidence'] = 0.5

            return result

        except Exception as e:
            return {
                "frame": "NEUTRAL",
                "confidence": 0.5,
                "reason": f"Error: {str(e)}"
            }

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = 'title',
        body_col: str = 'selftext',
        delay: float = 0.1
    ) -> pd.DataFrame:
        """
        Classify framing for posts in a DataFrame.

        Args:
            df: DataFrame with title and body columns
            title_col: Name of title column
            body_col: Name of body column
            delay: Delay between API calls (seconds) for rate limiting

        Returns:
            DataFrame with framing results added
        """
        results = []
        print(f"Classifying {len(df)} posts with OpenAI GPT-4...")

        for idx in tqdm(range(len(df)), desc="Framing Classification"):
            row = df.iloc[idx]
            title = str(row.get(title_col, ''))
            body = str(row.get(body_col, '')) if pd.notna(row.get(body_col)) else ''

            result = self.classify_post(title, body)
            results.append({
                'frame': result.get('frame', 'NEUTRAL'),
                'frame_confidence': result.get('confidence', 0.5),
                'frame_reason': result.get('reason', '')
            })

            # Rate limiting
            time.sleep(delay)

        # Add framing columns
        df = df.copy()
        df['frame'] = [r['frame'] for r in results]
        df['frame_confidence'] = [r['frame_confidence'] for r in results]
        df['frame_reason'] = [r['frame_reason'] for r in results]

        return df

    def calculate_frame_distribution(self, df: pd.DataFrame) -> dict:
        """
        Calculate distribution of frames in DataFrame.

        Args:
            df: DataFrame with 'frame' column

        Returns:
            Dictionary with frame counts and percentages
        """
        if 'frame' not in df.columns:
            raise ValueError("DataFrame must have 'frame' column")

        frame_counts = df['frame'].value_counts()
        total = len(df)

        distribution = {}
        for frame in self.categories:
            count = frame_counts.get(frame, 0)
            pct = (count / total * 100) if total > 0 else 0
            distribution[frame] = {
                'count': int(count),
                'percentage': float(pct)
            }

        return distribution

    def compare_framing(self, df1: pd.DataFrame, df2: pd.DataFrame,
                       label1: str = "Period 1", label2: str = "Period 2") -> dict:
        """
        Compare framing distributions between two DataFrames.

        Args:
            df1: First DataFrame with 'frame' column
            df2: Second DataFrame with 'frame' column
            label1: Label for first period
            label2: Label for second period

        Returns:
            Dictionary with comparison results including chi-square test
        """
        from scipy.stats import chi2_contingency

        # Get distributions
        dist1 = self.calculate_frame_distribution(df1)
        dist2 = self.calculate_frame_distribution(df2)

        # Prepare contingency table
        frames = self.categories
        counts1 = [dist1[f]['count'] for f in frames]
        counts2 = [dist2[f]['count'] for f in frames]

        contingency = np.array([counts1, counts2])

        # Chi-square test
        chi2, pvalue, dof, expected = chi2_contingency(contingency)

        return {
            label1: dist1,
            label2: dist2,
            'chi2_statistic': float(chi2),
            'p_value': float(pvalue),
            'degrees_of_freedom': int(dof)
        }
