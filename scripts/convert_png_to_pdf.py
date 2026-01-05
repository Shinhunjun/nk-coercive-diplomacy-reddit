#!/usr/bin/env python3
"""
Convert all PNG figures to PDF format for LaTeX.
"""

from PIL import Image
import os

figures_dir = '/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/paper/figures'

# List of PNG files to convert (exclude those that already have PDF versions)
png_files = [
    'fig1_timeline.png',
    'fig2_framing_trends.png',
    'fig3_frame_distribution.png',
    'fig4_did_visualization.png',
    'fig4b_sentiment_did_visualization.png',
    'fig5_frame_heatmap.png',
    'fig6_sentiment_trends.png',
    'fig7_sentiment_framing_corr.png',
    'fig_community_framing_distribution.png',
    'fig_magnitude_comparison.png',
    'fig_ratio_comparison_3p.png',
]

for png_file in png_files:
    png_path = os.path.join(figures_dir, png_file)
    if os.path.exists(png_path):
        pdf_path = png_path.replace('.png', '.pdf')
        try:
            img = Image.open(png_path)
            # Convert to RGB if necessary (RGBA can cause issues with PDF)
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(pdf_path, 'PDF', resolution=300)
            print(f"Converted: {png_file} -> {png_file.replace('.png', '.pdf')}")
        except Exception as e:
            print(f"Error converting {png_file}: {e}")
    else:
        print(f"Not found: {png_file}")

print("\nDone!")
