"""Collect RUSSIA posts only (full, no sampling)"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_control_balanced import BalancedControlCollector

collector = BalancedControlCollector()
results = collector.collect_balanced_posts(topic='russia', target_per_period=500)
collector.save_results(results, 'russia')
print("\n✓ RUSSIA collection complete!")
