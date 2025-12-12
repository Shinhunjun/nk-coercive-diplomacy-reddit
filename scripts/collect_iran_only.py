"""Collect IRAN posts only (full, no sampling)"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_control_balanced import BalancedControlCollector

collector = BalancedControlCollector()
results = collector.collect_balanced_posts(topic='iran', target_per_period=500)
collector.save_results(results, 'iran')
print("\n✓ IRAN collection complete!")
