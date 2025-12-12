"""Collect ALL IRAN comments only (with pagination)"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_all_comments import FullCommentsCollector, collect_full_comments_for_topic

if __name__ == '__main__':
    collector = FullCommentsCollector()
    collect_full_comments_for_topic(collector, 'iran')
