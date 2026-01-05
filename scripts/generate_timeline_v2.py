#!/usr/bin/env python3
"""
Generate a compact line-diagram style timeline for the NK Diplomacy paper.
v4: Fixed title overlap - removed title (will use LaTeX caption instead)
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure - wider and shorter
fig, ax = plt.subplots(figsize=(14, 3.5))

# Timeline parameters
y_main = 0.65  # Main timeline y-position
y_periods = 0.25  # Period lines y-position

# Date to x-position mapping
def date_to_x(year, month, day=1):
    return (year - 2017) * 12 + month + (day - 1) / 30

# Key events
events = [
    ("Summit Announcement", 2018, 3, 8, '#FF9800'),  # orange
    ("Singapore Summit", 2018, 6, 12, '#4CAF50'),   # green
    ("Hanoi Summit", 2019, 2, 27, '#F44336'),       # red
]

# Periods with exact dates
periods = [
    {'label': 'P1 (Pre-Announcement)', 'start': (2017, 1, 1), 'end': (2018, 2, 28), 'color': '#757575', 'y_offset': 0},
    {'label': 'Transition', 'start': (2018, 3, 1), 'end': (2018, 5, 31), 'color': '#FFC107', 'y_offset': 0},
    {'label': 'P2 (Singapore-Hanoi)', 'start': (2018, 6, 1), 'end': (2019, 1, 31), 'color': '#4CAF50', 'y_offset': -0.15},
    {'label': 'Transition', 'start': (2019, 2, 1), 'end': (2019, 2, 28), 'color': '#FFC107', 'y_offset': -0.15},
    {'label': 'P3 (Post-Hanoi)', 'start': (2019, 3, 1), 'end': (2019, 12, 31), 'color': '#F44336', 'y_offset': -0.30},
]

# Draw main timeline arrow
ax.annotate('', xy=(38, y_main), xytext=(0, y_main),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Draw year markers on main timeline
for year in [2017, 2018, 2019, 2020]:
    x = date_to_x(year, 1, 1)
    ax.plot([x, x], [y_main - 0.03, y_main + 0.03], 'k-', lw=1.5)
    if year < 2020:
        ax.text(x, y_main - 0.08, str(year), ha='center', va='top', fontsize=11, fontweight='bold')

# Draw events on main timeline - labels BELOW the line
label_positions = [
    (2018, 3, 8, 'Summit\nAnnouncement', '#FF9800', 'left'),
    (2018, 6, 12, 'Singapore\nSummit', '#4CAF50', 'center'),
    (2019, 2, 27, 'Hanoi\nSummit', '#F44336', 'right'),
]

for year, month, day, name, color, ha in label_positions:
    x = date_to_x(year, month, day)
    # Event circle on top
    ax.scatter([x], [y_main], s=120, c=color, edgecolors='black', linewidths=1.5, zorder=5)
    # Date label on top of circle
    ax.text(x, y_main + 0.08, f"{year}-{month:02d}-{day:02d}", ha='center', va='bottom', fontsize=8, color='gray')
    # Name label on top of date
    ax.text(x, y_main + 0.15, name, ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

# Draw period brackets
for p in periods:
    start_x = date_to_x(*p['start'])
    end_x = date_to_x(*p['end'])
    y = y_periods + p['y_offset']
    color = p['color']
    
    # Draw dashed line with brackets
    ax.plot([start_x, end_x], [y, y], '--', color=color, lw=2.5)
    ax.plot([start_x, start_x], [y - 0.04, y + 0.04], '-', color=color, lw=2.5)
    ax.plot([end_x, end_x], [y - 0.04, y + 0.04], '-', color=color, lw=2.5)
    
    # Period label centered above the line - smaller font for short periods
    mid_x = (start_x + end_x) / 2
    period_width = end_x - start_x
    fontsize = 7 if period_width < 3 else 9  # smaller font for narrow periods
    ax.text(mid_x, y + 0.06, p['label'], ha='center', va='bottom', fontsize=fontsize, color=color, fontweight='bold')

# Draw vertical connecting lines from events to their corresponding periods
connections = [
    (2018, 3, 8, '#FF9800', y_periods),        # Summit Announcement -> Transition
    (2018, 6, 12, '#4CAF50', y_periods - 0.15), # Singapore Summit -> P2
    (2019, 2, 27, '#F44336', y_periods - 0.30), # Hanoi Summit -> P3
]

for year, month, day, color, target_y in connections:
    x = date_to_x(year, month, day)
    ax.plot([x, x], [y_main - 0.03, target_y + 0.04], ':', color=color, lw=2, alpha=0.7)

# Set limits and remove axes
ax.set_xlim(-1, 39)
ax.set_ylim(-0.15, 1.0)
ax.axis('off')

# NO TITLE - will use LaTeX caption instead

# Save figure
plt.tight_layout()
plt.savefig('/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/paper/figures/fig1_timeline_v2.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/paper/figures/fig1_timeline_v2.pdf', 
            bbox_inches='tight', facecolor='white')
print("Saved: fig1_timeline_v2.png and fig1_timeline_v2.pdf")
plt.close()
