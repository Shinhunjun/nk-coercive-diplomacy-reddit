"""
Re-run event study with reference month = Dec 2017 (t=-6) for BOTH Iran and
China. The existing event_study_results.json had ref=-6 only for Iran; this
script archives the matching China run so the response letter's "94.2%"
persistence claim has a verifiable source.

Reference choice (Dec 2017) rationale:
- Nov 2017: NK's Hwasong-15 ICBM test (last major military provocation)
- Dec 2017: post-ICBM stable period, no diplomatic signals yet
- Jan 2018+: Kim's New Year speech, Olympic thaw, full anticipation

Mirrors revision_event_study.run_event_study() with ref_rel=-6.
"""
import json
import os
import sys

import pandas as pd

# Reuse the core function from the existing script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_event_study import run_event_study, load_monthly_framing, rel_to_date, plot_event_study

OUTPUT_DIR = 'data/results/revision'
FIGURES_DIR = 'paper_revision/paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    monthly = load_monthly_framing()
    out = {}

    for ctrl, label_text, fig_name in [
        (['iran'],          'Iran',         'event_study_iran_dec2017'),
        (['china'],         'China',        'event_study_china_dec2017'),
        (['china', 'iran'], 'China + Iran', 'event_study_pooled_dec2017'),
    ]:
        es_df, summary, _ = run_event_study(monthly, ctrl, -6,
                                             f'NK vs {label_text} (ref=Dec 2017)')
        key = f"ref_dec2017_{'_'.join(ctrl)}"
        out[key] = summary
        plot_event_study(es_df, -6,
                         f'NK Framing (Control: {label_text}, ref=Dec 2017)',
                         os.path.join(FIGURES_DIR, f'{fig_name}.pdf'),
                         show_title=False)
        plot_event_study(es_df, -6,
                         f'NK Framing (Control: {label_text}, ref=Dec 2017)',
                         os.path.join(FIGURES_DIR, f'{fig_name}.png'),
                         show_title=False)

        ps = summary['post_singapore']
        ph = summary['post_hanoi']
        pre = summary['pre_treatment']
        pr = summary['persistence_ratio']
        print(f"\n=== {label_text} | ref=Dec 2017 (t=-6) ===")
        print(f"  pre-treatment mean δ: {pre['mean_coef']:+.4f}  ({pre['n_significant_at_05']}/{pre['n_periods']} sig)")
        print(f"  post-Singapore mean δ: {ps['mean_effect']:+.4f}  ({ps['n_significant']}/{ps['n_periods']} sig)")
        print(f"  post-Hanoi mean δ: {ph['mean_effect']:+.4f}  ({ph['n_significant']}/{ph['n_periods']} sig)")
        print(f"  persistence ratio (post-Hanoi / post-Singapore): {pr:.4f}  ({pr*100:.1f}%)")

    with open(os.path.join(OUTPUT_DIR, 'event_study_dec2017_ref.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {OUTPUT_DIR}/event_study_dec2017_ref.json")


if __name__ == '__main__':
    main()
