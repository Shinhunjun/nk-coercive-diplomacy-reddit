"""
Corrected placebo test (R2 M3) with archived JSON output.

Design fix from initial revision_placebo_test.py:
- Original: ±6-month window around each candidate date, which overlaps with the
  actual Singapore treatment when the placebo date is late in P1.
- Corrected: limit data to within the relevant pre-treatment period (P1 for
  Singapore, P2 for Hanoi) so that no placebo's post-window crosses into the
  actual treatment.

Specification mirrors the paper's main level-change DiD:
    framing_mean ~ treated + post + time + treated:post
fit by OLS with HC1 robust SE. Iran is used as the control (the primary
control reported in the paper). For each placebo date d, the subsample is
restricted to {NK, Iran} within the relevant period and `post = 1[month >= d]`.

Also runs the 5-window transition robustness check using the same level-change
DiD spec on the full P1+P2 data, varying which transition months are excluded.

Outputs:
  data/results/revision/placebo_test_corrected.json
  data/results/revision/transition_robustness.json
"""
import json
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import statsmodels.formula.api as smf

FRAMING_DIR = 'data/framing'
OUTPUT_DIR = 'data/results/revision'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    dfs = []
    for c in ['nk', 'iran', 'china', 'russia']:
        df = pd.read_csv(os.path.join(FRAMING_DIR, f'{c}_monthly_framing.csv'))
        df['country'] = c
        df['treated'] = int(c == 'nk')
        df['month_dt'] = pd.to_datetime(df['month'] + '-01')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def did_level(df, controls, period_start, period_end, treat_date, exclude=None):
    """Run level-change DiD on data restricted to [period_start, period_end]
    (inclusive), with `post = 1[month >= treat_date]`. Optional `exclude` is a
    list of (start, end) month pairs to drop from the sample.
    """
    treat_dt = pd.Timestamp(treat_date + '-01')
    s = pd.Timestamp(period_start + '-01')
    e = pd.Timestamp(period_end + '-01')

    sub = df[(df['country'].isin(['nk'] + controls))
             & (df['month_dt'] >= s)
             & (df['month_dt'] <= e)].copy()

    if exclude:
        for (xs, xe) in exclude:
            xs_dt = pd.Timestamp(xs + '-01')
            xe_dt = pd.Timestamp(xe + '-01')
            sub = sub[~((sub['month_dt'] >= xs_dt) & (sub['month_dt'] <= xe_dt))]

    sub = sub.sort_values('month_dt').reset_index(drop=True)
    sub['post'] = (sub['month_dt'] >= treat_dt).astype(int)
    months = sorted(sub['month'].unique())
    sub['time'] = sub['month'].map({m: i for i, m in enumerate(months)})

    if sub['post'].nunique() < 2 or sub['treated'].nunique() < 2:
        return {'did_coef': None, 'se': None, 'p': None, 'n_obs': len(sub),
                'pre_months': int((sub['post'] == 0)['month'].sum() if False else 0),
                'post_months': 0, 'note': 'insufficient variation'}

    model = smf.ols('framing_mean ~ treated + post + time + treated:post',
                    data=sub).fit(cov_type='HC1')

    pre_months = sorted(sub[sub['post'] == 0]['month'].unique())
    post_months = sorted(sub[sub['post'] == 1]['month'].unique())

    return {
        'did_coef': float(model.params['treated:post']),
        'se': float(model.bse['treated:post']),
        'p': float(model.pvalues['treated:post']),
        'n_obs': int(len(sub)),
        'pre_months_n': len(pre_months),
        'post_months_n': len(post_months),
        'pre_range': f"{pre_months[0]}..{pre_months[-1]}" if pre_months else None,
        'post_range': f"{post_months[0]}..{post_months[-1]}" if post_months else None,
    }


def main():
    df = load_data()

    # ── Singapore placebo (P1-internal-only) ─────────────────────
    # P1 = 2017-01 .. 2018-02. Placebo dates within P1.
    p1_start, p1_end = '2017-01', '2018-02'
    singapore_placebos = ['2017-04', '2017-06', '2017-08', '2017-10', '2017-12']

    sing_results = {'design': 'P1-internal-only level-change DiD, NK vs Iran, HC1 SE',
                    'pre_period': p1_start, 'post_end': p1_end,
                    'placebos': {}, 'actual': None}
    for d in singapore_placebos:
        sing_results['placebos'][d] = did_level(df, ['iran'], p1_start, p1_end, d)

    # Actual Singapore (Jun 2018) using the paper's main P1→P2 spec with the
    # Mar-May 2018 transition excluded.
    sing_results['actual'] = did_level(df, ['iran'],
                                        period_start='2017-01',
                                        period_end='2019-01',
                                        treat_date='2018-06',
                                        exclude=[('2018-03', '2018-05')])

    # ── Hanoi placebo (P2-internal-only) ─────────────────────────
    # P2 = 2018-06 .. 2019-01. Placebo dates within P2.
    p2_start, p2_end = '2018-06', '2019-01'
    hanoi_placebos = ['2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

    han_results = {'design': 'P2-internal-only level-change DiD, NK vs Iran, HC1 SE',
                   'pre_period': p2_start, 'post_end': p2_end,
                   'placebos': {}, 'actual': None}
    for d in hanoi_placebos:
        han_results['placebos'][d] = did_level(df, ['iran'], p2_start, p2_end, d)

    # Actual Hanoi (Feb 2019) using P2→P3 spec with Feb 2019 itself as treatment.
    # P3 starts March 2019; the transition is just Feb 2019, so it's the dividing month.
    han_results['actual'] = did_level(df, ['iran'],
                                       period_start='2018-06',
                                       period_end='2019-12',
                                       treat_date='2019-02',
                                       exclude=[('2019-02', '2019-02')])

    placebo = {'singapore': sing_results, 'hanoi': han_results}
    with open(os.path.join(OUTPUT_DIR, 'placebo_test_corrected.json'), 'w') as f:
        json.dump(placebo, f, indent=2)
    print("Wrote placebo_test_corrected.json")

    # ── Transition robustness ────────────────────────────────────
    # Five transition definitions; main P1->P2 effect, NK vs Iran and NK vs China.
    transition_defs = [
        ('no_exclusion', []),
        ('Mar_only',     [('2018-03', '2018-03')]),
        ('Mar_Apr',      [('2018-03', '2018-04')]),
        ('Mar_May_ours', [('2018-03', '2018-05')]),
        ('Jan_May',      [('2018-01', '2018-05')]),
    ]

    trans = {'design': 'P1->P2 level-change DiD, full window, varying transition exclusion',
             'spec': 'framing_mean ~ treated + post + time + treated:post (HC1 SE)',
             'period': '2017-01..2019-01', 'treat_date': '2018-06',
             'iran': {}, 'china': {}}

    for label, ex in transition_defs:
        trans['iran'][label]  = did_level(df, ['iran'],  '2017-01', '2019-01', '2018-06', exclude=ex)
        trans['china'][label] = did_level(df, ['china'], '2017-01', '2019-01', '2018-06', exclude=ex)

    with open(os.path.join(OUTPUT_DIR, 'transition_robustness.json'), 'w') as f:
        json.dump(trans, f, indent=2)
    print("Wrote transition_robustness.json")

    # ── Summary print ────────────────────────────────────────────
    print("\n=== Singapore placebos (NK vs Iran, P1-internal only) ===")
    for d, r in sing_results['placebos'].items():
        print(f"  {d}: did={r['did_coef']:+.3f} se={r['se']:.3f} p={r['p']:.3f}  n={r['n_obs']} pre={r['pre_months_n']} post={r['post_months_n']}")
    a = sing_results['actual']
    print(f"  ACTUAL 2018-06: did={a['did_coef']:+.3f} se={a['se']:.3f} p={a['p']:.3f}  n={a['n_obs']}")

    print("\n=== Hanoi placebos (NK vs Iran, P2-internal only) ===")
    for d, r in han_results['placebos'].items():
        print(f"  {d}: did={r['did_coef']:+.3f} se={r['se']:.3f} p={r['p']:.3f}  n={r['n_obs']} pre={r['pre_months_n']} post={r['post_months_n']}")
    a = han_results['actual']
    print(f"  ACTUAL 2019-02: did={a['did_coef']:+.3f} se={a['se']:.3f} p={a['p']:.3f}  n={a['n_obs']}")

    print("\n=== Transition robustness (NK vs Iran) ===")
    for label, _ in transition_defs:
        r = trans['iran'][label]
        print(f"  {label:>14s}: did={r['did_coef']:+.3f} se={r['se']:.3f} p={r['p']:.3f}  n={r['n_obs']}")
    print("\n=== Transition robustness (NK vs China) ===")
    for label, _ in transition_defs:
        r = trans['china'][label]
        print(f"  {label:>14s}: did={r['did_coef']:+.3f} se={r['se']:.3f} p={r['p']:.3f}  n={r['n_obs']}")


if __name__ == '__main__':
    main()
