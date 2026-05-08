"""
Revision Analysis 1: Single Event-Study Specification
=====================================================
Addresses R2-M1 (non-independent DiD), R2-M2 (parallel trends), R2-M3 (no anticipation).

Model: Y_it = α_i + β*time + Σ_k δ_k(Treat_i × 1[t=k]) + ε_it

Key design choices:
- Linear time trend (avoids saturation with 2 units)
- OLS with HC1 robust SE
- Main reference: t=-4 (Feb 2018) = last month before diplomatic signals
- Robustness: multiple reference points (t=-5, t=-6, t=-7)

Justification for reference point:
  Trump accepted Kim's invitation on March 8, 2018.
  Our paper defines transition period as March-May 2018.
  Therefore Feb 2018 is the last "clean" pre-treatment month.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Configuration ──────────────────────────────────────────────
FRAMING_DIR = 'data/framing'
OUTPUT_DIR = 'data/results/revision'
FIGURES_DIR = 'paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SINGAPORE_SUMMIT = '2018-06'
HANOI_SUMMIT_REL_FROM_JUNE = 8  # Feb 2019 = 8 months after Jun 2018

# Reference points to test (rel_month relative to June 2018)
# t=-4 = Feb 2018, t=-5 = Jan 2018, t=-6 = Dec 2017, t=-7 = Nov 2017
MAIN_REF = -4           # Feb 2018: last month before diplomatic signals
ROBUSTNESS_REFS = [-5, -6, -7]  # Jan 2018, Dec 2017, Nov 2017


def load_monthly_framing():
    """Load monthly framing data for all countries."""
    dfs = []
    for country in ['nk', 'china', 'iran', 'russia']:
        path = os.path.join(FRAMING_DIR, f'{country}_monthly_framing.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['country'] = country
            df['treated'] = 1 if country == 'nk' else 0
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined['month_dt'] = pd.to_datetime(combined['month'] + '-01')

    ref = pd.Timestamp(SINGAPORE_SUMMIT + '-01')
    combined['rel_month'] = ((combined['month_dt'].dt.year - ref.year) * 12 +
                              (combined['month_dt'].dt.month - ref.month))

    min_dt = combined['month_dt'].min()
    combined['time'] = ((combined['month_dt'].dt.year - min_dt.year) * 12 +
                         (combined['month_dt'].dt.month - min_dt.month))

    return combined


def rel_to_date(rel_month):
    """Convert relative month to calendar month string."""
    ref = pd.Timestamp(SINGAPORE_SUMMIT + '-01')
    dt = ref + pd.DateOffset(months=rel_month)
    return dt.strftime('%Y-%m')


def run_event_study(monthly_df, controls, ref_rel, label):
    """
    Event study with linear time trend and specified reference period.
    """
    subset = monthly_df[monthly_df['country'].isin(['nk'] + controls)].copy()

    all_rm = sorted(subset['rel_month'].unique())

    # Create interaction dummies
    rm_to_col = {}
    for rm in all_rm:
        if rm == ref_rel:
            continue
        col = f'Dt_pos{rm}' if rm >= 0 else f'Dt_neg{abs(rm)}'
        subset[col] = ((subset['rel_month'] == rm) & (subset['treated'] == 1)).astype(int)
        rm_to_col[rm] = col

    terms = list(rm_to_col.values())
    formula = f"framing_mean ~ C(country) + time + {' + '.join(terms)}"

    model = smf.ols(formula, data=subset)
    result = model.fit(cov_type='HC1')

    # Extract coefficients
    es_results = [{'rel_month': ref_rel, 'coef': 0.0, 'se': 0.0, 'ci_lower': 0.0,
                   'ci_upper': 0.0, 'pvalue': np.nan, 'is_reference': True}]

    for rm in all_rm:
        if rm == ref_rel:
            continue
        col = rm_to_col[rm]
        if col in result.params:
            es_results.append({
                'rel_month': rm,
                'coef': float(result.params[col]),
                'se': float(result.bse[col]),
                'ci_lower': float(result.conf_int().loc[col][0]),
                'ci_upper': float(result.conf_int().loc[col][1]),
                'pvalue': float(result.pvalues[col]),
                'is_reference': False
            })

    es_df = pd.DataFrame(es_results).sort_values('rel_month').reset_index(drop=True)

    # Compute summary with periods defined relative to Singapore (rel_month=0)
    # Pre-treatment: before ref_rel (clean pre-period)
    pre = es_df[(es_df['rel_month'] < ref_rel) & (~es_df['is_reference'])]
    # Transition: between ref and Singapore
    transition = es_df[(es_df['rel_month'] > ref_rel) & (es_df['rel_month'] < 0)]
    # Post-Singapore (P2): rel_month 0 to 7
    post_sing = es_df[(es_df['rel_month'] >= 0) & (es_df['rel_month'] < HANOI_SUMMIT_REL_FROM_JUNE)]
    # Post-Hanoi (P3): rel_month >= 8
    post_hanoi = es_df[es_df['rel_month'] >= HANOI_SUMMIT_REL_FROM_JUNE]

    n_pre_sig = int(sum(pre['pvalue'] < 0.05)) if len(pre) > 0 else 0
    mean_sing = float(post_sing['coef'].mean()) if len(post_sing) > 0 else None
    mean_hanoi = float(post_hanoi['coef'].mean()) if len(post_hanoi) > 0 else None
    mean_transition = float(transition['coef'].mean()) if len(transition) > 0 else None

    if mean_sing and mean_hanoi and mean_sing != 0:
        persistence = mean_hanoi / mean_sing
    else:
        persistence = None

    summary = {
        'label': label,
        'controls': controls,
        'reference_month': ref_rel,
        'reference_date': rel_to_date(ref_rel),
        'n_obs': len(subset),
        'n_params': len(result.params),
        'df_resid': int(result.df_resid),
        'r_squared': float(result.rsquared),
        'pre_treatment': {
            'n_periods': len(pre),
            'n_significant_at_05': n_pre_sig,
            'mean_coef': float(pre['coef'].mean()) if len(pre) > 0 else None,
            'parallel_trends_pass': n_pre_sig <= 2
        },
        'transition': {
            'n_periods': len(transition),
            'mean_effect': mean_transition,
            'n_significant': int(sum(transition['pvalue'] < 0.05)) if len(transition) > 0 else 0
        },
        'post_singapore': {
            'mean_effect': mean_sing,
            'n_periods': len(post_sing),
            'n_significant': int(sum(post_sing['pvalue'] < 0.05)) if len(post_sing) > 0 else 0
        },
        'post_hanoi': {
            'mean_effect': mean_hanoi,
            'n_periods': len(post_hanoi),
            'n_significant': int(sum(post_hanoi['pvalue'] < 0.05)) if len(post_hanoi) > 0 else 0
        },
        'persistence_ratio': float(persistence) if persistence else None,
        'coefficients': es_df.to_dict('records')
    }

    return es_df, summary, result


def plot_event_study(es_df, ref_rel, label, save_path, show_title=True):
    """Publication-quality event-study plot."""
    fig, ax = plt.subplots(figsize=(13, 6))

    months = es_df['rel_month'].values
    coefs = es_df['coef'].values
    ci_lower = es_df['ci_lower'].values
    ci_upper = es_df['ci_upper'].values

    ax.fill_between(months, ci_lower, ci_upper, alpha=0.15, color='#2166AC')
    ax.plot(months, coefs, 'o-', color='#2166AC', markersize=5, linewidth=1.5, label='Treatment Effect')

    ax.axhline(y=0, color='black', linewidth=0.8)

    # Key event lines
    ax.axvline(x=-3.5, color='#999999', linewidth=1.2, linestyle='-.',
               alpha=0.7, label='Summit Announced (Mar 2018)')
    ax.axvline(x=-0.5, color='#B2182B', linewidth=1.5, linestyle='--',
               alpha=0.8, label='Singapore Summit (Jun 2018)')
    ax.axvline(x=HANOI_SUMMIT_REL_FROM_JUNE - 0.5, color='#EF8A62', linewidth=1.5,
               linestyle=':', alpha=0.8, label='Hanoi Summit (Feb 2019)')

    # Period shading
    ax.axvspan(min(months) - 0.5, ref_rel + 0.5, alpha=0.03, color='gray')  # pre
    ax.axvspan(ref_rel + 0.5, -0.5, alpha=0.08, color='#FEE08B',
               label='Transition (Mar-May 2018)')  # transition
    ax.axvspan(-0.5, HANOI_SUMMIT_REL_FROM_JUNE - 0.5, alpha=0.05, color='#2166AC')  # P2
    ax.axvspan(HANOI_SUMMIT_REL_FROM_JUNE - 0.5, max(months) + 0.5,
               alpha=0.05, color='#EF8A62')  # P3

    ax.set_xlabel('Months Relative to Singapore Summit (June 2018)', fontsize=12)
    ax.set_ylabel('Treatment Effect on Framing Score\n(relative to reference)', fontsize=12)
    if show_title:
        ax.set_title(f'Event Study: {label}\n(Reference: {rel_to_date(ref_rel)})', fontsize=13)

    # Reference annotation
    ax.annotate(f'Ref ({rel_to_date(ref_rel)})', xy=(ref_rel, 0),
                xytext=(ref_rel - 2, max(ci_upper) * 0.3 if max(ci_upper) > 0.5 else 0.3),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center', color='gray')

    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(min(months) - 0.5, max(months) + 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_robustness_comparison(all_summaries, save_path):
    """Plot Singapore and Hanoi effects across different reference points."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    refs = []
    sing_effects = []
    hanoi_effects = []
    persist_ratios = []
    pre_sigs = []

    for key, res in all_summaries.items():
        refs.append(f"{rel_to_date(res['reference_month'])}\n(t={res['reference_month']})")
        sing_effects.append(res['post_singapore']['mean_effect'] or 0)
        hanoi_effects.append(res['post_hanoi']['mean_effect'] or 0)
        persist_ratios.append(res['persistence_ratio'] or 0)
        pre_sigs.append(f"{res['pre_treatment']['n_significant_at_05']}/{res['pre_treatment']['n_periods']}")

    x = range(len(refs))

    # Panel A: Effects
    ax1.bar([i - 0.15 for i in x], sing_effects, 0.3, label='Post-Singapore δ', color='#2166AC', alpha=0.8)
    ax1.bar([i + 0.15 for i in x], hanoi_effects, 0.3, label='Post-Hanoi δ', color='#EF8A62', alpha=0.8)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(refs, fontsize=9)
    ax1.set_ylabel('Mean Treatment Effect', fontsize=11)
    ax1.set_title('Effect Magnitudes by Reference Point', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, axis='y')

    # Add pre-sig labels
    for i, sig in enumerate(pre_sigs):
        ax1.annotate(f'Pre-sig: {sig}', xy=(i, max(sing_effects + hanoi_effects) * 1.05),
                     ha='center', fontsize=8, color='gray')

    # Panel B: Persistence
    ax2.bar(x, [p * 100 for p in persist_ratios], color='#4DAF4A', alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(refs, fontsize=9)
    ax2.set_ylabel('Persistence Ratio (%)', fontsize=11)
    ax2.set_title('Asymmetric Persistence by Reference Point', fontsize=12)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary_row(res):
    """Print one-line summary of event study result."""
    pt = 'PASS' if res['pre_treatment']['parallel_trends_pass'] else 'WARN'
    ps = f"{res['post_singapore']['mean_effect']:+.3f}" if res['post_singapore']['mean_effect'] is not None else 'N/A'
    ph = f"{res['post_hanoi']['mean_effect']:+.3f}" if res['post_hanoi']['mean_effect'] is not None else 'N/A'
    pr = f"{res['persistence_ratio']*100:.1f}%" if res['persistence_ratio'] is not None else 'N/A'
    tr = f"{res['transition']['mean_effect']:+.3f}" if res['transition']['mean_effect'] is not None else 'N/A'
    sig = f"{res['pre_treatment']['n_significant_at_05']}/{res['pre_treatment']['n_periods']}"
    return f"{res['reference_date']:<10} {sig:>7} {pt:>5} {tr:>10} {ps:>10} {ph:>10} {pr:>10}"


def main():
    print("=" * 70)
    print("REVISION ANALYSIS 1: EVENT STUDY (MULTIPLE REFERENCE POINTS)")
    print("=" * 70)
    print(f"\nModel: Y_it = α_i + β*time + Σ_k δ_k(Treat × D_k)")
    print(f"Main ref: t={MAIN_REF} ({rel_to_date(MAIN_REF)}) = last month before diplomatic signals")
    print(f"Robustness refs: {[f't={r} ({rel_to_date(r)})' for r in ROBUSTNESS_REFS]}")

    monthly = load_monthly_framing()
    print(f"\nData: {len(monthly)} country-month obs")

    all_results = {}

    # ──────────────────────────────────────────────────
    # PART 1: MAIN SPECIFICATION (Iran, ref=Feb 2018)
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"MAIN: NK vs Iran, ref = {rel_to_date(MAIN_REF)} (t={MAIN_REF})")
    print("=" * 60)

    es_df, summary, model = run_event_study(monthly, ['iran'], MAIN_REF,
                                             f'NK vs Iran (ref={rel_to_date(MAIN_REF)})')
    all_results['main_iran'] = summary

    print(f"  N = {summary['n_obs']}, df = {summary['df_resid']}, R² = {summary['r_squared']:.4f}")
    print(f"  Pre-treatment sig: {summary['pre_treatment']['n_significant_at_05']}/{summary['pre_treatment']['n_periods']}")
    print(f"  Parallel trends: {'PASS' if summary['pre_treatment']['parallel_trends_pass'] else 'WARNING'}")
    if summary['transition']['mean_effect'] is not None:
        print(f"  Transition (Mar-May 2018) mean δ: {summary['transition']['mean_effect']:+.4f} "
              f"({summary['transition']['n_significant']}/{summary['transition']['n_periods']} sig)")
    if summary['post_singapore']['mean_effect'] is not None:
        print(f"  Post-Singapore mean δ: {summary['post_singapore']['mean_effect']:+.4f} "
              f"({summary['post_singapore']['n_significant']}/{summary['post_singapore']['n_periods']} sig)")
    if summary['post_hanoi']['mean_effect'] is not None:
        print(f"  Post-Hanoi mean δ: {summary['post_hanoi']['mean_effect']:+.4f} "
              f"({summary['post_hanoi']['n_significant']}/{summary['post_hanoi']['n_periods']} sig)")
    if summary['persistence_ratio'] is not None:
        print(f"  Persistence ratio: {summary['persistence_ratio']:.3f} ({summary['persistence_ratio']*100:.1f}%)")

    plot_event_study(es_df, MAIN_REF, 'NK Framing (Control: Iran)',
                     os.path.join(FIGURES_DIR, 'event_study_iran.pdf'))

    # Also with China
    print(f"\n--- NK vs China, ref = {rel_to_date(MAIN_REF)} ---")
    es_df_c, summary_c, _ = run_event_study(monthly, ['china'], MAIN_REF,
                                              f'NK vs China (ref={rel_to_date(MAIN_REF)})')
    all_results['main_china'] = summary_c
    print(f"  Pre-sig: {summary_c['pre_treatment']['n_significant_at_05']}/{summary_c['pre_treatment']['n_periods']}, "
          f"PT: {'PASS' if summary_c['pre_treatment']['parallel_trends_pass'] else 'WARN'}")
    if summary_c['post_singapore']['mean_effect'] is not None:
        print(f"  Post-Singapore: {summary_c['post_singapore']['mean_effect']:+.4f}, "
              f"Post-Hanoi: {summary_c['post_hanoi']['mean_effect']:+.4f}")
    if summary_c['persistence_ratio'] is not None:
        print(f"  Persistence: {summary_c['persistence_ratio']*100:.1f}%")
    plot_event_study(es_df_c, MAIN_REF, 'NK Framing (Control: China)',
                     os.path.join(FIGURES_DIR, 'event_study_china.pdf'))

    # Pooled
    print(f"\n--- NK vs China+Iran, ref = {rel_to_date(MAIN_REF)} ---")
    es_df_p, summary_p, _ = run_event_study(monthly, ['china', 'iran'], MAIN_REF,
                                              f'NK vs China+Iran (ref={rel_to_date(MAIN_REF)})')
    all_results['main_pooled'] = summary_p
    print(f"  Pre-sig: {summary_p['pre_treatment']['n_significant_at_05']}/{summary_p['pre_treatment']['n_periods']}, "
          f"PT: {'PASS' if summary_p['pre_treatment']['parallel_trends_pass'] else 'WARN'}")
    if summary_p['post_singapore']['mean_effect'] is not None:
        print(f"  Post-Singapore: {summary_p['post_singapore']['mean_effect']:+.4f}, "
              f"Post-Hanoi: {summary_p['post_hanoi']['mean_effect']:+.4f}")
    if summary_p['persistence_ratio'] is not None:
        print(f"  Persistence: {summary_p['persistence_ratio']*100:.1f}%")
    plot_event_study(es_df_p, MAIN_REF, 'NK Framing (Controls: China + Iran)',
                     os.path.join(FIGURES_DIR, 'event_study_pooled.pdf'))

    # ──────────────────────────────────────────────────
    # PART 2: ROBUSTNESS (Iran, multiple ref points)
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROBUSTNESS: NK vs Iran with different reference points")
    print("=" * 60)

    robustness_results = {'main': summary}

    for ref in ROBUSTNESS_REFS:
        ref_date = rel_to_date(ref)
        es_df_r, summary_r, _ = run_event_study(monthly, ['iran'], ref,
                                                  f'NK vs Iran (ref={ref_date})')
        robustness_results[f'ref_{ref}'] = summary_r
        all_results[f'robustness_iran_ref{ref}'] = summary_r

    print(f"\n{'Ref Date':<10} {'Pre-sig':>7} {'PT':>5} {'Transition':>10} {'Sing δ':>10} {'Hanoi δ':>10} {'Persist':>10}")
    print("-" * 70)
    for key, res in robustness_results.items():
        print(f"  {print_summary_row(res)}")

    # Robustness comparison plot
    plot_robustness_comparison(robustness_results,
                                os.path.join(FIGURES_DIR, 'event_study_robustness.pdf'))

    # ──────────────────────────────────────────────────
    # PART 3: OVERALL SUMMARY
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"\n{'Specification':<40} {'Ref':>10} {'Pre-sig':>9} {'PT':>5} {'Sing δ':>10} {'Hanoi δ':>10} {'Persist':>10}")
    print("-" * 100)

    for key, res in all_results.items():
        pt = 'PASS' if res['pre_treatment']['parallel_trends_pass'] else 'WARN'
        ps = f"{res['post_singapore']['mean_effect']:+.3f}" if res['post_singapore']['mean_effect'] is not None else 'N/A'
        ph = f"{res['post_hanoi']['mean_effect']:+.3f}" if res['post_hanoi']['mean_effect'] is not None else 'N/A'
        pr = f"{res['persistence_ratio']*100:.1f}%" if res['persistence_ratio'] is not None else 'N/A'
        sig = f"{res['pre_treatment']['n_significant_at_05']}/{res['pre_treatment']['n_periods']}"
        print(f"{res['label']:<40} {res['reference_date']:>10} {sig:>9} {pt:>5} {ps:>10} {ph:>10} {pr:>10}")

    # Save
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    output_path = os.path.join(OUTPUT_DIR, 'event_study_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
