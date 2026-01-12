
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

RESULTS_DIR = Path('data/results')
PROCESSED_DIR = Path('data/processed')
N_BOOTSTRAP = 1000

def assign_period(ts):
    SINGAPORE_DATE = pd.to_datetime('2018-06-12').tz_localize('UTC')
    HANOI_DATE = pd.to_datetime('2019-02-28').tz_localize('UTC')
    dt = pd.to_datetime(ts, unit='s', utc=True)
    if dt < SINGAPORE_DATE: return 'P1'
    elif dt < HANOI_DATE: return 'P2'
    else: return 'P3'

def calc_framing_metric(df, frame_col='frame'):
    if len(df) == 0: return 0
    # Metric: %Dip - %Thr
    n_dip = (df[frame_col] == 'DIPLOMACY').sum()
    n_thr = (df[frame_col] == 'THREAT').sum()
    return (n_dip / len(df) * 100) - (n_thr / len(df) * 100)

def calc_sentiment_metric(df):
    if len(df) == 0: return 0
    # Metric: Mean Sentiment
    return df['roberta_compound'].mean()


def bootstrap_ratio(data_p1, data_p2, data_p3, metric_func, label=""):
    # Calculate observed
    m1 = metric_func(data_p1)
    m2 = metric_func(data_p2)
    m3 = metric_func(data_p3)
    
    d12 = abs(m2 - m1)
    d23 = abs(m3 - m2)
    obs_ratio = d23 / d12 if d12 != 0 else np.inf
    
    # Bootstrap
    ratios = []
    
    def get_vals(df):
        if isinstance(df, pd.DataFrame):
            # Identifying column based on typical structure
            if 'frame' in df.columns: return df['frame'].values
            if 'dominant_frame' in df.columns: return df['dominant_frame'].values
            if 'roberta_compound' in df.columns: return df['roberta_compound'].values
        return df # assumes already array
        
    v1 = get_vals(data_p1)
    v2 = get_vals(data_p2)
    v3 = get_vals(data_p3)
    
    # Custom bootstrap for framing (categorical)
    # We need to map categories to numbers for fast bincount? Or just use random choice
    # Optimizing: 'DIPLOMACY'->1, 'THREAT'->-1, Others->0. Metric is Mean * 100.
    def optimize_framing_data(vals):
        mapped = np.zeros(len(vals))
        mapped[vals == 'DIPLOMACY'] = 1
        mapped[vals == 'THREAT'] = -1
        return mapped # metric is mean * 100
        
    is_framing = False
    if len(v1) > 0 and isinstance(v1[0], str):
        is_framing = True
        v1 = optimize_framing_data(v1)
        v2 = optimize_framing_data(v2)
        v3 = optimize_framing_data(v3)
        
    print(f"Bootstrapping {label}...")
    np.random.seed(42)
    with tqdm(total=N_BOOTSTRAP) as pbar:
        for _ in range(N_BOOTSTRAP):
            # Resample
            s1 = np.random.choice(v1, len(v1))
            s2 = np.random.choice(v2, len(v2))
            s3 = np.random.choice(v3, len(v3))
            
            if is_framing:
                bs_m1 = np.mean(s1) * 100
                bs_m2 = np.mean(s2) * 100
                bs_m3 = np.mean(s3) * 100
            else:
                bs_m1 = np.mean(s1)
                bs_m2 = np.mean(s2)
                bs_m3 = np.mean(s3)
                
            bs_d12 = abs(bs_m2 - bs_m1)
            bs_d23 = abs(bs_m3 - bs_m2)
            
            if bs_d12 < 1e-6:
                ratios.append(np.inf)
            else:
                ratios.append(bs_d23 / bs_d12)
            pbar.update(1)
            
    # CI
    ci_low = np.percentile(ratios, 2.5)
    ci_high = np.percentile(ratios, 97.5)
    
    return d12, d23, obs_ratio, ci_low, ci_high


def bootstrap_did_ratio(nk_df, ctrl_df, metric_func, label=""):
    # DiD = (NK_post - NK_pre) - (Ctrl_post - Ctrl_pre)
    # We need P1, P2, P3 for both
    
    def split_periods(df):
        p1 = df[df['period'] == 'P1'].copy()
        p2 = df[df['period'] == 'P2'].copy()
        p3 = df[df['period'] == 'P3'].copy()
        return p1, p2, p3

    nk_p1, nk_p2, nk_p3 = split_periods(nk_df)
    ctrl_p1, ctrl_p2, ctrl_p3 = split_periods(ctrl_df)
    
    # Calculate Observed DiD
    # P1 -> P2
    nk_m1, nk_m2, nk_m3 = metric_func(nk_p1), metric_func(nk_p2), metric_func(nk_p3)
    ctrl_m1, ctrl_m2, ctrl_m3 = metric_func(ctrl_p1), metric_func(ctrl_p2), metric_func(ctrl_p3)
    
    print(f"DEBUG {label}:")
    print(f"  NK Means: P1={nk_m1:.4f}, P2={nk_m2:.4f}, P3={nk_m3:.4f}")
    print(f"  Ctrl Means: P1={ctrl_m1:.4f}, P2={ctrl_m2:.4f}, P3={ctrl_m3:.4f}")
    
    nk_d12 = nk_m2 - nk_m1
    ctrl_d12 = ctrl_m2 - ctrl_m1
    did_p1p2 = nk_d12 - ctrl_d12
    
    # P2 -> P3
    nk_d23 = nk_m3 - nk_m2
    ctrl_d23 = ctrl_m3 - ctrl_m2
    did_p2p3 = nk_d23 - ctrl_d23
    
    print(f"  DiD P1->P2: {did_p1p2:.4f} (NK={nk_d12:.4f} - Ctrl={ctrl_d12:.4f})")
    print(f"  DiD P2->P3: {did_p2p3:.4f} (NK={nk_d23:.4f} - Ctrl={ctrl_d23:.4f})")
    
    obs_ratio = abs(did_p2p3) / abs(did_p1p2) if did_p1p2 != 0 else np.inf
    
    # Bootstrap
    ratios = []
    
    def get_vals(df):
        return df['roberta_compound'].values if len(df) > 0 else np.array([])
        
    n_v1, n_v2, n_v3 = get_vals(nk_p1), get_vals(nk_p2), get_vals(nk_p3)
    c_v1, c_v2, c_v3 = get_vals(ctrl_p1), get_vals(ctrl_p2), get_vals(ctrl_p3)
    
    print(f"Bootstrapping DiD {label}...")
    np.random.seed(42)
    with tqdm(total=N_BOOTSTRAP) as pbar:
        for _ in range(N_BOOTSTRAP):
            # Resample NK
            ns1 = np.random.choice(n_v1, len(n_v1))
            ns2 = np.random.choice(n_v2, len(n_v2))
            ns3 = np.random.choice(n_v3, len(n_v3))
            
            # Resample Ctrl
            cs1 = np.random.choice(c_v1, len(c_v1))
            cs2 = np.random.choice(c_v2, len(c_v2))
            cs3 = np.random.choice(c_v3, len(c_v3))
            
            # Calc Means
            nm1, nm2, nm3 = np.mean(ns1), np.mean(ns2), np.mean(ns3)
            cm1, cm2, cm3 = np.mean(cs1), np.mean(cs2), np.mean(cs3)
            
            # Calc DiD
            b_did12 = (nm2 - nm1) - (cm2 - cm1)
            b_did23 = (nm3 - nm2) - (cm3 - cm2)
            
            # Ratio
            if abs(b_did12) < 1e-6:
                ratios.append(np.inf)
            else:
                ratios.append(abs(b_did23) / abs(b_did12))
            pbar.update(1)
            
    ci_low = np.percentile(ratios, 2.5)
    ci_high = np.percentile(ratios, 97.5)
    
    return did_p1p2, did_p2p3, obs_ratio, ci_low, ci_high


def main():
    # 1. Content Framing
    print("Loading Content Data...")
    df_p1 = pd.read_csv(PROCESSED_DIR / 'nk_p1_framing_results.csv')
    df_p2p3 = pd.read_csv(PROCESSED_DIR / 'nk_p2_p3_framing_results.csv')
    
    meta_df = pd.read_csv(PROCESSED_DIR / 'nk_comments_recursive_roberta_final.csv')
    meta_df['period_calc'] = meta_df['created_utc'].apply(assign_period)
    id_to_period = meta_df.set_index('id')['period_calc'].to_dict()
    
    df_p2p3['period'] = df_p2p3['id'].map(id_to_period)
    df_p2p3 = df_p2p3.dropna(subset=['period'])
    df_p2 = df_p2p3[df_p2p3['period'] == 'P2']
    df_p3 = df_p2p3[df_p2p3['period'] == 'P3']
    
    c_d12, c_d23, c_ratio, c_low, c_high = bootstrap_ratio(df_p1, df_p2, df_p3, calc_framing_metric, "Content")
    
    # 2. Edge Framing
    print("Loading Edge Data...")
    e_p1 = pd.read_csv(RESULTS_DIR / 'edge_framing_P1_Recursive.csv')
    e_p2 = pd.read_csv(RESULTS_DIR / 'edge_framing_P2_Recursive.csv')
    e_p3 = pd.read_csv(RESULTS_DIR / 'edge_framing_P3_Recursive.csv')
    e_d12, e_d23, e_ratio, e_low, e_high = bootstrap_ratio(e_p1, e_p2, e_p3, calc_framing_metric, "Edge")
    
    # 3. Community Framing
    print("Loading Community Data...")
    com_p1 = pd.read_csv(RESULTS_DIR / 'community_framing_recursive_P1_Recursive.csv')
    if 'dominant_frame' in com_p1.columns: com_p1 = com_p1.rename(columns={'dominant_frame': 'frame'})
    com_p2 = pd.read_csv(RESULTS_DIR / 'community_framing_recursive_P2_Recursive.csv')
    if 'dominant_frame' in com_p2.columns: com_p2 = com_p2.rename(columns={'dominant_frame': 'frame'})
    com_p3 = pd.read_csv(RESULTS_DIR / 'community_framing_recursive_P3_Recursive.csv')
    if 'dominant_frame' in com_p3.columns: com_p3 = com_p3.rename(columns={'dominant_frame': 'frame'})
    com_d12, com_d23, com_ratio, com_low, com_high = bootstrap_ratio(com_p1, com_p2, com_p3, calc_framing_metric, "Community")
    
    # 4. Sentiment (DiD) - Monthly Aggregation Method
    print("Loading Sentiment Data (NK + China)...")
    nk_sent = pd.read_csv(PROCESSED_DIR / 'nk_comments_recursive_roberta_final.csv')
    nk_sent['period'] = nk_sent['created_utc'].apply(assign_period)
    
    # Load China Only (User Preference)
    def load_safe(path):
        df = pd.read_csv(path, on_bad_lines='skip')
        df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
        df = df.dropna(subset=['created_utc'])
        df['period'] = df['created_utc'].apply(assign_period)
        return df

    china = load_safe(PROCESSED_DIR / 'china_comments_recursive_roberta_final.csv')
    
    # Monthly DiD Bootstrap Function
    def bootstrap_did_ratio_monthly(nk_df, ctrl_df, label="Sentiment(Monthly)"):
        import statsmodels.formula.api as smf
        
        def agg_to_monthly(df):
            df = df.copy()
            df['date'] = pd.to_datetime(df['created_utc'], unit='s', utc=True)
            df['month'] = df['date'].dt.to_period('M')
            # Group by month and period to keep period labels
            agg = df.groupby(['month', 'period'])['roberta_compound'].mean().reset_index()
            return agg

        def calc_monthly_did_ols(nk_agg, ctrl_agg, p_pre, p_post):
            # Combine
            nk_sub = nk_agg[nk_agg['period'].isin([p_pre, p_post])].copy()
            nk_sub['is_treated'] = 1
            
            ctrl_sub = ctrl_agg[ctrl_agg['period'].isin([p_pre, p_post])].copy()
            ctrl_sub['is_treated'] = 0
            
            combined = pd.concat([nk_sub, ctrl_sub])
            combined['is_post'] = (combined['period'] == p_post).astype(int)
            
            if len(combined) < 4: return 0 
            
            try:
                mod = smf.ols("roberta_compound ~ is_treated * is_post", data=combined)
                res = mod.fit()
                return res.params.get('is_treated:is_post', 0)
            except:
                return 0

        # Observed
        print(f"Calculating Observed Monthly DiD choice...")
        nk_agg_obs = agg_to_monthly(nk_df)
        ctrl_agg_obs = agg_to_monthly(ctrl_df)
        
        did12 = calc_monthly_did_ols(nk_agg_obs, ctrl_agg_obs, 'P1', 'P2')
        did23 = calc_monthly_did_ols(nk_agg_obs, ctrl_agg_obs, 'P2', 'P3')
        
        ratio = abs(did23) / abs(did12) if did12 != 0 else np.inf
        print(f"  Observed: P1->P2={did12:.4f}, P2->P3={did23:.4f}, Ratio={ratio:.2f}")

        # Bootstrap
        print(f"Bootstrapping Monthly DiD {label}...")
        ratios = []
        
        # We resample rows of the Monthly Aggregations (Cluster Bootstrap equivalent-ish)
        # This assumes independence of months.
        n_months_obs = nk_agg_obs
        c_months_obs = ctrl_agg_obs
        
        np.random.seed(42)
        with tqdm(total=N_BOOTSTRAP) as pbar:
            for _ in range(N_BOOTSTRAP):
                def resample_period_months(df):
                    groups = []
                    for p in ['P1', 'P2', 'P3']:
                        sub = df[df['period'] == p]
                        if len(sub) > 0:
                            res = sub.sample(n=len(sub), replace=True)
                            groups.append(res)
                    return pd.concat(groups) if groups else df
                
                bs_n = resample_period_months(n_months_obs)
                bs_c = resample_period_months(c_months_obs)
                
                d12 = calc_monthly_did_ols(bs_n, bs_c, 'P1', 'P2')
                d23 = calc_monthly_did_ols(bs_n, bs_c, 'P2', 'P3')
                
                if abs(d12) < 1e-6:
                    ratios.append(np.inf)
                else:
                    ratios.append(abs(d23)/abs(d12))
                pbar.update(1)

        ci_low = np.percentile(ratios, 2.5)
        ci_high = np.percentile(ratios, 97.5)
        
        return did12, did23, ratio, ci_low, ci_high

    s_d12, s_d23, s_ratio, s_low, s_high = bootstrap_did_ratio_monthly(nk_sent, china, "Sent_Monthly_China")
    
    # Generate Table
    print("\\n" * 2)
    print("=" * 60)
    print("FINAL COMMENT RATCHET TABLE (LATEX)")
    print("=" * 60)
    
    print(r"\\begin{table}[t]")
    print(r"\\centering")
    print(r"\\small")
    print(r"\\setlength{\\tabcolsep}{3pt}")
    print(r"\\caption{Comment Ratchet Effect Validation: Unlike sentiment, framing shifts exhibit asymmetric persistence (Ratio $\\ll$ 1.0).}")
    print(r"\\resizebox{\\columnwidth}{!}{")
    print(r"\\begin{tabular}{lcccc}")
    print(r"\\toprule")
    print(r"\\textbf{Metric} & \\textbf{$\\Delta$(P1$\\to$P2)} & \\textbf{$\\Delta$(P2$\\to$P3)} & \\textbf{Ratio} & \\textbf{95\% CI} \\\\")
    print(r"\\midrule")
    
    def fmt_row(name, d12, d23, ratio, low, high, is_pct=True):
        sig = "*" if (low > 1.0 or high < 1.0) else ""
        if is_pct:
            return f"{name} & {d12:.1f}pp & {d23:.1f}pp & {ratio:.2f}{sig} & [{low:.2f}, {high:.2f}] \\\\"
        else:
            # Absolute Values for Sentiment
            return f"{name} & {abs(d12):.3f} & {abs(d23):.3f} & {ratio:.2f}{sig} & [{low:.2f}, {high:.2f}] \\\\"

    print(fmt_row("Content", c_d12, c_d23, c_ratio, c_low, c_high))
    print(fmt_row("Edge", e_d12, e_d23, e_ratio, e_low, e_high))
    print(fmt_row("Community", com_d12, com_d23, com_ratio, com_low, com_high))
    
    # Sentiment Row (Monthly Aggregation)
    print(fmt_row("Sentiment", s_d12, s_d23, s_ratio, s_low, s_high, is_pct=False))
    
    print(r"\\bottomrule")
    print(r"\\multicolumn{5}{l}{\\footnotesize pp = percentage points; * 95\% CI excludes 1.0 (significant asymmetry)} \\\\")
    print(r"\\end{tabular}")
    print(r"}")
    print(r"\\label{tab:app_comment_ratchet_validation}")
    print(r"\\end{table}")

if __name__ == "__main__":
    main()
