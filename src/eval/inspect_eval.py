import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

i = pd.read_parquet('data/processed/intervals.parquet')
i_eval = i[i['split'] == 'EVAL']
m = pd.read_parquet('data/processed/metrics.parquet')
b = pd.read_parquet('data/processed/bootstrap_results.parquet')
c = pd.read_parquet('data/processed/capacity_sensitivity.parquet')
pl = pd.read_parquet('data/processed/patch_level.parquet')

print('=== 1. SAMPLE COUNTS N (EVAL SPLIT) ===')
for gid, g in i_eval[(i_eval['arm']=='fixed') & (i_eval['level']==0.9) & (i_eval['model']=='lgbm')].groupby('game_id'):
    patches_eval = g[g['patch_id'] != 'none']['patch_id'].nunique()
    print(f'{gid}: total_eval_points={len(g)}, unique_patches={patches_eval}')

print('\n=== 2. PATCH EVENT COUNT IN 0-7D SHOCK WINDOW (EVAL) ===')
for gid, g in pl[pl['level']==0.9].groupby('game_id'):
    print(f'{gid}: independent patch events in 0-7d window = {g["patch_id"].nunique()}')
total_patches_0_7d = pl[pl['level']==0.9]['patch_id'].nunique()
print(f'POOLED TOTAL: {total_patches_0_7d} independent patch events')

print('\n=== 3. EVAL METRICS BY ARM (90% LEVEL, LGBM) ===')
m_lgbm = m[(m['model'] == 'lgbm') & (m['level'] == 0.9)]
cov_piv = m_lgbm.groupby(['arm', 'window'])['coverage'].mean().unstack() * 100
width_piv = m_lgbm.groupby(['arm', 'window'])['mean_width'].mean().unstack()
norm_w_piv = m_lgbm.groupby(['arm', 'window'])['norm_width'].mean().unstack() * 100

# Capacity cost & break-even
c_lgbm = c[(c['model'] == 'lgbm') & (c['level'] == 0.9)]
c_2x = c_lgbm[c_lgbm['penalty_ratio'] == 2.0].groupby('arm')['norm_cost'].mean()
c_5x = c_lgbm[c_lgbm['penalty_ratio'] == 5.0].groupby('arm')['norm_cost'].mean()
c_10x = c_lgbm[c_lgbm['penalty_ratio'] == 10.0].groupby('arm')['norm_cost'].mean()
r_star = c_lgbm.groupby('arm')['break_even_r_star'].mean()

table = pd.DataFrame({
    'Cov 0-7d': cov_piv['post_patch (0-7d)'],
    'Cov 8-14d': cov_piv['post_patch (8-14d)'],
    'Cov 15-30d': cov_piv['post_patch (15-30d)'],
    'Cov Base (>30d)': cov_piv['baseline (>30d)'],
    'Width (0-7d)': width_piv['post_patch (0-7d)'],
    'NormW % (0-7d)': norm_w_piv['post_patch (0-7d)'],
    'Cost @2x': c_2x,
    'Cost @5x': c_5x,
    'Cost @10x': c_10x,
    'Break-even r*': r_star
})
print(table.round(3))

print('\n=== 4. CLUSTER BOOTSTRAP OVER PATCH EVENTS (POOLED, 90% LEVEL, LGBM) ===')
b_lgbm = b[(b['model'] == 'lgbm') & (b['level'] == 0.9) & (b['scope'] == 'pooled')].copy()

def fmt_p(p):
    return "< 0.001" if p <= 0.001 else f"{p:.3f}"

b_lgbm['cov_diff'] = b_lgbm.apply(lambda r: f"{r['cov_diff_mean']*100:+.2f}% [{r['cov_diff_ci_lo']*100:+.2f}%, {r['cov_diff_ci_hi']*100:+.2f}%] (p {fmt_p(r['cov_p_val'])})", axis=1)
b_lgbm['width_diff'] = b_lgbm.apply(lambda r: f"{r['width_diff_mean']:+,.0f} [{r['width_diff_ci_lo']:+,.0f}, {r['width_diff_ci_hi']:+,.0f}]", axis=1)
if 'norm_width_diff_mean' in b_lgbm.columns:
    b_lgbm['norm_w_diff'] = b_lgbm.apply(lambda r: f"{r['norm_width_diff_mean']*100:+.1f}% [{r['norm_width_diff_ci_lo']*100:+.1f}%, {r['norm_width_diff_ci_hi']*100:+.1f}%]", axis=1)
else:
    b_lgbm['norm_w_diff'] = "N/A"
b_lgbm['cost5x_diff'] = b_lgbm.apply(lambda r: f"{r['cost5x_diff_mean']:+.3f} [{r['cost5x_diff_ci_lo']:+.3f}, {r['cost5x_diff_ci_hi']:+.3f}]", axis=1)

cols = ['arm_x', 'arm_y', 'is_primary', 'cov_diff', 'width_diff', 'norm_w_diff', 'cost5x_diff']
print(b_lgbm[cols].to_string(index=False))
