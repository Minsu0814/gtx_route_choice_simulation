"""매칭 정확도 검증 — 랜덤 100개 OD 수작업 검증 리포트 생성."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import sqlite3
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from module.similarity import parse_smartcard_trip, compute_all_metrics, compute_composite_similarity

ROOT = Path('../../')
DATA_DIR = ROOT / 'data' / 'training_set'
TCN_DIR = ROOT / 'data' / 'tcn' / '20250217'
IMG_DIR = ROOT / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

SIM_WEIGHTS = {'mode': 0.39, 'route': 0.40, 'sequence': 0.21}
N_SAMPLE = 38000  # 검증 샘플 수 (전체 OD의 ~7%, ±0.5% 오차)

# ========== 1. 데이터 로드 ==========
print('[1/4] Loading data...')
training = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
tcn = pd.read_parquet(TCN_DIR / 'TCN_20250217_route.parquet')
conn = sqlite3.connect(str(DATA_DIR / 'otp_cache.db'))

print(f'  Training: {len(training):,} rows, {training["od_pair"].nunique():,} ODs')
print(f'  TCN: {len(tcn):,} trips')

# ========== 2. 랜덤 OD 샘플링 (층화: 유사도 구간별) ==========
print(f'\n[2/4] Sampling {N_SAMPLE} ODs (stratified by similarity)...')

# 선택된 대안만 (choice_prob 최대)
best_alts = training.loc[training.groupby('od_pair')['choice_prob'].idxmax()].copy()

# 유사도 구간별 층화 샘플링
bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels = ['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
best_alts['sim_bin'] = pd.cut(best_alts['sim_composite'], bins=bins, labels=labels, right=False)

# TCN에 있는 OD만
tcn_ods = set(tcn['od_pair'].unique())
best_alts = best_alts[best_alts['od_pair'].isin(tcn_ods)]

# 구간별 비례 샘플링
sampled_list = []
for bin_label, group in best_alts.groupby('sim_bin', observed=True):
    n = min(len(group), max(5, int(N_SAMPLE * len(group) / len(best_alts))))
    sampled_list.append(group.sample(n=n, random_state=42))
sampled = pd.concat(sampled_list).reset_index(drop=True)

if len(sampled) > N_SAMPLE:
    sampled = sampled.sample(N_SAMPLE, random_state=42)

print(f'  Sampled: {len(sampled)} ODs')
print(f'  유사도 분포:')
print(sampled['sim_bin'].value_counts().sort_index().to_string())

# ========== 3. 상세 검증 ==========
print(f'\n[3/4] Validating matches...')

results = []
for _, row in sampled.iterrows():
    od = row['od_pair']
    alt_idx = int(row['alt_idx'])

    # OTP 캐시
    cur = conn.cursor()
    cur.execute('SELECT data FROM otp_cache WHERE od_pair = ?', (od,))
    cache_row = cur.fetchone()
    if cache_row is None:
        continue
    otp_data = pickle.loads(cache_row[0])
    if alt_idx >= len(otp_data['otp_parsed']):
        continue
    otp_parsed = otp_data['otp_parsed'][alt_idx]

    # SC (TCN) — 첫 번째 통행
    sc_trips = tcn[tcn['od_pair'] == od]
    if len(sc_trips) == 0:
        continue
    sc_sample = sc_trips.iloc[0]
    sc_parsed = parse_smartcard_trip(sc_sample)

    # 유사도 재계산 (상세)
    metrics = compute_all_metrics(otp_parsed, sc_parsed, skip_diagnostics=False)
    composite = compute_composite_similarity(metrics, weights=SIM_WEIGHTS)

    # 노선 비교
    otp_routes = sorted(otp_parsed.get('routes', []))
    sc_routes = sorted(sc_parsed.get('routes', []))
    route_match = set(otp_routes) == set(sc_routes)
    route_overlap = len(set(otp_routes) & set(sc_routes)) / max(len(set(otp_routes) | set(sc_routes)), 1)

    # 수단 비교
    otp_modes = sorted(otp_parsed.get('modes', set()))
    sc_modes = sorted(sc_parsed.get('modes', set()))
    mode_match = set(otp_modes) == set(sc_modes)

    # 정류장 비교
    otp_stops = otp_parsed.get('stops', [])
    sc_stops = sc_parsed.get('stops', [])

    # 환승 비교
    otp_transfers = otp_parsed.get('transfer_count', 0)
    sc_transfers = sc_parsed.get('transfer_count', 0)
    transfer_match = otp_transfers == sc_transfers

    # 자동 판정
    # 정확: 노선+수단+환승 모두 일치
    # 부분일치: 노선 70%+ 겹침 or 수단 일치
    # 불일치: 나머지
    if route_match and mode_match and transfer_match:
        auto_verdict = 'EXACT'
    elif route_overlap >= 0.7 or mode_match:
        auto_verdict = 'PARTIAL'
    elif route_overlap >= 0.3:
        auto_verdict = 'WEAK'
    else:
        auto_verdict = 'MISMATCH'

    results.append({
        'od_pair': od,
        'alt_idx': alt_idx,
        'sim_composite': composite.get('composite', row['sim_composite']),
        'sim_mode': metrics.get('mode', {}).get('mode_jaccard', 0),
        'sim_route': metrics.get('route', {}).get('route_jaccard', 0),
        'sim_sequence': metrics.get('sequence', {}).get('seq_lcs', 0),
        'otp_routes': ', '.join(otp_routes),
        'sc_routes': ', '.join(sc_routes),
        'otp_modes': ', '.join(otp_modes),
        'sc_modes': ', '.join(sc_modes),
        'otp_stops': ' → '.join(otp_stops[:5]),
        'sc_stops': ' → '.join(sc_stops[:5]),
        'otp_transfers': otp_transfers,
        'sc_transfers': sc_transfers,
        'route_match': route_match,
        'mode_match': mode_match,
        'transfer_match': transfer_match,
        'route_overlap': route_overlap,
        'n_total': row['n_total'],
        'choice_prob': row['choice_prob'],
        'auto_verdict': auto_verdict,
    })

df_results = pd.DataFrame(results)
print(f'  검증 완료: {len(df_results)} ODs')

# ========== 4. 결과 분석 ==========
print(f'\n[4/4] Results...')

# 전체 정확도
print('\n' + '=' * 70)
print('매칭 정확도 검증 결과')
print('=' * 70)

verdict_counts = df_results['auto_verdict'].value_counts()
total = len(df_results)
print(f'\n=== 자동 판정 결과 (n={total}) ===')
for v in ['EXACT', 'PARTIAL', 'WEAK', 'MISMATCH']:
    cnt = verdict_counts.get(v, 0)
    pct = cnt / total * 100
    print(f'  {v:<12}: {cnt:>4} ({pct:>5.1f}%)')

exact_partial = verdict_counts.get('EXACT', 0) + verdict_counts.get('PARTIAL', 0)
print(f'\n  정확+부분일치: {exact_partial}/{total} ({exact_partial/total*100:.1f}%)')

# 유사도 구간별 정확도
print(f'\n=== 유사도 구간별 정확도 ===')
df_results['sim_bin'] = pd.cut(df_results['sim_composite'], bins=bins, labels=labels, right=False)
print(f'{"구간":<12} {"EXACT":>8} {"PARTIAL":>8} {"WEAK":>8} {"MISMATCH":>8} {"정확률":>8}')
print('-' * 60)
for bin_label in labels:
    subset = df_results[df_results['sim_bin'] == bin_label]
    if len(subset) == 0:
        continue
    n = len(subset)
    exact = (subset['auto_verdict'] == 'EXACT').sum()
    partial = (subset['auto_verdict'] == 'PARTIAL').sum()
    weak = (subset['auto_verdict'] == 'WEAK').sum()
    mismatch = (subset['auto_verdict'] == 'MISMATCH').sum()
    acc = (exact + partial) / n * 100
    print(f'{bin_label:<12} {exact:>8} {partial:>8} {weak:>8} {mismatch:>8} {acc:>7.1f}%')

# 개별 항목별 일치율
print(f'\n=== 개별 항목 일치율 ===')
print(f'  노선 완전 일치:  {df_results["route_match"].mean()*100:.1f}%')
print(f'  수단 일치:       {df_results["mode_match"].mean()*100:.1f}%')
print(f'  환승횟수 일치:   {df_results["transfer_match"].mean()*100:.1f}%')
print(f'  노선 겹침 평균:  {df_results["route_overlap"].mean()*100:.1f}%')

# 유사도별 일치율
print(f'\n=== 유사도 vs 일치율 상관 ===')
print(f'  sim_composite vs route_match:    r={df_results["sim_composite"].corr(df_results["route_match"].astype(float)):.3f}')
print(f'  sim_composite vs mode_match:     r={df_results["sim_composite"].corr(df_results["mode_match"].astype(float)):.3f}')
print(f'  sim_composite vs transfer_match: r={df_results["sim_composite"].corr(df_results["transfer_match"].astype(float)):.3f}')

# 불일치 사례 상세
mismatches = df_results[df_results['auto_verdict'] == 'MISMATCH']
if len(mismatches) > 0:
    print(f'\n=== 불일치 사례 (상위 10개) ===')
    for _, m in mismatches.head(10).iterrows():
        print(f'  OD={m["od_pair"]}, sim={m["sim_composite"]:.3f}')
        print(f'    OTP: {m["otp_routes"]} ({m["otp_modes"]})')
        print(f'    SC:  {m["sc_routes"]} ({m["sc_modes"]})')
        print()

# CSV 저장
out_path = DATA_DIR / 'matching_validation.csv'
df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'\n상세 결과 저장: {out_path}')

# 시각화
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:/Windows/Fonts/malgun.ttf'
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) 판정 분포
ax = axes[0]
colors = {'EXACT': '#2196F3', 'PARTIAL': '#4CAF50', 'WEAK': '#FF9800', 'MISMATCH': '#F44336'}
verdicts = ['EXACT', 'PARTIAL', 'WEAK', 'MISMATCH']
counts = [verdict_counts.get(v, 0) for v in verdicts]
bars = ax.bar(verdicts, counts, color=[colors[v] for v in verdicts], edgecolor='white')
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{cnt}\n({cnt/total*100:.0f}%)', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Count')
ax.set_title('(a) Matching Verdict Distribution')

# (b) 유사도 vs 정확 판정
ax = axes[1]
for v, c in colors.items():
    subset = df_results[df_results['auto_verdict'] == v]
    ax.scatter(subset['sim_composite'], subset['route_overlap'],
              c=c, label=v, alpha=0.6, s=20, edgecolors='none')
ax.set_xlabel('Composite Similarity')
ax.set_ylabel('Route Overlap Ratio')
ax.set_title('(b) Similarity vs Route Overlap')
ax.legend(fontsize=8)

# (c) 구간별 정확률
ax = axes[2]
bin_acc = []
bin_labels_used = []
for bl in labels:
    subset = df_results[df_results['sim_bin'] == bl]
    if len(subset) > 0:
        acc = ((subset['auto_verdict'] == 'EXACT').sum() +
               (subset['auto_verdict'] == 'PARTIAL').sum()) / len(subset) * 100
        bin_acc.append(acc)
        bin_labels_used.append(bl)
ax.bar(bin_labels_used, bin_acc, color='steelblue', edgecolor='navy', alpha=0.8)
for i, (bl, acc) in enumerate(zip(bin_labels_used, bin_acc)):
    ax.text(i, acc + 1, f'{acc:.0f}%', ha='center', fontsize=9)
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Composite Similarity Bin')
ax.set_title('(c) Accuracy by Similarity Range')
ax.set_ylim(0, 105)

plt.tight_layout()
fig_path = IMG_DIR / 'fig_matching_validation.png'
plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
plt.close()
print(f'시각화 저장: {fig_path}')

print('\nDone.')
