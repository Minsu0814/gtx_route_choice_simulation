# -*- coding: utf-8 -*-
"""
Paper Figure 3: 2D Grid Search Heatmap — (a) Test ρ² (b) FPR-1
Clean version: no cell numbers, colorbar only, fine-grid optimal marked.
Usage: python paper_fig3_heatmap.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pathlib import Path

# ── 한글 폰트 ──
font_path = 'C:/Windows/Fonts/malgun.ttf'
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / 'data' / 'sensitivity'
IMG_DIR = ROOT / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──
df_coarse = pd.read_csv(DATA_DIR / 'sensitivity_2d_grid_step005.csv')
df_fine = pd.read_csv(DATA_DIR / 'sensitivity_2d_grid_step001_rt08-10_sq00-02.csv')
print(f'Coarse grid (step=0.05): {len(df_coarse)} scenarios')
print(f'Fine grid   (step=0.01): {len(df_fine)} scenarios')

# 채택 최적점 (B 클러스터: seq 가중치 논문 방어용으로 의도적 선택)
# A 클러스터(0.96/0.03)가 ρ² 최대이나, seq=3%는 방어 어려움
# B 클러스터(0.90/0.08)는 ρ² 0.005 손해로 seq 10%급 확보
adopted = df_fine[(df_fine['route'] == 0.90) & (df_fine['sequence'] == 0.08)]
if len(adopted) > 0:
    best = adopted.iloc[0]
else:
    best = df_fine.loc[df_fine['test_rho_sq'].idxmax()]
opt_rt, opt_sq = best['route'], best['sequence']
opt_rho, opt_fpr = best['test_rho_sq'], best['test_fpr']
opt_mode = 1.0 - opt_rt - opt_sq
print(f'Adopted: w_route={opt_rt:.2f}, w_seq={opt_sq:.2f}, w_mode={opt_mode:.2f}')
print(f'         ρ²={opt_rho:.4f}, FPR-1={opt_fpr:.4f}')

# ── Pivot (coarse grid) ──
pivot_rho = df_coarse.pivot(index='route', columns='sequence', values='test_rho_sq')
pivot_fpr = df_coarse.pivot(index='route', columns='sequence', values='test_fpr')
route_vals = list(pivot_rho.index)
seq_vals = list(pivot_rho.columns)


def draw_heatmap(ax, pivot, cmap, title, label_fmt, opt_val, opt_label):
    data = pivot.values
    valid = data[~np.isnan(data)]
    vmin, vmax = valid.min(), valid.max()

    im = ax.imshow(data, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax, origin='lower',
                   interpolation='nearest')

    # ── 최적점 (fine grid) ──
    # Coarse grid에서 가장 가까운 셀 좌표 찾기
    closest_rt = min(route_vals, key=lambda v: abs(v - opt_rt))
    closest_sq = min(seq_vals, key=lambda v: abs(v - opt_sq))
    oi = route_vals.index(closest_rt)
    oj = seq_vals.index(closest_sq)

    ax.plot(oj, oi, marker='*', markersize=22, color='#FFD700',
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    # 주석 — 화살표 방향을 패널에 맞게
    ax.annotate(
        f'Adopted\n'
        f'$w_{{route}}$={opt_rt:.2f}, $w_{{seq}}$={opt_sq:.2f}\n'
        f'{label_fmt.format(opt_val)}',
        xy=(oj, oi),
        xytext=(oj - 4, oi - 4),
        fontsize=9, fontweight='bold', color='#1A1A1A',
        ha='center', va='top',
        arrowprops=dict(arrowstyle='->', color='#333', lw=1.8,
                        connectionstyle='arc3,rad=-0.2'),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#333', alpha=0.95, linewidth=1),
        zorder=11,
    )

    # ── 축 ──
    # 틱을 적당히 간격 띄워서 표시 (0.00, 0.20, 0.40, ...)
    xtick_idx = [i for i, v in enumerate(seq_vals) if v % 0.20 < 0.001 or i == 0]
    ytick_idx = [i for i, v in enumerate(route_vals) if v % 0.20 < 0.001 or i == 0]

    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([f'{seq_vals[i]:.2f}' for i in xtick_idx], fontsize=9)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([f'{route_vals[i]:.2f}' for i in ytick_idx], fontsize=9)

    ax.set_xlabel(r'$w_{sequence}$', fontsize=13, labelpad=6)
    ax.set_ylabel(r'$w_{route}$', fontsize=13, labelpad=6)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cbar.ax.tick_params(labelsize=9)

    # 테두리
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    return im


# ══════════════════════════════════════════════
# Figure 생성
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

draw_heatmap(axes[0], pivot_rho, 'YlOrRd',
             r'(a) Test McFadden $\rho^2$',
             r'$\rho^2$ = {:.3f}', opt_rho, 'Optimal')

draw_heatmap(axes[1], pivot_fpr, 'YlGn',
             '(b) First Preference Recovery',
             'FPR-1 = {:.1%}', opt_fpr, 'Optimal')

plt.tight_layout(w_pad=3)
out_path = IMG_DIR / 'fig3_2d_grid_heatmap.png'
fig.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'\nSaved: {out_path}')
