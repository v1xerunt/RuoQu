"""Figure 3: independent reproduction of philological verdicts (blind test, Shangshu, Liezi)."""
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from style import *

setup()
rng = np.random.default_rng(1)
bt = pd.read_csv(os.path.join(PKG, 'runs', 'blind_ensemble_table.tsv'), sep='\t')

# Liezi passages scored by the fold-0 Ruoqu-L model (liezi.py)
lz = pd.read_csv(os.path.join(PKG, 'runs', 'liezi_passages_l3_fold0.tsv'), sep='\t').rename(
    columns={'passage': 'pi', 'n_chars': 'n', 'share_pre220': 'borrowed', 'median_date': 'median',
             'p_after_200': 'p_after', 'llr_300_vs_m400': 'llr', 'text_start': 'text'})
lz['kind'] = np.where(lz.borrowed >= 0.3, 'parallel', np.where(lz.borrowed < 0.05, 'unique', 'mixed'))
print(lz.groupby('kind').agg(n=('pi', 'size'), llr=('llr', 'median'), date=('median', 'median')))
print('p', mannwhitneyu(lz[lz.kind == 'unique'].llr, lz[lz.kind == 'parallel'].llr, alternative='greater').pvalue)

EN = {'列子': 'Liezi', '關尹子': 'Guanyinzi', '亢倉子': 'Kangcangzi', '子華子': 'Zihuazi', '元經': 'Yuanjing',
      '李衛公問對': 'Li Weigong wendui', '孔叢子': 'Kongcongzi', '古文孝經孔傳': 'Kong comm. on Xiaojing',
      '忠經': 'Zhongjing', '西京雜記': 'Xijing zaji', '神異經': 'Shenyi jing', '漢武帝內傳': 'Han Wudi neizhuan',
      '洞冥記': 'Dongming ji', '列仙傳': 'Liexian zhuan', '偽古文尚書': 'Old Text Shangshu',
      '尉繚子': 'Weiliaozi', '六韜': 'Liutao', '晏子春秋': 'Yanzi chunqiu', '鶡冠子': 'Heguanzi', '文子': 'Wenzi',
      '孫子': 'Sunzi', '老子': 'Laozi', '穆天子傳': 'Mu tianzi zhuan'}

fig = plt.figure(figsize=(W2, 104 * MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.0], height_ratios=[1, 1], hspace=0.62, wspace=0.55,
                      left=0.13, right=0.985, top=0.95, bottom=0.10)

# ---------------- a: blind test ----------------
ax = fig.add_subplot(gs[:, 0])
label(ax, 'a', x=-0.62, y=1.0)
d = bt[bt.cat.isin(['A', 'B'])].copy()
d['name'] = d.work.map(EN)
A = d[d.cat == 'A'].sort_values('p_later')
B = d[d.cat == 'B'].sort_values('p_later')
yA = np.arange(len(A)) + len(B) + 1.5
yB = np.arange(len(B))
ax.axvspan(0.5, 1.02, color=WINE, alpha=0.06, lw=0)
ax.axvline(0.5, color=GREYS[2], lw=0.5, ls=(0, (3, 2)))
ax.scatter(A.p_later, yA, s=14, color=WINE, edgecolor='white', lw=0.3, zorder=3)
ax.scatter(B.p_later, yB, s=14, color=BLUE, edgecolor='white', lw=0.3, zorder=3, marker='s')
for (yy, r) in zip(np.r_[yB, yA], pd.concat([B, A]).itertuples()):
    ax.text(-0.04, yy, f'{r.name} {r.work}', fontsize=5.0, ha='right', va='center',
            color=WINE if r.cat == 'A' else BLUE)
ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.8, yA.max() + 1.3)
ax.set_xlabel('posterior probability that the text\npostdates its traditional attribution')
ax.text(1.0, yA.max() + 0.9, 'forgeries (15)\n13 read later than attributed', fontsize=5.2, ha='right', va='bottom',
        color=WINE, linespacing=1.15)
ax.text(1.0, len(B) - 0.35, 'once doubted and vindicated\nby excavated manuscripts (8)\n7 read as early', fontsize=5.2,
        ha='right', va='bottom', color=BLUE, linespacing=1.15)
ax.text(0.02, yA.max() + 0.9, 'AUC 0.983', fontsize=6.2, fontweight='bold', color=INK, va='bottom')

# ---------------- b: Shangshu ----------------
ax = fig.add_subplot(gs[0, 1])
label(ax, 'b', x=-0.30, y=1.04)
names = ['Ruoqu-L', 'discriminative\n(Aeneas-style)', 'character LR', 'period n-gram', 'function words']
auc = [0.815, 0.381, 0.315, 0.436, 0.475]
cols = [BLUE, WINE, GREYS[0], GREYS[1], GREYS[2]]
y = np.arange(len(names))[::-1]
ax.barh(y, np.array(auc) - 0.5, left=0.5, color=cols, height=0.62, lw=0)
for yy, v in zip(y, auc):
    ax.text(v + (0.012 if v > 0.5 else -0.012), yy, f'{v:.3f}', va='center', ha='left' if v > 0.5 else 'right',
            fontsize=5.2)
ax.axvline(0.5, color=INK, lw=0.5)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.tick_params(axis='y', length=0)
ax.set_xlim(0.2, 0.92)
ax.set_xlabel('AUC, forged vs authentic chapters')
ax.set_title('Shangshu 《尚書》 by chapter', fontsize=6.3, loc='left', pad=3)
ax.text(0.80, 3.5, r'$p=2\times10^{-5}$', fontsize=5.0, ha='right', va='center', color=BLUE)

# ---------------- c: Shangshu anatomy ----------------
ax = fig.add_subplot(gs[0, 2])
label(ax, 'c', x=-0.34, y=1.04)
cls = ['forged\n(Old Text)', 'authentic\n(New Text)', 'Zhou\nannouncements']
own = [-0.0370, -0.0458, -0.0395]
share = [0.055, 0.336, 0.078]
x = np.arange(3)
ax.bar(x, own, color=[WINE, BLUE, BLUE_L], width=0.58, lw=0)
for xx, v in zip(x, own):
    ax.text(xx, v - 0.0012, f'{v:.4f}', ha='center', va='top', fontsize=5.0)
ax.axhline(0, color=INK, lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([f'{c}\n{100 * s_:.0f}% quoted' for c, s_ in zip(cls, share)], fontsize=5.0,
                                     linespacing=1.1)
ax.tick_params(axis='x', length=0)
ax.set_ylim(-0.058, 0.009)
ax.set_ylabel('per-character evidence,\n300 CE vs 400 BCE (nats)', fontsize=5.8)
ax.set_title("the chapters' own wording", fontsize=6.3, loc='left', pad=3)
ax.plot([0, 0, 1, 1], [0.002, 0.004, 0.004, 0.002], color=GREYS[0], lw=0.5)
ax.text(0.5, 0.0045, 'p = 0.038', fontsize=5.0, ha='center', va='bottom', color=GREYS[0])

# ---------------- d: Liezi passages ----------------
ax = fig.add_subplot(gs[1, 1])
label(ax, 'd', x=-0.30, y=1.04)
kc = {'unique': WINE, 'mixed': SLATE, 'parallel': BLUE}
kn = {'unique': 'no early parallel (<5%)', 'mixed': 'partial (5–30%)', 'parallel': 'parallel (≥30%)'}
for k in ['parallel', 'mixed', 'unique']:
    s_ = lz[lz.kind == k]
    ax.scatter(s_.borrowed + (rng.uniform(0, 0.012, len(s_)) if k == 'unique' else 0), s_.llr, s=8, color=kc[k],
               alpha=0.85, edgecolor='white', lw=0.25, label=kn[k], zorder=3)
ax.axhline(0, color=INK, lw=0.5)
ax.set_xlabel('share of passage found in\ntexts dated before 220 CE')
ax.set_ylabel('per-character evidence,\n300 CE vs 400 BCE (nats)', fontsize=5.8)
ax.set_xlim(-0.03, 0.8)
ax.set_ylim(-0.17, 0.06)
ax.set_title('119 passages of the Liezi 《列子》', fontsize=6.3, loc='left', pad=3)
ax.legend(loc='lower right', fontsize=4.8, handletextpad=0.1, borderaxespad=0.1, markerscale=1.2)
ax.text(0.79, 0.055, 'reads later', fontsize=4.9, color=GREYS[1], va='top', ha='right')

# ---------------- e: Liezi summary ----------------
ax = fig.add_subplot(gs[1, 2])
label(ax, 'e', x=-0.34, y=1.04)
for i, k in enumerate(['unique', 'mixed', 'parallel']):
    v = lz[lz.kind == k].llr.values
    ax.boxplot(v, positions=[i], widths=0.5, patch_artist=True, showfliers=False,
               medianprops=dict(color='white', lw=1.0), whiskerprops=dict(lw=0.6, color=kc[k]),
               capprops=dict(lw=0), boxprops=dict(lw=0, facecolor=kc[k]))
    ax.scatter(i + 0.36 + rng.uniform(-0.06, 0.06, len(v)), v, s=2.5, color=kc[k], alpha=0.6, lw=0, zorder=3)
    ax.text(i, -0.165, f'n = {len(v)}', fontsize=4.8, ha='center', va='bottom', color=GREYS[1])
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['no early\nparallel', 'partial', 'parallel'], fontsize=5.2)
ax.tick_params(axis='x', length=0)
ax.axhline(0, color=INK, lw=0.5)
ax.set_ylim(-0.17, 0.06)
ax.set_ylabel('per-character evidence,\n300 CE vs 400 BCE (nats)', fontsize=5.8)
ax.plot([0, 0, 2, 2], [0.035, 0.042, 0.042, 0.035], color=GREYS[0], lw=0.5)
ax.text(1, 0.044, r'$p=1.2\times10^{-4}$ ($3.3\times10^{-6}$ for Ruoqu-XL)', fontsize=4.9, ha='center', va='bottom')
ax.set_title('passages without early parallels', fontsize=6.3, loc='left', pad=3)

save(fig, 'fig3_philology')
print('ok')
