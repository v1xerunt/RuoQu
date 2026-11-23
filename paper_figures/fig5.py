"""Figure 5: what the model reveals about the history of written Chinese."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from style import *

setup()
rng = np.random.default_rng(5)
LT = os.path.join(PKG, 'runs', 'lexical_timeline')
wave = pd.read_csv(os.path.join(LT, 'fig_wave_data.tsv'), sep='\t')
pairs = pd.read_csv(os.path.join(LT, 'fig_pairs_data.tsv'), sep='\t')
views = pd.read_csv(os.path.join(PKG, 'runs', 'views_skel_lex.tsv'), sep='\t')
canon = pd.read_csv(os.path.join(PKG, 'runs', 'l3_canon.tsv'), sep='\t')
yrs = [c for c in wave.columns if c.lstrip('-').isdigit()]
Y = np.array([int(c) for c in yrs])

fig = plt.figure(figsize=(W2, 132 * MM))
outer = fig.add_gridspec(1, 2, width_ratios=[0.88, 1.0], wspace=0.10, left=0.012, right=0.985, top=0.95, bottom=0.075)
left = outer[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.62], wspace=0.0)
right = outer[0, 1].subgridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.62, wspace=0.55)

# ---------------- a: wave ----------------
ax = fig.add_subplot(left[0, 0])
lax = fig.add_subplot(left[0, 1], sharey=ax)
lax.axis('off')
label(ax, 'a', x=-0.03, y=1.035)
M = wave[yrs].values
M = M / np.abs(M).max(1, keepdims=True)
M = np.sign(M) * np.abs(M) ** 1.4  # soften the noise floor so the peaks carry the picture
cmap = LinearSegmentedColormap.from_list('div', ['#9fb7d3', '#d6e1ee', '#f3f6fa', '#ffffff', '#f6e3e6', '#dc9aa6',
                                                 '#a63d52'])
im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=-1, vmax=1, extent=(Y[0] - 50, Y[-1] + 50, len(M) - 0.5, -0.5),
               interpolation='nearest')
ax.set_yticks([])
year_axis(ax, 'x', ticks=(-500, 0, 500, 1000, 1500))
ax.set_xlabel('century on which the model is conditioned')
ax.spines['left'].set_visible(False)
CAT = {'g': (INK, 'grammar'), 'i': (BLUE, 'offices'), 'b': (WINE, 'Buddhist idiom'),
       'r': (PURPLE, 'reign titles and dynasty'), 'l': (TEAL, 'literary life')}
show = {'已矣': ('g', 'final particles'), '先王': ('l', 'former kings'),
        '諸侯': ('i', 'feudal lords'), '丞相': ('i', 'chancellor'),
        '時佛': ('b', '"once the Buddha"'), '太尉': ('i', 'grand commandant'),
        '正真': ('b', 'early sūtra term'), '佛說': ('b', '"the Buddha said"'),
        '爾時': ('b', '"at that time"'), '答曰': ('g', '"answered"'),
        '貞觀': ('r', 'reign 627–649'), '開元': ('r', 'reign 713–741'),
        '宰相': ('i', 'chief minister'), '節度': ('i', 'military governor'),
        '樞密': ('i', 'Military Affairs'), '祖師': ('b', 'patriarch (Chan)'),
        '紹興': ('r', 'reign 1131–1162'), '次韻': ('l', 'matching rhymes'),
        '只是': ('g', '"merely"'), '至正': ('r', 'reign 1341–1368'),
        '洪武': ('r', 'reign 1368–1398'), '國朝': ('r', '"our dynasty"'),
        '朱子': ('l', 'Master Zhu (Xi)'), '巡撫': ('i', 'provincial governor'),
        '皇上': ('r', '"His Majesty"'), '地方': ('i', 'locality')}
rows = list(wave['item'])
items = sorted([(rows.index(k), k) for k in show if k in rows])
pos, last = [], -10
for r, k in items:
    p = max(r, last + 3.0)
    pos.append(p); last = p
over = pos[-1] - (len(rows) - 1)
if over > 0:
    pos = [p - over * (i / (len(pos) - 1)) for i, p in enumerate(pos)]
for (r, k), p in zip(items, pos):
    cat, gl = show[k]
    c = CAT[cat][0]
    lax.plot([0.0, 0.08, 0.16], [r, r, p], color=c, lw=0.35, clip_on=False)
    lax.text(0.19, p, f'{k}  {gl}', fontsize=4.7, color=c, va='center', ha='left', clip_on=False)
lax.set_xlim(0, 1)
ax.set_xlim(Y[0] - 50, Y[-1] + 50)
ax.set_ylim(len(M) - 0.5, -0.5)
cax = ax.inset_axes([0.0, 1.012, 0.45, 0.013])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_ticks([-1, 1]); cb.set_ticklabels(['less', 'more expected'])
cb.ax.tick_params(labelsize=4.8, length=1.2, pad=1)
cb.outline.set_linewidth(0.3)
ax.text(1.0, 1.018, '156 most time-specific words', transform=ax.transAxes, fontsize=5.0, va='center', ha='right',
        color=GREYS[0])
handles = [plt.Line2D([], [], ls='', marker='s', ms=3.2, color=v[0]) for v in CAT.values()]
ax.legend(handles, [v[1] for v in CAT.values()], loc='upper left', bbox_to_anchor=(-0.02, -0.075), fontsize=4.8,
          ncol=3, handletextpad=0.1, columnspacing=0.6, borderaxespad=0, frameon=False)

# ---------------- b: replacement curves ----------------
sub = right[0, :].subgridspec(1, 4, wspace=0.42)
groups = [('之／底／的', 'attributive'), ('此／這', '"this"'), ('何／甚／麼', '"what"'),
          ('其／他', '3rd person')]
cols = [BLUE, WINE, PURPLE]
for k, (g, gl) in enumerate(groups):
    ax = fig.add_subplot(sub[0, k])
    if k == 0:
        label(ax, 'b', x=-0.62, y=1.06)
    d = pairs[pairs.group == g]
    for j in range(len(d)):
        r = d.iloc[j]
        ax.plot(Y, r[yrs].values.astype(float), color=cols[j], lw=0.9, label=f'{r["char"]}')
    ax.axhline(0, color=GREYS[2], lw=0.4)
    ax.set_title(f'{gl}', fontsize=5.8, pad=2)
    ax.legend(loc='upper left', fontsize=5.6, handlelength=0.8, handletextpad=0.25, borderaxespad=0.05,
              labelspacing=0.12)
    ax.set_xlim(-650, 1950)
    ax.set_xticks([-500, 500, 1500]); ax.set_xticklabels(['500\nBCE', '500', '1500'], fontsize=4.9)
    ax.tick_params(axis='y', labelsize=4.9)
    ax.axvspan(589, 1279, color=SLATE, alpha=0.07, lw=0)
    if k == 0:
        ax.set_ylabel('relative log-probability\nin the same context', fontsize=5.4)

# ---------------- c: grammar vs vocabulary ----------------
ax = fig.add_subplot(right[1, 0])
label(ax, 'c', x=-0.40, y=1.04)
order = ['秦西漢', '東漢', '魏晉南北朝', '隋唐', '五代北宋', '南宋', '元', '明', '清']
en = ['Qin–W.Han', 'E. Han', 'Six Dyn.', 'Sui–Tang', 'N. Song', 'S. Song', 'Yuan', 'Ming', 'Qing']
mid = [-100, 120, 400, 750, 1020, 1200, 1320, 1500, 1780]
rat, lo, hi = [], [], []
for p in order:
    g = views[views.period == p]
    rat.append(g.e_skel.median() / g.e_lex.median())
    bs = []
    for _ in range(800):
        s = g.sample(len(g), replace=True, random_state=rng.integers(1e9))
        bs.append(s.e_skel.median() / max(s.e_lex.median(), 1))
    a, b = np.percentile(bs, [2.5, 97.5]); lo.append(a); hi.append(b)
rat, lo, hi = map(np.array, (rat, lo, hi))
ax.axhline(1, color=GREYS[2], lw=0.6, ls=(0, (3, 2)))
ax.vlines(mid, np.clip(lo, 0.5, 2.25), np.clip(hi, 0.5, 2.25), color=BLUE_L, lw=1.4)
ax.plot(mid, rat, color=BLUE, lw=1.0, marker='o', ms=3, mec='white', mew=0.3, zorder=4)
ax.axvspan(780, 1100, color=WINE, alpha=0.08, lw=0)
ax.text(940, 2.18, 'Ancient-style prose\nmovement', fontsize=4.9, color=WINE, ha='center', va='top', linespacing=1.0)
for x_, r_, n_ in zip(mid, rat, en):
    pass
ax.set_xticks(mid); ax.set_xticklabels(en, rotation=45, ha='right', rotation_mode='anchor', fontsize=5)
ax.set_ylabel('dating error, grammar view /\ndating error, vocabulary view', fontsize=5.4)
ax.set_ylim(0.5, 2.25)
ax.set_xlim(-250, 1900)

# ---------------- d: canonization ----------------
ax = fig.add_subplot(right[1, 1])
label(ax, 'd', x=-0.40, y=1.04)
canon['cl'] = canon.cls.replace({'none': '0', 'df0(4g)': '0', 'df1-4': '1–4', 'df5-19': '5–19', 'df20-99': '20–99',
                                 'df100+': '≥100'})
liji = canon.title.str.contains('禮記')
dx = canon.chapter.str.contains('大學|中庸')
canon['grp'] = np.where(liji & dx, 'dz', np.where(liji, 'liji', np.where(canon.title.str.contains('論語'), 'ly', 'oth')))
cls = ['0', '1–4', '5–19', '20–99', '≥100']
spec = [('oth', 'other early works', GREYS[1], 'o'), ('liji', 'Liji, other chapters', BLUE_L, 'o'),
        ('ly', 'Analects 論語', PURPLE, 's'), ('dz', 'Great Learning 大學 and\nDoctrine of the Mean 中庸', WINE, 'D')]
for gk, gl, c, mk in spec:
    d = canon[canon.grp == gk].groupby('cl').apply(lambda g: pd.Series(dict(n=g.n.sum(), v=g.llr.sum() / g.n.sum())))
    d = d.reindex(cls)
    ok = d.n >= 100
    ax.plot(np.arange(5)[ok.values], d.v[ok].values, color=c, marker=mk, ms=3, lw=0.9, label=gl, mec='white',
            mew=0.3)
ax.axhline(0, color=GREYS[2], lw=0.5)
ax.set_xticks(range(5)); ax.set_xticklabels(cls)
ax.set_xlabel('Song–Qing works reusing the phrase')
ax.set_ylabel('per-character evidence,\nSong vs Warring States (nats)', fontsize=5.6)
ax.legend(loc='upper left', fontsize=4.6, handletextpad=0.3, borderaxespad=0.05, labelspacing=0.2, handlelength=1.4)
ax.set_ylim(-0.075, 0.16)
ax.text(0.98, 0.03, 'stricter quotation removal\nshrinks the gap by 10%',
        transform=ax.transAxes, fontsize=4.7, ha='right', va='bottom', color=GREYS[0], linespacing=1.1)

save(fig, 'fig5_language')
print('ok', np.round(rat, 2))
