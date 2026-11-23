"""Figure 2: the limits of imitation, human and machine."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from style import *

setup()
rng = np.random.default_rng(0)
h = pd.read_csv(os.path.join(PKG, 'runs', 'l3_imit_robust.tsv'), sep='\t')
a = pd.read_csv(os.path.join(PKG, 'runs', 'aiimit_pieces.tsv'), sep='\t')
h['shift'] = h.ctrl_app - h.app
a['shift'] = a.own - a.app
a['need'] = a.own - a.target

fig = plt.figure(figsize=(W2, 122 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], width_ratios=[1.05, 1.0, 1.0], hspace=0.38, wspace=0.42,
                      left=0.06, right=0.985, top=0.965, bottom=0.085)

# ---------------- a: concept ----------------
ax = fig.add_subplot(gs[0, 0])
label(ax, 'a', x=-0.10, y=1.02)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
y = 0.42
ax.annotate('', xy=(0.98, y), xytext=(0.02, y), arrowprops=dict(arrowstyle='-|>', lw=0.6, color=INK, mutation_scale=6))
ax.text(0.98, y + 0.03, 'year', fontsize=5.2, ha='right', va='bottom', color=GREYS[1])
pts = {'target': 0.12, 'imit': 0.66, 'own': 0.84}
ax.plot([pts['target']], [y], marker='o', ms=5, color=GREYS[0], zorder=5)
ax.text(pts['target'], y - 0.07, 'exemplar\n(e.g. Tao Yuanming)', fontsize=5.2, ha='center', va='top',
        color=GREYS[0], linespacing=1.15)
ax.plot([pts['own']], [y], marker='o', ms=5, color=TEAL, zorder=5)
ax.text(pts['own'], y - 0.07, "writer's\nordinary\npieces", fontsize=5.2, ha='center', va='top', color=TEAL,
        linespacing=1.15)
ax.plot([pts['imit']], [y], marker='D', ms=4.2, color=TEAL, mfc='white', mew=1.0, zorder=5)
ax.text(pts['imit'], y - 0.07, 'imitation', fontsize=5.2, ha='center', va='top', color=TEAL)
ax.add_patch(FancyArrowPatch((pts['own'], y + 0.20), (pts['target'], y + 0.20), arrowstyle='-|>', mutation_scale=6,
                             lw=0.6, color=GREYS[2], ls=(0, (3, 2))))
ax.text((pts['own'] + pts['target']) / 2, y + 0.24, 'distance to cover', fontsize=5.4, ha='center', color=GREYS[1])
ax.add_patch(FancyArrowPatch((pts['own'], y + 0.08), (pts['imit'], y + 0.08), arrowstyle='-|>', mutation_scale=6,
                             lw=1.0, color=TEAL))
ax.text((pts['own'] + pts['imit']) / 2, y + 0.115, 'covered', fontsize=5.4, ha='center', color=TEAL)
ax.text(0.02, 1.0, 'All dates are dates the model assigns,\nso each writer serves as their own control',
        fontsize=5.4, va='top', color=INK, linespacing=1.3)
ax.text(0.5, 0.84, r'fraction covered $=\dfrac{\mathrm{covered}}{\mathrm{distance\ to\ cover}}$', fontsize=6,
        ha='center', va='top')
ax.text(0.02, 0.04, '1,047 dated imitations in 225 collections (Tang–Qing)\n'
        '2,578 imitations by six large language models', fontsize=5.0, va='bottom', color=GREYS[0], linespacing=1.3)


# ---------------- b: distance covered vs distance required ----------------
def binned(x, yv, edges, groups=None, n=600):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x > lo) & (x <= hi)
        if m.sum() < 8:
            continue
        v = yv[m]
        bs = [np.median(rng.choice(v, len(v))) for _ in range(n)]
        rows.append(((lo + hi) / 2, np.median(v), *np.percentile(bs, [2.5, 97.5]), m.sum()))
    return np.array(rows)


ax = fig.add_subplot(gs[0, 1])
label(ax, 'b', x=-0.24)
edges = np.array([0, 300, 600, 900, 1200, 1500, 2100])
lim = 1900
ax.fill_between([0, lim], [0, 0], [0, lim], color=GRID, alpha=0.5, lw=0, zorder=0)
ax.plot([0, lim], [0, lim], color=GREYS[2], lw=0.6, ls=(0, (3, 2)))
ax.text(1180, 1330, 'perfect imitation', rotation=45, fontsize=5.2, color=GREYS[1], ha='center', va='center',
        rotation_mode='anchor')
ax.axhline(0, color=GREYS[2], lw=0.5)
rh = binned(h.gap.values, h['shift'].values, edges)
ra = binned(a.need.values, a['shift'].values, edges)
for r, c, mk, lab, dx in [(rh, TEAL, 'o', 'human imitators', -18), (ra, PURPLE, 's', 'language models (LLMs)', 18)]:
    ax.errorbar(r[:, 0] + dx, r[:, 1], yerr=[r[:, 1] - r[:, 2], r[:, 3] - r[:, 1]], fmt=mk, ms=3, color=c,
                ecolor=c, elinewidth=0.7, capsize=0, mec='white', mew=0.3, label=lab, zorder=4)
xx = np.array([0, lim])
ax.plot(xx, 0.082 * xx, color=TEAL, lw=0.7, alpha=0.8)
ax.text(620, 120, 'slope 0.082\n(8.2 years per century)', fontsize=5, color=TEAL, ha='center', va='bottom',
        linespacing=1.0)
ax.set_xlim(0, lim); ax.set_ylim(-150, lim)
ax.set_xlabel('distance to cover (years)')
ax.set_ylabel('distance covered (years)')
ax.set_aspect('equal')
ax.set_xticks([0, 500, 1000, 1500]); ax.set_yticks([0, 500, 1000, 1500])
ax.legend(loc='upper left', fontsize=5.2, handletextpad=0.3, borderaxespad=0.2)

# ---------------- c: fraction covered, with intervals ----------------
ax = fig.add_subplot(gs[0, 2])
label(ax, 'c', x=-0.40)
rows = [
    ('HEAD', 'human writers', TEAL),
    ('20–60 characters', 0.109, 0.064, 0.144, 0.25, TEAL),
    ('60–150 characters', 0.051, 0.024, 0.099, 0.21, TEAL),
    ('150+ characters', 0.044, 0.019, 0.090, 0.11, TEAL),
    ('placebo', 0.0, np.nan, np.nan, np.nan, GREYS[1]),
    ('HEAD', 'language models', PURPLE),
    ('Claude Sonnet', 0.114, 0.085, 0.146, np.nan, AI_COL['sonnet']),
    ('DeepSeek v4.1', 0.115, 0.070, 0.146, np.nan, AI_COL['deepseek-v41-flash']),
    ('GPT Sol 6.2', 0.065, 0.039, 0.091, np.nan, AI_COL['gpt-sol62']),
    ('Gemini 4 Neon', 0.054, 0.025, 0.101, np.nan, AI_COL['gemini4-neon']),
    ('Qwen3-32B', 0.051, np.nan, np.nan, np.nan, AI_COL['qwen3-32b']),
    ('Qwen2.5-32B', 0.047, np.nan, np.nan, np.nan, AI_COL['qwen25-32b']),
]
ys, labs = [], []
for i, r in enumerate(rows):
    yy = len(rows) - i
    if r[0] == 'HEAD':
        ax.text(-0.012, yy, r[1], fontsize=5.8, fontweight='bold', color=r[2], ha='right', va='center',
                transform=ax.get_yaxis_transform(), clip_on=False)
        continue
    n, m, lo, hi, nrm, c = r
    if not np.isnan(lo):
        ax.plot([lo, hi], [yy, yy], color=c, lw=1.0, solid_capstyle='butt')
    ax.plot(m, yy, 'o', ms=3.4, color=c, mec='white', mew=0.3, zorder=4)
    if not np.isnan(nrm):
        ax.plot(nrm, yy, 'o', ms=3.2, mfc='white', mec=c, mew=0.8, zorder=4)
    ys.append(yy); labs.append(n)
ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=5.6)
ax.tick_params(axis='y', length=0)
ax.axvline(0, color=GREYS[2], lw=0.5)
ax.axvline(1, color=GREYS[2], lw=0.5)
ax.set_xlim(-0.04, 0.32)
ax.set_xlabel('fraction of the distance covered')
ax.set_ylim(0.3, len(rows) + 0.7)
ax.legend([Line2D([], [], marker='o', ls='', color=GREYS[0], ms=3.4),
           Line2D([], [], marker='o', ls='', mfc='white', mec=GREYS[0], ms=3.2)],
          ['median (95% CI)', 'relative to genuine pieces\nof the exemplar\'s era'], loc='lower right', fontsize=4.9,
          handletextpad=0.2, borderaxespad=0.1)

# ---------------- d: AI imitation tracking ----------------
ax = fig.add_subplot(gs[1, 0:2])
label(ax, 'd', x=-0.085, y=1.03)
pass
lab_en = {'詩經': 'Shijing', '孟子': 'Mencius', '莊子': 'Zhuangzi', '楚辭': 'Chuci', '司馬相如': 'Sima Xiangru',
          '史記': 'Shiji', '古詩十九首': 'Nineteen Old Poems', '曹植': 'Cao Zhi', '阮籍': 'Ruan Ji',
          '陶淵明': 'Tao Yuanming', '謝靈運': 'Xie Lingyun', '鮑照': 'Bao Zhao', '庾信賦': 'Yu Xin (fu)',
          '庾信': 'Yu Xin', '李白': 'Li Bai', '杜甫': 'Du Fu', '柳宗元': 'Liu Zongyuan', '韓愈': 'Han Yu',
          '韓愈文': 'Han Yu (prose)', '白居易': 'Bai Juyi', '李商隱': 'Li Shangyin', '歐陽修': 'Ouyang Xiu',
          '蘇軾': 'Su Shi', '蘇軾賦': 'Su Shi (fu)', '蘇軾文': 'Su Shi (prose)', '陸游': 'Lu You'}
fcol = {'verse': PURPLE, 'prose': BLUE, 'fu': WINE}
fname = {'verse': 'verse', 'prose': 'prose', 'fu': 'rhapsody (fu)'}
# model defaults (no target named)
own = a.groupby(['model', 'gclass']).own.first().unstack()
for gcl, c in fcol.items():
    lo, hi = own[gcl].min(), own[gcl].max()
    ax.axhspan(lo, hi, color=c, alpha=0.13, lw=0, zorder=1)
ax.text(-690, own['verse'].max() + 18, "LLMs' own verse (no exemplar named)", fontsize=5, color=PURPLE, va='bottom')
ax.text(-690, own['prose'].min() - 22, "LLMs' own prose", fontsize=5, color=BLUE, va='top')
ax.plot([-750, 1300], [-750, 1300], color=GREYS[2], lw=0.6, ls=(0, (3, 2)), zorder=2)
ax.text(-560, -470, 'perfect imitation', rotation=0, fontsize=5, color=GREYS[1], ha='left')
g = a.groupby(['label', 'gclass']).agg(t=('target', 'first'), app=('app', 'median'),
                                       q1=('app', lambda v: v.quantile(.25)), q3=('app', lambda v: v.quantile(.75)),
                                       n=('app', 'size')).reset_index()
for r in g.itertuples():
    c = fcol[r.gclass]
    ax.plot([r.t, r.t], [r.q1, r.q3], color=c, lw=0.7, alpha=0.7, zorder=3)
    ax.plot(r.t, r.app, 'o' if r.gclass == 'verse' else ('s' if r.gclass == 'prose' else 'D'), ms=3.2, color=c,
            mec='white', mew=0.3, zorder=4)
show = {'詩經': (0, 40), '楚辭': (0, 40), '史記': (0, -95), '古詩十九首': (0, 45), '陶淵明': (-10, 40),
        '杜甫': (0, 40), '韓愈文': (0, -95), '歐陽修': (0, -95), '陸游': (0, 40), '莊子': (5, -95), '庾信': (0, 40)}
for r in g.itertuples():
    if r.label in show:
        dx, dy = show[r.label]
        ax.text(r.t + dx, (r.q3 if dy > 0 else r.q1) + (12 if dy > 0 else -12), f'{lab_en[r.label]}\n{r.label}',
                fontsize=4.8, ha='center', va='bottom' if dy > 0 else 'top', color=GREYS[0], linespacing=1.0)
ax.set_xlim(-760, 1260); ax.set_ylim(-600, 1650)
year_axis(ax, 'x', ticks=(-500, 0, 500, 1000)); year_axis(ax, 'y', ticks=(-500, 0, 500, 1000, 1500))
ax.set_xlabel('date of the exemplar')
ax.set_ylabel('date assigned to the imitation')
for i, (p0, p1, n) in enumerate(PERIODS):
    if p1 < -600 or p0 > 1650:
        continue
    pass
ax.legend([Line2D([], [], marker=m, ls='', color=c, ms=3.2) for m, c in zip(['o', 's', 'D'], fcol.values())],
          [fname[k] for k in fcol], loc='lower right', fontsize=5.2, title='imitations (median, IQR over six LLMs)',
          title_fontsize=5.2, ncol=3, handletextpad=0.2, columnspacing=0.8)

# ---------------- e: where the own era leaks ----------------
ax = fig.add_subplot(gs[1, 2])
label(ax, 'e', x=-0.33, y=1.03)
groups = [('human\nwriters', h, TEAL)] + [(AI_NAME[m].replace(' ', '\n', 1), a[a.model == m], AI_COL[m]) for m in
                                          ['sonnet', 'gpt-sol62', 'gemini4-neon', 'deepseek-v41-flash', 'qwen3-32b',
                                           'qwen25-32b']]
x = np.arange(len(groups))
for i, (n, d, c) in enumerate(groups):
    for j, (col, hatch, off) in enumerate([('res_func', None, -0.19), ('res_cont', None, 0.19)]):
        v = d[col].dropna().values
        m = np.median(v)
        bs = [np.median(rng.choice(v, len(v))) for _ in range(500)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.bar(i + off, m, width=0.36, color=c, alpha=0.45 if j == 0 else 1.0, lw=0)
        ax.plot([i + off, i + off], [lo, hi], color=INK, lw=0.5)
ax.axhline(0, color=INK, lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([g[0].replace(chr(10), ' ') for g in groups], fontsize=5.0, rotation=40, ha='right', rotation_mode='anchor')
ax.tick_params(axis='x', length=0, pad=1.5)
ax.set_ylabel("per-character evidence for the writer's own era\n" + r"(nats, $\log p(x\mid t_\mathrm{own}) - \log p(x\mid t_\mathrm{exemplar})$)",
              fontsize=5.6)
from matplotlib.patches import Patch
ax.legend([Patch(fc=GREYS[0], alpha=0.45), Patch(fc=GREYS[0])], ['function words', 'content characters'],
          loc='upper right', fontsize=5.2, borderaxespad=0.1)
ax.set_ylim(-0.05, 0.21)
ax.set_xlim(-0.6, len(groups) - 0.4)

save(fig, 'fig2_imitation')
print('ok')
