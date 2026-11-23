"""Figure 4: auditing and completing the catalogue."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from style import *

setup()
rng = np.random.default_rng(2)
R = lambda *p: os.path.join(PKG, *p)
cons = pd.read_csv(R('data', 'consensus_dates_v0.tsv'), sep='\t')
cons['cat_mid'] = (cons.cat_lo + cons.cat_hi) / 2
cbc = pd.read_csv(R('runs', 'cbc_attr.tsv'), sep='\t')
cbc['mid'] = (cbc.lo + cbc.hi) / 2
siku = pd.read_csv(R('runs', 'siku_attr.tsv'), sep='\t')
ub = pd.read_csv(R('runs', 'undated_byline_eval.tsv'), sep='\t')
nd = pd.read_csv(R('data', 'new_dates_v0.tsv'), sep='\t')

fig = plt.figure(figsize=(W2, 112 * MM))
gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 0.8, 0.85, 0.85], hspace=0.48, wspace=0.62,
                      left=0.06, right=0.985, top=0.955, bottom=0.085)

# ---------------- a: consensus ----------------
ax = fig.add_subplot(gs[0, 0])
label(ax, 'a', x=-0.25, y=1.03)
lim = (-700, 1950)
ax.plot(lim, lim, color=GREYS[2], lw=0.6, ls=(0, (3, 2)), zorder=1)
cm = cons.cat_mid.clip(-650, None)
for r, c_ in zip(cons.itertuples(), cm):
    ax.plot([r.sch_mid, r.sch_mid], [c_, r.model_median], color=GREYS[3], lw=0.45, zorder=2)
ax.scatter(cons.sch_mid, cm, s=9, facecolor='white', edgecolor=GREYS[0], lw=0.6, zorder=3,
           label='catalogue / traditional date')
ax.scatter(cons.sch_mid, cons.model_median, s=9, color=BLUE, edgecolor='white', lw=0.3, zorder=4,
           label='Ruoqu (never saw the work)')
ax.set_xlim(*lim); ax.set_ylim(*lim)
year_axis(ax, 'x', ticks=(-500, 0, 500, 1000, 1500)); year_axis(ax, 'y', ticks=(-500, 0, 500, 1000, 1500))
ax.set_xlabel('date established by scholarship')
ax.set_ylabel('date')
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=5.0, handletextpad=0.2, borderaxespad=0.1)

# ---------------- b: precision by rank ----------------
ax = fig.add_subplot(gs[0, 1])
label(ax, 'b', x=-0.42, y=1.03)
ranks = ['1–30', '31–60', '61–97']
prec = [22 / 24, 19 / 22, 24 / 30]
nn = ['22/24', '19/22', '24/30']
x = np.arange(3)
ax.bar(x, prec, color=[BLUE, '#4a7fb5', BLUE_L], width=0.62, lw=0)
for xx, p, s in zip(x, prec, nn):
    ax.text(xx, p + 0.015, f'{100 * p:.0f}%', ha='center', va='bottom', fontsize=5.4)
    ax.text(xx, 0.04, s, ha='center', va='bottom', fontsize=4.8, color='white')
ax.set_xticks(x); ax.set_xticklabels(ranks)
ax.set_xlabel('rank among 97 works flagged\nas later than catalogued')
ax.set_ylabel('share confirmed on checking')
ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%'])
ax.tick_params(axis='x', length=0)


# ---------------- c, d: systematic tests ----------------
def strip(ax, groups, ylab, title, note):
    for i, (n, v, c) in enumerate(groups):
        v = np.asarray(v)
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=2.2 if len(v) > 100 else 4, color=c, alpha=0.5, lw=0,
                   zorder=2)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.28, i + 0.28], [q2, q2], color=INK, lw=1.1, zorder=4)
        ax.plot([i, i], [q1, q3], color=INK, lw=0.6, zorder=4)
        ax.text(i + 0.3, q2, f'{q2:+.0f}', fontsize=5.0, va='center', ha='left', zorder=5)
    ax.axhline(0, color=GREYS[2], lw=0.5)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=5.1)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(ylab, fontsize=5.8)
    ax.set_title(title, fontsize=6.0, loc='left', pad=3)
    ax.text(0.5, 0.99, note, transform=ax.transAxes, fontsize=4.8, ha='center', va='top', linespacing=1.2)


ax = fig.add_subplot(gs[0, 2])
label(ax, 'c', x=-0.42, y=1.03)
w = cbc[(cbc.genre == 5) & cbc.mid.between(400, 589)]
strip(ax, [('Fei Changfang\nre-attributions', w[w.fei].offset, WINE), ('other\nattributions', w[~w.fei].offset, BLUE)],
      'model date − catalogue date (years)', 'Buddhist translations of 400–589',
      'AUC 0.74 (p = 9×10$^{-5}$)\nflagged 22% vs 7%')
ax.set_ylim(-600, 700)

ax = fig.add_subplot(gs[0, 3])
label(ax, 'd', x=-0.42, y=1.03)
strip(ax, [('"old edition\nattributed to"', siku[siku.jiu].offset, WINE),
           ('all other\nworks', siku[~siku.jiu].offset, BLUE)],
      'model date − catalogue date (years)', 'Siku editors (1773–1782)', 'AUC 0.65 (p = 4×10$^{-5}$)\nflagged 19% vs 1%')
ax.set_ylim(-1200, 1500)

# ---------------- e: undated texts, validation ----------------
ax = fig.add_subplot(gs[1, 0])
label(ax, 'e', x=-0.25, y=1.03)
ax.plot([-100, 1950], [-100, 1950], color=GREYS[2], lw=0.6, ls=(0, (3, 2)), zorder=1)
st = ub[ub.kind == 'stated'].dropna(subset=['b_mid'])
at = ub[ub.kind == 'attached'].dropna(subset=['b_mid'])
ax.scatter(at.b_mid + rng.uniform(-8, 8, len(at)), at['median'], s=6, color=WINE, alpha=0.6, lw=0, zorder=3,
           label=f'catalogue-assigned period (n = {len(at)})')
ax.scatter(st.b_mid + rng.uniform(-8, 8, len(st)), st['median'], s=6, color=BLUE, alpha=0.75, lw=0, zorder=4,
           label=f'named translator or author (n = {len(st)})')
ax.set_xlim(-100, 1950); ax.set_ylim(-100, 1950)
year_axis(ax, 'x', ticks=(0, 500, 1000, 1500)); year_axis(ax, 'y', ticks=(0, 500, 1000, 1500))
ax.set_xlabel('date in the text’s own heading (midpoint)')
ax.set_ylabel('model date')
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=4.9, handletextpad=0.1, borderaxespad=0.1, markerscale=1.4)
ax.text(0.97, 0.03, 'median difference\n38 yr for named persons\n100 yr for assigned periods', transform=ax.transAxes, fontsize=4.9,
        ha='right', va='bottom', linespacing=1.25)

# ---------------- f: Later Han catalogue ----------------
ax = fig.add_subplot(gs[1, 1])
label(ax, 'f', x=-0.42, y=1.03)
lh = at[(at.b_lo == 25) & (at.b_hi == 220)]
v = lh['median'].values
ax.axhspan(25, 220, color=WINE, alpha=0.12, lw=0)
ax.text(0.03, 60, 'Eastern Han\n(assigned)', fontsize=4.9, color=WINE, ha='left', va='center', linespacing=1.0,
        transform=ax.get_yaxis_transform())
ax.scatter(rng.uniform(-0.2, 0.2, len(v)), v, s=10, color=WINE, edgecolor='white', lw=0.3, zorder=3)
ax.plot([-0.32, 0.32], [np.median(v)] * 2, color=INK, lw=1.0)
ax.text(0.34, np.median(v), f'{np.median(v):.0f}', fontsize=5, va='center')
ax.set_xlim(-0.6, 0.6); ax.set_xticks([])
ax.set_ylim(-100, 900)
year_axis(ax, 'y', ticks=(0, 200, 400, 600, 800))
ax.set_ylabel('model date')
ax.set_title(f'{len(v)} anonymous sūtras\nfiled under the Later Han', fontsize=5.8, loc='left', pad=3, linespacing=1.1)
ax.spines['bottom'].set_visible(False)

# ---------------- g: released dates ----------------
ax = fig.add_subplot(gs[1, 2:])
label(ax, 'g', x=-0.13, y=1.03)
edges = np.arange(-450, 1951, 50)
ctr = (edges[:-1] + edges[1:]) / 2
tiers = [('A', BLUE, 'A'), ('B', '#4a7fb5', 'B'), ('C', BLUE_L, 'C')]
bottom = np.zeros(len(ctr))
for t, c, n in tiers:
    hst = np.histogram(nd[nd.tier == t]['median'], edges)[0]
    ax.bar(ctr, hst, width=46, bottom=bottom, color=c, lw=0, label=f'{n} ({(nd.tier == t).sum()})')
    bottom += hst
year_axis(ax, 'x', ticks=(0, 500, 1000, 1500))
ax.set_xlim(-450, 1950)
ax.set_xlabel('model date')
ax.set_ylabel('texts per 50 years')
ax.set_title('635 texts that no standard catalogue dates', fontsize=6.0, loc='left', pad=3)
ax.legend(loc='upper left', fontsize=4.9, handlelength=0.9, borderaxespad=0.1, title='reliability tier',
          title_fontsize=5.0)
cnt = nd.cat.str[:3].value_counts()
ax.text(0.98, 0.97, f'Buddhist {cnt.get("KR6", 0)}\nDaoist {cnt.get("KR5", 0)}\nother {len(nd) - cnt.get("KR6", 0) - cnt.get("KR5", 0)}',
        transform=ax.transAxes, fontsize=4.9, va='top', ha='right', linespacing=1.2, color=GREYS[0])

save(fig, 'fig4_audit')
print('ok', len(v), (v > 220).mean())
