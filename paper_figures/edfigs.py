"""Extended Data figures."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from style import *

setup()
rng = np.random.default_rng(11)
R = lambda *p: os.path.join(PKG, *p)


def boot_med(v, groups=None, n=1000):
    v = np.asarray(v, float)
    if groups is None:
        bs = [np.median(rng.choice(v, len(v))) for _ in range(n)]
    else:
        g = np.asarray(groups); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}
        bs = [np.median(v[np.concatenate([idx[k] for k in rng.choice(u, len(u))])]) for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


# =============== ED1: accuracy, calibration, length ===============
oof = pd.read_csv(R('runs', 'l3_oof.tsv'), sep='\t')
aud = pd.read_csv(R('runs', 'l3_audit.tsv'), sep='\t')
oof = oof.merge(aud[['id', 'n_pass']], on='id')
oof['mid'] = (oof.lo + oof.hi) / 2
oof['err'] = (oof['median'] - oof.mid).abs()
oof['w80'] = oof.q90 - oof.q10
bins = [-2000, -221, 220, 589, 907, 1279, 1368, 1644, 3000]
per = ['Pre-Qin', 'Qin–Han', 'Six Dyn.', 'Sui–Tang', 'Song', 'Yuan', 'Ming', 'Qing']
oof['per'] = pd.cut(oof.mid, bins, labels=per)

fig, axs = plt.subplots(1, 4, figsize=(W2, 52 * MM), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1, 1.3, 0.9]))
ax = axs[0]; label(ax, 'a', x=-0.42)
cov = {'Ruoqu-L': [0.651, 0.865, 0.942], 'discriminative (Aeneas-style)': [0.644, 0.858, 0.919], 'character LR': [0.659, 0.881, 0.945],
       'period n-gram': [0.609, 0.856, 0.937], 'function words': [0.648, 0.893, 0.955]}
cc = {'Ruoqu-L': BLUE, 'discriminative (Aeneas-style)': WINE, 'character LR': GREYS[0], 'period n-gram': GREYS[1],
      'function words': GREYS[2]}
ax.plot([0.4, 1], [0.4, 1], color=GREYS[2], lw=0.6, ls=(0, (3, 2)))
for k, v in cov.items():
    ax.plot([0.5, 0.8, 0.9], v, marker='o', ms=2.6, lw=0.8, color=cc[k], label=k)
ax.set_xlim(0.45, 0.95); ax.set_ylim(0.45, 1.0)
ax.set_xlabel('nominal coverage'); ax.set_ylabel('empirical coverage')
ax.legend(fontsize=4.8, loc='upper left', handlelength=1.2)
ax.set_title('calibration (fold 0)', fontsize=5.8, loc='left')

ax = axs[1]; label(ax, 'b', x=-0.42)
lb = pd.cut(oof.n_pass, [0, 1, 2, 4, 8, 16, 24])
t = oof.groupby(lb, observed=True).err
med = t.median(); n = t.size()
ci = np.array([boot_med(g.values) for _, g in t])
x = np.arange(len(med))
ax.bar(x, med.values, color=BLUE, width=0.65, lw=0)
ax.vlines(x, ci[:, 0], ci[:, 1], color=INK, lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(['1', '2', '3–4', '5–8', '9–16', '17–24'], rotation=45, ha='right', rotation_mode='anchor')
ax.set_xlabel('256-character passages per work')
ax.set_ylabel('median absolute error (years)')
for xx, nn in zip(x, n.values):
    ax.text(xx, 6, f'{nn}', ha='center', va='bottom', fontsize=4.6, color='white', rotation=90)
ax.set_title('error falls with length', fontsize=5.8, loc='left')

ax = axs[2]; label(ax, 'c', x=-0.30)
t = oof.groupby('per', observed=True)
med = t.err.median().reindex(per); w = t.w80.median().reindex(per)
cv = t.apply(lambda g: ((g.mid >= g.q10 - 12.5) & (g.mid <= g.q90 + 12.5)).mean()).reindex(per)
nn = t.size().reindex(per)
x = np.arange(len(per))
ax.bar(x - 0.2, med.values, width=0.38, color=BLUE, lw=0, label='median error')
ax.bar(x + 0.2, w.values, width=0.38, color=BLUE_L, lw=0, label='80% interval width')
ax2 = ax.twinx()
ax2.plot(x, cv.values, color=WINE, marker='o', ms=2.6, lw=0.8)
ax2.axhline(0.8, color=WINE, lw=0.5, ls=(0, (3, 2)))
ax2.set_ylim(0.5, 1.0); ax2.set_ylabel('80% interval coverage', color=WINE, fontsize=5.6)
ax2.tick_params(axis='y', colors=WINE, labelsize=5)
ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(WINE)
ax.set_xticks(x); ax.set_xticklabels(per, rotation=45, ha='right', rotation_mode='anchor', fontsize=5)
ax.set_ylabel('years')
ax.legend(fontsize=4.8, loc='upper right', handlelength=0.9)
ax.set_title('all 6,390 works', fontsize=5.8, loc='left')

ax = axs[3]; label(ax, 'd', x=-0.36)
names = ['M\n(34 M)', 'L\n(98 M)', 'XL\n(219 M)']
medv = [87.0, 72.5, 72.5]
meanv = [174.1, 140.8, 124.0]
x = np.arange(3)
ax.bar(x, medv, color=[BLUE_L, BLUE, '#0b2545'], width=0.6, lw=0, label='median')
ax.plot(x, meanv, 'D', color=WINE, ms=3.2, label='mean')
for xx, v in zip(x, medv):
    ax.text(xx, v + 3, f'{v:g}', ha='center', fontsize=5)
for xx, v in zip(x, meanv):
    ax.text(xx + 0.12, v, f'{v:g}', ha='left', va='center', fontsize=5, color=WINE)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=5.2)
ax.set_ylabel('absolute error (years)')
ax.set_ylim(0, 200)
ax.legend(fontsize=4.8, loc='upper left', handlelength=0.9)
ax.set_title('model size (fold 0)', fontsize=5.8, loc='left')
save(fig, 'ed1_accuracy')

# =============== ED2: imitation robustness ===============
h = pd.read_csv(R('runs', 'l3_imit_robust.tsv'), sep='\t')
fig, axs = plt.subplots(1, 3, figsize=(W2, 52 * MM), gridspec_kw=dict(wspace=0.45, width_ratios=[1.1, 1, 1]))
ax = axs[0]; label(ax, 'a', x=-0.30)
lbs = ['20-60', '60-150', '150+']
for i, l in enumerate(lbs):
    v = h[h.lb == l].pull.values
    v = v[np.isfinite(v)]
    parts = ax.violinplot(np.clip(v, -2, 2), positions=[i], widths=0.7, showextrema=False)
    for b in parts['bodies']:
        b.set_facecolor(TEAL_L); b.set_alpha(0.8); b.set_edgecolor('none')
    ax.plot(i, np.median(v), 'o', color=TEAL, ms=3.2)
    ax.text(i, 1.85, f'n = {len(v)}', ha='center', fontsize=4.8, color=GREYS[1])
ax.axhline(0, color=INK, lw=0.5); ax.axhline(1, color=GREYS[2], lw=0.5, ls=(0, (3, 2)))
ax.set_xticks(range(3)); ax.set_xticklabels(['20–60', '60–150', '150+'])
ax.set_xlabel('length of imitation (characters)')
ax.set_ylabel('fraction of the distance covered\n(clipped to ±2)')
ax.set_ylim(-2.1, 2.05)
ax.set_title('individual imitations', fontsize=5.8, loc='left')

ax = axs[1]; label(ax, 'b', x=-0.32)
h['era'] = pd.cut(h.year, [0, 907, 1279, 1368, 1644, 1912], labels=['Tang\nor earlier', 'Song', 'Yuan', 'Ming', 'Qing'])
eras = ['Tang\nor earlier', 'Song', 'Yuan', 'Ming', 'Qing']
for i, e in enumerate(eras):
    s = h[h.era == e]
    m = s.pull.median(); lo, hi = boot_med(s.pull.values, s.work.values)
    ax.plot([i, i], [lo, hi], color=TEAL, lw=1.0)
    ax.plot(i, m, 'o', color=TEAL, ms=3.4, mec='white', mew=0.3)
    ax.text(i, -0.22, f'{len(s)}', ha='center', fontsize=4.8, color=GREYS[1])
ax.axhline(0, color=INK, lw=0.5)
ax.set_xticks(range(5)); ax.set_xticklabels(eras, fontsize=5.2)
ax.set_ylabel('fraction covered (median, 95% CI)')
ax.set_ylim(-0.25, 0.35)
ax.set_title("by the imitator's period", fontsize=5.8, loc='left')

ax = axs[2]; label(ax, 'c', x=-0.32)
for i, g in enumerate(['verse', 'prose']):
    s = h[h.gclass == g]
    m = s.pull.median(); lo, hi = boot_med(s.pull.values, s.work.values)
    ax.plot([i, i], [lo, hi], color=TEAL, lw=1.0)
    ax.plot(i, m, 'o', color=TEAL, ms=3.4, mec='white', mew=0.3)
    ax.text(i, -0.07, f'{len(s)}', ha='center', fontsize=4.8, color=GREYS[1])
ax.axhline(0, color=INK, lw=0.5)
ax.set_xticks(range(2)); ax.set_xticklabels(['verse', 'prose'])
ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.08, 0.2)
ax.set_ylabel('fraction covered (median, 95% CI)')
ax.set_title('by form', fontsize=5.8, loc='left')
save(fig, 'ed2_imitation')

# =============== ED3: imitator model sensitivity ===============
lam = [1.0, 0.99, 0.98, 0.95, 0.90]
auc = [0.973, 1.000, 1.000, 0.946, 0.920]
a_ok = [11, 11, 12, 12, 12]
b_ok = [8, 8, 8, 6, 6]
fig, axs = plt.subplots(1, 2, figsize=(W1 * 1.5, 45 * MM), gridspec_kw=dict(wspace=0.55))
ax = axs[0]; label(ax, 'a', x=-0.35)
ax.plot(lam, auc, 'o-', color=BLUE, ms=3)
ax.set_xlabel(r'probability that a character imitates, $\lambda$')
ax.set_ylabel('AUC, forgeries vs vindicated texts')
ax.invert_xaxis(); ax.set_ylim(0.85, 1.01)
ax.axhline(0.967, color=GREYS[2], lw=0.5, ls=(0, (3, 2)))
ax.text(0.905, 0.962, 'plain dating', fontsize=4.8, color=GREYS[1], va='top')
ax = axs[1]; label(ax, 'b', x=-0.35)
ax.plot(lam, np.array(a_ok) / 14, 'o-', color=WINE, ms=3, label='forgeries read later (of 14)')
ax.plot(lam, np.array(b_ok) / 8, 's-', color=BLUE, ms=3, label='vindicated texts read early (of 8)')
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('share correct')
ax.invert_xaxis(); ax.set_ylim(0.6, 1.05)
ax.legend(fontsize=4.8, loc='lower left')
save(fig, 'ed3_imitator')

# =============== ED4: audit overview ===============
fig, axs = plt.subplots(1, 3, figsize=(W2, 58 * MM), gridspec_kw=dict(wspace=0.45, width_ratios=[1.25, 1, 1]))
ax = axs[0]; label(ax, 'a', x=-0.25)
aud['mid'] = (aud.lo + aud.hi) / 2
n = aud[~aud.flag_late & ~aud.flag_early]
ax.scatter(n.mid.clip(-650), n['median'], s=1.2, color=GREYS[3], lw=0, rasterized=True)
fl = aud[aud.flag_late]; fe = aud[aud.flag_early]
ax.scatter(fe.mid.clip(-650), fe['median'], s=3.5, color=BLUE, lw=0, label=f'earlier than catalogued ({len(fe)})')
ax.scatter(fl.mid.clip(-650), fl['median'], s=3.5, color=WINE, lw=0, label=f'later than catalogued ({len(fl)})')
ax.plot([-650, 1950], [-650, 1950], color=GREYS[1], lw=0.5, ls=(0, (3, 2)))
ax.set_xlim(-650, 1950); ax.set_ylim(-650, 1950); ax.set_aspect('equal')
year_axis(ax, 'x', ticks=(-500, 0, 500, 1000, 1500)); year_axis(ax, 'y', ticks=(-500, 0, 500, 1000, 1500))
ax.set_xlabel('catalogue date (midpoint)'); ax.set_ylabel('model date')
ax.legend(fontsize=4.8, loc='upper left', markerscale=1.5)
ax = axs[1]; label(ax, 'b', x=-0.36)
cats = {'KR1': 'Classics', 'KR2': 'Histories', 'KR3': 'Masters', 'KR4': 'Collections', 'KR5': 'Daoist',
        'KR6': 'Buddhist'}
aud['part'] = aud.cat.str[:3]
t = aud.groupby('part').agg(late=('flag_late', 'mean'), early=('flag_early', 'mean'), n=('id', 'size')).reindex(cats)
y = np.arange(len(t))[::-1]
ax.barh(y + 0.2, 100 * t.late, height=0.38, color=WINE, lw=0, label='later')
ax.barh(y - 0.2, 100 * t.early, height=0.38, color=BLUE, lw=0, label='earlier')
ax.set_yticks(y); ax.set_yticklabels([f'{cats[k]} ({t.n[k]})' for k in t.index], fontsize=5.2)
ax.set_xlabel('works flagged (%)')
ax.legend(fontsize=4.8, loc='lower right')
ax.tick_params(axis='y', length=0)
ax = axs[2]; label(ax, 'c', x=-0.36)
ev = pd.read_csv(R('runs', 'l3_audit_evidence_late.tsv'), sep='\t')
ax.hist(ev.func_share, bins=np.linspace(0, 0.45, 19), color=BLUE, lw=0)
ax.axvline(ev.func_share.median(), color=WINE, lw=0.8)
ax.text(ev.func_share.median() + 0.01, ax.get_ylim()[1] * 0.92, f'median {ev.func_share.median():.2f}', fontsize=5,
        color=WINE)
ax.set_xlabel('share of the late evidence carried\nby function words')
ax.set_ylabel('flagged works')
save(fig, 'ed4_audit')

# =============== ED5: registers ===============
reg = pd.read_csv(os.path.join(PKG, 'runs', 'lexical_timeline', 'fig_registers_data.tsv'), sep='\t')
yrs = [c for c in reg.columns if c.lstrip('-').isdigit()]
Y = np.array([int(c) for c in yrs])
RN = {'別集': 'literary collections', '史部': 'histories', '子部': 'masters', '佛典譯經': 'Buddhist translations',
      '佛教撰述': 'Buddhist native works', '禪宗語錄': 'Chan records', '小說筆記': 'fiction, notes', '詞曲': 'song lyrics'}
RC = {'別集': TEAL, '史部': BLUE, '子部': BLUE_L, '佛典譯經': WINE, '佛教撰述': '#d58b97', '禪宗語錄': PURPLE,
      '小說筆記': GREYS[1], '詞曲': '#3f8f7f'}
chars = ['之', '也', '矣', '其', '曰', '我', '是', '底', '了']
fig, axs = plt.subplots(2, 5, figsize=(W2, 72 * MM), gridspec_kw=dict(wspace=0.38, hspace=0.45))
for k, ch in enumerate(chars):
    ax = axs.flat[k]
    d = reg[reg['char'] == ch]
    for r in d.itertuples():
        row = d.loc[r.Index, yrs].values.astype(float)
        ax.plot(Y, row, color=RC.get(r.register, GREYS[2]), lw=0.7)
    ax.axhline(0, color=GREYS[2], lw=0.4)
    ax.set_title(ch, fontsize=7, pad=2)
    ax.set_xticks([-500, 500, 1500]); ax.set_xticklabels(['500 BCE', '500', '1500'], fontsize=4.8)
    ax.tick_params(axis='y', labelsize=4.8)
axs.flat[-1].axis('off')
axs.flat[-1].legend([Line2D([], [], color=c, lw=1) for c in RC.values()], [RN[k] for k in RC], fontsize=5,
                    loc='center')
axs[0, 0].set_ylabel('relative log-probability', fontsize=5.4)
axs[1, 0].set_ylabel('relative log-probability', fontsize=5.4)
save(fig, 'ed5_registers')

# =============== ED6: undated validation ===============
val = pd.DataFrame([('CBETA date range', 402, 83, 0.76, 0.88), ('date in heading', 155, 38, 0.90, 0.80),
                    ("author's life (DILA)", 228, 107, 0.90, 0.79), ('Daoist database', 54, 169, 0.87, 0.66),
                    ('catalogue-assigned period', 127, 100, 0.87, 0.44)],
                   columns=['src', 'n', 'err', 'cov', 'rho'])
fig, axs = plt.subplots(1, 3, figsize=(W2, 45 * MM), gridspec_kw=dict(wspace=0.9))
for k, (col, lab, c) in enumerate([('err', 'median difference (years)', BLUE), ('cov', '80% interval overlaps\nexternal range', TEAL),
                                   ('rho', 'rank correlation', PURPLE)]):
    ax = axs[k]; label(ax, 'abc'[k], x=-0.95 if k == 0 else -0.25)
    y = np.arange(len(val))[::-1]
    ax.barh(y, val[col], color=[c] * 4 + [WINE], height=0.6, lw=0)
    for yy, v in zip(y, val[col]):
        ax.text(v, yy, f' {v:g}', va='center', fontsize=5)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{s} ({n})' for s, n in zip(val.src, val.n)] if k == 0 else [], fontsize=5.2)
    ax.set_xlabel(lab, fontsize=5.6)
    ax.tick_params(axis='y', length=0)
    if col != 'err':
        ax.set_xlim(0, 1.1)
save(fig, 'ed6_undated')
print('ok')
