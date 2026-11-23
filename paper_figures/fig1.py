"""Figure 1: study design, the model, and out-of-sample accuracy (full page)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from style import *

setup()
oof = pd.read_csv(os.path.join(PKG, 'runs', 'l3_oof.tsv'), sep='\t')
oof['mid'] = (oof.lo + oof.hi) / 2
oof['err'] = (oof['median'] - oof.mid).abs()

FW, FH = W2, 228 * MM
ASP = FW / FH
fig = plt.figure(figsize=(FW, FH))
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis('off')
T = ov.transAxes
BG = '#f3f3f3'
LAV = '#eeecf6'
EDGE = '#cfcfcf'
HEAD = 8.6


def rbox(x, y, w, h, fc, ec='none', lw=0.6, r=0.012, ls='-', z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls,
                       transform=T, mutation_aspect=ASP, zorder=z)
    ov.add_patch(p)
    return p


def tx(x, y, s, **k):
    k.setdefault('fontsize', 6.2)
    k.setdefault('va', 'center')
    return ov.text(x, y, s, transform=T, **k)


def arr(x0, y0, x1, y1, c=GREYS[1], lw=0.9, ls='-', ms=7, cs='arc3,rad=0'):
    ov.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 transform=T, zorder=6, connectionstyle=cs))


def ax_at(x, y, w, h):
    return fig.add_axes([x, y, w, h], zorder=5)


# =============================== panel a ===============================
A0, A1 = 0.728, 0.985
SB = '#f3f3f3'
tx(0.006, A1 - 0.002, 'a', fontsize=11, fontweight='bold')
for x0, x1 in [(0.026, 0.318), (0.330, 0.716), (0.728, 0.992)]:
    rbox(x0, A0 - 0.006, x1 - x0, A1 - A0 + 0.016, SB, r=0.010)

# left: data and analyses
tx(0.037, A1 - 0.004, 'Data and analyses', fontsize=7.4, fontweight='bold')
data = [('6,390 dated works', '571 million characters'), ('1,169 undated works', 'none used in training'),
        ('34 blind-test works', ''), ('1,047 human imitations', 'from 225 collections'),
        ('2,578 LLM imitations', 'from six models')]
ana = ['accuracy', 'audit of the canon', 'language history', 'new dates', 'authentication', 'limits of imitation']
links = [(0, 0), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
dys = [A1 - 0.044 - i * 0.0415 for i in range(len(data))]
ays = [dys[0] + 0.013, dys[0], dys[0] - 0.013, dys[1], dys[2], (dys[3] + dys[4]) / 2]
DX0, DX1, AX0 = 0.037, 0.180, 0.207
for (a_, b_) in links:
    ov.plot([DX1 + 0.001, AX0 - 0.004], [dys[a_], ays[b_]], color=GREYS[2], lw=0.6, transform=T, zorder=3)
for y_, (n, d) in zip(dys, data):
    rbox(DX0, y_ - 0.0150, DX1 - DX0, 0.030, 'white', ec='#dadada', lw=0.4, r=0.004, z=3)
    tx(DX0 + 0.006, y_ + (0.0055 if d else 0.0), n, fontsize=6.1, fontweight='bold', zorder=4)
    if d:
        tx(DX0 + 0.006, y_ - 0.0070, d, fontsize=5.3, color=GREYS[1], zorder=4)
for y_, n in zip(ays, ana):
    ov.plot(AX0, y_, 'o', ms=2.6, color=BLUE, transform=T, zorder=4)
    tx(AX0 + 0.007, y_, n, fontsize=6.0, color=BLUE, zorder=4)

# middle: three dates for one work
MX = 0.342
tx(MX, A1 - 0.004, 'Three dates for the Lingqi jing 《靈棋經》', fontsize=7.4, fontweight='bold')
ax = ax_at(MX + 0.004, A0 + 0.094, 0.355, 0.118)
ax.set_facecolor('none')
ax.set_xlim(-650, 1950); ax.set_ylim(0.55, 3.75)
for s in ['left', 'right', 'top']:
    ax.spines[s].set_visible(False)
ax.set_yticks([])
year_axis(ax, 'x', ticks=(-500, 0, 500, 1000, 1500))
ax.tick_params(axis='x', labelsize=5.6)
ax.set_xlabel('year', fontsize=5.8, labelpad=1)
rows = [(3.0, 'traditional attribution', WINE), (2.0, 'scholarship', TEAL), (1.0, 'Ruoqu (out of sample)', BLUE)]
for yv, lab, c in rows:
    ax.axhline(yv, color='#dcdcdc', lw=0.5, zorder=0)
    ax.text(-640, yv + 0.28, lab, fontsize=5.8, color=c, va='bottom', ha='left')
ax.plot([-134, -93], [3, 3], color=WINE, lw=4.5, solid_capstyle='butt')
ax.text(-40, 3, 'Dongfang Shuo (Western Han)', fontsize=5.6, color=WINE, va='center')
ax.plot([317, 420], [2, 2], color=TEAL, lw=4.5, solid_capstyle='butt')
ax.text(300, 2, 'Eastern Jin core ', fontsize=5.6, color=TEAL, va='center', ha='right')
ax.plot(917, 2, 'o', color=TEAL, ms=3.8, mfc='white', mew=0.9)
ax.text(940, 2, ' postface dated by events of 917', fontsize=5.6, color=TEAL, va='center')
ax.plot([950, 1250], [1, 1], color=BLUE_L, lw=4.5, solid_capstyle='butt')
ax.plot(975, 1, 'o', color=BLUE, ms=4.2)
ax.text(1270, 1, ' 975 (80% interval\n 950–1250)', fontsize=5.6, color=BLUE, va='center', linespacing=1.0)
tx(MX + 0.004, A0 + 0.050, 'characters that most favour 975 over 100 BCE', fontsize=5.8, color=GREYS[0])
line = '臨從來未遂平生志今日逢君始稱心'
wts = {5: .55, 6: .7, 7: .45, 8: .9, 9: 1.0, 10: .75}
cm_ev = LinearSegmentedColormap.from_list('ev', ['#ffffff', WINE])
rng = np.random.default_rng(3)
cw = 0.0178
for j, s in enumerate(line):
    w = wts.get(j, 0.05 + 0.08 * rng.random())
    rbox(MX + 0.004 + j * cw, A0 + 0.016, cw * 0.9, 0.021, cm_ev(w), ec='#dcdcdc', lw=0.3, r=0.002, z=3)
    tx(MX + 0.004 + j * cw + cw * 0.45, A0 + 0.0265, s, fontsize=7.0, ha='center',
       color='white' if w > 0.6 else INK, zorder=4)

# right: authentication tests
RX = 0.739
tx(RX, A1 - 0.004, 'Independent tests of authentication', fontsize=7.4, fontweight='bold')
ax = ax_at(0.868, A0 + 0.048, 0.110, 0.165)
ax.set_facecolor('none')
tests = [('forgeries vs texts vindicated\nby excavated manuscripts', 0.983),
         ('forged vs authentic\nShangshu chapters', 0.815),
         ("Fei Changfang's batch\nattributions of 598", 0.737),
         ("Siku editors' 'old edition'\nformula (1773–1782)", 0.646)]
y = np.arange(len(tests))[::-1]
ax.axvline(0.5, color=GREYS[2], lw=0.6, ls=(0, (3, 2)))
ax.hlines(y, 0.5, [t[1] for t in tests], color=BLUE_L, lw=1.4)
ax.plot([t[1] for t in tests], y, 'o', color=BLUE, ms=4.4)
for yy, (n, v) in zip(y, tests):
    ax.text(v + 0.035, yy, f'{v:.2f}', fontsize=5.6, va='center', color=BLUE)
ax.set_yticks(y); ax.set_yticklabels([t[0] for t in tests], fontsize=5.6, linespacing=1.05)
ax.tick_params(axis='y', length=0, pad=2)
ax.tick_params(axis='x', labelsize=5.6)
ax.set_xlim(0.45, 1.1); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(['0.5', '0.75', '1'])
ax.set_xlabel('AUC (0.5 is chance)', fontsize=5.8, labelpad=1)
ax.set_ylim(-0.6, len(tests) - 0.4)
ax.spines['left'].set_visible(False)

# =============================== panel b ===============================
B0, B1 = 0.375, 0.705
tx(0.006, B1 - 0.004, 'b', fontsize=11, fontweight='bold')
tx(0.030, B1 - 0.004, 'A generative model of how each century wrote', fontsize=7.6, fontweight='bold')
ov.plot([0.745, 0.775], [B1 - 0.004] * 2, color=GREYS[1], lw=1.2, transform=T)
tx(0.780, B1 - 0.004, 'dating', fontsize=6.0, color=GREYS[0])
ov.plot([0.835, 0.865], [B1 - 0.004] * 2, color=PURPLE, lw=1.2, ls=(0, (3, 2)), transform=T)
tx(0.870, B1 - 0.004, 'training only', fontsize=6.0, color=GREYS[0])
BY0, BY1 = B0 + 0.018, B0 + 0.293
TOP = BY1 - 0.016

# sub-panel backgrounds
rbox(0.026, BY0, 0.252, BY1 - BY0, LAV, r=0.010)
rbox(0.290, BY0, 0.188, BY1 - BY0, SB, r=0.010)
rbox(0.490, BY0, 0.168, BY1 - BY0, SB, ec='#a9a9a9', lw=0.6, ls=(0, (3, 2)), r=0.010)
rbox(0.670, BY0, 0.322, BY1 - BY0, SB, r=0.010)

# --- training
tx(0.038, TOP, 'Training', fontsize=7.2, fontweight='bold')
tx(0.038, TOP - 0.020, 'next-character prediction on\n571 million dated characters', fontsize=5.9, va='top',
   color=GREYS[0], linespacing=1.2)
tx(0.038, B0 + 0.200, 'earlier quotations removed', fontsize=6.2, fontweight='bold', color=PURPLE)
qs = '子曰學而時習之不亦說乎有朋自遠方來'
for j, s in enumerate(qs):
    tx(0.040 + j * 0.0128, B0 + 0.178, s, fontsize=6.6, ha='left', color=GREYS[2] if 2 <= j <= 13 else INK)
ov.plot([0.040 + 2 * 0.0128, 0.040 + 14 * 0.0128 - 0.003], [B0 + 0.178] * 2, color=PURPLE, lw=0.9, transform=T,
        zorder=5)
tx(0.038, B0 + 0.138, 'five folds split by author', fontsize=6.2, fontweight='bold', color=PURPLE)
for k in range(5):
    yy = B0 + 0.110 - k * 0.0150
    held = (k == 1)
    for j in range(16):
        ov.add_patch(Rectangle((0.040 + j * 0.0105, yy), 0.0078, 0.0098, fc=WINE if held else PURPLE_L, ec='none',
                               transform=T, alpha=1 if held else 0.55 + 0.45 * rng.random(), zorder=3))
    tx(0.040 + 16 * 0.0105 + 0.004, yy + 0.005, 'held out' if held else f'fold {k + 1}', fontsize=5.3,
       color=WINE if held else GREYS[1])

# --- tokens and year embedding
CX = 0.302
tx(CX, TOP, 'Character tokens', fontsize=7.2, fontweight='bold')
toks = [('t', BLUE, 'white'), ('g', SLATE, 'white')] + [(c, 'white', INK) for c in '見故園春色好']
tw, tg = 0.0158, 0.0185
for j, (s, fc, c) in enumerate(toks):
    xx = CX + j * tg
    rbox(xx, B0 + 0.205, tw, 0.024, fc, ec=fc if j < 2 else GREYS[2], lw=0.5, r=0.003, z=3)
    tx(xx + tw / 2, B0 + 0.217, s, fontsize=6.8, ha='center', color=c, style='italic' if j < 2 else 'normal', zorder=4)
tx(CX, B0 + 0.198, 'year and register', fontsize=5.3, color=BLUE, va='top')
tx(CX + tg * 4.7, B0 + 0.198, 'characters', fontsize=5.3, color=GREYS[1], va='top')
arr(0.279, B0 + 0.217, CX - 0.003, B0 + 0.217, c=PURPLE, ls=(0, (3, 2)))
tx(CX, B0 + 0.150, 'Smooth year embedding', fontsize=6.4, fontweight='bold', color=BLUE)
# stacked periodic features of the year, read off at one year
EX0, EY0, EW, EH = CX + 0.002, B0 + 0.040, 0.115, 0.090
iax = ax_at(EX0, EY0, EW, EH)
iax.set_facecolor('none')
periods = [5000, 2000, 900, 450]
shades = ['#1b4f8a', '#3f72a8', '#6f97c4', '#9fbbdc']
ty = 975
yr_ = np.linspace(-600, 1900, 800)
gap = 1.25
for k, (P, c) in enumerate(zip(periods, shades)):
    base = (len(periods) - 1 - k) * gap
    iax.plot(yr_, base + 0.45 * np.sin(2 * np.pi * (yr_ + 600) / P), color=c, lw=0.8)
    iax.plot(ty, base + 0.45 * np.sin(2 * np.pi * (ty + 600) / P), 'o', ms=2.4, color=c, mec='white', mew=0.4,
             zorder=5)
iax.axvline(ty, color=GREYS[1], lw=0.5, ls=(0, (2, 1.5)))
iax.set_xlim(-600, 1900); iax.set_ylim(-0.7, (len(periods) - 1) * gap + 0.7)
iax.axis('off')
tx(EX0 + EW * (ty + 600) / 2500, EY0 - 0.004, 'year t', fontsize=5.3, color=GREYS[1], ha='center', va='top')
cmap_e = LinearSegmentedColormap.from_list('e', ['#eef3f9', BLUE])
for k, (P, c) in enumerate(zip(periods, shades)):
    base = (len(periods) - 1 - k) * gap
    yfig = EY0 + EH * (base + 0.7) / ((len(periods) - 1) * gap + 1.4)
    val = 0.5 + 0.5 * np.sin(2 * np.pi * (ty + 600) / P)
    ov.add_patch(Rectangle((EX0 + EW + 0.012, yfig - 0.0055), 0.011, 0.011, fc=cmap_e(val), ec='white', lw=0.4,
                           transform=T, zorder=4))
    ov.plot([EX0 + EW * (ty + 600) / 2500 + 0.003, EX0 + EW + 0.010], [yfig] * 2, color=GREYS[2], lw=0.4,
            ls=(0, (1, 1)), transform=T, zorder=3)
tx(EX0 + EW + 0.0175, EY0 - 0.004, 'features', fontsize=5.3, color=GREYS[1], ha='center', va='top')

# --- transformer
X0, X1 = 0.490, 0.658
tx((X0 + X1) / 2, TOP, 'Transformer decoder', fontsize=7.2, fontweight='bold', ha='center')
bl = X0 + 0.009
for k, off in enumerate([0.008, 0.004, 0.0]):
    rbox(bl + off, B0 + 0.076 + off, 0.118, 0.170, 'white' if k == 2 else '#f7f8fb',
         ec=BLUE if k == 2 else '#b9c8e0', lw=0.9 if k == 2 else 0.5, r=0.008, z=2 + k)
bx = bl + 0.011
bw = 0.096
cxb = bx + bw / 2
rbox(bx, B0 + 0.196, bw, 0.030, '#e8eef7', ec=BLUE, lw=0.6, r=0.004, z=6)
tx(cxb, B0 + 0.211, 'Feed-forward', fontsize=6.4, ha='center', color=BLUE, zorder=7)
rbox(bx + 0.006, B0 + 0.174, bw - 0.012, 0.016, 'white', ec=GREYS[2], lw=0.5, r=0.003, z=6)
tx(cxb, B0 + 0.182, 'Layer norm', fontsize=5.6, ha='center', color=GREYS[0], zorder=7)
rbox(bx, B0 + 0.120, bw, 0.036, '#e8eef7', ec=BLUE, lw=0.6, r=0.004, z=6)
tx(cxb - 0.008, B0 + 0.138, 'Causal\nself-attention', fontsize=6.1, ha='center', color=BLUE, zorder=7,
   linespacing=1.0)
for i_ in range(4):
    for j_ in range(4):
        if j_ <= i_:
            ov.add_patch(Rectangle((bx + 0.077 + j_ * 0.0040, B0 + 0.147 - i_ * 0.0040), 0.0036, 0.0036,
                                   fc=BLUE if j_ == i_ else BLUE_L, ec='none', transform=T, zorder=8))
rbox(bx + 0.006, B0 + 0.097, bw - 0.012, 0.016, 'white', ec=GREYS[2], lw=0.5, r=0.003, z=6)
tx(cxb, B0 + 0.105, 'Layer norm', fontsize=5.6, ha='center', color=GREYS[0], zorder=7)
for yy in [B0 + 0.165, B0 + 0.236]:
    ov.add_patch(Circle((cxb, yy), 0.0045, fc='white', ec=INK, lw=0.5, transform=T, zorder=7))
    tx(cxb, yy, '+', fontsize=6, ha='center', zorder=8)
ov.plot([cxb] * 2, [B0 + 0.067, B0 + 0.248], color=INK, lw=0.4, transform=T, zorder=5)
bx2 = bl + 0.118 + 0.011
ov.plot([bx2 - 0.004, bx2, bx2, bx2 - 0.004], [B0 + 0.245, B0 + 0.245, B0 + 0.078, B0 + 0.078], color=INK,
        lw=0.6, transform=T, zorder=5)
tx(bx2 + 0.003, B0 + 0.162, '×12', fontsize=7.2, fontweight='bold')
rbox(bx, B0 + 0.045, bw, 0.022, '#e8eef7', ec=BLUE, lw=0.6, r=0.004, z=6)
tx(cxb, B0 + 0.056, 'Token embedding', fontsize=6.0, ha='center', color=BLUE, zorder=7)
tx((X0 + X1) / 2, B0 + 0.030, '98 million parameters', fontsize=5.3, ha='center', color=GREYS[1])
ov.plot([CX + 8 * tg - 0.002, 0.484, 0.484], [B0 + 0.217, B0 + 0.217, B0 + 0.056], color=GREYS[1], lw=0.9,
        transform=T, zorder=6)
arr(0.484, B0 + 0.056, bx - 0.002, B0 + 0.056, c=GREYS[1])

# --- outputs
OX = 0.682
tx(OX, TOP, 'Likelihood by year', fontsize=7.2, fontweight='bold')
g = np.linspace(-600, 1900, 300)
ll = -((g - 975) / 330) ** 2
iax = ax_at(OX + 0.004, B0 + 0.180, 0.112, 0.062)
iax.set_facecolor('none')
iax.plot(g, ll, color=BLUE, lw=0.9)
iax.set_xticks([]); iax.set_yticks([])
for s in ['top', 'right']:
    iax.spines[s].set_visible(False)
iax.set_xlabel('year', fontsize=5.2, labelpad=0.5)
arr(X1 + 0.001, B0 + 0.208, OX - 0.004, B0 + 0.208, c=GREYS[1])
arr(OX + 0.121, B0 + 0.210, OX + 0.147, B0 + 0.210, c=GREYS[1])
tx(OX + 0.134, B0 + 0.224, 'Bayes', fontsize=5.4, ha='center', color=GREYS[1])
tx(OX + 0.158, TOP, 'Posterior date', fontsize=7.2, fontweight='bold')
post = np.exp(-0.5 * ((g - 975) / 95) ** 2)
c_ = np.cumsum(post); c_ /= c_[-1]
q10, q90 = g[np.searchsorted(c_, .1)], g[np.searchsorted(c_, .9)]
iax = ax_at(OX + 0.160, B0 + 0.180, 0.135, 0.062)
iax.set_facecolor('none')
iax.fill_between(g, 0, post, where=(g >= q10) & (g <= q90), color=BLUE_L, lw=0)
iax.plot(g, post, color=BLUE, lw=0.9)
iax.set_xticks([]); iax.set_yticks([])
for s in ['top', 'right', 'left']:
    iax.spines[s].set_visible(False)
iax.set_xlabel('year', fontsize=5.2, labelpad=0.5)
rbox(OX, B0 + 0.092, 0.300, 0.050, 'white', ec='#dadada', lw=0.4, r=0.006, z=3)
tx(OX + 0.010, B0 + 0.128, 'Character-level evidence', fontsize=6.8, fontweight='bold', zorder=4)
tx(OX + 0.010, B0 + 0.106, r'$\log p(x_i \mid x_{<i}, t_1) - \log p(x_i \mid x_{<i}, t_0)$', fontsize=6.2, zorder=4)
rbox(OX, B0 + 0.030, 0.300, 0.050, 'white', ec='#dadada', lw=0.4, r=0.006, z=3)
tx(OX + 0.010, B0 + 0.066, 'Model of imitation', fontsize=6.8, fontweight='bold', zorder=4)
tx(OX + 0.010, B0 + 0.044, r"$\lambda\, p(x_i \mid x_{<i}, t') + (1-\lambda)\, p(x_i \mid x_{<i}, t)$", fontsize=6.2,
   zorder=4)
arr(X1 + 0.001, B0 + 0.117, OX - 0.002, B0 + 0.117, c=GREYS[1])
arr(X1 + 0.001, B0 + 0.055, OX - 0.002, B0 + 0.055, c=GREYS[1])

# =============================== panels c–f ===============================
gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.0, 1.15, 0.85], wspace=0.55, left=0.06, right=0.99,
                      bottom=0.045, top=0.315)

# c: corpus
ax = fig.add_subplot(gs[0, 0])
label(ax, 'c', x=-0.30, y=1.04)
gnames = ['Classics', 'Histories', 'Masters', 'Collections', 'Daoist', 'Buddhist transl.', 'Buddhist native']
gcols = ['#1b4f8a', '#4a7fb5', '#8fb3d9', '#1f7a72', '#5b4b8a', '#9e2a3a', '#d58b97']
edges = np.arange(-600, 1951, 50)
H = []
for gi in range(7):
    s = oof[oof.genre == gi]
    h = np.zeros(len(edges) - 1)
    for a, b in zip(s.lo.clip(-600, 1949), s.hi.clip(-600, 1949)):
        if b <= a:
            b = a + 1
        w = np.clip(np.minimum(edges[1:], b) - np.maximum(edges[:-1], a), 0, None)
        h += w / w.sum()
    H.append(h)
ctr = (edges[:-1] + edges[1:]) / 2
ax.stackplot(ctr, np.array(H), colors=gcols, lw=0)
ax.set_xlim(-600, 1900); ax.set_ylim(0, None)
year_axis(ax, ticks=(-500, 0, 500, 1000, 1500))
ax.set_ylabel('works per 50 years'); ax.set_xlabel('catalogue date')
hs = [Rectangle((0, 0), 1, 1, fc=c) for c in gcols]
ax.legend(hs, gnames, loc='upper left', fontsize=4.8, handlelength=0.8, handleheight=0.8, borderaxespad=0)

# d: OOF vs catalogue
ax = fig.add_subplot(gs[0, 1])
label(ax, 'd', x=-0.36, y=1.04)
cmh = LinearSegmentedColormap.from_list('b', ['#dfe9f5', '#8fb3d9', BLUE, '#0b2545'])
hb = ax.hexbin(oof.mid.clip(-600, 1900), oof['median'], gridsize=30, extent=(-600, 1900, -600, 1900), cmap=cmh,
               norm=LogNorm(1, 300), mincnt=1, lw=0.1, edgecolors='face')
ax.plot([-600, 1900], [-600, 1900], color=WINE, lw=0.6, ls=(0, (3, 2)))
ax.set_xlim(-600, 1900); ax.set_ylim(-600, 1900); ax.set_aspect('equal')
year_axis(ax, 'x', ticks=(-500, 0, 500, 1000, 1500)); year_axis(ax, 'y', ticks=(-500, 0, 500, 1000, 1500))
ax.set_xlabel('catalogue date'); ax.set_ylabel('model date (out of sample)')
cax = ax.inset_axes([0.60, 0.08, 0.35, 0.04])
cb = fig.colorbar(hb, cax=cax, orientation='horizontal')
cb.set_ticks([1, 10, 100]); cb.set_ticklabels(['1', '10', '100'])
cb.ax.tick_params(labelsize=4.6, length=1.2, pad=1); cb.set_label('works', fontsize=4.8, labelpad=1)
cb.ax.xaxis.set_label_position('top'); cb.outline.set_linewidth(0.4)
ax.text(0.03, 0.97, 'n = 6,390\nmedian error 72.5 yr\n80% coverage 0.88', transform=ax.transAxes, fontsize=5.0,
        va='top', linespacing=1.25)

# e: error by period
ax = fig.add_subplot(gs[0, 2])
label(ax, 'e', x=-0.28, y=1.04)
per = ['Pre-Qin', 'Qin–Han', 'Six Dyn.', 'Sui–Tang', 'Song', 'Yuan', 'Ming', 'Qing']
fold0 = {'Ruoqu-L': [25.0, 77.5, 99.5, 49.5, 81.0, 58.25, 68.5, 72.5],
         'discriminative (Aeneas-style)': [1325.0, 252.5, 65.5, 37.5, 57.5, 56.0, 68.5, 77.5],
         'character LR': [150.0, 327.5, 68.25, 68.5, 155.5, 43.75, 100.0, 101.5],
         'period n-gram': [1400.0, 502.5, 225.0, 100.5, 106.0, 37.75, 84.75, 277.5],
         'function words': [113.0, 427.5, 99.5, 100.5, 219.5, 66.0, 161.0, 330.5]}
sty = {'Ruoqu-L': (BLUE, 1.4, 'o', 3), 'discriminative (Aeneas-style)': (WINE, 1.1, 's', 2.6),
       'character LR': (GREYS[0], 0.7, '^', 2.4), 'period n-gram': (GREYS[1], 0.7, 'v', 2.4),
       'function words': (GREYS[2], 0.7, 'D', 2.0)}
x = np.arange(len(per))
for k, v in fold0.items():
    c, lw, mk, ms = sty[k]
    ax.plot(x, v, color=c, lw=lw, marker=mk, ms=ms, label=k, mec='white', mew=0.3, zorder=5 if k == 'Ruoqu-L' else 3)
ax.set_yscale('log'); ax.set_ylim(15, 4000)
ax.set_yticks([25, 50, 100, 250, 500, 1000]); ax.set_yticklabels(['25', '50', '100', '250', '500', '1000'])
ax.minorticks_off()
ax.set_xticks(x); ax.set_xticklabels(per, rotation=40, ha='right', rotation_mode='anchor')
ax.set_ylabel('median absolute error (years)')
ax.legend(loc='upper right', fontsize=4.8, handlelength=1.6, bbox_to_anchor=(1.03, 1.05))
ax.axvspan(-0.5, 1.5, color=SLATE, alpha=0.07, lw=0)
ax.text(0.5, 16.5, 'sparse\nearly data', fontsize=4.8, ha='center', va='bottom', color=GREYS[1], linespacing=1.0)
ax.set_xlim(-0.5, 7.5)

# f: overall
ax = fig.add_subplot(gs[0, 3])
label(ax, 'f', x=-0.42, y=1.04)
names = ['Ruoqu-L', 'discrim.\n(Aeneas-style)', 'char. LR', 'n-gram', 'func. words']
work = [72.5, 67.0, 106.0, 150.5, 187.5]
pas = [112.5, 104.0, 131.0, 149.5, 232.0]
cols = [BLUE, WINE, GREYS[0], GREYS[1], GREYS[2]]
y = np.arange(len(names))[::-1]
ax.barh(y + 0.18, work, height=0.34, color=cols, lw=0)
ax.barh(y - 0.18, pas, height=0.34, color=cols, alpha=0.38, lw=0)
for yy, a, b in zip(y, work, pas):
    ax.text(a + 4, yy + 0.18, f'{a:g}', va='center', fontsize=4.9)
    ax.text(b + 4, yy - 0.18, f'{b:g}', va='center', fontsize=4.9, color=GREYS[1])
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlabel('median absolute error (years)')
ax.set_xlim(0, 300); ax.tick_params(axis='y', length=0)
from matplotlib.patches import Patch
ax.legend([Patch(fc=GREYS[0]), Patch(fc=GREYS[0], alpha=0.38)], ['whole work', 'one 256-character passage'],
          loc='upper center', bbox_to_anchor=(0.45, 1.16), fontsize=4.8, handlelength=0.9, borderaxespad=0)

save(fig, 'fig1_instrument')
print('ok')
