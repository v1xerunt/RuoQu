"""Shared figure style (Nature-like): Arial 6-7 pt, CJK fallback, muted palette without orange."""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
# PKG: the unpacked results archive from Zenodo (folders data/ and runs/); OUT: where figures are written
PKG = os.environ.get('RUOQU_RESULTS', os.path.join(os.path.dirname(HERE), 'results'))
OUT = os.environ.get('RUOQU_FIGURES', os.path.join(HERE, 'output'))
os.makedirs(OUT, exist_ok=True)

MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM  # single / double column

# palette (no orange)
INK = '#1f1f1f'
GRID = '#e6e6e6'
BLUE = '#1b4f8a'      # Ruoqu
BLUE_L = '#8fb3d9'
WINE = '#9e2a3a'      # discriminative control / forgeries
WINE_L = '#e3a6ae'
TEAL = '#1f7a72'      # humans
TEAL_L = '#9fd1cb'
PURPLE = '#5b4b8a'    # AI
PURPLE_L = '#bdb3dc'
GREEN = '#4f7f3a'
GREYS = ['#4d4d4d', '#7f7f7f', '#a6a6a6', '#cfcfcf']
SLATE = '#5d6d7e'
MAGENTA = '#a1307a'

AI_COL = {
    'sonnet': '#3e2f73', 'gpt-sol62': '#2f6f9f', 'gemini4-neon': '#6f9fcf',
    'deepseek-v41-flash': '#3f8f7f', 'qwen3-32b': '#8f7fbf', 'qwen25-32b': '#b8acd8'}
AI_NAME = {
    'sonnet': 'Claude Sonnet', 'gpt-sol62': 'GPT Sol 6.2', 'gemini4-neon': 'Gemini 4 Neon',
    'deepseek-v41-flash': 'DeepSeek v4.1', 'qwen3-32b': 'Qwen3-32B', 'qwen25-32b': 'Qwen2.5-32B'}

PERIODS = [(-600, -221, 'Pre-Qin'), (-221, 220, 'Qin–Han'), (220, 589, 'Six Dyn.'), (589, 907, 'Sui–Tang'),
           (907, 1279, 'Song'), (1279, 1368, 'Yuan'), (1368, 1644, 'Ming'), (1644, 1912, 'Qing')]


def setup():
    for f in fm.findSystemFonts():
        pass
    mpl.rcParams.update({
        'font.family': ['Arial', 'Microsoft JhengHei'],
        'font.size': 6.5,
        'axes.titlesize': 7,
        'axes.labelsize': 6.5,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.pad': 2, 'ytick.major.pad': 2,
        'axes.labelpad': 2.5,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.handlelength': 1.2, 'legend.borderpad': 0.2, 'legend.labelspacing': 0.3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 400, 'figure.dpi': 150,
        'lines.linewidth': 0.9,
        'patch.linewidth': 0.5,
        'hatch.linewidth': 0.5,
        'axes.unicode_minus': True,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'mathtext.sf': 'Arial',
    })


def year_fmt(y, pos=None):
    y = int(round(y))
    if y < 0:
        return f'{-y} BCE'
    if y == 0:
        return '0'
    return f'{y}'


def year_axis(ax, which='x', ticks=(-500, 0, 500, 1000, 1500), ce=True):
    from matplotlib.ticker import FixedLocator, FuncFormatter
    loc = FixedLocator(ticks)
    fmt = FuncFormatter(lambda y, p: (f'{-int(y)} BCE' if y < 0 else ('1 CE' if y == 0 and ce else f'{int(y)}')))
    fmt = FuncFormatter(lambda y, p: (f'{-int(y)} BCE' if y < 0 else f'{int(y)}'))
    a = ax.xaxis if which == 'x' else ax.yaxis
    a.set_major_locator(loc); a.set_major_formatter(fmt)


def label(ax, s, x=-0.18, y=1.06, fig=None):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left', color=INK)


def period_bands(ax, lo=-650, hi=1950, alpha=0.06, text=True, y=1.0, fs=5, which='x'):
    for i, (a, b, n) in enumerate(PERIODS):
        a, b = max(a, lo), min(b, hi)
        if i % 2 == 0:
            if which == 'x':
                ax.axvspan(a, b, color=SLATE, alpha=alpha, lw=0, zorder=0)
            else:
                ax.axhspan(a, b, color=SLATE, alpha=alpha, lw=0, zorder=0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', pad_inches=0.02, dpi=250)
    plt.close(fig)
