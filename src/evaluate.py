"""Text-level dating evaluation (E1) for the TC-LM and baselines on a common year grid.

Posterior over years for a text: softmax(tau * sum_passages score(year)), uniform prior over the grid.
tau is fitted by 2-fold cross-fitting over texts (maximize log posterior mass of the label interval).
Metrics: median absolute error to the label midpoint, distance to the label interval, coverage of
central credible intervals, per period.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, json
import numpy as np
import pandas as pd

C = (WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
R = (WORK + '/runs')
GRID = np.arange(-1000, 1901, 25).astype(float)
# training data before -600 is two works (Shijing, Zhouyi); the model cannot resolve earlier dates
GMIN = float(__import__('os').environ.get('LETTER_GMIN', -600))
PERIODS = [(-2000, -221, '先秦'), (-221, 220, '秦漢'), (220, 589, '魏晉南北朝'), (589, 960, '隋唐五代'),
           (960, 1279, '宋'), (1279, 1368, '元'), (1368, 1644, '明'), (1644, 3000, '清以後')]


def load_scores(path, grid=GRID):
    z = np.load(path)
    ll = z['ll'].astype(np.float64)
    if 'grid' in z.files:  # TC-LM: last column = date unknown
        ll = ll[:, :len(z['grid'])]
    else:  # baseline bins -> grid (density within bin)
        mids = z['mids']
        edges = np.concatenate([[-1100], (mids[1:] + mids[:-1]) / 2, [1950]])
        from baselines import EDGES
        b = np.clip(np.searchsorted(EDGES, grid, side='right') - 1, 0, len(mids) - 1)
        width = np.bincount(b, minlength=len(mids)).astype(float)
        ll = ll[:, b] - np.log(np.maximum(width[b], 1))[None]
    ll[:, grid < GMIN] = -1e9
    return pd.DataFrame({'id': z['ids'], 'pi': z['pi'], 'nt': z['nt']}), ll


def text_scores(meta, ll):
    g = meta.groupby('id').indices
    ids = list(g)
    S = np.stack([ll[g[i]].sum(0) for i in ids])
    n = np.array([len(g[i]) for i in ids])
    return ids, S, n


def posterior(S, tau):
    z = tau * (S - S.max(1, keepdims=True))
    p = np.exp(z); return p / p.sum(1, keepdims=True)


def quantile(p, q):
    c = np.cumsum(p, 1)
    return GRID[np.minimum((c < q).sum(1), len(GRID) - 1)]


def interval_mass(p, lo, hi):
    m = (GRID[None] >= lo[:, None] - 12.5) & (GRID[None] <= hi[:, None] + 12.5)
    return (p * m).sum(1)


def fit_tau(S, lo, hi):
    taus = np.exp(np.linspace(np.log(1e-4), np.log(2), 60))
    best = max(taus, key=lambda t: np.mean(np.log(interval_mass(posterior(S, t), lo, hi) + 1e-6)))
    return best


def metrics(p, lo, hi):
    med = quantile(p, 0.5)
    mid = (lo + hi) / 2
    out = dict(n=len(lo), mae_mid=np.median(np.abs(med - mid)), mean_ae_mid=np.mean(np.abs(med - mid)),
               dist_int=np.median(np.maximum(0, np.maximum(lo - med, med - hi))),
               mean_dist_int=np.mean(np.maximum(0, np.maximum(lo - med, med - hi))))
    for level in (0.5, 0.8, 0.9):
        a, b = quantile(p, (1 - level) / 2), quantile(p, 1 - (1 - level) / 2)
        out[f'cov{int(level * 100)}'] = np.mean((mid >= a - 12.5) & (mid <= b + 12.5))
    out['width80'] = np.median(quantile(p, 0.9) - quantile(p, 0.1))
    return out


def evaluate(path, split='heldout', label=''):
    meta, ll = load_scores(path)
    idx = pd.read_csv(f'{C}/{split}.tsv', sep='\t').set_index('id')
    ids, S, n = text_scores(meta, ll)
    lo, hi = idx.loc[ids, 'lo'].values.astype(float), idx.loc[ids, 'hi'].values.astype(float)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 2, len(ids))
    P = np.zeros_like(S); taus = []
    for f in (0, 1):
        t = fit_tau(S[fold != f], lo[fold != f], hi[fold != f]); taus.append(t)
        P[fold == f] = posterior(S[fold == f], t)
    rows = [dict(model=label, period='all', **metrics(P, lo, hi))]
    mid = (lo + hi) / 2
    for a, b, name in PERIODS:
        s = (mid >= a) & (mid < b)
        if s.sum(): rows.append(dict(model=label, period=name, **metrics(P[s], lo[s], hi[s])))
    # passage-level (single passage, tau fitted the same way on passages)
    pid = meta.id.values
    plo, phi = idx.loc[pid, 'lo'].values.astype(float), idx.loc[pid, 'hi'].values.astype(float)
    pf = fold[np.searchsorted(np.array(ids), pid)] if False else rng.integers(0, 2, len(pid))
    PP = np.zeros_like(ll)
    for f in (0, 1):
        t = fit_tau(ll[pf != f], plo[pf != f], phi[pf != f])
        PP[pf == f] = posterior(ll[pf == f], t)
    rows.append(dict(model=label, period='passage', **metrics(PP, plo, phi)))
    return pd.DataFrame(rows), dict(ids=ids, P=P, taus=taus)


if __name__ == '__main__':
    res = []
    for arg in sys.argv[1:]:
        label, path = arg.split('=')
        df, _ = evaluate(path, label=label); res.append(df)
    df = pd.concat(res)
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))
