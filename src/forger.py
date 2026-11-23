"""Forger model: a text written at t may imitate the language of an earlier era t' <= t.

Each character is either imitation (drawn from p(.|t')) or leaks the writer's own era (p(.|t)):
    log L(t, t') = sum_i log[ lam * p(x_i | ctx, t') + (1 - lam) * p(x_i | ctx, t) ]
Prior: no imitation (t' = t) with prob 1 - rho, otherwise t' uniform over earlier grid years.
Posterior over the writing date t marginalizes t'. Archaic-looking characters cost the 'late' hypothesis
at most -log(lam) each, while a character that is far more probable late than early can push it a lot:
the one-sided logic of philological dating ("not earlier than").
Token log-probs come from score.py --tokens 4 (grid every 100 years).
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, pickle, json
import numpy as np
import pandas as pd

R = (WORK + '/runs')
GRID = np.arange(-1000, 1901, 25).astype(float)
YEARS = GRID[::4]  # -1000, -900, ..., 1900
KEEP = YEARS >= -600
Y = YEARS[KEEP]


def text_lp(entries):
    """concatenate token log-probs (years x tokens) of all passages, dropping pad/sep."""
    L = []
    for e in entries:
        tok = e['tok'].astype(int); lp = e['lp'][:len(YEARS)].astype(np.float64)[KEEP]
        m = (tok != 0) & (tok != 1)
        L.append(lp[:, m])
    return np.concatenate(L, 1)


def forger_posterior(L, lam, rho, tau):
    """L: years x tokens. Returns posterior over writing date t (len(Y))."""
    G = len(Y)
    logL = np.full((G, G), -np.inf)  # [t, t']
    for t in range(G):
        for s in range(t + 1):
            if s == t:
                logL[t, s] = L[t].sum()
            else:
                a = np.log(lam) + L[s]; b = np.log(1 - lam) + L[t]
                logL[t, s] = np.logaddexp(a, b).sum()
    prior = np.zeros((G, G))
    for t in range(G):
        prior[t, t] = 1 - rho if t > 0 else 1.0
        if t > 0: prior[t, :t] = rho / t
    with np.errstate(divide='ignore'):
        z = tau * logL + np.log(prior)
    z -= z[np.isfinite(z)].max()
    p = np.exp(z).sum(1)
    return p / p.sum()


def summarize(p, attr_hi=None):
    c = np.cumsum(p)
    q = lambda a: Y[min((c < a).sum(), len(Y) - 1)]
    out = dict(median=q(.5), q10=q(.1), q90=q(.9))
    if attr_hi is not None: out['p_later'] = float(p[Y > attr_hi + 50].sum())
    return out


def blind(name, lam, rho, tau):
    T = pickle.load(open(f'{R}/{name}/scores_blind_full_unk_tok.pkl', 'rb'))
    by = {}
    for e in T: by.setdefault(e['id'], []).append(e)
    bl = pd.read_csv((PROJECT + '/data/blind_v0.tsv'), sep='\t')
    rows = []
    for g, sub in bl[bl['cat'].isin(['A', 'B', 'D'])].groupby('group', sort=False):
        mem = [i for i in sub.id if i in by]
        if not mem: continue
        # copies of the same work: score the first (longest) copy only
        i = max(mem, key=lambda k: len(by[k]))
        p = forger_posterior(text_lp(by[i]), lam, rho, tau)
        r = sub.iloc[0]
        rows.append(dict(work=g, cat=r['cat'], attr=f'{r.attr_lo}~{r.attr_hi}', cons=f'{r.cons_lo}~{r.cons_hi}',
                         **summarize(p, float(r.attr_hi))))
    return pd.DataFrame(rows)


def heldout(name, lam, rho, tau, tag='early'):
    T = pickle.load(open(f'{R}/{name}/scores_heldout_unk_{tag}_tok.pkl', 'rb'))
    by = {}
    for e in T: by.setdefault(e['id'], []).append(e)
    idx = pd.read_csv((WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0') + '/heldout.tsv', sep='\t').set_index('id')
    rows = []
    for i, es in by.items():
        p = forger_posterior(text_lp(es), lam, rho, tau)
        lo, hi = idx.loc[i, 'lo'], idx.loc[i, 'hi']
        rows.append(dict(id=i, title=idx.loc[i, 'title'], lo=lo, hi=hi, **summarize(p, float(hi))))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    from blind import auc
    name = sys.argv[1]
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0234
    pd.set_option('display.width', 200)
    for lam in [1.0 - 1e-9, 0.99, 0.98, 0.95, 0.9]:
        for rho in [0.2]:
            df = blind(name, lam, rho, tau)
            a, b = df[df.cat == 'A'], df[df.cat == 'B']
            print(f'lam={lam:.3f} rho={rho}: A later-than-attributed {(a.p_later > .5).sum()}/{len(a)}, B {(b.p_later < .5).sum()}/{len(b)}, '
                  f'AUC {auc(a.p_later, b.p_later):.3f}, A median-cons {np.median(a["median"] - a.cons.map(lambda c: np.mean([float(x) for x in c.split("~")]))):.0f}')
            if lam in (0.98,):
                print(df.to_string(index=False))
