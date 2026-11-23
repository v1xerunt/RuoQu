"""Limits of imitation: analysis of imitation vs control pieces (same author), out-of-fold models.

pull      = (apparent date of imitation - median apparent date of the author's controls) / (target - author date)
            0 = no movement toward the model, 1 = reaches the model's era
residue   = per-character log p(x | author date) - log p(x | target date) for imitations (> 0: still reads
            like the author's own time); split into function words and content characters
Usage: python imit_eval.py <prefix>   (runs <prefix>, <prefix>_f1..f4)
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
VER = __import__('os').environ.get('LETTER_V', '2')

R = (WORK + '/runs')
D = (WORK + '/data')
GRID = np.arange(-1000, 1901, 25).astype(float)
KEEP = GRID >= -600
YEARS = GRID[::4]
EFF = 300  # evidence of a 300-character piece, so long and short pieces are comparable


def load(prefix):
    sys.path.insert(0, os.path.dirname(__file__))
    from views import FUNC
    vocab = json.load(open(f'{D}/corpus{VER}_f0/vocab.json')) + ['X']
    isf = np.array([c in FUNC for c in vocab])
    out = []
    for f in range(5):
        run = prefix if f == 0 else f'{prefix}_f{f}'
        p = f'{R}/{run}/scores_imit_unk.npz'
        if not os.path.exists(p): continue
        z = np.load(p); idx = pd.read_csv(f'{D}/corpus{VER}_f{f}/imit.tsv', sep='\t').set_index('id')
        T = pickle.load(open(p.replace('.npz', '_tok.pkl'), 'rb'))
        tok = {}
        for e in T: tok.setdefault(e['id'], []).append(e)
        meta = pd.DataFrame({'id': z['ids'], 'nt': z['nt']}); ll = z['ll'][:, :len(GRID)].astype(np.float64)
        for i, ii in meta.groupby('id').indices.items():
            n = meta.nt.values[ii].sum(); r = idx.loc[i]
            mll = ll[ii].sum(0) / max(n, 1)
            # token-level split at the 100-year grid
            fsum = np.zeros(len(YEARS)); csum = np.zeros(len(YEARS)); nf = nc = 0
            for e in tok.get(i, []):
                t = e['tok'].astype(int); lp = e['lp'][:len(YEARS)].astype(np.float64); m = t > 1
                fm = m & isf[t]; cm = m & ~isf[t]
                fsum += lp[:, fm].sum(1); csum += lp[:, cm].sum(1); nf += fm.sum(); nc += cm.sum()
            out.append(dict(id=i, fold=f, work=r.work, author=r.author, kind=r.kind, gclass=r.gclass, title=r.title,
                            year=(r.lo + r.hi) / 2, target=r.target, n=int(n), mll=mll,
                            fll=fsum / max(nf, 1), cll=csum / max(nc, 1), nf=nf, nc=nc))
    return pd.DataFrame(out)


def apparent(mll, tau):
    z = tau * EFF * mll[KEEP]; z -= z.max(); p = np.exp(z); p /= p.sum()
    return GRID[KEEP][min((np.cumsum(p) < .5).sum(), KEEP.sum() - 1)]


def ceiling(prefix, tau):
    """apparent dates of genuine pieces by authors near each imitated model's date (scored by their held-out fold)."""
    rows = []
    for f in range(5):
        run = prefix if f == 0 else f'{prefix}_f{f}'
        p = f'{R}/{run}/scores_genuine_unk.npz'
        if not os.path.exists(p): continue
        z = np.load(p); idx = pd.read_csv(f'{D}/corpus{VER}_f{f}/genuine.tsv', sep='\t').set_index('id')
        meta = pd.DataFrame({'id': z['ids'], 'nt': z['nt']}); ll = z['ll'][:, :len(GRID)].astype(np.float64)
        for i, ii in meta.groupby('id').indices.items():
            n = meta.nt.values[ii].sum()
            if n < 20: continue
            r = idx.loc[i]
            rows.append(dict(target=r.target, gclass=r.gclass, year=(r.lo + r.hi) / 2, n=n,
                             app=apparent(ll[ii].sum(0) / n, tau)))
    return pd.DataFrame(rows)


def main(prefix, tau=0.02):
    df = load(prefix)
    df = df[df.n >= 20]
    df['app'] = df.mll.map(lambda v: apparent(v, tau))
    ctrl = df[df.kind == 'ctrl'].groupby(['work', 'gclass']).app.median()
    ctrl_all = df[df.kind == 'ctrl'].groupby('work').app.median()
    im = df[df.kind == 'imit'].copy()
    im['ctrl_app'] = [ctrl.get((w, g), ctrl_all.get(w, np.nan)) for w, g in zip(im.work, im.gclass)]
    im = im[im.ctrl_app.notna()]
    im['gap'] = im.year - im.target
    im['shift'] = im.app - im.ctrl_app
    im['pull'] = -im['shift'] / im.gap
    gi = lambda y: int(np.argmin(np.abs(GRID - max(y, -600))))
    gy = lambda y: int(np.argmin(np.abs(YEARS - max(y, -600))))
    im['residue'] = [v[gi(a)] - v[gi(t)] for v, a, t in zip(im.mll, im.year, im.target)]
    im['res_func'] = [v[gy(a)] - v[gy(t)] for v, a, t in zip(im.fll, im.year, im.target)]
    im['res_cont'] = [v[gy(a)] - v[gy(t)] for v, a, t in zip(im.cll, im.year, im.target)]
    # the same residue for the author's controls (baseline: how far the author's normal writing is from the target era)
    ct = df[df.kind == 'ctrl'].drop(columns='target').merge(im[['work', 'target']].drop_duplicates(), on='work')
    ct['residue'] = [v[gi(a)] - v[gi(t)] for v, a, t in zip(ct.mll, ct.year, ct.target)]
    ct['res_func'] = [v[gy(a)] - v[gy(t)] for v, a, t in zip(ct.fll, ct.year, ct.target)]
    ct['res_cont'] = [v[gy(a)] - v[gy(t)] for v, a, t in zip(ct.cll, ct.year, ct.target)]
    pd.set_option('display.width', 220)
    print(f'{len(im)} imitations, {im.work.nunique()} collections; controls {len(ct)} (paired by collection and target)')
    print(f'shift toward the model (years): median {-im["shift"].median():.0f}; Wilcoxon p={wilcoxon(im["shift"]).pvalue:.2g}')
    print(f'pull fraction: median {im.pull.median():.2f}, IQR {im.pull.quantile(.25):.2f}~{im.pull.quantile(.75):.2f}')
    im['gapbin'] = pd.cut(im.gap, [100, 300, 600, 1000, 3000])
    print(im.groupby('gapbin', observed=True).agg(n=('pull', 'size'), shift=('shift', 'median'), pull=('pull', 'median'),
                                                   residue=('residue', 'median'), res_func=('res_func', 'median'),
                                                   res_cont=('res_cont', 'median')).round(3).to_string())
    print(im.groupby('gclass').agg(n=('pull', 'size'), pull=('pull', 'median'), residue=('residue', 'median')).round(3).to_string())
    print('residue per character, imitations vs the same authors\' ordinary pieces (target-matched):')
    for col in ['residue', 'res_func', 'res_cont']:
        print(f'  {col}: imitations {im[col].median():+.4f}  controls {ct[col].median():+.4f}  '
              f'fraction of control residue remaining {im[col].median() / ct[col].median():.2f}')
    print(f'Spearman(gap, pull) = {spearmanr(im.gap, im.pull)[0]:.3f}')
    # ceiling calibration: how early can the model date a genuine piece of the target era (same form class)?
    ce = ceiling(prefix, tau)
    if len(ce):
        cg = ce.groupby(['target', 'gclass']).app.median(); ct_ = ce.groupby('target').app.median()
        print('genuine target-era pieces: apparent date - true date, median by target:',
              (ce.app - ce.year).groupby(ce.target).median().round(0).to_dict())
        im['ceil'] = [cg.get((t, g), ct_.get(t, np.nan)) for t, g in zip(im.target, im.gclass)]
        im['room'] = im.ctrl_app - im.ceil
        ok = im.ceil.notna() & (im.room >= 100)
        im['pull_norm'] = np.where(ok, (im.ctrl_app - im.app) / im.room, np.nan)
        k = im[ok]
        print(f'normalized pull (fraction of the distance to genuine target-era pieces): n={len(k)} median {k.pull_norm.median():.2f} '
              f'IQR {k.pull_norm.quantile(.25):.2f}~{k.pull_norm.quantile(.75):.2f}; raw pull on the same pieces {k.pull.median():.2f}; '
              f'Spearman(gap, pull_norm) = {spearmanr(k.gap, k.pull_norm)[0]:.3f}')
        print(k.groupby('gclass').pull_norm.agg(['size', 'median']).round(2).to_string())
    im['lenbin'] = pd.cut(im.n, [0, 60, 150, 10000])
    print('pull by length:', im.groupby('lenbin', observed=True).pull.agg(['size', 'median']).round(3).to_dict('index'))
    im.drop(columns=['mll', 'fll', 'cll']).to_csv(f'{R}/{prefix}_imit_pieces.tsv', sep='\t', index=False)
    return im, ct


if __name__ == '__main__':
    main(sys.argv[1])
