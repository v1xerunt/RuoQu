"""AI imitations vs human imitations: how far toward the target era does an imitation get, and what remains.

Each piece is scored by every finished v3 fold model (ensemble: mean per-character log-likelihood over models).
A model's own classical Chinese = its control pieces (same form and title, no target named).
pull    = (control date - imitation date) / (control date - target date): 0 = no movement, 1 = reaches the target
residue = per-character log p(x | control date) - log p(x | target date), imitations vs controls, split into
          function words and content characters (fraction of the control's residue remaining)
tracking: Spearman and slope of apparent date on target date across a model's imitations.
Usage: python aiimit_eval.py [tau]
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

R = (WORK + '/runs')
D = (WORK + '/data')
GRID = np.arange(-1000, 1901, 25).astype(float)
KEEP = GRID >= -600
YEARS = GRID[::4]
EFF = 300
RUNS = ['l3', 'l3_f1', 'l3_f2', 'l3_f3', 'l3_f4']


def load():
    sys.path.insert(0, os.path.dirname(__file__))
    from views import FUNC
    vocab = json.load(open(f'{D}/corpus3_f0/vocab.json')) + ['X']
    isf = np.array([c in FUNC for c in vocab])
    idx = pd.read_csv(f'{D}/corpus3_f0/aiimit.tsv', sep='\t').set_index('id')
    M, F, C, used = {}, {}, {}, []
    for run in RUNS:
        p = f'{R}/{run}/scores_aiimit_unk.npz'
        if not os.path.exists(p): continue
        used.append(run)
        z = np.load(p); ll = z['ll'][:, :len(GRID)].astype(np.float64)
        meta = pd.DataFrame({'id': z['ids'], 'nt': z['nt']})
        for i, ii in meta.groupby('id').indices.items():
            M.setdefault(i, []).append(ll[ii].sum(0) / max(meta.nt.values[ii].sum(), 1))
        for e in pickle.load(open(p.replace('.npz', '_tok.pkl'), 'rb')):
            t = e['tok'].astype(int); lp = e['lp'][:len(YEARS)].astype(np.float64); m = t > 1
            fm, cm = m & isf[t], m & ~isf[t]
            F.setdefault((e['id'], run), [np.zeros(len(YEARS)), 0]); C.setdefault((e['id'], run), [np.zeros(len(YEARS)), 0])
            F[(e['id'], run)][0] += lp[:, fm].sum(1); F[(e['id'], run)][1] += fm.sum()
            C[(e['id'], run)][0] += lp[:, cm].sum(1); C[(e['id'], run)][1] += cm.sum()
    rows = []
    for i, v in M.items():
        fl = np.mean([F[(i, r)][0] / max(F[(i, r)][1], 1) for r in used if (i, r) in F], 0)
        cl = np.mean([C[(i, r)][0] / max(C[(i, r)][1], 1) for r in used if (i, r) in C], 0)
        rows.append(dict(id=i, mll=np.mean(v, 0), fll=fl, cll=cl))
    df = pd.DataFrame(rows).set_index('id').join(idx)
    print('models used for scoring:', used)
    return df.reset_index()


def apparent(mll, tau):
    z = tau * EFF * mll[KEEP]; z -= z.max(); p = np.exp(z); p /= p.sum()
    return GRID[KEEP][min((np.cumsum(p) < .5).sum(), KEEP.sum() - 1)]


def main(tau=0.02):
    df = load()
    df = df[(df.n_raw >= 20)]
    df['app'] = df.mll.map(lambda v: apparent(v, tau))
    gi = lambda y: int(np.argmin(np.abs(GRID - max(y, -600))))
    gy = lambda y: int(np.argmin(np.abs(YEARS - max(y, -600))))
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
    print('pieces with >= 20 characters:', df.groupby('model').size().to_dict())
    print('verbatim copying (share of characters in a 12-gram found in the corpus), mean / share of pieces > 20%:')
    print(df.groupby('model').copied.agg(['mean', lambda v: (v > .2).mean()]).round(3).to_string())
    df = df[df.copied <= .2]
    ctrl = df[df.kind == 'ctrl']
    own = ctrl.groupby(['model', 'gclass']).app.median()
    print("\n== each model's own classical Chinese (controls, no target named): apparent date")
    print(ctrl.groupby(['model', 'gclass']).app.agg(['size', 'median', lambda v: v.quantile(.25), lambda v: v.quantile(.75)]).to_string())
    im = df[df.kind == 'imit'].copy()
    im['own'] = [own.get((m, g), np.nan) for m, g in zip(im.model, im.gclass)]
    im['gap'] = im.own - im.target
    im['pull'] = (im.own - im.app) / im.gap
    im.loc[im.gap < 100, 'pull'] = np.nan
    im['residue'] = [v[gi(a)] - v[gi(t)] for v, a, t in zip(im.mll, im.own, im.target)]
    im['res_func'] = [v[gy(a)] - v[gy(t)] for v, a, t in zip(im.fll, im.own, im.target)]
    im['res_cont'] = [v[gy(a)] - v[gy(t)] for v, a, t in zip(im.cll, im.own, im.target)]
    # controls evaluated at the same targets (pairs by model, form class and title)
    ct = ctrl.merge(im[['model', 'gclass', 'title', 'target', 'own']].drop_duplicates(), on=['model', 'gclass', 'title'],
                    suffixes=('_x', ''))
    for col, v, g in (('residue', 'mll', gi), ('res_func', 'fll', gy), ('res_cont', 'cll', gy)):
        ct[col] = [x[g(a)] - x[g(t)] for x, a, t in zip(ct[v], ct.own, ct.target)]
    print('\n== imitations: movement toward the target')
    for m, g in im.groupby('model'):
        r = spearmanr(g.target, g.app)[0]; b = np.polyfit(g.target, g.app, 1)[0]
        s = g.app - g.own
        print(f'{m:20s} n={len(g):3d}  shift {-s.median():+5.0f}y (p={wilcoxon(s).pvalue:.1g})  pull median {g.pull.median():.2f} '
              f'IQR {g.pull.quantile(.25):.2f}~{g.pull.quantile(.75):.2f}  tracking rho={r:.2f} slope={b:.2f}')
        c = ct[ct.model == m]
        print(' ' * 21 + '  per-character residue (imitation | own control): ' + '  '.join(
            f'{k} {g[k].median():+.4f} | {c[k].median():+.4f}' for k in ('residue', 'res_func', 'res_cont')))
    import imit_eval
    imit_eval.VER = '3'
    ce = imit_eval.ceiling('l3', tau)
    if len(ce):
        cg = ce.groupby(['target', 'gclass']).app.median(); ct_ = ce.groupby('target').app.median()
        im['ceil'] = [cg.get((t, g), ct_.get(t, np.nan)) for t, g in zip(im.target, im.gclass)]
        room = im.own - im.ceil
        im['pull_norm'] = np.where(im.ceil.notna() & (room >= 100), (im.own - im.app) / room, np.nan)
        print('\n== normalized pull (distance to genuine target-era pieces):')
        print(im.groupby('model').pull_norm.agg(['count', 'median', lambda v: v.quantile(.25), lambda v: v.quantile(.75)]).round(2).to_string())
    print('\n== pull by form class')
    print(im.pivot_table(index='model', columns='gclass', values='pull', aggfunc='median').round(2).to_string())
    print('\n== apparent date by target (median over titles)')
    print(im.pivot_table(index='label', columns='model', values='app', aggfunc='median').join(
        im.groupby('label').target.first()).sort_values('target').to_string())
    im.drop(columns=['mll', 'fll', 'cll']).to_csv(f'{R}/aiimit_pieces.tsv', sep='\t', index=False)
    return im, ct


if __name__ == '__main__':
    main(*(float(a) for a in sys.argv[1:]))
