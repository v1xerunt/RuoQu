"""Robustness of the imitation result (v3, five folds), humans and AI on the same footing.

1. pull by length bin (short pieces carry little evidence: measurement noise shrinks the apparent movement)
2. ceiling-normalized pull with genuine target-era pieces matched on form class AND length bin
3. residue (function / content) of imitations vs the same author's ordinary pieces of the same form class and length
   bin, evaluated at the imitation's target
4. placebo: ordinary pieces given the target of an imitation in the same collection (their pull should be ~0)
5. bootstrap over collections (humans) / prompts (AI): 95% intervals of the median pull
Usage: python imit_robust.py [tau]
"""
import sys
import numpy as np
import pandas as pd
import imit_eval as ie
import aiimit_eval as ae

ie.VER = '3'
GRID, KEEP, YEARS = ie.GRID, ie.KEEP, ie.YEARS
LB = [20, 60, 150, 10000]
LBL = ['20-60', '60-150', '150+']
gi = lambda y: int(np.argmin(np.abs(GRID - max(y, -600))))
gy = lambda y: int(np.argmin(np.abs(YEARS - max(y, -600))))
rng = np.random.default_rng(0)


def boot(v, groups, n=1000):
    v = np.asarray(v, float); g = np.asarray(groups); u = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in u}
    meds = [np.nanmedian(v[np.concatenate([idx[k] for k in rng.choice(u, len(u))])]) for _ in range(n)]
    return np.percentile(meds, [2.5, 97.5])


def residues(df, own_col, tgt_col):
    df['residue'] = [v[gi(a)] - v[gi(t)] for v, a, t in zip(df.mll, df[own_col], df[tgt_col])]
    df['res_func'] = [v[gy(a)] - v[gy(t)] for v, a, t in zip(df.fll, df[own_col], df[tgt_col])]
    df['res_cont'] = [v[gy(a)] - v[gy(t)] for v, a, t in zip(df.cll, df[own_col], df[tgt_col])]
    return df


def main(tau=0.02):
    pd.set_option('display.width', 220)
    # ---------- humans ----------
    df = ie.load('l3'); df = df[df.n >= 20].copy()
    df['app'] = df.mll.map(lambda v: ie.apparent(v, tau))
    df['lb'] = pd.cut(df.n, LB, labels=LBL)
    ce = ie.ceiling('l3', tau); ce['lb'] = pd.cut(ce.n, LB, labels=LBL)
    ceil = ce.groupby(['target', 'gclass', 'lb'], observed=True).app.median()
    ceil2 = ce.groupby(['target', 'lb'], observed=True).app.median()
    ctrl = df[df.kind == 'ctrl']
    # baseline matched on collection, form class and length bin (the dating model has a length bias)
    cm3 = ctrl.groupby(['work', 'gclass', 'lb'], observed=True).app.agg(['median', 'size'])
    cmed = ctrl.groupby(['work', 'gclass']).app.median()
    def base(w, g, l):
        if (w, g, l) in cm3.index and cm3.loc[(w, g, l), 'size'] >= 3: return cm3.loc[(w, g, l), 'median']
        return np.nan
    im = df[df.kind == 'imit'].copy()
    im['ctrl_app'] = [base(w, g, l) for w, g, l in zip(im.work, im.gclass, im.lb)]
    im = im[im.ctrl_app.notna()].copy()
    im['gap'] = im.year - im.target
    im['pull'] = (im.ctrl_app - im.app) / im.gap
    im['ceil'] = [ceil.get((t, g, l), ceil2.get((t, l), np.nan)) for t, g, l in zip(im.target, im.gclass, im.lb)]
    room = im.ctrl_app - im.ceil
    im['pull_norm'] = np.where(im.ceil.notna() & (room >= 100), (im.ctrl_app - im.app) / room, np.nan)
    print('=== HUMANS:', len(im), 'imitations,', im.work.nunique(), 'collections')
    t = im.groupby('lb', observed=True).agg(n=('pull', 'size'), pull=('pull', 'median'), n_norm=('pull_norm', 'count'),
                                            pull_norm=('pull_norm', 'median'))
    for l in LBL:
        s = im[im.lb == l]
        if len(s) > 20:
            t.loc[l, 'ci_lo'], t.loc[l, 'ci_hi'] = boot(s.pull.values, s.work.values)
    print('pull by length (95% CI over collections, raw pull):'); print(t.round(3).to_string())
    print('ceiling: genuine target-era pieces, apparent - true date, by length:',
          (ce.app - ce.year).groupby(ce.lb, observed=True).median().round(0).to_dict())
    # residues: imitations vs same-author controls of the same form class and length bin, at the imitation's target
    im = residues(im, 'year', 'target')
    pairs = []
    for r in im.itertuples():
        c = ctrl[(ctrl.work == r.work) & (ctrl.gclass == r.gclass) & (ctrl.lb == r.lb)]
        if len(c) < 3: continue
        rc = np.median([v[gi(r.year)] - v[gi(r.target)] for v in c.mll])
        fc = np.median([v[gy(r.year)] - v[gy(r.target)] for v in c.fll])
        cc = np.median([v[gy(r.year)] - v[gy(r.target)] for v in c.cll])
        pairs.append(dict(lb=r.lb, gclass=r.gclass, work=r.work, d_all=r.residue - rc, d_func=r.res_func - fc,
                          d_cont=r.res_cont - cc, c_all=rc, c_func=fc, c_cont=cc,
                          i_all=r.residue, i_func=r.res_func, i_cont=r.res_cont))
    P = pd.DataFrame(pairs)
    print(f'matched residue (imitation minus matched controls; negative = imitation reads more like the target), n={len(P)}:')
    print(P.groupby('lb', observed=True)[['i_func', 'c_func', 'd_func', 'i_cont', 'c_cont', 'd_cont']].median().round(4).to_string())
    print('fraction of control residue remaining (median imitation / median control):',
          {k: round(P[f'i_{k}'].median() / P[f'c_{k}'].median(), 2) for k in ('all', 'func', 'cont')})
    # placebo: ordinary pieces of collections with imitations, given the target of a random imitation of that collection
    pl = []
    tg = im.groupby('work').target.apply(list).to_dict()
    for w, g in ctrl[ctrl.work.isin(tg)].groupby('work'):
        if len(g) < 6: continue
        k = rng.permutation(len(g)); test, rest = g.iloc[k[:len(k) // 3]], g.iloc[k[len(k) // 3:]]
        bs = rest.groupby(['gclass', 'lb'], observed=True).app.agg(['median', 'size'])
        for r in test.itertuples():
            if (r.gclass, r.lb) not in bs.index or bs.loc[(r.gclass, r.lb), 'size'] < 3: continue
            t_ = rng.choice(tg[w]); b = bs.loc[(r.gclass, r.lb), 'median']
            if r.year - t_ >= 100: pl.append(dict(work=w, lb=r.lb, pull=(b - r.app) / (r.year - t_)))
    PL = pd.DataFrame(pl)
    print('placebo (ordinary pieces with a fake target): median pull by length',
          PL.groupby('lb', observed=True).pull.median().round(3).to_dict(), f'overall {PL.pull.median():.3f} (n={len(PL)})')
    im.drop(columns=['mll', 'fll', 'cll']).to_csv(f'{ie.R}/l3_imit_robust.tsv', sep='\t', index=False)

    # ---------- AI ----------
    a = ae.load(); a = a[a.n_raw >= 20].copy(); a = a[a.copied <= .2]
    a['app'] = a.mll.map(lambda v: ae.apparent(v, tau)); a['lb'] = pd.cut(a.n_raw, LB, labels=LBL)
    ac = a[a.kind == 'ctrl']; own3 = ac.groupby(['model', 'gclass', 'lb'], observed=True).app.agg(['median', 'size'])
    own = ac.groupby(['model', 'gclass']).app.median()
    ai = a[a.kind == 'imit'].copy()
    ai['own'] = [own3.loc[(m, g, l), 'median'] if (m, g, l) in own3.index and own3.loc[(m, g, l), 'size'] >= 3 else np.nan
                 for m, g, l in zip(ai.model, ai.gclass, ai.lb)]
    ai['pull'] = np.where(ai.own - ai.target >= 100, (ai.own - ai.app) / (ai.own - ai.target), np.nan)
    ai['ceil'] = [ceil.get((t, g, l), ceil2.get((t, l), np.nan)) for t, g, l in zip(ai.target, ai.gclass, ai.lb)]
    room = ai.own - ai.ceil
    ai['pull_norm'] = np.where(ai.ceil.notna() & (room >= 100), (ai.own - ai.app) / room, np.nan)
    print('\n=== AI: pull by length (median; n)')
    print(ai.pivot_table(index='model', columns='lb', values='pull', aggfunc='median', observed=True).round(3).to_string())
    print(ai.pivot_table(index='model', columns='lb', values='pull', aggfunc='count', observed=True).to_string())
    print('AI normalized pull by length:')
    print(ai.pivot_table(index='model', columns='lb', values='pull_norm', aggfunc='median', observed=True).round(3).to_string())
    print('humans normalized pull by length:', im.groupby('lb', observed=True).pull_norm.median().round(3).to_dict())
    for m, g in ai.groupby('model'):
        lo, hi = boot(g.pull.dropna().values, g.dropna(subset=['pull']).pid.values)
        print(f'  {m:20s} pull median {g.pull.median():.3f} (95% CI {lo:.3f}~{hi:.3f}), normalized {g.pull_norm.median():.3f}')
    # same targets, side by side (pieces of 60+ characters)
    hs = im[im.n > 60].groupby('target').agg(h_n=('pull', 'size'), h_pull=('pull', 'median'), h_norm=('pull_norm', 'median'))
    as_ = ai[ai.n_raw > 60].groupby('target').agg(a_n=('pull', 'count'), a_pull=('pull', 'median'), a_norm=('pull_norm', 'median'))
    print('\nsame targets, pieces > 60 characters (humans vs all AI models pooled):')
    print(hs.join(as_, how='inner').round(3).to_string())


if __name__ == '__main__':
    main(*(float(x) for x in sys.argv[1:]))
