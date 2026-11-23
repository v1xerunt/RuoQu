"""Cross-fitted (out-of-fold) evaluation over all folds and a fold ensemble for the blind sets.

Runs: <prefix> (fold 0) and <prefix>_f1..f4. Each fold's held-out posterior uses that fold model's tau
(fitted by 2-fold cross-fitting within the fold, as in evaluate.py).
Blind ensemble: average the per-passage log-likelihoods of the available fold models.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys
import numpy as np
import pandas as pd
from evaluate import evaluate, metrics, PERIODS, load_scores, text_scores, posterior, quantile, GRID
from blind import work_table, summarize

R = (WORK + '/runs')
D = (WORK + '/data')


def runs(prefix):
    out = []
    for f in range(5):
        r = prefix if f == 0 else f'{prefix}_f{f}'
        if os.path.exists(f'{R}/{r}/scores_heldout_unk.npz'): out.append((f, r))
    return out


def oof(prefix):
    rows, Ps, lo, hi = [], [], [], []
    taus = []
    for f, r in runs(prefix):
        cp = os.environ.get('LETTER_CPREFIX', 'corpus_f')
        import evaluate as ev; ev.C = f'{D}/{cp}{f}'
        df, res = evaluate(f'{R}/{r}/scores_heldout_unk.npz', label=r)
        idx = pd.read_csv(f'{D}/{cp}{f}/heldout.tsv', sep='\t').set_index('id').loc[res['ids']]
        Ps.append(res['P']); lo.append(idx.lo.values); hi.append(idx.hi.values); taus += res['taus']
        rows.append(pd.DataFrame({'id': res['ids'], 'fold': f, 'title': idx.title.values, 'cat': idx.cat.values, 'genre': idx.genre.values,
                                  'lo': idx.lo.values, 'hi': idx.hi.values, 'date_src': idx.date_src.values,
                                  'median': quantile(res['P'], .5), 'q10': quantile(res['P'], .1), 'q90': quantile(res['P'], .9)}))
    P = np.concatenate(Ps); lo = np.concatenate(lo).astype(float); hi = np.concatenate(hi).astype(float)
    tab = pd.concat(rows)
    res = [dict(period='all', **metrics(P, lo, hi))]
    mid = (lo + hi) / 2
    for a, b, name in PERIODS:
        s = (mid >= a) & (mid < b)
        if s.sum(): res.append(dict(period=name, **metrics(P[s], lo[s], hi[s])))
    return tab, pd.DataFrame(res), float(np.exp(np.mean(np.log(taus))))


def ensemble_scores(prefix, split, genre='unk'):
    """average per-passage log-likelihoods across fold models -> a temporary npz"""
    lls = []
    for f, r in runs(prefix):
        p = f'{R}/{r}/scores_{split}_{genre}.npz'
        if os.path.exists(p):
            z = np.load(p); lls.append(z['ll']); base = z
    if not lls: return None
    out = f'{R}/{prefix}_ens_{split}_{genre}.npz'
    np.savez_compressed(out, ids=base['ids'], pi=base['pi'], ll=np.mean(lls, 0), nt=base['nt'], grid=base['grid'])
    return out, len(lls)


if __name__ == '__main__':
    prefix = sys.argv[1]
    pd.set_option('display.width', 220)
    tab, res, tau = oof(prefix)
    print(f'cross-fitted over {tab.fold.nunique()} folds, {len(tab)} texts')
    print(res.round(3).to_string(index=False))
    tab.to_csv(f'{R}/{prefix}_oof.tsv', sep='\t', index=False)
    cp0 = f"{D}/{os.environ.get('LETTER_CPREFIX', 'corpus_f')}0"
    import evaluate as ev; ev.C = cp0
    import blind as bl; bl.C = cp0
    for split in ['blind_full', 'blind_res']:
        e = ensemble_scores(prefix, split)
        if e:
            df = work_table(split, e[0], tau)
            print(f'== {split}: ensemble of {e[1]} fold models (tau {tau:.4f})')
            print(df[['work', 'cat', 'attr', 'cons', 'median', 'q10', 'q90', 'p_later']].round(2).to_string(index=False))
            summarize(df)


def blind_v1(prefix):
    """second (out-of-fold) blind list: each work is scored by the fold model that held it out"""
    from blind import auc
    tab = pd.read_csv(f'{R}/{prefix}_oof.tsv', sep='\t').set_index('id')
    bl = pd.read_csv((PROJECT + '/data/blind_v1_oof.tsv'), sep='\t')
    rows = []
    for _, r in bl.iterrows():
        if r.id not in tab.index: continue
        t = tab.loc[r.id]
        rows.append(dict(work=r.group, conf=r.conf, attr=f'{r.attr_lo}~{r.attr_hi}', cons=f'{r.cons_lo}~{r.cons_hi}',
                         fold=int(t.fold), median=t['median'], q10=t.q10, q90=t.q90,
                         later_than_attr=t['median'] > r.attr_hi + 100, in_cons=r.cons_lo - 100 <= t['median'] <= r.cons_hi + 100))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f'later than attributed (+100y): {df.later_than_attr.sum()}/{len(df)};  within consensus (±100y): {df.in_cons.sum()}/{len(df)}')
    return df
