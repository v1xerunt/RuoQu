"""Dates for the undated texts (ensemble of the fold models that scored the 'undated' split), and validation
against the scholarly date statements of the 道教文化中心資料庫 (data/daoinfo_dates.tsv) for Daoist texts.
Also validates the daoinfo labels themselves on Daoist texts that do have catalogue dates (OOF predictions).
Texts whose 12-grams the scoring models largely saw in training (runs/undated_overlap.tsv, seen12 > 0.2) are
reported separately: their dates partly echo the catalogue labels of the trained copies.
Output: runs/undated_dates.tsv, runs/dao_dated_disagree.tsv (dated Daoist texts where catalogue and daoinfo differ)
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, numpy as np, pandas as pd
os.environ.setdefault('LETTER_CORPUS', 'corpus3_f0')
from evaluate import load_scores, text_scores, posterior, quantile, GRID
R = (WORK + '/runs'); D = (WORK + '/data'); P = (PROJECT + '/data')
TAU = 0.0223  # five-fold ensemble temperature fitted on held-out texts (oof.py)


def main():
    S = []
    for run in ['l3', 'l3_f1', 'l3_f2', 'l3_f3', 'l3_f4']:
        p = f'{R}/{run}/scores_undated_unk.npz'
        if os.path.exists(p):
            meta, ll = load_scores(p); ids, s, n = text_scores(meta, ll); S.append(pd.DataFrame(s, index=ids))
    print(len(S), 'models')
    Sm = sum(S) / len(S)
    p = posterior(Sm.values, TAU)
    idx = pd.read_csv(f'{D}/corpus3_f0/undated.tsv', sep='\t').set_index('id')
    out = idx.loc[Sm.index, ['title', 'cat', 'dyn', 'author', 'n_raw']].copy()
    out['median'] = quantile(p, .5); out['q10'] = quantile(p, .1); out['q90'] = quantile(p, .9); out['q25'] = quantile(p, .25); out['q75'] = quantile(p, .75)
    dao = pd.read_csv(f'{P}/daoinfo_dates.tsv', sep='\t').set_index('id') if os.path.exists(f'{P}/daoinfo_dates.tsv') else None
    if dao is not None:
        out = out.join(dao[['statement', 'lo', 'hi']].rename(columns={'lo': 'dao_lo', 'hi': 'dao_hi'}))
    if os.path.exists(f'{R}/undated_overlap.tsv'):
        out = out.join(pd.read_csv(f'{R}/undated_overlap.tsv', sep='\t').set_index('id')[['seen12']])
    out.to_csv(f'{R}/undated_dates.tsv', sep='\t')
    pd.set_option('display.width', 220)
    print(len(out), 'undated texts dated; by category (median of model medians):')
    print(out.groupby(out.cat.str[:4]).agg(n=('median', 'size'), median=('median', 'median')).sort_values('n', ascending=False).head(15).to_string())
    if dao is not None:
        v = out[out.dao_lo.notna()].copy(); v['dao_mid'] = (v.dao_lo + v.dao_hi) / 2
        v['inside'] = (v['median'] >= v.dao_lo - 50) & (v['median'] <= v.dao_hi + 50)
        v['outside80'] = (v.q10 > v.dao_hi + 50) | (v.q90 < v.dao_lo - 50)  # the model's 80% interval misses the stated period
        for name, g in [('all', v), ('seen12<=0.2', v[v.seen12 <= .2]) if 'seen12' in v else ('all', v)]:
            print(f'\nundated Daoist texts with a daoinfo date ({name}): n={len(g)}; model within ±50 y of the stated period {g.inside.mean():.2f}; '
                  f'80% interval misses it {g.outside80.mean():.2f}; median |model - midpoint| {(g["median"] - g.dao_mid).abs().median():.0f}; '
                  f'Spearman {g[["median", "dao_mid"]].corr("spearman").iloc[0, 1]:.2f}')
        # the labels' own reliability: Daoist texts with catalogue dates (OOF)
        o = pd.read_csv(f'{R}/l3_oof.tsv', sep='\t').set_index('id').join(dao[['statement', 'lo', 'hi']].rename(columns={'lo': 'dao_lo', 'hi': 'dao_hi'}), how='inner')
        o = o[o.dao_lo.notna()]
        if len(o):
            o['cat_mid'] = (o.lo + o.hi) / 2; o['dao_mid'] = (o.dao_lo + o.dao_hi) / 2
            print(f'dated Daoist texts with a daoinfo date: n={len(o)}; |catalogue - daoinfo| median {(o.cat_mid - o.dao_mid).abs().median():.0f}; '
                  f'|model - daoinfo| median {(o["median"] - o.dao_mid).abs().median():.0f}; |model - catalogue| median {(o["median"] - o.cat_mid).abs().median():.0f}')
            # where the two scholarly labels disagree (ranges more than 100 y apart), whom does the out-of-fold model side with?
            gap = np.maximum(o.dao_lo - o.hi, o.lo - o.dao_hi)
            d = o[gap > 100].copy()
            d['side'] = np.where((d['median'] - d.dao_mid).abs() < (d['median'] - d.cat_mid).abs(), 'daoinfo', 'catalogue')
            print(f'catalogue and daoinfo >100 y apart: n={len(d)}; model closer to daoinfo {(d.side == "daoinfo").mean():.2f}')
            d[['title', 'lo', 'hi', 'dao_lo', 'dao_hi', 'median', 'q10', 'q90', 'side', 'statement']].to_csv(f'{R}/dao_dated_disagree.tsv', sep='\t')
            print(d[['title', 'lo', 'hi', 'dao_lo', 'dao_hi', 'median', 'side']].to_string())
        print('\nundated Daoist texts whose 80% interval misses the daoinfo period:')
        print(v[v.outside80].sort_values('median')[['title', 'statement', 'dao_lo', 'dao_hi', 'median', 'q10', 'q90'] + (['seen12'] if 'seen12' in v else [])].to_string())

if __name__ == '__main__':
    main()
