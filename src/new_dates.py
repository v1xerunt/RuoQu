"""Dates for texts nobody dates. Joins the five-model ensemble dates of the undated texts
(runs/undated_dates.tsv) with every external date we could find: CBETA work metadata (data/cbeta_dates.tsv), the
byline of the first juan (data/undated_bylines.tsv), the 道教文化中心資料庫 statements (data/daoinfo_dates.tsv), and the
life dates of the CBETA creators from the DILA person authority (data/creator_dates.tsv; modern editors skipped).
1. Texts with an external date validate the released intervals: agreement and the real coverage of the 80% interval
   (share of texts whose external range overlaps it).
2. Texts with none of them are released, with reliability tiers fixed before looking at the dates:
   A: >= 2,000 characters, 80% interval <= 400 y wide, <= 20% of 12-grams seen in training
   B: 80% interval <= 700 y wide and <= 50% seen;  C: the rest (listed, not used)
   Commentaries/compilations (title with 注 註 疏 解 箋 鈔 抄 集解 科) are marked: their dates mix layers.
   Apocrypha (KR6u, Taishō 疑似部) are marked: they imitate the translation idiom, so their dates lean early.
Output: data/new_dates_v0.tsv, runs/new_dates_validation.tsv
Usage: python new_dates.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import re, numpy as np, pandas as pd

R = (WORK + '/runs'); D = (WORK + '/data'); P = (PROJECT + '/data')
LAYER = re.compile(r'注|註|疏|解|箋|鈔|抄|科|集解|音義|講|釋')


def stats(x, lo, hi, name):
    mid = (x[lo] + x[hi]) / 2
    inside = (x['median'] >= x[lo] - 50) & (x['median'] <= x[hi] + 50)
    cover = (x.q10 <= x[hi]) & (x.q90 >= x[lo])
    print(f'{name}: n={len(x)}; |model - midpoint| median {(x["median"] - mid).abs().median():.0f} y; within ±50 y {inside.mean():.2f}; '
          f'80% interval overlaps the external range {cover.mean():.2f}; Spearman {pd.concat([x["median"], mid], axis=1).corr("spearman").iloc[0, 1]:.2f}')
    return dict(set=name, n=len(x), med_err=(x['median'] - mid).abs().median(), within50=inside.mean(), cover80=cover.mean())


def main():
    d = pd.read_csv(f'{R}/undated_dates.tsv', sep='\t', index_col=0)
    print(len(d), 'undated texts')
    cb = pd.read_csv(f'{P}/cbeta_dates.tsv', sep='\t').set_index('id')
    bl = pd.read_csv(f'{P}/undated_bylines.tsv', sep='\t').set_index('id')
    src = pd.read_csv(f'{D}/meta.tsv', sep='\t').set_index('id').source
    d = d.join(cb[['work', 'byline', 'dynasty', 'c_from', 'c_to']]).join(bl[['kind', 'b_lo', 'b_hi']])
    try:
        cr = pd.read_csv(f'{P}/creator_dates.tsv', sep='\t').dropna(subset=['a_lo'])
        d = d.join(cr.groupby('id').agg(a_lo=('a_lo', 'min'), a_hi=('a_hi', 'max'), creators=('creator', ','.join)))
    except FileNotFoundError:
        d['a_lo'] = d['a_hi'] = np.nan
    d['source'] = src.reindex(d.index)
    d['width'] = d.q90 - d.q10
    # CBETA single years (time_from == time_to) and spans; drop implausible ones (from > to)
    d.loc[d.c_from > d.c_to, ['c_from', 'c_to']] = np.nan
    rows = [stats(d[d.c_from.notna()], 'c_from', 'c_to', 'CBETA years'),
            stats(d[d.c_from.notna() & (d.seen12 <= .2)], 'c_from', 'c_to', 'CBETA years, seen12<=0.2'),
            stats(d[d.kind.eq('stated')], 'b_lo', 'b_hi', 'byline stated'),
            stats(d[d.kind.eq('attached')], 'b_lo', 'b_hi', 'byline attached (今附X錄)')]
    if 'dao_lo' in d:
        rows.append(stats(d[d.dao_lo.notna()], 'dao_lo', 'dao_hi', 'daoinfo'))
    if d.a_lo.notna().any():
        rows.append(stats(d[d.a_lo.notna()], 'a_lo', 'a_hi', 'author life dates (DILA)'))
        rows.append(stats(d[d.a_lo.notna() & d.c_from.isna()], 'a_lo', 'a_hi', 'author life dates, no CBETA years'))
    pd.DataFrame(rows).to_csv(f'{R}/new_dates_validation.tsv', sep='\t', index=False)
    # per-category reliability from every externally dated undated text (CBETA years > byline > author life > daoinfo)
    ext_lo = d.c_from.where(d.c_from.notna(), d.b_lo.where(d.kind.eq('stated'), d.a_lo))
    ext_hi = d.c_to.where(d.c_from.notna(), d.b_hi.where(d.kind.eq('stated'), d.a_hi))
    if 'dao_lo' in d: ext_lo, ext_hi = ext_lo.fillna(d.dao_lo), ext_hi.fillna(d.dao_hi)
    v = d.assign(e_lo=ext_lo, e_hi=ext_hi).dropna(subset=['e_lo'])
    v = v.assign(cover=(v.q10 <= v.e_hi) & (v.q90 >= v.e_lo), off=v['median'] - (v.e_lo + v.e_hi) / 2)
    cat = v.groupby(v.cat.str[:4]).agg(n=('cover', 'size'), cover80=('cover', 'mean'), err=('off', lambda s: s.abs().median()),
                                       bias=('off', 'median'))
    print('\nper-category check on externally dated texts (n >= 10):'); print(cat[cat.n >= 10].round(2).to_string())
    bad = set(cat[(cat.n >= 10) & ((cat.cover80 < .6) | (cat.bias.abs() > 150))].index)
    print('categories flagged unreliable (80% interval covers < 60% of dated texts, or median bias > 150 y):', sorted(bad))
    # Buddhist texts the model places in 600-907: how often are externally dated ones of that kind really post-Tang?
    b = v[v.cat.str.startswith('KR6') & v['median'].between(600, 907)]
    print(f'Buddhist texts dated 600-907 by the model: {len(b)}; externally post-907: {(b.e_lo > 907).mean():.2f}')
    # nobody dates it: no CBETA years or dynasty, no dated/attached byline, no daoinfo statement
    known = d.c_from.notna() | (d.dynasty.notna() & ~d.dynasty.isin(['unknown'])) | d.kind.isin(['stated', 'attached'])
    if 'dao_lo' in d: known |= d.dao_lo.notna()
    known |= d.a_lo.notna()
    new = d[~known].copy()
    # modern scholarly articles printed in 藏外佛教文獻: not texts to date, but a positive control (should come out latest)
    art = new.cat.eq('KR6v') & new.title.astype(str).str.contains(r'《|關於|概貌|價值|新資料|敘錄|研究|若干問題|考$')
    print(f'\npositive control, modern articles in 藏外佛教文獻: n={int(art.sum())}; model medians {sorted(new[art]["median"].tolist())}')
    new = new[~art]
    new['layered'] = new.title.astype(str).str.contains(LAYER)
    new['apocryphal'] = new.cat.eq('KR6u')
    new['unreliable_cat'] = new.cat.str[:4].isin(bad)  # flag found by validation, after the tiers were fixed
    new['tier'] = np.where((new.n_raw >= 2000) & (new.width <= 400) & (new.seen12 <= .2), 'A',
                           np.where((new.width <= 700) & (new.seen12 <= .5), 'B', 'C'))
    cols = ['title', 'cat', 'source', 'work', 'byline', 'n_raw', 'seen12', 'median', 'q25', 'q75', 'q10', 'q90', 'width', 'tier', 'layered', 'apocryphal', 'unreliable_cat']
    new[[c for c in cols if c in new]].sort_values(['tier', 'median']).to_csv(f'{P}/new_dates_v0.tsv', sep='\t')
    print(f'\nnobody dates: {len(new)} texts; tiers {new.tier.value_counts().to_dict()}; layered {int(new.layered.sum())}; apocryphal {int(new.apocryphal.sum())}; '
          f'unreliable category {int(new.unreliable_cat.sum())}; tier A without any flag '
          f'{int(((new.tier == "A") & ~new.layered & ~new.apocryphal & ~new.unreliable_cat).sum())}')
    era = pd.cut(new['median'], [-2000, -221, 220, 589, 907, 1279, 1368, 1644, 2000],
                 labels=['先秦', '秦漢', '魏晉南北朝', '隋唐五代', '宋', '元', '明', '清以後'])
    print(pd.crosstab(era, new.tier).to_string())
    print(new.groupby(new.cat.str[:4]).agg(n=('median', 'size'), A=('tier', lambda s: (s == 'A').sum()), median=('median', 'median'))
          .sort_values('n', ascending=False).head(15).to_string())
    pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 30)
    print(new[new.tier == 'A'].sort_values('median')[['title', 'source', 'n_raw', 'median', 'q10', 'q90', 'layered']].to_string())


if __name__ == '__main__':
    main()
