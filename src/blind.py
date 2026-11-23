"""Blind test (E7) and archaizing test (E4): date posteriors vs attributed and consensus dates.

tau (posterior temperature) comes from the E1 held-out calibration of the same model.
For each work: posterior median, 80% interval, terminus post quem (10% quantile), posterior mass on
the consensus vs the attributed interval, and the verdict (which one the model supports).
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, json
import numpy as np
import pandas as pd
from evaluate import load_scores, text_scores, posterior, quantile, interval_mass, GRID, evaluate

P = (PROJECT + '/data')
C = (WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
SHANG = {'guwen': ('偽古文尚書', 'A', -1600, -700, 250, 322),
         'jinwen': ('今文尚書(虞夏商及其他)', 'C', -1600, -700, -1100, -250),
         'zhougao': ('今文尚書·周初誥命', 'C', -1046, -950, -1046, -900)}


def work_table(split, path, tau):
    meta, ll = load_scores(path)
    ids, S, n = text_scores(meta, ll)
    idx = pd.read_csv(f'{C}/{split}.tsv', sep='\t').set_index('id')
    S = pd.DataFrame(S, index=ids)
    bl = pd.read_csv(f'{P}/blind_v0.tsv', sep='\t')
    groups = []
    if split.startswith('blind'):
        for g, sub in bl[bl['cat'].isin(['A', 'B', 'D'])].groupby('group', sort=False):
            members = [i for i in sub.id if i in S.index]
            if not members: continue
            r = sub.iloc[0]
            groups.append(dict(work=g, cat=r['cat'], members=members, attr=(float(r.attr_lo), float(r.attr_hi)),
                               cons=(float(r.cons_lo), float(r.cons_hi)), conf=r['conf']))
        ss = idx[idx.date_src == 'shangshu']
        for cls, (name, cat, alo, ahi, clo, chi) in SHANG.items():
            members = [i for i in ss.index[ss.author == cls] if i in S.index]
            groups.append(dict(work=name, cat=cat, members=members, attr=(alo, ahi), cons=(clo, chi), conf='high', agg='sum'))
    else:
        E4 = {'KR3a0009': ('法言', -479, -380), 'KR3g0001': ('太玄', -900, -250), 'KR5f0017': ('太玄', -900, -250),
              'KR3a0014': ('中說', -479, -380), 'KR3g0004': ('潛虛', -33, 18), 'KR3g0003': ('元包', -1100, -700)}
        for i in S.index:
            w, ilo, ihi = E4[i]
            groups.append(dict(work=w + '/' + i, cat='E4', members=[i], attr=(ilo, ihi),
                               cons=(float(idx.loc[i, 'lo']), float(idx.loc[i, 'hi'])), conf='secure'))
    rows = []
    for gr in [g for g in groups if g['members']]:
        s = S.loc[gr['members']].values.sum(0, keepdims=True)
        # each member text is a copy/edition: average rather than sum to avoid double counting
        if gr.get('agg') != 'sum':
            s = s / max(1, len(gr['members']))
        p = posterior(s, tau)
        alo, ahi = gr['attr']; clo, chi = gr['cons']
        m_c = interval_mass(p, np.array([clo]), np.array([chi]))[0]
        m_a = interval_mass(p, np.array([alo]), np.array([ahi]))[0]
        med = quantile(p, 0.5)[0]
        p_later = float((p * (GRID[None] > ahi + 12.5)).sum())  # P(written after the attributed date)
        rows.append(dict(work=gr['work'], cat=gr['cat'], conf=gr['conf'], n_text=len(gr['members']), p_later=p_later,
                         n_pass=int(sum(len(meta[meta.id == i]) for i in gr['members'])),
                         attr=f'{alo:.0f}~{ahi:.0f}', cons=f'{clo:.0f}~{chi:.0f}',
                         median=med, q10=quantile(p, 0.1)[0], q90=quantile(p, 0.9)[0],
                         mass_cons=m_c, mass_attr=m_a,
                         d_cons=max(0, clo - med, med - chi), d_attr=max(0, alo - med, med - ahi),
                         verdict='cons' if max(0, clo - med, med - chi) < max(0, alo - med, med - ahi) else 'attr'))
    out = pd.DataFrame(rows)
    if split.startswith('blind'):  # per-chapter table for the Shangshu
        ss = idx[idx.date_src == 'shangshu']
        ch = [i for i in ss.index if i in S.index]
        if ch:
            p = posterior(S.loc[ch].values, tau)
            out.attrs['shangshu'] = pd.DataFrame({'title': ss.loc[ch, 'title'].values, 'cls': ss.loc[ch, 'author'].values,
                                                  'n_char': ss.loc[ch, 'n_raw'].values, 'median': quantile(p, 0.5),
                                                  'q10': quantile(p, 0.1), 'q90': quantile(p, 0.9)}, index=ch)
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return np.nan
    return ((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def summarize(df):
    """A: forgery (consensus later than attribution). B: once suspected as late, vindicated as early.
    The model sides with the consensus on A if P(later than attributed) > .5, on B if P(later than the
    suspected date) < .5. AUC compares the two classes on that same score."""
    a = df[df.cat == 'A']; b = df[df.cat == 'B']
    print(f'A sides with consensus: {(a.p_later > .5).sum()}/{len(a)}   B sides with consensus: {(b.p_later < .5).sum()}/{len(b)}   '
          f'AUC(A vs B) {auc(a.p_later, b.p_later):.3f}   A: median date - consensus mid {np.median(a["median"] - a.cons.map(lambda c: np.mean([float(x) for x in c.split("~")]))):.0f}y')


if __name__ == '__main__':
    name = sys.argv[1]; genre = sys.argv[2] if len(sys.argv) > 2 else 'unk'
    R = (WORK + '/runs')
    BL = 'baselines' + ('' if C.endswith('corpus_f0') else '_' + C.split('/')[-1])
    path = (lambda s: f'{R}/{BL}/{name[3:]}_{s}.npz') if name.startswith('bl:') else (lambda s: f'{R}/{name}/scores_{s}_{genre}.npz')
    _, cal = evaluate(path('heldout'))
    tau = float(np.exp(np.mean(np.log(cal['taus']))))
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    for split in ['blind_full', 'blind_res', 'e4_full', 'e4_res']:
        try:
            df = work_table(split, path(split), tau)
        except FileNotFoundError:
            continue
        print(f'== {split} (tau={tau:.4f})')
        print(df.round(2).to_string(index=False))
        if split.startswith('blind'):
            summarize(df)
            if 'shangshu' in df.attrs:
                t = df.attrs['shangshu']
                print(t.groupby('cls')[['median', 'q10']].median().round(0).to_string())
                from scipy.stats import mannwhitneyu
                g, j = t[t.cls == 'guwen']['median'], t[t.cls != 'guwen']['median']
                print(f'Shangshu chapters: guwen median {g.median():.0f} vs jinwen {j.median():.0f};  AUC(guwen later) {auc(g, j):.3f}  p={mannwhitneyu(g, j, alternative="greater").pvalue:.2g}')
