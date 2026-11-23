"""Audit of the canon with cross-fitted predictions (each text dated by the fold model that never saw it).

Flags works whose language is later than the catalog date allows: terminus post quem (q10 of the
posterior) later than the label's upper bound. Ranks by posterior mass after the label interval.
Usage: python audit.py <run prefix, e.g. tc_m> (expects runs <prefix>_f0..f4, or <prefix> for fold 0)
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, os
import numpy as np
import pandas as pd
from evaluate import load_scores, text_scores, posterior, quantile, fit_tau, GRID, interval_mass

D = (WORK + '/data')
R = (WORK + '/runs')


CP = __import__('os').environ.get('LETTER_CPREFIX', 'corpus_f')


def fold_table(run, fold):
    meta, ll = load_scores(f'{R}/{run}/scores_heldout_unk.npz')
    ids, S, n = text_scores(meta, ll)
    idx = pd.read_csv(f'{D}/{CP}{fold}/heldout.tsv', sep='\t').set_index('id')
    lo, hi = idx.loc[ids, 'lo'].values.astype(float), idx.loc[ids, 'hi'].values.astype(float)
    tau = fit_tau(S, lo, hi)
    p = posterior(S, tau)
    df = idx.loc[ids, ['title', 'author', 'cat', 'lo', 'hi', 'date_src']].copy()
    df['fold'] = fold; df['n_pass'] = n
    df['median'] = quantile(p, 0.5); df['q10'] = quantile(p, 0.1); df['q90'] = quantile(p, 0.9)
    df['mass_label'] = interval_mass(p, lo, hi)
    df['mass_later'] = (p * (GRID[None] > hi[:, None] + 12.5)).sum(1)
    df['mass_earlier'] = (p * (GRID[None] < lo[:, None] - 12.5)).sum(1)
    return df


if __name__ == '__main__':
    prefix = sys.argv[1]
    tabs = []
    for f in range(5):
        run = f'{prefix}_f{f}' if os.path.exists(f'{R}/{prefix}_f{f}') else (prefix if f == 0 else None)
        if run and os.path.exists(f'{R}/{run}/scores_heldout_unk.npz'):
            tabs.append(fold_table(run, f))
    df = pd.concat(tabs)
    df['flag_late'] = df.q10 > df.hi + 50
    df['flag_early'] = df.q90 < df.lo - 50
    df.to_csv(f'{R}/{prefix}_audit.tsv', sep='\t')
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    print(len(df), 'texts;', df.flag_late.sum(), 'flagged late,', df.flag_early.sum(), 'flagged early')
    cols = ['title', 'author', 'lo', 'hi', 'date_src', 'q10', 'median', 'q90', 'mass_later', 'n_pass']
    print(df[df.flag_late].sort_values('mass_later', ascending=False)[cols].head(80).to_string())
