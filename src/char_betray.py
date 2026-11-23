"""Which characters betray the imitator's own era?  Humans vs AI, and the anatomy of the AI's default verse.

Per character in an imitation: llr = log p(x | own era) - log p(x | target era) (100-year token grid), own era =
author date (humans) or the model's own default date (AI controls, by model and form class).
Aggregated by character: mean llr and count; top 'betraying' characters (high mean llr, count >= MINC), split into
function words and content characters. Then the AI default verse: per-character llr(1225 vs 750) in AI control verse
vs genuine Tang verse (target-era genuine pieces 745-850): characters that make AI verse read Southern Song.
Usage: python char_betray.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, pickle
import numpy as np
import pandas as pd
from views import FUNC

R = (WORK + '/runs')
D = (WORK + '/data')
YEARS = np.arange(-1000, 1901, 25)[::4].astype(float)
MINC = 30
yi = lambda y: int(np.argmin(np.abs(YEARS - max(float(y), -600))))


def acc(entries, own, tgt, S, N):
    for e in entries:
        t = e['tok'].astype(int); lp = e['lp'][:len(YEARS)].astype(np.float64); m = t > 1
        l = lp[yi(own)] - lp[yi(tgt)]
        np.add.at(S, t[m], l[m]); np.add.at(N, t[m], 1)


def table(S, N, vocab, isf, k=25):
    mean = np.where(N >= MINC, S / np.maximum(N, 1), np.nan)
    df = pd.DataFrame({'char': vocab[:len(S)], 'n': N, 'mean': mean, 'func': isf[:len(S)]}).dropna()
    out = {}
    for f, name in ((False, 'content'), (True, 'function')):
        d = df[df.func == f].sort_values('mean', ascending=False)
        out[name] = ' '.join(f'{r.char}{r.mean:+.2f}' for r in d.head(k).itertuples())
    return out, df


def main():
    vocab = json.load(open(f'{D}/corpus3_f0/vocab.json')) + ['X']
    V = len(vocab); isf = np.array([c in FUNC for c in vocab])
    # humans: imitations, each scored by the fold model that held its collection out
    im = pd.read_csv(f'{R}/l3_imit_robust.tsv', sep='\t')
    S, N = np.zeros(V), np.zeros(V)
    for f in range(5):
        run = 'l3' if f == 0 else f'l3_f{f}'
        sub = im[im.fold == f].set_index('id')
        for e in pickle.load(open(f'{R}/{run}/scores_imit_unk_tok.pkl', 'rb')):
            if e['id'] in sub.index:
                r = sub.loc[e['id']]; acc([e], r.year, r.target, S, N)
    h, hdf = table(S, N, vocab, isf)
    print('== HUMAN imitations: characters that most betray the author\'s own era (mean llr own vs target)')
    print('content :', h['content']); print('function:', h['function'])
    # AI: pieces scored by fold model l3 (not in any training set)
    ai = pd.read_csv(f'{R}/aiimit_pieces.tsv', sep='\t').set_index('id')
    S2, N2 = np.zeros(V), np.zeros(V)
    T = pickle.load(open(f'{R}/l3/scores_aiimit_unk_tok.pkl', 'rb'))
    for e in T:
        if e['id'] in ai.index:
            r = ai.loc[e['id']]
            if np.isfinite(r.own) and np.isfinite(r.target): acc([e], r.own, r.target, S2, N2)
    a, adf = table(S2, N2, vocab, isf)
    print('\n== AI imitations (six models pooled): characters that most betray the AI\'s own default era')
    print('content :', a['content']); print('function:', a['function'])
    j = hdf.merge(adf, on=['char', 'func'], suffixes=('_h', '_a'))
    print(f'\nshared characters (n>={MINC} in both): {len(j)}; Spearman of mean llr humans vs AI: '
          f'{j[["mean_h", "mean_a"]].corr("spearman").iloc[0, 1]:.2f}')
    for f, name in ((False, 'content'), (True, 'function')):
        jj = j[j.func == f]
        print(f'  {name}: mean llr humans {np.average(jj.mean_h, weights=jj.n_h):+.3f}, AI {np.average(jj.mean_a, weights=jj.n_a):+.3f}')
    # anatomy of the AI default verse: llr(1225 vs 750) in AI control verse vs genuine Tang verse
    ctrl_ids = set(pd.read_csv(f'{D}/corpus3_f0/aiimit.tsv', sep='\t').query("kind == 'ctrl' and gclass == 'verse'").id)
    S3, N3 = np.zeros(V), np.zeros(V)
    for e in T:
        if e['id'] in ctrl_ids: acc([e], 1225, 750, S3, N3)
    S4, N4 = np.zeros(V), np.zeros(V)  # genuine Tang verse: ordinary pieces of Tang collections (imit split controls)
    for f in range(5):
        run = 'l3' if f == 0 else f'l3_f{f}'
        g = pd.read_csv(f'{D}/corpus3_f{f}/imit.tsv', sep='\t')
        g = g[(g.kind == 'ctrl') & (g.gclass == 'verse') & ((g.lo + g.hi) / 2).between(700, 900)]
        ids = set(g.id)
        for e in pickle.load(open(f'{R}/{run}/scores_imit_unk_tok.pkl', 'rb')):
            if e['id'] in ids: acc([e], 1225, 750, S4, N4)
    print('Tang verse characters:', int(N4.sum()))
    tot = lambda S_, N_: S_.sum() / max(N_.sum(), 1)
    print(f'\n== AI default verse vs genuine Tang verse: mean per-character llr(1225 vs 750): AI {tot(S3, N3):+.4f}, Tang {tot(S4, N4):+.4f}')
    m3 = np.where(N3 >= 20, S3 / np.maximum(N3, 1), np.nan); m4 = np.where(N4 >= 20, S4 / np.maximum(N4, 1), np.nan)
    d = pd.DataFrame({'char': vocab[:V], 'ai_n': N3, 'ai': m3, 'tang_n': N4, 'tang': m4, 'func': isf})
    # contribution to the AI-Tang gap: frequency difference x late-leaning llr
    fa, ft = N3 / max(N3.sum(), 1), N4 / max(N4.sum(), 1)
    d['contrib'] = fa * np.nan_to_num(m3) - ft * np.nan_to_num(m4)
    d = d[(d.ai_n >= 20)]
    print('characters contributing most to the Song-like reading of AI verse (char: AI freq per 1k / Tang freq per 1k, llr):')
    print(' '.join(f'{r.char}({1000 * fa[vocab.index(r.char)]:.1f}/{1000 * ft[vocab.index(r.char)]:.1f},{r.ai:+.2f})'
                   for r in d.sort_values('contrib', ascending=False).head(40).itertuples()))
    print('characters AI over-uses relative to Tang verse (ratio of frequencies, AI count >= 20):')
    d['ratio'] = (fa[d.index] + 1e-5) / (ft[d.index] + 1e-5)
    print(' '.join(f'{r.char}×{r.ratio:.1f}' for r in d.sort_values('ratio', ascending=False).head(40).itertuples()))


if __name__ == '__main__':
    main()
