"""Character-level anatomy of the Shangshu chapters: stitched quotations vs the forger's own filler.

Each character is 'shared' if it lies inside an 8-gram also found in a text dated before 220 CE (other than
the Shangshu and its commentaries), else 'own'. Per-character LLR = log p(x | 300 CE) - log p(x | -400)
(positive = reads like the Eastern Jin). Compares guwen (forged) and jinwen chapters, shared vs own.
Usage: python shangshu_anatomy.py <run> [corpus] [varmap version]
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, json, pickle
import numpy as np
import pandas as pd
from liezi import early_hashes
from corpus import ngram_hash, cover

R = (WORK + '/runs')
D = (WORK + '/data')
YEARS = np.arange(-1000, 1901, 25)[::4]


def main(run, corpus='corpus_f0', v=''):
    vocab = json.load(open(f'{D}/{corpus}/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    vm = json.load(open(f'{D}/varmap{v}.json'))
    meta = pd.read_csv(f'{D}/meta.tsv', sep='\t')
    excl = set(meta[meta.cat == 'KR1b'].id)  # all Shangshu editions and commentaries
    H, n = early_hashes(c2i, vm, exclude=excl)
    print('early texts', n)
    idx = pd.read_csv(f'{D}/{corpus}/blind_full.tsv', sep='\t').set_index('id')
    ss = idx[idx.date_src == 'shangshu']
    T = [e for e in pickle.load(open(f'{R}/{run}/scores_blind_full_unk_tok.pkl', 'rb')) if e['id'] in ss.index]
    ia, ib = int(np.argmin(np.abs(YEARS + 400))), int(np.argmin(np.abs(YEARS - 300)))
    rows = []
    for e in T:
        tok = e['tok'].astype(np.int64); lp = e['lp'].astype(np.float64)
        m = tok > 1
        h = ngram_hash(tok, 8); j = np.searchsorted(H, h); j[j >= len(H)] = 0
        sh = cover(H[j] == h, 8, len(tok)) & m
        llr = lp[ib] - lp[ia]
        for kind, mask in (('shared', sh), ('own', m & ~sh)):
            if mask.sum():
                rows.append(dict(id=e['id'], cls=ss.loc[e['id'], 'author'], title=ss.loc[e['id'], 'title'], kind=kind,
                                 n=int(mask.sum()), llr_sum=llr[mask].sum()))
    df = pd.DataFrame(rows)
    g = df.groupby(['cls', 'kind']).agg(chars=('n', 'sum'), llr=('llr_sum', 'sum'))
    g['per_char'] = g.llr / g.chars
    print(g.round(4).to_string())
    # per chapter, own-character LLR, guwen vs jinwen
    ch = df[df.kind == 'own'].groupby(['id', 'cls']).agg(n=('n', 'sum'), s=('llr_sum', 'sum')).reset_index()
    ch['per_char'] = ch.s / ch.n
    from scipy.stats import mannwhitneyu
    a, b = ch[ch.cls == 'guwen'].per_char, ch[ch.cls != 'guwen'].per_char
    print(f'own characters, per chapter: guwen median {a.median():+.4f} vs jinwen {b.median():+.4f}, one-sided p={mannwhitneyu(a, b, alternative="greater").pvalue:.2g}')
    sh_ = df[df.kind == 'shared']
    frac = df.pivot_table(index='id', columns='kind', values='n', aggfunc='sum').fillna(0)
    frac['shared_frac'] = frac.get('shared', 0) / frac.sum(1)
    frac['cls'] = ss.loc[frac.index, 'author']
    print('fraction of characters shared with pre-220 texts:', frac.groupby('cls').shared_frac.median().round(3).to_dict())
    df.to_csv(f'{R}/{run}/shangshu_anatomy.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main(*sys.argv[1:])
