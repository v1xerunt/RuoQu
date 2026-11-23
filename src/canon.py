"""Why do canonized classics read late?  Character-level decomposition by later reuse.

For every chapter of the classics split by chapters.py (scored by the fold model that held the work out,
score.py --tokens 4), each character gets a reuse intensity: the largest number of later (Song and after,
mid-date >= 960) texts that contain an 8-gram covering it (classes none / df0(4g) = only a reused 4-gram / df1-4 /
df5-19 / df20-99 / df100+). Canonized chapters (大學, 中庸) are compared with the other 禮記 chapters at matched
reuse intensity: if they read late only because they are quoted, the difference vanishes within classes.
Later texts exclude the work's own commentary categories (e.g. 禮記: KR1d, KR1h), so editions and
commentaries that contain the whole text do not count. Training already deletes spans covered by 12-grams first
attested earlier, so reuse shorter than 12 characters is what remains in the training data.
Per character: LLR = log p(x | 1100) - log p(x | -300) (positive = reads like the Song).
Usage: python canon.py <run prefix, e.g. l3>
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, pickle
import numpy as np
import pandas as pd
from corpus import ngram_hash, cover

R = (WORK + '/runs')
D = (WORK + '/data')
YEARS = np.arange(-1000, 1901, 25)[::4]
IA, IB = int(np.argmin(np.abs(YEARS + 300))), int(np.argmin(np.abs(YEARS - 1100)))
BINS = [1, 5, 20, 100]  # 8-gram document frequency in later texts
LAB = {-1: 'none', 0: 'df0(4g)', 1: 'df1-4', 2: 'df5-19', 3: 'df20-99', 4: 'df100+'}
EXCL = {'KR1d0052': {'KR1d', 'KR1h'}, 'KR1h0004': {'KR1h'}, 'KR1a0001': {'KR1a'}, 'KR1h0006': {'KR1h'}}
VER = os.environ.get('LETTER_V', '3')


def chapters(prefix):
    out = []
    for f in range(5):
        run = prefix if f == 0 else f'{prefix}_f{f}'
        p = f'{R}/{run}/scores_chapters_unk_tok.pkl'
        if not os.path.exists(p): continue
        idx = pd.read_csv(f'{D}/corpus{VER}_f{f}/chapters.tsv', sep='\t').set_index('id')
        for e in pickle.load(open(p, 'rb')):
            r = idx.loc[e['id']]
            out.append(dict(id=e['id'], fold=f, work=r.work, title=r.title, chapter=r.chapter, part=r.part,
                            tok=e['tok'].astype(np.int64), llr=(e['lp'][IB] - e['lp'][IA]).astype(np.float64)))
    return out


def later_df(rows, vm, c2i):
    """document frequency, among later texts, of every 4- and 8-gram of the chapters (per work, own categories excluded)."""
    meta = pd.read_csv(f'{D}/meta.tsv', sep='\t')
    late = meta[((meta.lo + meta.hi) / 2 >= 960) & ~meta.foreign.astype(bool)].set_index('id')
    works = sorted({r['work'] for r in rows})
    wcat = meta.set_index('id').cat.to_dict()
    U = {n: np.unique(np.concatenate([ngram_hash(r['tok'], n) for r in rows])) for n in (4, 8)}
    df = {(w, n): np.zeros(len(U[n]), np.int32) for w in works for n in (4, 8)}
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']:
        for line in open(f'{D}/extract/{part}.jsonl'):
            r = json.loads(line)
            if r['id'] not in late.index or not r.get('juans'): continue
            cat = late.at[r['id'], 'cat']
            t = np.array([c2i.get(vm.get(c, c), 2) for c in ''.join(r['juans'])], np.int64)
            for n in (4, 8):
                h = ngram_hash(t, n); j = np.searchsorted(U[n], h); j[j >= len(U[n])] = 0
                hit = np.unique(j[U[n][j] == h])
                for w in works:
                    if cat not in EXCL.get(w, {wcat.get(w)}) and r['id'] != w: df[(w, n)][hit] += 1
    return U, df


def main(prefix):
    vocab = json.load(open(f'{D}/corpus{VER}_f0/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    vm = json.load(open(f'{D}/varmap{VER}.json'))
    rows = chapters(prefix)
    print(len(rows), 'chapters from', sorted({r['work'] for r in rows}))
    U, df = later_df(rows, vm, c2i)
    out = []
    for r in rows:
        t = r['tok']; m = t > 1
        # reuse intensity of each character: max number of later texts containing an 8-gram that covers it
        h = ngram_hash(t, 8); j = np.searchsorted(U[8], h); j[j >= len(U[8])] = 0
        d = np.where(U[8][j] == h, df[(r['work'], 8)][j], 0)
        inten = np.zeros(len(t), np.int64)
        for k in range(8):
            if len(d): np.maximum.at(inten, np.arange(len(d)) + k, d)
        b4 = ngram_hash(t, 4); j4 = np.searchsorted(U[4], b4); j4[j4 >= len(U[4])] = 0
        d4 = np.where(U[4][j4] == b4, df[(r['work'], 4)][j4], 0)
        i4 = np.zeros(len(t), np.int64)
        for k in range(4):
            if len(d4): np.maximum.at(i4, np.arange(len(d4)) + k, d4)
        bins = np.digitize(inten, BINS)
        bins[(inten == 0) & (i4 == 0)] = -1  # not even a reused 4-gram
        for k in np.unique(bins[m]):
            mk = m & (bins == k)
            out.append(dict(work=r['work'], title=r['title'], chapter=r['chapter'], part=r['part'], cls=LAB[k],
                            n=int(mk.sum()), llr=float(r['llr'][mk].sum())))
    o = pd.DataFrame(out)
    o.to_csv(f'{R}/{prefix}_canon.tsv', sep='\t', index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 300)
    g = o.groupby(['title', 'chapter', 'cls']).agg(n=('n', 'sum'), llr=('llr', 'sum')).reset_index()
    g['per_char'] = g.llr / g.n.clip(lower=1)
    p = g.pivot_table(index=['title', 'chapter'], columns='cls', values=['per_char', 'n'])
    tot = o.groupby(['title', 'chapter']).agg(n=('n', 'sum'), llr=('llr', 'sum'))
    p[('per_char', 'all')] = tot.llr / tot.n
    print(p.round(3).to_string())
    w = o[o.work == 'KR1d0052'].copy(); w['canon'] = w.chapter.isin(['大學', '中庸'])
    t = w.groupby(['cls', 'canon']).agg(n=('n', 'sum'), llr=('llr', 'sum'))
    t['per_char'] = t.llr / t.n
    print('禮記: per-character LLR(1100 vs -300), canonized chapters (大學, 中庸) vs the rest, by reuse intensity')
    print(t.round(4).unstack().to_string())


if __name__ == '__main__':
    main(sys.argv[1])
