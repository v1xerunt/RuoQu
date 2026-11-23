"""First attestation ('首見書證') of the words in flagged texts: an automated version of the philologist's check.

For every 2- and 3-character n-gram of a target text, find the earliest catalogue date (label lower bound, v3 labels)
of any OTHER dated text in the corpus that contains it, and how many dated texts contain it. An n-gram is
'late' for the target if its earliest other attestation is >= the target's catalogue upper bound + GAP years and it
occurs in >= MINDF dated texts (a real word, not a chance juxtaposition). Rate = late n-grams per 1,000 characters.
Because early periods have less surviving text, the same rate is computed for control texts of similar date that the
audit did not flag; a flagged text is notable only if its rate is far above controls of its period.
Main statistic: common two-character words (in >= 10 dated texts) first attested late, per 1,000 characters.
Usage: python firstattest.py [n_controls=300]
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json
import numpy as np
import pandas as pd
from corpus import ngram_hash

R = (WORK + '/runs')
D = (WORK + '/data')
P = (PROJECT + '/data')
GAP, MINDF, NS = 100, 3, (2, 3)
BIG = 10 ** 6


def text_of(r, vm):
    return ''.join(vm.get(c, c) for c in ''.join(r['juans']))


def main(n_controls=300):
    vocab = json.load(open(f'{D}/corpus3_f0/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    vm = json.load(open(f'{D}/varmap3.json'))
    lab = pd.concat([pd.read_csv(f'{D}/corpus3_f{f}/heldout.tsv', sep='\t', usecols=['id', 'title', 'author', 'cat', 'lo', 'hi'])
                     for f in range(5)]).drop_duplicates('id').set_index('id')
    lab = lab[(lab.hi - lab.lo) <= 150]
    a = pd.read_csv(f'{R}/l3_audit.tsv', sep='\t').set_index('id')
    targets = list(a[a.flag_late].index)
    rev = pd.read_csv(f'{P}/audit_cases_v1.tsv', sep='\t')
    targets += [i for i in rev.id if i not in targets and i in a.index]
    rng = np.random.default_rng(0)
    pool = a[~a.flag_late & ~a.flag_early & a.index.isin(lab.index)]
    controls = list(rng.choice(pool.index, min(int(n_controls), len(pool)), replace=False))
    want = set(targets) | set(controls)
    toks, H = {}, {n: [] for n in NS}
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']:
        for line in open(f'{D}/extract/{part}.jsonl'):
            r = json.loads(line)
            if r['id'] in want and r.get('juans'):
                t = np.array([c2i.get(c, 2) for c in text_of(r, vm)], np.int64)
                toks[r['id']] = t
                for n in NS: H[n].append(np.unique(ngram_hash(t, n)))
    U = {n: np.unique(np.concatenate(H[n])) for n in NS}
    print('targets', len(targets), 'controls', len(controls), {n: len(U[n]) for n in NS}, flush=True)
    m1 = {n: np.full(len(U[n]), BIG, np.int32) for n in NS}; i1 = {n: np.full(len(U[n]), -1, np.int32) for n in NS}
    m2 = {n: np.full(len(U[n]), BIG, np.int32) for n in NS}
    df = {n: np.zeros(len(U[n]), np.int32) for n in NS}
    ids = list(lab.index); pos = {k: j for j, k in enumerate(ids)}
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']:
        for line in open(f'{D}/extract/{part}.jsonl'):
            r = json.loads(line)
            if r['id'] not in pos or not r.get('juans'): continue
            lo = int(lab.at[r['id'], 'lo']); me = pos[r['id']]
            t = toks.get(r['id'])
            if t is None: t = np.array([c2i.get(c, 2) for c in text_of(r, vm)], np.int64)
            for n in NS:
                h = ngram_hash(t, n); j = np.searchsorted(U[n], h); j[j >= len(U[n])] = 0
                u = np.unique(j[U[n][j] == h])
                df[n][u] += 1
                new1 = lo < m1[n][u]
                k = u[new1]; m2[n][k] = m1[n][k]; m1[n][k] = lo; i1[n][k] = me
                k = u[~new1]; ok = lo < m2[n][k]; m2[n][k[ok]] = lo
    rows = []
    for tid in targets + controls:
        if tid not in toks: continue
        t = toks[tid]; me = pos.get(tid, -2); hi = a.at[tid, 'hi']
        late, nlate, nw = [], 0, 0
        for n in NS:
            h = ngram_hash(t, n); j = np.searchsorted(U[n], h)
            first = np.where(i1[n][j] == me, m2[n][j], m1[n][j])
            # df counts the target itself when it is a dated text
            d = df[n][j] - (1 if tid in pos else 0)
            bad = (first >= hi + GAP) & (first < BIG) & (d >= MINDF)
            nlate += int(bad.sum())
            if n == 2: nw += int((bad & (d >= 10)).sum())  # common two-character words first attested late
            if tid in targets:
                for s in np.flatnonzero(bad)[:4000]:
                    late.append((''.join(vocab[x] for x in t[s:s + n]), int(first[s]), int(d[s])))
        seen = {}
        for w, f, d in late:
            if w not in seen: seen[w] = (f, d)
        top = sorted([kv for kv in seen.items() if len(kv[0]) == 2], key=lambda kv: -kv[1][1])[:30]
        rows.append(dict(id=tid, kind='target' if tid in targets else 'control', title=a.at[tid, 'title'], lo=a.at[tid, 'lo'],
                         hi=hi, median=a.at[tid, 'median'], n=len(t), late_per_k=1000 * nlate / max(len(t), 1),
                         words_per_k=1000 * nw / max(len(t), 1),
                         words=' '.join(f'{w}({f},{d})' for w, (f, d) in top)))
    o = pd.DataFrame(rows)
    o['period'] = pd.cut(o.hi, [-2000, 220, 589, 907, 1127, 1279, 1368, 1644, 2000],
                         labels=['漢以前', '魏晉南北朝', '隋唐', '五代北宋', '南宋', '元', '明', '清'])
    base = o[o.kind == 'control'].groupby('period', observed=True).words_per_k.agg(['median', lambda v: v.quantile(.9), 'size'])
    base.columns = ['ctrl_median', 'ctrl_q90', 'ctrl_n']
    o = o.join(base, on='period')
    o['ratio'] = o.words_per_k / o.ctrl_median.clip(lower=0.05)
    o['pct'] = [(o[(o.kind == 'control') & (o.period == p_)].words_per_k < v).mean() for p_, v in zip(o.period, o.words_per_k)]
    o.to_csv(f'{R}/l3_firstattest.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 160); pd.set_option('display.max_rows', 300)
    print(base.round(2).to_string())
    t = o[o.kind == 'target'].sort_values('ratio', ascending=False)
    print(t[['id', 'title', 'lo', 'hi', 'median', 'n', 'late_per_k', 'words_per_k', 'ctrl_median', 'ctrl_q90', 'ratio', 'pct']].round(2).to_string(index=False))
    for r in t.itertuples():
        print(f'## {r.title} ({r.lo:.0f}~{r.hi:.0f}; model {r.median:.0f}; common late words {r.words_per_k:.2f}/k vs controls {r.ctrl_median:.2f}, '
              f'percentile {r.pct:.2f}): {r.words}')


if __name__ == '__main__':
    main(*sys.argv[1:])
