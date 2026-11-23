"""Controlled anachronism test (E8a).

Take held-out passages from texts dated before 900 CE and plant one word into each:
  late    : a character bigram/trigram first attested (in >= 3 training texts) after 1000 CE and common later
  control : a bigram/trigram attested before 400 CE and common throughout
The anachronism map should rank planted late words at the top, and not planted controls.
Writes corpus_f0/synth.{bin,tsv} and synth_meta.json; score with score.py --split synth --tokens 1.
"""
import json, os
import numpy as np
import pandas as pd
from baselines import nhash, bin_of, MIDS, load, C, SEP
from views import FUNC

rng = np.random.default_rng(0)
P = 256


def first_attest(n, min_texts=3):
    data, idx = load('train')
    H, B = [], []
    for i in range(len(idx)):
        t = np.asarray(data[idx.offset[i]:idx.offset[i] + idx.length[i]])
        h, ok = nhash(t, n); h = np.unique(h[ok])
        H.append(h); B.append(np.full(len(h), bin_of(idx.lo[i]), np.int16))
    H = np.concatenate(H); B = np.concatenate(B)
    o = np.lexsort((B, H)); H, B = H[o], B[o]
    st = np.concatenate([[0], np.flatnonzero(H[1:] != H[:-1]) + 1]); cnt = np.diff(np.concatenate([st, [len(H)]]))
    ok = cnt >= min_texts
    return H[st][ok], B[np.minimum(st + min_texts - 1, len(H) - 1)][ok], cnt[ok]


def main():
    data, idx = load('train')
    vocab = json.load(open(f'{C}/vocab.json'))
    # candidate words: count n-gram occurrences in a sample of training text, keep frequent ones
    words = {}
    for n in (2, 3):
        u, fb, ntext = first_attest(n)
        sample = np.concatenate([np.asarray(data[o:o + min(l, 20000)]) for o, l in zip(idx.offset, idx.length)])
        h, ok = nhash(sample, n)
        hu, hc = np.unique(h[ok], return_counts=True)
        # map hash -> token sequence (first occurrence)
        pos = np.flatnonzero(ok)
        first_pos = {}
        hh = h[pos]
        order = np.argsort(hh, kind='stable'); hs = hh[order]
        st = np.concatenate([[0], np.flatnonzero(hs[1:] != hs[:-1]) + 1])
        for a in st:
            first_pos[int(hs[a])] = pos[order[a]]
        j = np.searchsorted(u, hu); j[j >= len(u)] = 0; m = u[j] == hu
        for hv, c, b, nt in zip(hu[m], hc[m], fb[j[m]], ntext[j[m]]):
            if c < 200: continue
            p0 = first_pos[int(hv)]
            words[tuple(sample[p0:p0 + n].tolist())] = (int(b), int(c), int(nt))
    late = [w for w, (b, c, nt) in words.items() if MIDS[b] >= 1000 and nt >= 20]
    isf = lambda w: all(vocab[t] in FUNC for t in w)
    late_f = [w for w, (b, c, nt) in words.items() if MIDS[b] >= 800 and nt >= 20 and isf(w)]
    ctrl_f = [w for w, (b, c, nt) in words.items() if MIDS[b] <= 400 and nt >= 200 and isf(w)]
    print('late function-word n-grams', len(late_f), [''.join(vocab[t] for t in w) for w in late_f[:80]])
    ctrl = [w for w, (b, c, nt) in words.items() if MIDS[b] <= 400 and nt >= 200]
    print('late words', len(late), 'control words', len(ctrl))
    print('late examples', [''.join(vocab[t] for t in w) for w in late[:60]])
    held = pd.read_csv(f'{C}/heldout.tsv', sep='\t')
    held = held[(held.lo + held.hi) / 2 < 900]
    hd = np.memmap(f'{C}/heldout.bin', dtype=np.uint16, mode='r')
    arrs, rows, meta, off = [], [], [], 0
    for _, r in held.iterrows():
        tok = np.asarray(hd[r.offset:r.offset + r.length])
        n = len(tok) // P
        for k in rng.choice(n, size=min(n, 6), replace=False) if n else []:
            p = tok[k * P:(k + 1) * P].copy()
            if (p == SEP).sum() > 4: continue
            for kind, pool in (('orig', None), ('late', late), ('ctrl', ctrl), ('late_func', late_f), ('ctrl_func', ctrl_f)):
                q = p.copy(); w = None; at = -1
                if pool is not None:
                    w = pool[rng.integers(len(pool))]; at = int(rng.integers(20, P - 20 - len(w)))
                    q[at:at + len(w)] = w
                arrs.append(q)
                rows.append(dict(id=f'{r.id}#{k}#{kind}', title=r.title, lo=r.lo, hi=r.hi, genre=r.genre,
                                 offset=off, length=P))
                meta.append(dict(id=f'{r.id}#{k}#{kind}', kind=kind, at=at, n=len(w) if w else 0,
                                 word=''.join(vocab[t] for t in w) if w else ''))
                off += P
    np.concatenate(arrs).astype(np.uint16).tofile(f'{C}/synth.bin')
    pd.DataFrame(rows).to_csv(f'{C}/synth.tsv', sep='\t', index=False)
    json.dump(meta, open(f'{C}/synth_meta.json', 'w'), ensure_ascii=False)
    print('synth passages', len(rows))


if __name__ == '__main__':
    main()
