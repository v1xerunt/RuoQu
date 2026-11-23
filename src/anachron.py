"""Anachronism maps (E8) and evidence decomposition (E5).

For a work with attributed date a and candidate (later) date c, each character gets
    llr_i = log p(x_i | x_<i, c) - log p(x_i | x_<i, a)
Positive values are evidence that the character was written at c rather than a.
We report (1) the top-scoring short spans with context, (2) how the total evidence splits between
function words and content characters, normalized per character.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, json, pickle
import numpy as np
import pandas as pd
from views import FUNC

C = (WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
R = (WORK + '/runs')
GRID = np.arange(-1000, 1901, 25).astype(float)


def load_tok(name, split, genre='unk', k=4):
    T = pickle.load(open(f'{R}/{name}/scores_{split}_{genre}_tok.pkl', 'rb'))
    years = GRID[list(range(0, len(GRID), k))]
    return T, years


def gi(years, y):
    return int(np.argmin(np.abs(years - y)))


def spans(llr, tok, vocab, width=(1, 2, 3, 4), top=15, ctx=6):
    out = []
    cs = np.concatenate([[0], np.cumsum(llr)])
    for w in width:
        s = cs[w:] - cs[:-w]
        for i in np.argsort(-s)[:top * 2]:
            out.append((s[i] / w ** 0.5, i, w))
    out.sort(reverse=True)
    seen, res = set(), []
    for sc, i, w in out:
        if any(j in seen for j in range(i, i + w)): continue
        seen.update(range(i, i + w))
        a, b = max(0, i - ctx), min(len(tok), i + w + ctx)
        s = ''.join(vocab[t] for t in tok[a:i]) + '【' + ''.join(vocab[t] for t in tok[i:i + w]) + '】' + ''.join(vocab[t] for t in tok[i + w:b])
        res.append((round(float(llr[i:i + w].sum()), 2), s))
        if len(res) >= top: break
    return res


def decompose(T, ids, years, a, c, vocab):
    """total and per-character evidence by class, over all passages of the given texts"""
    ia, ic = gi(years, a), gi(years, c)
    tot = {'func': [], 'content': []}
    for e in T:
        if e['id'] not in ids: continue
        tok = e['tok'].astype(int); lp = e['lp'].astype(np.float64)
        llr = lp[ic] - lp[ia]
        m = (tok != 0) & (tok != 1)
        isf = np.array([vocab[t] in FUNC for t in tok]) & m
        tot['func'].append(llr[isf]); tot['content'].append(llr[m & ~isf])
    f = np.concatenate(tot['func']) if tot['func'] else np.zeros(0)
    k = np.concatenate(tot['content']) if tot['content'] else np.zeros(0)
    return dict(n_func=len(f), n_content=len(k), sum_func=f.sum(), sum_content=k.sum(),
                per_char_func=f.mean() if len(f) else np.nan, per_char_content=k.mean() if len(k) else np.nan,
                frac_func=f.sum() / (f.sum() + k.sum()) if len(f) + len(k) else np.nan)


def top_spans(T, ids, years, a, c, vocab, top=20):
    ia, ic = gi(years, a), gi(years, c)
    res = []
    for e in T:
        if e['id'] not in ids: continue
        tok = e['tok'].astype(int); lp = e['lp'].astype(np.float64)
        llr = lp[ic] - lp[ia]; llr[(tok == 0) | (tok == 1)] = 0
        res += [(s, t, e['id'], e['pi']) for s, t in spans(llr, tok, vocab, top=5)]
    res.sort(key=lambda x: -x[0])
    return res[:top]


if __name__ == '__main__':
    name, split = sys.argv[1], sys.argv[2]
    vocab = json.load(open(f'{C}/vocab.json')) + ['X']
    T, years = load_tok(name, split)
    bl = pd.read_csv((PROJECT + '/data/blind_v0.tsv'), sep='\t')
    idx = pd.read_csv(f'{C}/{split}.tsv', sep='\t')
    for g, sub in bl[bl['cat'].isin(['A', 'B'])].groupby('group', sort=False):
        ids = set(sub.id) & set(idx.id)
        if not ids: continue
        r = sub.iloc[0]
        a = (float(r.attr_lo) + float(r.attr_hi)) / 2; c = (float(r.cons_lo) + float(r.cons_hi)) / 2
        d = decompose(T, ids, years, a, c, vocab)
        print(f'== {g} [{r["cat"]}] attr {a:.0f} vs cons {c:.0f}:', {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()})
        for s, t, i, pi in top_spans(T, ids, years, a, c, vocab, top=8):
            print(f'   {s:+.2f}  {t}')
