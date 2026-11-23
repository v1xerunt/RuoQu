"""Passage-level anatomy of a blind work (default 列子 KR5c0125).

For each 256-char passage scored with per-token log-probs (score.py --tokens 4): a passage-level posterior
(passage tau from held-out calibration), and the fraction of its characters inside 8-grams shared with
texts dated before 220 CE (borrowed / parallel material, from the raw extracted corpus).
Compares the dates of passages with and without early parallels, and reports flagged passages.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, json, pickle
import numpy as np
import pandas as pd

R = (WORK + '/runs')
D = (WORK + '/data')
YEARS = np.arange(-1000, 1901, 25)[::4]
KEEP = YEARS >= -600


def early_hashes(vocab_c2i, vm, before=220, exclude=()):
    from corpus import ngram_hash
    meta = pd.read_csv(f'{D}/meta.tsv', sep='\t')
    ids = set(meta[((meta.lo + meta.hi) / 2 < before) & ~meta.foreign.astype(bool)].id) - set(exclude)
    man = pd.read_csv((PROJECT + '/data/manual_dates.tsv'), sep='\t')
    ids |= set(man[(man.lo + man.hi) / 2 < before].id)
    ids -= set(exclude)
    H = []
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']:
        for line in open(f'{D}/extract/{part}.jsonl'):
            r = json.loads(line)
            if r['id'] in ids and r.get('juans'):
                s = ''.join(vm.get(c, c) for c in ''.join(r['juans']))
                t = np.array([vocab_c2i.get(c, 2) for c in s], np.int64)
                H.append(np.unique(ngram_hash(t, 8)))
    return np.unique(np.concatenate(H)), len(ids)


def main(run, wid='KR5c0125', group_ids=('KR5c0125', 'KR5c0124', 'KR5c0049'), corpus='corpus_f0', v=''):
    from corpus import ngram_hash, cover
    vocab = json.load(open(f'{D}/{corpus}/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    vm = json.load(open(f'{D}/varmap{v}.json'))
    H, n_src = early_hashes(c2i, vm, exclude=group_ids)
    print('early texts', n_src, 'unique 8-grams', len(H))
    T = [e for e in pickle.load(open(f'{R}/{run}/scores_blind_full_unk_tok.pkl', 'rb')) if e['id'] == wid]
    tau_p = 0.2  # passage-level temperature (held-out calibration gave ~0.2-0.23)
    rows = []
    for e in T:
        tok = e['tok'].astype(np.int64); lp = e['lp'][:len(YEARS)].astype(np.float64)[KEEP]
        m = (tok > 1)
        h = ngram_hash(tok, 8); j = np.searchsorted(H, h); j[j >= len(H)] = 0
        shared = cover(H[j] == h, 8, len(tok)) & m
        ll = lp[:, m].sum(1)
        z = tau_p * (ll - ll.max()); p = np.exp(z); p /= p.sum()
        c = np.cumsum(p); Y = YEARS[KEEP]
        rows.append(dict(pi=e['pi'], text=''.join(vocab[t] for t in tok[m][:40]), full=''.join(vocab[t] for t in tok[m]), n=int(m.sum()),
                         borrowed=shared.sum() / max(m.sum(), 1), median=Y[min((c < .5).sum(), len(Y) - 1)],
                         p_after_200=float(p[Y >= 200].sum()), llr=(ll[np.argmin(np.abs(Y - 300))] - ll[np.argmin(np.abs(Y + 400))]) / max(m.sum(), 1)))
    df = pd.DataFrame(rows).sort_values('pi')
    return df


if __name__ == '__main__':
    from scipy.stats import mannwhitneyu
    run = sys.argv[1]
    corpus = sys.argv[2] if len(sys.argv) > 2 else 'corpus_f0'
    v = sys.argv[3] if len(sys.argv) > 3 else ''
    df = main(run, corpus=corpus, v=v)
    df['kind'] = np.where(df.borrowed >= 0.3, 'parallel', np.where(df.borrowed < 0.05, 'unique', 'mixed'))
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 400)
    print(df.groupby('kind').agg(n=('pi', 'size'), median_date=('median', 'median'), p_after_200=('p_after_200', 'mean'),
                                 llr_per_char=('llr', 'median')).round(3))
    a, b = df[df.kind == 'unique'].llr, df[df.kind == 'parallel'].llr
    print('unique vs parallel: per-char LLR(300 vs -400) one-sided p =', mannwhitneyu(a, b, alternative='greater').pvalue)
    print('passages with diagnostic content (scholars cite these as late / Buddhist-influenced):')
    for kw in ['偃師', '西方之人', '西極之國', '化人', '楊朱曰']:
        s = df[df.full.str.contains(kw)]
        print(f'  {kw}: n={len(s)}  dates {s["median"].tolist()}  P(after 200 CE) {s.p_after_200.round(2).tolist()}  borrowed {s.borrowed.round(2).tolist()}')
    df.to_csv(f'{R}/{run}/liezi_passages.tsv', sep='\t', index=False)
    print(df[['pi', 'n', 'borrowed', 'median', 'p_after_200', 'llr', 'text']].round(3).to_string(index=False, max_colwidth=30))
