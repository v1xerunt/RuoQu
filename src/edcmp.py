"""E10: paired comparison of the same work in two editions (posterior median per edition)."""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys
import numpy as np, pandas as pd
from evaluate import load_scores, text_scores, posterior, quantile, evaluate
R = (WORK + '/runs')
name = sys.argv[1]
_, cal = evaluate(f'{R}/{name}/scores_heldout_unk.npz')
tau = float(np.exp(np.mean(np.log(cal['taus']))))
mm, ml = load_scores(f'{R}/{name}/scores_ed_master_unk.npz')
mid, mS, _ = text_scores(mm, ml)
med_m = dict(zip(mid, quantile(posterior(mS, tau), 0.5)))
C = (WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
for ed in ['WYG', 'ZTDZ', 'tls', 'SBCK']:
    try: em, el = load_scores(f'{R}/{name}/scores_ed_{ed}_unk.npz')
    except FileNotFoundError: continue
    eid, eS, _ = text_scores(em, el)
    med_e = dict(zip(eid, quantile(posterior(eS, tau), 0.5)))
    idx = pd.read_csv(f'{C}/ed_{ed}.tsv', sep='\t').set_index('id')
    d = []
    for i in eid:
        k = f'{i}|{ed}'
        if k in med_m:
            lo, hi = idx.loc[i, 'lo'], idx.loc[i, 'hi']
            d.append((med_e[i] - med_m[k], abs(med_e[i] - (lo + hi) / 2), abs(med_m[k] - (lo + hi) / 2)))
    d = np.array(d)
    if len(d):
        print(f'{name} {ed:5s} n={len(d):3d}  shift(ed - master): median {np.median(d[:,0]):+.0f}  mean {d[:,0].mean():+.0f}  |shift| median {np.median(np.abs(d[:,0])):.0f};  AE ed {np.median(d[:,1]):.0f} vs master {np.median(d[:,2]):.0f}')
