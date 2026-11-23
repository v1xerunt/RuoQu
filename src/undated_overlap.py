"""How much of each undated text the scoring models saw in training: share of its 12-grams that occur in the
training split of fold 0 or fold 1 (the two models that dated the undated texts). A high share means the text is
another edition or a compilation of trained material, so its date partly echoes a catalogue label.
Output: runs/undated_overlap.tsv (id, n12, seen12)
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import numpy as np, pandas as pd
from corpus import ngram_hash, member
D = (WORK + '/data'); R = (WORK + '/runs')
N = 12


def main():
    u = pd.read_csv(f'{D}/corpus3_f0/undated.tsv', sep='\t')
    a = np.fromfile(f'{D}/corpus3_f0/undated.bin', np.uint16)
    H = [ngram_hash(a[o:o + l], N) for o, l in zip(u.offset, u.length)]
    seen = [np.zeros(len(h), bool) for h in H]
    for f in (0, 1):
        t = np.fromfile(f'{D}/corpus3_f{f}/train.bin', np.uint16)
        U = np.unique(ngram_hash(t, N)); del t
        print('fold', f, len(U), 'distinct training 12-grams', flush=True)
        for k, h in enumerate(H): seen[k] |= member(h, U)
        del U
    u['n12'] = [len(h) for h in H]; u['seen12'] = [s.mean() if len(s) else np.nan for s in seen]
    u[['id', 'title', 'cat', 'n12', 'seen12']].to_csv(f'{R}/undated_overlap.tsv', sep='\t', index=False)
    print(u.seen12.describe().to_string())
    print('share of texts with >50% of 12-grams seen in training:', f'{(u.seen12 > .5).mean():.3f}')
    print(u.sort_values('seen12', ascending=False)[['id', 'title', 'n12', 'seen12']].head(25).to_string(index=False))


if __name__ == '__main__':
    main()
