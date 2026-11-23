"""Undated texts (no catalogue date, never in training): build a scoring split so the fold models can date them.
Same vocabulary and variant map as v3; no masking (the texts were never trained on). Written for every fold
directory (identical files) so that any fold model can score it.
Usage: python undated.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import json
import numpy as np
import pandas as pd

D = (WORK + '/data')


def main():
    m = pd.read_csv(f'{D}/meta.tsv', sep='\t')
    held = set(pd.concat([pd.read_csv(f'{D}/corpus3_f{f}/heldout.tsv', sep='\t', usecols=['id']) for f in range(5)]).id)
    want = m[m.lo.isna() & ~m.id.isin(held)].set_index('id')
    vocab = json.load(open(f'{D}/corpus3_f0/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    vm = json.load(open(f'{D}/varmap3.json'))
    rows, arrs, off = [], [], 0
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']:
        for line in open(f'{D}/extract/{part}.jsonl'):
            r = json.loads(line)
            if r['id'] not in want.index or not r.get('juans'): continue
            t = ''.join(vm.get(c, c) for c in 'Ｓ'.join(r['juans']))
            tok = np.array([1 if c == 'Ｓ' else c2i.get(c, 2) for c in t], np.uint16)
            if len(tok) < 200: continue
            w = want.loc[r['id']]
            rows.append(dict(id=r['id'], title=w['title'], cat=w['cat'], dyn=w['dyn'], author=w['author'], lo=np.nan, hi=np.nan,
                             genre=7, n_raw=len(tok), offset=off, length=len(tok)))
            arrs.append(tok); off += len(tok)
    a = np.concatenate(arrs)
    df = pd.DataFrame(rows)
    for f in range(5):
        a.tofile(f'{D}/corpus3_f{f}/undated.bin'); df.to_csv(f'{D}/corpus3_f{f}/undated.tsv', sep='\t', index=False)
    print(len(df), 'texts', a.size / 1e6, 'M tokens'); print(df.cat.str[:3].value_counts().to_dict())


if __name__ == '__main__':
    main()
