"""Find edition-marker characters: usage explained by the base edition rather than by time.

For texts dated 600-1300 CE (where HFL, WYG, SBCK and CBETA all have material), count characters per
edition group. A character is a marker if its rate in one edition group is >= 8x its rate in every other
group, and the variant tables link it to a character that is common in the other groups.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import json, collections, sys
import numpy as np
import pandas as pd
sys.path.insert(0, (PROJECT + '/src'))
from varmap import load_pairs

D = (WORK + '/data')
GROUP = {'HFL': 'daozang', 'CK-KZ': 'daozang', 'WYG': 'siku', 'SBCK': 'sbck', 'T': 'cbeta', 'X': 'cbeta', 'J': 'cbeta'}


def main():
    meta = pd.read_csv(f'{D}/meta.tsv', sep='\t').set_index('id')
    vm = json.load(open(f'{D}/varmap.json'))
    cnt = collections.defaultdict(collections.Counter)
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']:
        for line in open(f'{D}/extract/{part}.jsonl'):
            r = json.loads(line)
            g = GROUP.get(r.get('edition', ''))
            if not g or r['id'] not in meta.index: continue
            lo, hi = meta.loc[r['id'], 'lo'], meta.loc[r['id'], 'hi']
            if not (pd.notna(lo) and 600 <= (lo + hi) / 2 <= 1300): continue
            for j in r['juans']:
                cnt[g].update(vm.get(c, c) for c in j)
    tot = {g: sum(c.values()) for g, c in cnt.items()}
    print({g: f'{n / 1e6:.1f}M' for g, n in tot.items()})
    tables = load_pairs()
    rate = lambda g, c: cnt[g][c] / tot[g]
    rows = []
    allc = set().union(*[set(c) for c in cnt.values()])
    for c in allc:
        rs = {g: rate(g, c) for g in cnt}
        g0 = max(rs, key=rs.get)
        if rs[g0] < 2e-5: continue
        others = [rs[g] for g in rs if g != g0]
        if rs[g0] >= 8 * max(others) + 1e-9:
            partners = set().union(*[t.get(c, set()) for t in tables])
            common = [(p, min(rate(g, p) for g in cnt if g != g0)) for p in partners]
            common = [(p, r) for p, r in common if r > 5e-5]
            rows.append(dict(char=c, group=g0, rate_per10k=rs[g0] * 1e4, max_other=max(others) * 1e4,
                             partner=''.join(p for p, _ in sorted(common, key=lambda x: -x[1])[:3])))
    df = pd.DataFrame(rows).sort_values('rate_per10k', ascending=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
    print(df.head(80).to_string(index=False))
    df.to_csv(f'{D}/edition_markers.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
