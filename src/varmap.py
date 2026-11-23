"""Build a variant-character map: edition-specific graphic variants -> the form used by CBETA.

A pair v->p from the variant dictionaries (twedu, hydzd) is accepted only if v is rare in the
normalized CBETA text relative to p (so genuine distinct words such as 云/雲, 于/於, 后/後 survive).
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import pickle, collections, json

D = (WORK + '/data')
V = (WORK + '/raw/cjkvi-variants')
REF_EDS = {('KR6', 'T'), ('KR6', 'X'), ('KR6', 'J')}
# Qing taboo substitutions that cannot be told apart from ordinary use are left alone (e.g. 元 for 玄);
# 缺筆 forms are handled by the dictionaries. Manual additions:
MANUAL = {'𤣥': '玄', '玅': '妙', '𢎞': '弘', '甯': '寧', '邱': '丘', '𤓰': '瓜', '夘': '卯',
          # edition markers found by edmarkers.py / error analysis (v2)
          '无': '無', '炁': '氣', '虗': '虛', '为': '為', '㢤': '哉', '䰟': '魂', '伹': '但'}
PROTECT = set('于云后余才台里斗丑干谷了面咸系范沈发几只叶万与亏广卜千')


def load_pairs():
    """Ordered list of variant tables: [{variant: {standard candidates}}], most trusted first."""
    import glob, os
    def read(fn, tag=None, sym=False):
        d = collections.defaultdict(set)
        for l in open(f'{V}/{fn}', encoding='utf8'):
            if l.startswith('#') or l.count(',') < 2: continue
            a, t, b = l.strip().split(',')[:3]
            if '<' in t or a == b or len(a) != 1 or len(b) != 1: continue
            if tag and t != tag: continue
            d[b].add(a)
            if sym: d[a].add(b)
        return d
    tables = [read('twedu-variants.txt', 'twedu/variant'), read('hydzd-variants.txt', 'hydzd/variant')]
    rest = collections.defaultdict(set)
    for fn in ['jp-old-style.txt', 'jisx0213-variants.txt', 'jisx0212-variants.txt', 'cjkvi-variants.txt',
               'koseki-variants.txt', 'dypytz-variants.txt', 'radical-variants.txt', 'ucs-scs.txt']:
        for k, v in read(fn, sym=True).items(): rest[k] |= v
    tables.append(rest)
    return tables


def context_vectors(chars, n_lines=4000):
    """prev/next-char count vectors for the given chars, from a sample of non-CBETA text."""
    import itertools
    vec = {c: collections.Counter() for c in chars}
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5']:
        with open(f'{D}/extract/{part}.jsonl') as f:
            for line in itertools.islice(f, 0, None, 3):
                r = json.loads(line)
                for j in r.get('juans', [])[:8]:
                    for i in range(1, len(j) - 1):
                        c = j[i]
                        if c in vec:
                            vec[c]['L' + j[i - 1]] += 1; vec[c]['R' + j[i + 1]] += 1
    return vec


def cos(a, b):
    import math
    num = sum(v * b.get(k, 0) for k, v in a.items())
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def main():
    st = pickle.load(open(f'{D}/charstats.pkl', 'rb'))
    ref, tot = collections.Counter(), collections.Counter()
    for part, eds in st.items():
        for ed, c in eds.items():
            tot.update(c)
            if (part, ed) in REF_EDS: ref.update(c)
    tables = load_pairs()
    m = {}
    for v in set().union(*tables):
        if v in PROTECT: continue
        fv = ref.get(v, 0)
        if fv > 30: continue  # v must be (nearly) unused in normalized CBETA
        for t in tables:
            ps = [p for p in t.get(v, ()) if ref.get(p, 0) >= 20 and fv <= 0.002 * ref.get(p, 0)]
            if ps:
                m[v] = max(ps, key=lambda p: ref.get(p, 0)); break
    # context check: a graphic variant must occur in the same contexts as its standard form
    ctx = context_vectors({c for c in m if tot[c] >= 300} | set(m.values()))
    bad = []
    for v, p in list(m.items()):
        if v in ctx and p in ctx:
            cs = cos(ctx[v], ctx[p])
            if cs < 0.3:
                bad.append((tot[v], v, p, round(cs, 2))); del m[v]
    print('rejected by context:', sorted(bad, reverse=True)[:60])
    m.update(MANUAL)
    # resolve chains
    for v in list(m):
        seen = {v}; p = m[v]
        while p in m and p not in seen and m[p] != p:
            seen.add(p); p = m[p]
        m[v] = p
    m = {v: p for v, p in m.items() if v != p}
    json.dump(m, open(f"{D}/varmap{__import__('os').environ.get('LETTER_V', '')}.json", 'w'), ensure_ascii=False)
    hit = sorted(((tot[v], v, p) for v, p in m.items() if tot[v] > 0), reverse=True)
    print(len(m), 'mappings,', len(hit), 'occur;', sum(h[0] for h in hit) / sum(tot.values()) * 100, '% of chars changed')
    print(' '.join(f'{v}>{p}:{n // 1000}k' for n, v, p in hit[:150]))
    # chars frequent overall but missing from the reference after mapping
    after = collections.Counter()
    for c, n in tot.items(): after[m.get(c, c)] += n
    miss = [(n, c) for c, n in after.most_common() if ref.get(c, 0) == 0][:80]
    print('frequent but absent from CBETA:', ' '.join(f'{c}:{n // 1000}k' for n, c in miss))
    print('types before/after', len(tot), len(after))


if __name__ == '__main__':
    main()
