"""Assign a composition-date interval to every Kanripo text.

Priority: catalog DATES of the primary person > unique CBDB match (name + dynasty) > dynasty span.
Output: data/meta.tsv
"""
import json, re, sqlite3, collections
import pandas as pd
from catalog import parse_dates, parse_text_date, OUT, RAW

# Composition-date spans (years CE) for catalog dynasty labels.
DYN = {
    '周': (-550, -221), '春秋': (-770, -476), '戰國': (-475, -221), '秦': (-221, -206),
    '漢': (-206, 220), '前漢': (-206, 8), '西漢': (-206, 8), '新': (9, 25),
    '後漢': (25, 220), '東漢': (25, 220),
    '魏': (220, 265), '曹魏': (220, 265), '吳': (222, 280), '蜀': (221, 263), '三國': (220, 280),
    '晉': (265, 420), '西晉': (265, 316), '東晉': (317, 420), '六朝': (220, 589),
    '劉宋': (420, 479), '南朝宋': (420, 479), '蕭齊': (479, 502), '南齊': (479, 502), '齊': (479, 502),
    '梁': (502, 557), '陳': (557, 589),
    '元魏': (386, 534), '北魏': (386, 534), '後魏': (386, 534), '東魏': (534, 550), '西魏': (535, 557),
    '高齊': (550, 577), '北齊': (550, 577), '北周': (557, 581),
    '姚秦': (384, 417), '後秦': (384, 417), '符秦': (351, 394), '苻秦': (351, 394), '前秦': (351, 394),
    '北涼': (397, 439), '西秦': (385, 431), '乞伏秦': (385, 431), '前涼': (317, 376),
    '隋': (581, 618), '唐': (618, 907), '唐初期': (618, 712), '唐後期': (756, 907),
    '五代': (907, 960), '南唐': (937, 975), '後蜀': (934, 965), '前蜀': (903, 925), '吳越': (907, 978),
    '後唐': (923, 936), '後晉': (936, 947), '後周': (951, 960), '後梁': (907, 923),
    '宋': (960, 1279), '北宋': (960, 1127), '南宋': (1127, 1279), '遼': (916, 1125), '金': (1115, 1234),
    '西夏': (1038, 1227), '元': (1271, 1368), '明': (1368, 1644), '清': (1644, 1911), '淸': (1644, 1911),
    '民國': (1912, 1949),
}
FOREIGN = {'日本', '新羅', '高麗', '朝鮮', '百濟', '韓國', '越南', '安南'}
# Functions that make a person the author of the main text layer.
AUTHOR_FN = ('撰', '著', '譯', '述', '說', '造', '作', '記', '錄', '編', '集', '輯', '纂', '等譯', '等編',
             '奉敕撰', '奉敕編', '等錄', '主編', '重編', '傳', '')
COMMENT_FN = ('注', '註', '疏', '解', '箋', '音義', '釋', '序', '校', '補', '評', '點', '考證', '正義', '集解', '整理', '監修')


def norm_dyn(d):
    d = (d or '').strip().replace('?', '').replace('？', '')
    return d


def cbdb_index():
    c = sqlite3.connect(f'{RAW}/cbdb/cbdb_20261114.sqlite3')
    dy = {r[0]: (r[1], r[2], r[3]) for r in c.execute('select c_dy,c_dynasty_chn,c_start,c_end from DYNASTIES')}
    idx = collections.defaultdict(list)
    for pid, name, iy, by, dyr, fe, fl, cdy in c.execute(
            'select c_personid,c_name_chn,c_index_year,c_birthyear,c_deathyear,c_fl_earliest_year,c_fl_latest_year,c_dy from BIOG_MAIN'):
        if not name: continue
        idx[name].append(dict(pid=pid, iy=iy, by=by, dy=dyr, fe=fe, fl=fl, dyn=dy.get(cdy, ('', 0, 0))))
    return idx


def cbdb_span(cands, dyn):
    """Active span from a unique CBDB person whose dynasty span overlaps the catalog dynasty."""
    lo_d, hi_d = DYN.get(dyn, (-9999, 9999))
    ok = []
    for p in cands:
        _, s, e = p['dyn']
        if s or e:
            if e < lo_d - 30 or s > hi_d + 30: continue
        if p['by'] and p['dy'] and 0 < p['dy'] - p['by'] < 110:
            span = (min(p['by'] + 20, p['dy']), p['dy'], 'cbdb_bd')
        elif p['fe'] or p['fl']:
            a = p['fe'] or p['fl']; b = p['fl'] or p['fe']
            span = (a - 10, b + 10, 'cbdb_fl')
        elif p['iy']:
            span = (p['iy'] + 20, p['iy'] + 65, 'cbdb_iy')
        else:
            continue
        if span[1] < lo_d - 30 or span[0] > hi_d + 30: continue
        ok.append(span)
    if len(ok) == 1:
        return ok[0]
    if len(ok) > 1:  # accept if candidates agree within 30 years
        mids = [(a + b) / 2 for a, b, _ in ok]
        if max(mids) - min(mids) <= 30:
            return (min(a for a, _, _ in ok), max(b for _, b, _ in ok), ok[0][2] + '_multi')
    return None


def main():
    rows = [json.loads(l) for l in open(f'{OUT}/catalog_raw.jsonl')]
    cb = cbdb_index()
    out = []
    for r in rows:
        persons = r['persons']
        prim = None
        for p in persons:
            fn = p.get('function', '')
            if fn in AUTHOR_FN or (fn and not any(fn.startswith(c) for c in COMMENT_FN)):
                prim = p; break
        commentators = [p for p in persons if p is not prim and any(p.get('function', '').startswith(c) for c in COMMENT_FN)]
        dyn = norm_dyn((prim or {}).get('dynasty') or r['props'].get('DYNASTY') or r['dyn_label'])
        uncertain = '?' in ((prim or {}).get('dynasty') or r['dyn_label'] or '')
        name = (prim or {}).get('name', r['auth_label'])
        span, src = None, ''
        if r['props'].get('DATE'):
            d = parse_text_date(r['props']['DATE'])
            if d: span, src = d, 'text_date'
        if span is None and prim and prim.get('dates'):
            d = parse_dates(prim['dates'])
            if d: span, src = (d[0], d[1]), 'kr_' + d[2]
        if span is None and name and dyn in DYN:
            c = cbdb_span(cb.get(name, []), dyn)
            if c: span, src = (c[0], c[1]), c[2]
        if span is None and dyn in DYN:
            span, src = DYN[dyn], 'dyn'
        # clip person span to the dynasty span (+/- 20y) when both known
        if span and dyn in DYN and src not in ('dyn', 'text_date'):
            lo_d, hi_d = DYN[dyn]
            a, b = max(span[0], lo_d - 20), min(span[1], hi_d + 20)
            if a <= b: span = (a, b)
        out.append(dict(id=r['id'], title=r['title'], cat=r['cat'], part=r['id'][:3], dyn=dyn,
                        author=name, function=(prim or {}).get('function', ''),
                        n_commentators=len(commentators),
                        lo=span[0] if span else None, hi=span[1] if span else None, date_src=src,
                        dyn_uncertain=uncertain, foreign=dyn in FOREIGN,
                        genre=r['props'].get('GENRE', ''), source=r['props'].get('SOURCE', '')))
    df = pd.DataFrame(out)
    df.to_csv(f'{OUT}/meta.tsv', sep='\t', index=False)
    print(df.date_src.value_counts())
    print(df[df.lo.isna()].dyn.value_counts().head(20))
    df['w'] = df.hi - df.lo
    print(df.groupby('date_src').w.median())


if __name__ == '__main__':
    main()
