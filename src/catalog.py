"""Parse the Kanripo catalog (KR-Catalog/KR/*.txt) into a metadata table.

One row per text: id, title, category, persons (name, dynasty, function, dates).
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import re, glob, json, os, unicodedata

RAW = (WORK + '/raw')
OUT = (WORK + '/data')


def parse_dates(s):
    """Return (lo, hi, kind) of an author's active span, or None."""
    s = s.strip().replace('–', '-').replace('—', '-')
    s = s.replace('?', '').replace('ca.', '').replace('c.', '').strip()
    m = re.match(r'^fl\.\s*(-?\d+)\s*(?:-\s*(-?\d+))?$', s)
    if m:
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        return (a - 10, b + 10, 'fl')
    m = re.match(r'^d\.\s*(-?\d+)$', s)
    if m:
        d = int(m.group(1)); return (d - 40, d, 'd')
    m = re.match(r'^b\.\s*(-?\d+)$', s)
    if m:
        b = int(m.group(1)); return (b + 20, b + 60, 'b')
    m = re.match(r'^(-?\d+)\s*-\s*(-?\d+)$', s)
    if m:
        b, d = int(m.group(1)), int(m.group(2))
        if d < b or d - b > 110:
            return None
        lo = min(b + 20, d)
        return (lo, d, 'bd')
    return None


REIGN = {'順治': 1644, '康熙': 1662, '雍正': 1723, '乾隆': 1736, '嘉慶': 1796, '道光': 1821,
         '咸豐': 1851, '同治': 1862, '光緒': 1875, '宣統': 1909}
CNUM = {'〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}


def cn2int(s):
    if s == '元': return 1
    n, cur = 0, 0
    for ch in s:
        if ch in CNUM: cur = CNUM[ch]
        elif ch == '十': n += (cur or 1) * 10; cur = 0
        elif ch == '百': n += (cur or 1) * 100; cur = 0
    return n + cur


def parse_text_date(s):
    """Text-level :DATE: (e.g. TTC dating of Daoist texts) -> (lo, hi) or None."""
    s0 = unicodedata.normalize('NFKC', s).strip()
    unc = 15 if '?' in s0 else 0
    s = re.sub(r'\s*\([^)]*\)', '', s0.replace('?', '')).strip()
    if s.startswith('after '):
        m = re.match(r'after (\d{3,4})$', s)
        if m: y = int(m.group(1)); return (y, y + 30)
    s = re.sub(r'\s+or\s+', ' - ', s)
    m = re.match(r'^(\d{1,2})(?:st|nd|rd|th)\s*-\s*(\d{1,2})(?:st|nd|rd|th)\s*cent', s)
    if m:
        return ((int(m.group(1)) - 1) * 100 - unc, int(m.group(2)) * 100 + unc)
    for k, y0 in REIGN.items():
        m = re.match(k + r'(元|[〇一二三四五六七八九十百]+)年', s)
        if m:
            y = y0 + cn2int(m.group(1)) - 1; return (y - 3, y)
    m = re.match(r'^(?:ca\.\s*)?(-?\d{2,4})$', s)
    if m:
        y = int(m.group(1)); w = 25 if 'ca' in s else 5
        return (y - w - unc, y + (w if 'ca' in s else 0) + unc)
    m = re.match(r'^(-?\d{2,4})\s*-\s*(-?\d{2,4})$', s)
    if m:
        return (int(m.group(1)) - unc, int(m.group(2)) + unc)
    m = re.match(r'^(?:end of the\s*)?(\d{1,2})(?:st|nd|rd|th)\s*cent(?:ury)?\s*(前期|後期|前半|後半)?(?:\s*(?:-|or)\s*(\d{1,2})(?:st|nd|rd|th)\s*cent(?:ury)?\s*(前期|後期|前半|後半)?)?', s)
    if m:
        a = (int(m.group(1)) - 1) * 100; b = a + 100
        if m.group(2) in ('前期', '前半'): b = a + 50
        if m.group(2) in ('後期', '後半'): a = a + 50
        if s.startswith('end of'): a = b - 30
        if m.group(3):
            b = int(m.group(3)) * 100
            if m.group(4) in ('前期', '前半'): b -= 50
        return (a - unc, b + unc)
    return None


def parse_catalog():
    rows = []
    for fn in sorted(glob.glob(f'{RAW}/KR-Catalog/KR/KR*.txt')):
        cur = None; person = None; in_appendix = False; section = ''
        for line in open(fn, encoding='utf8'):
            s = line.rstrip('\n')
            m = re.match(r'^\*\*\* (KR\d[a-z]\d{4})\s*(.*)$', s)
            if m:
                if cur: rows.append(cur)
                kid, rest = m.group(1), m.group(2)
                parts = rest.split('-')
                cur = dict(id=kid, label=rest, title=parts[0], cat=kid[:4],
                           dyn_label=parts[1] if len(parts) > 1 else '',
                           auth_label='-'.join(parts[2:]) if len(parts) > 2 else '',
                           persons=[], props={})
                person = None; in_appendix = False; section = ''
                continue
            if cur is None:
                continue
            m = re.match(r'^(\*{4,})\s*(.*)$', s)
            if m:
                depth, name = len(m.group(1)), m.group(2).strip()
                if depth == 4:
                    in_appendix = name.startswith('(附)')
                    section = name
                    person = None
                elif section == '人物' and not in_appendix and depth == 5:
                    person = dict(name=name)
                    cur['persons'].append(person)
                elif depth >= 5:
                    person = None
                continue
            m = re.match(r'^\s*:([A-Z_]+):\s*(.*)$', s)
            if m:
                k, v = m.group(1), m.group(2).strip()
                if person is not None and k in ('DYNASTY', 'FUNCTION', 'DATES'):
                    person[k.lower()] = v
                elif person is None and not in_appendix and k in ('SOURCE', 'EXTENT', '_RESP', 'DYNASTY', 'GENRE', 'CBETA_ID', 'BASEEDITION', 'DATE'):
                    cur['props'][k] = v
        if cur: rows.append(cur)
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rows = parse_catalog()
    with open(f'{OUT}/catalog_raw.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n_dates = sum(any('dates' in p for p in r['persons']) for r in rows)
    print(len(rows), 'texts;', n_dates, 'with some person dates')
    from collections import Counter
    print(Counter(p.get('function', '') for r in rows for p in r['persons']).most_common(30))
