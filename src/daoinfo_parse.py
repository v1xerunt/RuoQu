"""Parse scholarly date statements in the 道教文化中心資料庫 entries (raw/daoinfo/entries.jsonl) into year ranges.
The dating sentence is the first clause that says when/by whom the text was made (出於, 成書, 似為, 約為, 撰於, ...);
the range spans every period named in it (e.g. 唐宋間 -> 618-1279; 南北朝 -> 420-589; 晚唐 -> 820-907).
Output: data/daoinfo_dates.tsv (id, title, statement, lo, hi). Ambiguous 宋 is read as Zhao Song unless 劉宋/南朝宋.
Only comma-delimited sub-clauses carrying a cue count (so '唐人某某註解' after '北齊某撰' does not widen the range), and
sub-clauses reporting a traditional attribution (相傳, 舊題, 原題, 題為, 託名) are skipped. Explicit years in brackets in a cue
sub-clause (e.g. 淳祐年間（1241-1252）) override the period names.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import re, json
import pandas as pd

RAW = (WORK + '/raw/daoinfo/entries.jsonl')
P = (PROJECT + '/data')
PER = [  # longer names first
    ('唐宋', 618, 1279), ('元明', 1271, 1644), ('明清', 1368, 1911), ('金元', 1115, 1368), ('宋金', 960, 1234), ('晉唐', 265, 907),
    ('南朝宋', 420, 479), ('劉宋', 420, 479), ('北宋末', 1100, 1127), ('南宋末', 1230, 1279), ('中晚唐', 756, 907), ('唐末五代', 860, 960), ('南北朝末', 550, 589),
    ('北齊', 550, 577), ('北周', 557, 581), ('東魏', 534, 550), ('西魏', 535, 557), ('十六國', 304, 439), ('齊梁', 479, 557),
    ('宋齊', 420, 502), ('秦漢', -221, 220), ('兩漢', -206, 220), ('唐初', 618, 712), ('宋初', 960, 1022), ('宋末', 1230, 1279),
    ('元初', 1271, 1300), ('元末', 1340, 1368), ('明末', 1600, 1644), ('清初', 1644, 1722), ('遼', 916, 1125),
    ('先秦', -770, -221), ('戰國', -475, -221), ('春秋', -770, -476), ('西漢', -206, 8), ('東漢', 25, 220), ('漢魏', -206, 265),
    ('漢末', 160, 220), ('三國', 220, 280), ('魏晉', 220, 420), ('西晉', 265, 316), ('東晉', 317, 420),
    ('六朝', 220, 589), ('南北朝', 420, 589), ('南朝', 420, 589), ('北朝', 386, 581), 
    ('北魏', 386, 534), ('蕭齊', 479, 502), ('南齊', 479, 502), ('梁', 502, 557), ('陳', 557, 589), ('隋唐', 581, 907),
    ('隋', 581, 618), ('初唐', 618, 712), ('盛唐', 712, 756), ('中唐', 756, 835), ('晚唐', 835, 907), ('唐末', 860, 907),
    ('唐代', 618, 907), ('唐', 618, 907), ('五代', 907, 960), ('北宋', 960, 1127), ('南宋', 1127, 1279), ('宋元', 960, 1368),
    ('宋代', 960, 1279), ('宋', 960, 1279), ('金', 1115, 1234), ('元代', 1271, 1368), ('元', 1271, 1368),
    ('明代', 1368, 1644), ('明初', 1368, 1435), ('明', 1368, 1644), ('清', 1644, 1911), ('晉', 265, 420), ('漢', -206, 220),
]
CUE = re.compile(r'(出於|成書|書成於|似為|似係|約為|蓋為|蓋係|當為|應為|撰於|約出|大約|編集|所作|所撰|所造|撰人不詳|作者不詳|年代)')
TRAD = re.compile(r'(相傳|舊題|原題|題為|題作|託名|假託|傳為)')
YEARS = re.compile(r'[（(](?<!前)(\d{3,4})(?:\s*[-－–~～至]\s*(\d{3,4}))?[）)]')


def _hit(s, name):
    """multi-character period names anywhere; a one-character dynasty name only next to a period word"""
    if len(name) > 1: return name in s
    return re.search(rf'(?:[於為係是自約在]){name}|{name}(?:代|朝|人|時|間|初|末|中葉|中期|以後|以前|之際|道士|中|、)', s) is not None


def periods(s):
    found = []
    for name, lo, hi in PER:
        if _hit(s, name):
            found.append((lo, hi)); s = s.replace(name, '　')
    return found


def parse(text, title=''):
    t = re.sub(r'[《》「」〈〉]', '', title)
    for clause in re.split(r'[。；;]', text):
        if not CUE.search(clause): continue
        s = re.sub(r'[《》「」〈〉]', '', clause).replace(t, '　')
        subs = [x for x in re.split(r'[，,]', s) if CUE.search(x) and not TRAD.search(x)]
        yrs = [int(y) for x in subs for m in YEARS.finditer(x) for y in m.groups() if y]
        if yrs:
            return clause, min(yrs), max(yrs)
        found = [f for x in subs for f in periods(x)]
        if not found and not TRAD.search(s):
            found = periods(s)  # cue and period in different sub-clauses, e.g. '唐代道書，撰人不詳'
        if found:
            return clause, min(f[0] for f in found), max(f[1] for f in found)
    return None, None, None


def main():
    rows = []
    for l in open(RAW):
        r = json.loads(l)
        st, lo, hi = parse(r.get('text', '') or '', re.sub(r'\\(.*?\\)', '', r['title']))
        rows.append(dict(id=r['id'], title=r['title'], statement=st, lo=lo, hi=hi, has_entry=bool(r.get('text'))))
    df = pd.DataFrame(rows)
    df.to_csv(f'{P}/daoinfo_dates.tsv', sep='\t', index=False)
    print(len(df), 'entries;', int(df.has_entry.sum()), 'with text;', int(df.lo.notna().sum()), 'with a parsed date')
    print(df[df.lo.notna()].sample(min(12, int(df.lo.notna().sum())), random_state=0)[['title', 'statement', 'lo', 'hi']].to_string(index=False))


if __name__ == '__main__':
    main()
