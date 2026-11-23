"""Systematic test: does the model's out-of-fold date flag doubtful translator attributions in the Chinese canon?

External labels from the CBC@ database (Radich & Norrish; raw HTML from cbc_fetch.py). For each Taishō text:
  disputed  = some scholarly source (not the Taishō [T] entry) assigns other people / Anonymous for this text, or its
              argument says the traditional ascription is doubtful (keywords)
  confirmed = a scholarly source lists the same translator as [T] and nothing disputes it
Model side: OOF median minus label midpoint (v3 labels), |offset|, and the audit flags (fixed before this test).
Tests: AUC of |offset| (and of early/late flags) for disputed vs confirmed; direction by translator period.
Usage: python cbc_attr.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, json, html
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RAW = (WORK + '/raw/cbc_dl')
R = (WORK + '/runs')
DOUBT = re.compile(r'doubt|unreliab|spurious|not (?:be )?(?:by|the work of)|cannot be|wrong|questionab|problemat|reascrib|re-ascrib|'
                   r'reattribut|misattribut|should be (?:re)?classified|apocryph|forg|incorrect|false|dubious|anonymous', re.I)
ACCEPT = re.compile(r'reliabl|secure|authentic|accept|safely|genuine|confirm', re.I)


def parse(path):
    s = open(path, encoding='utf8').read()
    ident = re.search(r'Text: (T\d{4}\w?)', s)
    rows = []
    t = s[s.find('assertion-table'):]
    for tr in re.findall(r'<tr>(.*?)</tr>', t, re.S)[1:]:
        tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)
        if len(tds) < 5: continue
        src = re.search(r'\[(?:<a[^>]*>)?([^<\]]+)', tds[1]); src = src.group(1).strip() if src else ''
        arg = html.unescape(re.sub(r'<[^>]+>', ' ', tds[3]))
        det = html.unescape(re.sub(r'<[^>]+>', ' ', tds[4]))
        m = re.search(r'People:(.*?)(?:Identifier:|$)', det, re.S); sec = m.group(1) if m else ''
        ppl = re.findall(r'[\u3400-\u9fff]{2,}', sec); anon = bool(re.search(r'Anonymous', sec))
        rows.append(dict(src=src, arg=re.sub(r'\s+', ' ', arg).strip(), people=ppl, anon=anon,
                         pref=('Yes' in tds[0]), npert=len(re.findall(r'<li>', tds[2]))))
    return ident.group(1) if ident else None, rows


def label(rows):
    """specific dispute: a scholarly assertion about at most 5 texts that gives other people / Anonymous for this text
    or argues the ascription is doubtful; fei: the text is in Radich 2019 (Fei Changfang's batch ascriptions);
    confirmed: a scholarly source lists the same translator and there is no specific dispute."""
    t = [r for r in rows if r['src'] == 'T']
    tp = set(p for r in t for p in r['people'])
    spec = fei = conf = False
    for r in rows:
        if r['src'] == 'T': continue
        other = set(r['people'])
        if r['src'].startswith('Radich 2019'): fei = True
        if r['npert'] <= 5 and (r['anon'] or (other and tp and not (other & tp)) or DOUBT.search(r['arg'])): spec = True
        if other & tp and not r['anon']: conf = True
    return spec, conf and not spec, fei, ';'.join(sorted(tp))


def main():
    want = json.load(open(f'{RAW}/want.json'))
    oof = pd.read_csv(f'{R}/l3_oof.tsv', sep='\t').set_index('id')
    aud = pd.read_csv(f'{R}/l3_audit.tsv', sep='\t').set_index('id')
    out = []
    for kr, tn, cid in want:
        p = f'{RAW}/text/{cid}.html'
        if not os.path.exists(p) or kr not in oof.index: continue
        ident, rows = parse(p)
        d, c, fei, tp = label(rows)
        r = oof.loc[kr]
        out.append(dict(id=kr, taisho=tn, title=r.title, genre=r.genre, lo=r.lo, hi=r.hi, median=r['median'], translator=tp,
                        n_assert=len(rows), disputed=d, confirmed=c, fei=fei,
                        flag_early=bool(aud.flag_early.get(kr, False)), flag_late=bool(aud.flag_late.get(kr, False))))
    df = pd.DataFrame(out)
    df['offset'] = df['median'] - (df.lo + df.hi) / 2
    df['absoff'] = df.offset.abs()
    df.to_csv(f'{R}/cbc_attr.tsv', sep='\t', index=False)
    pd.set_option('display.width', 200)
    tr = df[df.genre == 5].copy()
    tr['mid'] = (tr.lo + tr.hi) / 2
    tr['period'] = pd.cut(tr.mid, [0, 220, 317, 420, 589, 907, 1300], labels=['東漢', '三國西晉', '東晉十六國', '南北朝', '隋唐', '宋元'])
    print(len(df), 'texts;', 'translations', len(tr), '; specific dispute', int(tr.disputed.sum()), '; Fei batch (Radich 2019)', int(tr.fei.sum()),
          '; confirmed', int(tr.confirmed.sum()))
    print('offset (OOF median - label midpoint) by period and label:')
    print(tr.groupby(['period', 'disputed'], observed=True).offset.agg(['size', 'median']).round(0).unstack().to_string())
    print(tr.groupby(['period', 'fei'], observed=True).offset.agg(['size', 'median']).round(0).unstack().to_string())
    # directional tests: early translators (<= 280) misattributed texts read later; 5th-6th c. Fei-type read earlier
    for name, sub, alt in [('label <= 280 (Eastern Han, Wu, W. Jin early)', tr[tr.mid <= 280], 'greater'),
                           ('label 400-589', tr[tr.mid.between(400, 589)], 'less')]:
        for col in ['disputed', 'fei']:
            a, b = sub[sub[col]], sub[~sub[col]]
            if len(a) >= 5 and len(b) >= 5:
                u = mannwhitneyu(a.offset, b.offset, alternative=alt)
                auc = u.statistic / len(a) / len(b)
                print(f'{name}: {col} n={len(a)} vs {len(b)}; median offset {a.offset.median():.0f} vs {b.offset.median():.0f}; '
                      f'AUC({alt}) {auc if alt == "greater" else 1 - auc:.3f}, p={u.pvalue:.2g}; audit-flag rate {(a.flag_early | a.flag_late).mean():.3f} vs {(b.flag_early | b.flag_late).mean():.3f}')
    print('audit-flagged translations: share with a specific dispute', round(tr[tr.flag_early | tr.flag_late].disputed.mean(), 3),
          'vs unflagged', round(tr[~(tr.flag_early | tr.flag_late)].disputed.mean(), 3),
          '; Fei batch', round(tr[tr.flag_early | tr.flag_late].fei.mean(), 3), 'vs', round(tr[~(tr.flag_early | tr.flag_late)].fei.mean(), 3))


if __name__ == '__main__':
    main()
