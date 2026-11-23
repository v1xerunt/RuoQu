"""Traditional attributions of the undated Buddhist texts, read from the byline of each text's first juan
(raw Kanripo files, e.g. '北涼沙門道泰譯', '三藏法師玄奘奉　詔譯', '失譯人名今附東晉錄'), compared with the model dates
(runs/undated_dates.tsv). Kinds:
  stated   - a dynasty in the byline, or a translator/author whose dated works give his period
  attached - 失譯 'translator lost', attached by a later catalogue to a period (今附X錄)
  lost     - 失譯 with no attachment: the model's date is the only one
  japan    - Taishō vols 56-84 (Japanese authors; Japanese kanbun, outside the training language)
'宋' alone is Liu Song (420-479) for the translators in LIU_SONG, Zhao Song when the byline carries the translation-bureau
titles (西天, 奉詔, 臣, 大師); otherwise ambiguous and left out.
Output: data/undated_bylines.tsv, runs/undated_byline_eval.tsv
Usage: python byline.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import re, zipfile
import numpy as np, pandas as pd
from meta import DYN

D = (WORK + '/data'); R = (WORK + '/runs')
P = (PROJECT + '/data'); KZ = (WORK + '/raw/krzip')
LIU_SONG = {'求那跋摩', '求那跋陀羅', '寶雲', '法海', '畺良耶舍', '曇摩蜜多', '功德直', '先公', '沮渠京聲', '慧簡', '翔公',
            '智嚴', '曇無竭', '慧嚴', '僧伽跋摩', '佛陀什', '求那毘地'}
ZHAO = re.compile(r'西天|奉\s*詔|臣|大師|光祿|朝散|朝奉')
ATTACH = {'後漢': (25, 220), '漢': (25, 220), '前魏': (220, 265), '曹魏': (220, 265), '周': (557, 581), '唐': (618, 907), '吳': (222, 280), '魏': (220, 265), '魏吳': (220, 280), '西晉': (265, 316),
          '東晉': (317, 420), '晉': (265, 420), '三秦': (351, 431), '秦': (351, 431), '涼': (317, 439), '北涼': (397, 439),
          '宋': (420, 479), '齊': (479, 502), '梁': (502, 557), '陳': (557, 589)}
DYN2 = dict(DYN, 大唐=(618, 907), 大宋=(960, 1279), 大元=(1271, 1368), 大明=(1368, 1644), 大清=(1644, 1911), 有唐=(618, 907),
            皇宋=(960, 1279), 趙宋=(960, 1279), 元魏=(386, 534), 宇文周=(557, 581), 周宇文=(557, 581), 大周=(690, 705),
            秦=(351, 417), 漢=(25, 220), 新羅=(600, 935), 高麗=(918, 1392), 朝鮮=(1392, 1910))
del DYN2['周'], DYN2['新']
KEYS = sorted(DYN2, key=len, reverse=True)
NOT_DYN = ('清辯', '元曉', '陳那', '漢陽', '元慶', '明教', '明佺', '宋本', '元本', '明本', '晉譯', '唐譯')  # names/notes that start with a dynasty character
END = re.compile(r'(譯|造|撰|述|集|編|錄|說|記|出|抄|鈔|著|纂|輯|註|注|疏|解|頌|作|重治|刪定|譯出|等譯)[　 ]*$')


def header_lines(i):
    """the lines between the title and the start of the text, from the first juan file that has any"""
    try:
        z = zipfile.ZipFile(f'{KZ}/{i}.zip')
    except Exception:
        return []
    for name in sorted(n for n in z.namelist() if n.endswith('.txt'))[:3]:
        out = []
        for l in z.read(name).decode('utf8', 'ignore').split('\n'):
            l = re.sub(r'[〔【][^〕】]*[〕】]|[＝＋－＊]', '', l.replace('¶', '')).strip()  # drop the Taishō variant apparatus
            if not l or l.startswith('#') or l.startswith('No.') or l.startswith('('): continue
            if l.startswith('*') or l.startswith('「') or re.match(r'^(如是我聞|聞如是|一時|爾時)', l): break
            out.append(l)
            if len(out) >= 8: break
        if len(out) >= 2: return out
    return []


def attached(l):
    l = re.sub(r'[（()）]', '', l)
    for m in re.finditer(r'(?:附|在)(.{1,3}?)錄', l):
        for k in sorted(ATTACH, key=len, reverse=True):
            if m.group(1).startswith(k): return ATTACH[k]
    d = next((k for k in KEYS if l.startswith(k)), None)  # 'X失譯', 'X代失譯人名': the catalogue placed the lost translator in X
    if d and l.find('失譯') <= len(d) + 1: return DYN2[d]
    return None


def parse(lines, authors, title):
    """returns byline, kind, lo, hi"""
    carry = None  # a dynasty on its own line ('北涼天竺沙門浮陀跋摩' / '共道泰等譯')
    for l in lines:
        if '《' in l or '卷第' in l or '。' in l or len(l) > 45: continue
        if title and (l.startswith(title[:3]) or title.startswith(l[:3])): continue  # the title line
        if '失譯' in l or re.search(r'(?:附|在).{1,3}錄', l):
            a = attached(l)
            if a: return l, 'attached', *a
            if '失譯' in l: return l, 'lost', np.nan, np.nan
        d = None if l.startswith(NOT_DYN) else next((k for k in KEYS if l.startswith(k)), None)
        if not END.search(l):
            if d and re.search(r'三藏|沙門|天竺|西域|罽賓|月支|月氏|安息|康居|于闐|龜茲|優婆塞|居士|國', l): carry = d  # not a place name (吳門, 山陰)
            continue
        d = d or carry
        if d == '宋':
            if any(n in l for n in LIU_SONG): return l, 'stated', 420, 479
            if ZHAO.search(l): return l, 'stated', 960, 1127
            return l, 'ambiguous', np.nan, np.nan
        if d: return l, 'stated', *DYN2[d]
        if l.rstrip('　 ').endswith('造'): continue  # Indian author line; the translator follows
        # a translator/author name right before the closing verb (…玄奘奉　詔譯, …道泰等譯)
        # translators only: author names of compilations (編, 集) collide with homonyms in the dated table
        name = next((a for a in authors if a in l[-(len(a) + 6):]), None) if re.search(r'譯[　 ]*$', l) else None
        if name: return l, 'stated', *authors[name]
    return (lines[1] if len(lines) > 1 else ''), 'none', np.nan, np.nan


def main():
    u = pd.read_csv(f'{D}/corpus3_f0/undated.tsv', sep='\t')
    u = u[u.cat.str.startswith('KR6')]
    m = pd.read_csv(f'{D}/meta.tsv', sep='\t')
    src = m.set_index('id').source
    # translator/author periods from the dated Buddhist texts (names of 2+ characters, consistent spans only)
    a = m[m.part.eq('KR6') & m.lo.notna() & m.author.notna() & (m.author.str.len() >= 2)]
    g = a.groupby('author').agg(lo=('lo', 'median'), hi=('hi', 'median'), n=('lo', 'size'), sd=('lo', 'std'))
    g = g[(g.sd.fillna(0) < 60) & (g.hi - g.lo < 400)]
    authors = {k: (r.lo, r.hi) for k, r in g.sort_index(key=lambda s: -s.str.len()).iterrows()}
    rows = []
    for r in u.itertuples():
        s = str(src.get(r.id, ''))
        vol = re.search(r'Taisho Tripitaka Vol\. (\d+)', s)
        if vol and 56 <= int(vol.group(1)) <= 84:
            rows.append(dict(id=r.id, title=r.title, source=s, byline='', kind='japan', b_lo=np.nan, b_hi=np.nan)); continue
        bl, kind, lo, hi = parse(header_lines(r.id), authors, re.sub(r'^(佛說|大方廣|大乘)', '', str(r.title)))
        rows.append(dict(id=r.id, title=r.title, source=s, byline=bl, kind=kind, b_lo=lo, b_hi=hi))
    b = pd.DataFrame(rows)
    b.to_csv(f'{P}/undated_bylines.tsv', sep='\t', index=False)
    print(b.kind.value_counts().to_dict())
    d = pd.read_csv(f'{R}/undated_dates.tsv', sep='\t', index_col=0)
    e = b.set_index('id').join(d[['median', 'q10', 'q90', 'seen12']])
    e['b_mid'] = (e.b_lo + e.b_hi) / 2; e['off'] = e['median'] - e.b_mid
    e['inside'] = (e['median'] >= e.b_lo - 50) & (e['median'] <= e.b_hi + 50)
    e['miss80'] = (e.q10 > e.b_hi + 50) | (e.q90 < e.b_lo - 50)
    e.to_csv(f'{R}/undated_byline_eval.tsv', sep='\t')
    pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 40)
    for kind in ['stated', 'attached']:
        for name, x in [('all', e[e.kind == kind]), ('seen12<=0.2', e[(e.kind == kind) & (e.seen12 <= .2)])]:
            print(f'{kind} ({name}): n={len(x)}; |model - byline midpoint| median {x.off.abs().median():.0f}; within ±50 y {x.inside.mean():.2f}; '
                  f'80% interval misses {x.miss80.mean():.2f}; model later than the byline period by median {x.off.median():.0f}; '
                  f'Spearman {x[["median", "b_mid"]].corr("spearman").iloc[0, 1]:.2f}')
    at = e[e.kind == 'attached']
    print('\nattached (今附X錄) by attached period: model median and share inside')
    print(at.groupby(['b_lo', 'b_hi']).agg(n=('median', 'size'), model=('median', 'median'), inside=('inside', 'mean'),
                                            later=('off', lambda s: (s > 50).mean())).to_string())
    lo_ = e[e.kind == 'lost']
    print(f'\nlost (失譯, no attachment): n={len(lo_)}; model medians quartiles {lo_["median"].quantile([.25, .5, .75]).tolist()}')
    print(lo_.sort_values('median')[['title', 'byline', 'median', 'q10', 'q90']].to_string())
    print('\nstated texts whose 80% interval misses the byline period:')
    print(e[(e.kind == 'stated') & e.miss80].sort_values('off')[['title', 'byline', 'b_lo', 'b_hi', 'median', 'q10', 'q90', 'seen12']].to_string())
    # same era only (byline midpoint 25-589), so that the two kinds are compared on the same footing
    for kind in ['stated', 'attached']:
        x = e[(e.kind == kind) & e.b_mid.between(25, 589) & (e.seen12 <= .2)]
        print(f'{kind}, byline period 25-589, seen12<=0.2: n={len(x)}; |model - midpoint| median {x.off.abs().median():.0f}; within ±50 y {x.inside.mean():.2f}')
    # text length: attached texts are short; median regression of |error| on kind and log length, same era
    import statsmodels.formula.api as smf
    n = pd.read_csv(f'{D}/corpus3_f0/undated.tsv', sep='\t').set_index('id').n_raw
    x = e[e.kind.isin(['stated', 'attached']) & e.b_mid.between(25, 589) & (e.seen12 <= .2)].copy()
    x['abserr'] = x.off.abs(); x['att'] = (x.kind == 'attached').astype(int); x['logn'] = np.log(n.reindex(x.index))
    q = smf.quantreg('abserr ~ att + logn', x).fit(q=.5)
    print(f'median regression |error| ~ attached + log length: attached +{q.params.att:.0f} y (p={q.pvalues.att:.4f}), n={len(x)}')
    x['lbin'] = pd.qcut(x.logn, 4, labels=False)
    print(x.groupby(['lbin', 'kind']).abserr.agg(['size', 'median']).unstack().round(0).to_string())
    jp = e[e.kind == 'japan']
    print(f'\nJapanese texts (Taishō 56-84): n={len(jp)}; model medians quartiles {jp["median"].quantile([.25, .5, .75]).tolist()}')


if __name__ == '__main__':
    main()
