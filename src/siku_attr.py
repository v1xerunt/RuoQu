"""The model vs the Siku editors (1773-1782): do the books whose attribution the 四庫提要 doubts read later?

External label from each work's 提要 (juan 000 of the Kanripo Siku files, removed from the training text):
  formula 舊本題 (the editors' standard phrasing for an attribution they do not vouch for)
  doubt words (依託, 偽託, 偽撰, 託名, 殆出, 後人所作, 宋以後書, ... in the 提要)
  doubted = doubt words, or 舊本題 together with a doubt word or 未必/不可考/莫知
Model side: OOF median - label midpoint, and the cross-fitted audit flags (fixed beforehand).
Usage: python siku_attr.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import re, zipfile
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from extract import juan_key

RAW = (WORK + '/raw')
R = (WORK + '/runs')
DOUBT = re.compile(r'依託|依托|偽託|僞託|偽撰|僞撰|偽作|僞作|偽書|僞書|贋|假託|託名|嫁名|殆出|後人所(作|託|撰|增|竄|附)|後人(依託|偽|僞|附益|竄入)|'
                   r'(宋|元|明)以後(之)?書|(宋|元|明)人所(作|撰|託|為)|非其(原|舊)本|非.{1,4}所(作|撰|著)|不出於?.{1,4}之手|真偽|真僞')
SOFT = re.compile(r'未必|不可考|莫知|疑|附會')


def tiyao(zid):
    try:
        z = zipfile.ZipFile(f'{RAW}/krzip/{zid}.zip')
    except Exception:
        return None
    names = sorted([n for n in z.namelist() if re.search(r'_\d+[a-z]?\.txt$', n)], key=juan_key)
    j0 = [n for n in names if juan_key(n)[0] == 0]
    if not j0: return None
    t = z.read(j0[0]).decode('utf8', 'replace')
    t = re.sub(r'<pb:[^>]*>|¶|^#.*$|\([^)]*\)', '', t, flags=re.M)
    t = re.sub(r'\s+', '', t)
    i = t.find('提要')
    return t[i:] if i >= 0 else None


def main():
    o = pd.read_csv(f'{R}/l3_oof.tsv', sep='\t')
    a = pd.read_csv(f'{R}/l3_audit.tsv', sep='\t').set_index('id')
    rows = []
    for r in o.itertuples():
        if r.cat[:3] not in ('KR1', 'KR2', 'KR3', 'KR4', 'KR5'): continue
        t = tiyao(r.id)
        if not t or len(t) < 50: continue
        jiu = '舊本題' in t[:120]
        dw = DOUBT.findall(t); soft = bool(SOFT.search(t))
        rows.append(dict(id=r.id, title=r.title, cat=r.cat, lo=r.lo, hi=r.hi, median=r.median, date_src=r.date_src,
                         jiu=jiu, doubt=bool(dw), doubted=bool(dw) or (jiu and soft),
                         flag_late=bool(a.flag_late.get(r.id, False)), flag_early=bool(a.flag_early.get(r.id, False))))
    df = pd.DataFrame(rows)
    df['mid'] = (df.lo + df.hi) / 2; df['offset'] = df['median'] - df.mid
    df.to_csv(f'{R}/siku_attr.tsv', sep='\t', index=False)
    pd.set_option('display.width', 200)
    print(len(df), 'Siku works with a 提要 and an OOF date;', int(df.jiu.sum()), '舊本題;', int(df.doubted.sum()), 'doubted')
    for name, sub in [('all', df), ('label before 960', df[df.mid < 960]), ('label 960+', df[df.mid >= 960])]:
        x, y = sub[sub.doubted], sub[~sub.doubted]
        if len(x) < 5: continue
        u = mannwhitneyu(x.offset, y.offset, alternative='greater')
        print(f'== {name}: doubted n={len(x)} vs {len(y)}; median offset {x.offset.median():.0f} vs {y.offset.median():.0f}; '
              f'AUC {u.statistic / len(x) / len(y):.3f}, p={u.pvalue:.2g}; late-flag rate {x.flag_late.mean():.3f} vs {y.flag_late.mean():.3f}; '
              f'early-flag {x.flag_early.mean():.3f} vs {y.flag_early.mean():.3f}')
    fl = df[df.flag_late]
    print(f'late-flagged Siku works: {len(fl)}; doubted by the editors {fl.doubted.mean():.2f} (base rate {df.doubted.mean():.2f})')
    print('late-flagged but NOT doubted by the editors (candidates the editors missed, or model errors):')
    print(fl[~fl.doubted][['id', 'title', 'lo', 'hi', 'median', 'date_src']].to_string(index=False))
    print('doubted by the editors, model agrees with the catalogue (|offset| < 50):')
    print(df[df.doubted & (df.offset.abs() < 50)][['id', 'title', 'lo', 'hi', 'median']].head(30).to_string(index=False))


if __name__ == '__main__':
    main()
