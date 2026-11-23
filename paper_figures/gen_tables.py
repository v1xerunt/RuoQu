"""Generate LaTeX table fragments from the result tables."""
import os, re
import numpy as np
import pandas as pd
from style import PKG, OUT as FIG_OUT

OUT = os.path.join(os.path.dirname(FIG_OUT), 'tables')
os.makedirs(OUT, exist_ok=True)
R = lambda *p: os.path.join(PKG, *p)


def esc(s):
    s = '' if (s is None or (isinstance(s, float) and np.isnan(s))) else str(s)
    s = s.replace('\\', '/').replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#')
    s = s.replace('~', '–').replace('^', '').replace('$', '\\$')
    return s


def yr(v):
    v = int(round(v))
    return f'{-v} BCE' if v < 0 else f'{v}'


def rng_(s):
    a, b = [float(x) for x in str(s).split('~')]
    return f'{yr(a)}–{yr(b)}'


# ---------------- Extended Data Table 2: blind test ----------------
EN = {'列子': 'Liezi', '關尹子': 'Guanyinzi', '亢倉子': 'Kangcangzi', '子華子': 'Zihuazi', '元經': 'Yuanjing',
      '李衛公問對': 'Li Weigong wendui', '孔叢子': 'Kongcongzi', '古文孝經孔傳': 'Kong commentary, Xiaojing',
      '忠經': 'Zhongjing', '西京雜記': 'Xijing zaji', '神異經': 'Shenyi jing', '漢武帝內傳': 'Han Wudi neizhuan',
      '洞冥記': 'Dongming ji', '列仙傳': 'Liexian zhuan', '偽古文尚書': 'Old Text Shangshu',
      '尉繚子': 'Weiliaozi', '六韜': 'Liutao', '晏子春秋': 'Yanzi chunqiu', '鶡冠子': 'Heguanzi', '文子': 'Wenzi',
      '孫子': 'Sunzi', '老子': 'Laozi', '穆天子傳': 'Mu tianzi zhuan', '孔子家語': 'Kongzi jiayu',
      '鬼谷子': 'Guiguzi', '尹文子': 'Yinwenzi', '鄧析子': 'Dengxizi', '鬻子': 'Yuzi', '三略': 'Sanlüe',
      '握奇經': 'Woqi jing', '竹書紀年': 'Zhushu jinian', '參同契': 'Cantong qi', '焦氏易林': 'Jiaoshi yilin',
      '搜神記': 'Soushen ji'}
bt = pd.read_csv(R('runs', 'blind_ensemble_table.tsv'), sep='\t')
lines = []
heads = {'A': 'Forgeries (scholarly consensus later than the attribution)',
         'B': 'Texts once doubted and vindicated by excavated manuscripts',
         'D': 'Disputed works (no consensus; model dates reported for reference)'}
for cat in ['A', 'B', 'D']:
    d = bt[bt.cat == cat].sort_values('p_later', ascending=False)
    lines.append(f'\\multicolumn{{6}}{{l}}{{\\textit{{{heads[cat]}}}}}\\\\')
    for r in d.itertuples():
        lines.append(f'{EN.get(r.work, r.work)} {r.work} & {rng_(r.attr)} & {rng_(r.cons)} & {r.p_later:.2f} & '
                     f'{yr(r.median)} & {yr(r.q10)} to {yr(r.q90)}\\\\')
    lines.append('\\addlinespace')
open(os.path.join(OUT, 'ed_blind.tex'), 'w', encoding='utf-8').write('\n'.join(lines))

# ---------------- Supplementary Table 1: audit cases ----------------
CAT = {'1': 'Confirmed', '2': 'Contested', '3': 'New candidate', '4': 'Catalogue or edition',
       '5a': 'Model error, explained', '5b': 'Unexplained'}
c = pd.read_csv(R('data', 'audit_cases_v1.tsv'), sep='\t')
c['k'] = c.category.str.split(' ').str[0]
order = ['1', '2', '3', '4', '5a', '5b']
lines = []
for k in order:
    d = c[c.k == k]
    lines.append(f'\\multicolumn{{6}}{{l}}{{\\textbf{{{CAT[k]}}} ({len(d)})}}\\\\')
    for r in d.itertuples():
        side = 'later' if r.side == '晚于目录' else 'earlier'
        cat_ = esc(r.catalogue).replace('–', '–')
        lines.append(f'{esc(r.title)} & {side} & {esc(r.attribution)} & {cat_} & {esc(r.model)} & '
                     f'{{\\scriptsize {esc(r.scholarly)}}}\\\\')
    lines.append('\\addlinespace')
open(os.path.join(OUT, 'supp_audit.tex'), 'w', encoding='utf-8').write('\n'.join(lines))

# ---------------- Supplementary Table 2: tier-A new dates ----------------
nd = pd.read_csv(R('data', 'new_dates_v0.tsv'), sep='\t')
nd = nd.rename(columns={nd.columns[0]: 'id'})
A = nd[(nd.tier == 'A') & (~nd.layered) & (~nd.apocryphal) & (~nd.unreliable_cat)].sort_values('median')


def src(s):
    s = str(s)
    m = re.match(r'Taisho Tripitaka Vol\. (\d+), No\. (\S+)', s)
    if m: return f'T{m[2]}'
    m = re.match(r'.*Xuzangjing Vol\. (\d+), No\. (\S+)', s)
    if m: return f'X{m[2]}'
    m = re.match(r'Buddhist Texts not contained in the Tripitaka Vol\. (\d+), No\. (\S+)', s)
    if m: return f'ZW{m[2]}'
    if s == 'nan': return ''
    return esc(s[:28])


lines = []
for r in A.itertuples():
    lines.append(f'{r.id} & {esc(r.title)} & {src(r.source)} & {r.n_raw:,} & {yr(r.median)} & {yr(r.q10)} to {yr(r.q90)}\\\\')
open(os.path.join(OUT, 'supp_newdates.tex'), 'w', encoding='utf-8').write('\n'.join(lines))
print(len(A), 'tier-A unflagged;', len(c), 'audit cases')
