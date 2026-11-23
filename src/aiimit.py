"""AI imitations of classical Chinese: the same frozen prompt table for every model (closed and open).

build    : writes data/aiimit_prompts_v0.tsv (target style, target year, form, title, prompt) -- frozen before generation
gptparts : writes the prompt sheets for models run by hand (GPT), in parts
prepare  : collects generated pieces (data/aiimit/<model>*.jsonl or *.txt, lines "id<TAB>text"), normalizes them
           (punctuation removed, simplified -> traditional where the traditional form is in the vocabulary, varmap),
           flags verbatim copying (12-grams found in the corpus), writes corpus3_f{k}/aiimit.{bin,tsv} for score.py
Usage: python aiimit.py build | gptparts | prepare
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, sys, json, glob
import numpy as np
import pandas as pd

P = (PROJECT + '/data')
D = (WORK + '/data')
RAW = (WORK + '/raw')

# (label, target year, form, style phrase); the year conventions follow imit.py TARGETS
VERSE = [('詩經', -700, '四言詩', '《詩經》國風'), ('古詩十九首', 150, '五言詩', '漢代《古詩十九首》'),
         ('曹植', 210, '五言詩', '曹植'), ('阮籍', 255, '五言詩', '阮籍《詠懷詩》'), ('陶淵明', 405, '五言詩', '陶淵明'),
         ('謝靈運', 430, '五言詩', '謝靈運'), ('鮑照', 455, '樂府詩', '鮑照'), ('庾信', 540, '五言詩', '庾信'),
         ('李白', 745, '七言古詩', '李白'), ('杜甫', 760, '五言律詩', '杜甫'), ('韓愈', 810, '七言古詩', '韓愈'),
         ('白居易', 820, '七言古詩', '白居易'), ('李商隱', 850, '七言律詩', '李商隱'), ('蘇軾', 1085, '七言古詩', '蘇軾'),
         ('陸游', 1180, '七言律詩', '陸游')]
FU = [('楚辭', -290, '騷體辭賦', '屈原《離騷》《九歌》'), ('司馬相如', -130, '賦', '司馬相如'),
      ('庾信賦', 540, '賦', '庾信'), ('蘇軾賦', 1085, '賦', '蘇軾《赤壁賦》')]
PROSE = [('莊子', -300, '散文', '《莊子》'), ('孟子', -300, '散文', '《孟子》'), ('史記', -95, '散文', '司馬遷《史記》'),
         ('柳宗元', 785, '散文', '柳宗元'), ('韓愈文', 810, '散文', '韓愈'), ('歐陽修', 1050, '散文', '歐陽修'),
         ('蘇軾文', 1085, '散文', '蘇軾')]
T_VERSE = ['秋夜', '送別', '登高', '春日', '江上', '月夜', '思歸', '田家', '遊山', '贈友', '夜雨', '懷古']
T_FU = ['秋風賦', '江山賦', '夜月賦', '雪賦', '歸田賦', '登臺賦']
T_PROSE = ['論學', '說樹', '遊西山記', '送友人序', '讀書說', '論交', '記夢', '隱者傳']
TAIL = '要求：使用繁體字；只輸出作品正文，不要標題、解說或註釋；須為原創，不得抄錄或拼接古人原句。'


def build():
    rows = []
    def add(kind, label, year, form, style, title, gclass):
        if style:
            p = f'請模仿{style}的風格，用文言創作一{"篇" if gclass != "verse" else "首"}{form}，題為《{title}》。'
        else:
            p = f'請用文言創作一{"篇" if gclass != "verse" else "首"}{form}，題為《{title}》。'
        rows.append(dict(id=f'ai{len(rows):04d}', kind=kind, label=label, target=year, form=form, gclass=gclass,
                         title=title, prompt=p + TAIL))
    for label, y, form, style in VERSE:
        for t in T_VERSE: add('imit', label, y, form, style, t, 'verse')
    for form in sorted({v[2] for v in VERSE}):
        for t in T_VERSE: add('ctrl', '', np.nan, form, '', t, 'verse')
    for label, y, form, style in FU:
        for t in T_FU: add('imit', label, y, form, style, t, 'fu')
    for t in T_FU: add('ctrl', '', np.nan, '賦', '', t, 'fu')
    for label, y, form, style in PROSE:
        for t in T_PROSE: add('imit', label, y, form, style, t, 'prose')
    for t in T_PROSE: add('ctrl', '', np.nan, '散文', '', t, 'prose')
    df = pd.DataFrame(rows)
    out = f'{P}/aiimit_prompts_v0.tsv'
    assert not os.path.exists(out), 'frozen: ' + out
    df.to_csv(out, sep='\t', index=False)
    print(len(df), 'prompts', df.groupby(['gclass', 'kind']).size().to_dict())


GPT_HEAD = ('下面有 {n} 道文言寫作題，每行一題，格式為「編號<TAB>題目」。請逐題獨立創作，全部完成。\n'
            '輸出格式：每篇作品佔一行，寫成「編號<TAB>正文」，正文中間不要換行（詩句之間用標點隔開即可）；'
            '不要輸出任何其他文字。不要聯網搜索。\n\n')


def gptparts(k=4):
    df = pd.read_csv(f'{P}/aiimit_prompts_v0.tsv', sep='\t').sample(frac=1, random_state=0)  # mixed order
    os.makedirs(f'{P}/aiimit', exist_ok=True)
    for j, ix in enumerate(np.array_split(np.arange(len(df)), k)):
        part = df.iloc[ix]
        body = '\n'.join(f'{r.id}\t{r.prompt}' for r in part.itertuples())
        open(f'{P}/aiimit/gpt_prompt_part{j + 1}.txt', 'w').write(GPT_HEAD.format(n=len(part)) + body + '\n')
        print(j + 1, len(part))


def s2t_map(vocab):
    """simplified -> traditional, only where the simplified form is outside the vocabulary and a traditional is in it."""
    inv = {c: k for k, c in enumerate(vocab)}; m = {}  # vocabulary is frequency-ordered: prefer the commonest
    for l in open(f'{RAW}/cjkvi-variants/cjkvi-simplified.txt', encoding='utf8'):
        if l.startswith('#') or not l.strip(): continue
        a, rel, b = l.rstrip('\n').split(',')[:3]
        if rel.endswith('simplified') and b not in inv and a in inv:
            if b not in m or inv[a] < inv[m[b]]: m[b] = a
    return m


def read_outputs():
    out = []
    for p in sorted(glob.glob(f'{P}/aiimit/out_*')):
        model = re.sub(r'^out_|(_(part|file)\d+)?\.(jsonl|txt)$', '', os.path.basename(p))
        for l in open(p, encoding='utf8'):
            l = l.strip()
            if not l: continue
            if p.endswith('.jsonl'):
                r = json.loads(l); i, t = r['id'], r['text']
            else:
                m = re.match(r'^(ai\d{4})\s+(.*)$', l)
                if not m: continue
                i, t = m.groups()
            out.append(dict(model=model, sample=0, id=i, text=t))
    df = pd.DataFrame(out)
    df['sample'] = df.groupby(['model', 'id']).cumcount()
    return df


def prepare():
    sys.path.insert(0, os.path.dirname(__file__))
    from corpus import ngram_hash, cover
    vocab = json.load(open(f'{D}/corpus3_f0/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    vm = json.load(open(f'{D}/varmap3.json')); s2t = s2t_map(vocab)
    pr = pd.read_csv(f'{P}/aiimit_prompts_v0.tsv', sep='\t').set_index('id')
    df = read_outputs()
    df = df[df.id.isin(pr.index)].reset_index(drop=True)
    toks, n_s2t, n_unk = [], [], []
    for t, ti in zip(df.text, df.id.map(pr.title)):
        t = re.sub(r'<think>.*?</think>', '', t, flags=re.S)
        ls = [l.strip() for l in re.split(r'\n+', t) if l.strip()]
        if ls and ti in ls[0] and len(ls[0]) <= len(ti) + 6: ls = ls[1:]  # a title line despite the instruction
        ls = [l for l in ls if not re.match(r'^[（(]?(註|注|說明|按|解析|賞析)', l)]  # explanatory notes
        t = '\n'.join(ls)
        cs = [c for c in t if '\u3400' <= c <= '\u9fff' or '\U00020000' <= c <= '\U0002ffff' or '\uf900' <= c <= '\ufaff']
        n_s2t.append(sum(c in s2t for c in cs))
        cs = [s2t.get(c, c) for c in cs]; cs = [vm.get(c, c) for c in cs]
        ids = np.array([c2i.get(c, 2) for c in cs[:600]], np.uint16)
        toks.append(ids); n_unk.append(int((ids == 2).sum()))
    df['n'] = [len(x) for x in toks]; df['n_s2t'] = n_s2t; df['n_unk'] = n_unk
    # verbatim copying: share of characters inside a 12-gram that also occurs in the corpus (fold-0 train + held-out)
    hs = [ngram_hash(x.astype(np.int64), 12) for x in toks]
    U = np.unique(np.concatenate(hs)); found = np.zeros(len(U), bool)
    for s in ['train', 'heldout']:
        data = np.memmap(f'{D}/corpus3_f0/{s}.bin', dtype=np.uint16, mode='r'); step = 50_000_000
        for o in range(0, len(data), step):
            h = ngram_hash(np.asarray(data[o:o + step + 11]).astype(np.int64), 12)
            j = np.searchsorted(U, h); j[j >= len(U)] = 0
            found[np.unique(j[U[j] == h])] = True
    F = U[found]
    cp = []
    for x, h in zip(toks, hs):
        if len(h) == 0: cp.append(0.0); continue
        j = np.searchsorted(F, h); j[j >= len(F)] = 0
        cp.append(float(cover(F[j] == h, 12, len(x)).mean()) if len(F) else 0.0)
    df['copied'] = cp
    df = df.join(pr[['kind', 'label', 'target', 'form', 'gclass', 'title']], on='id')
    df['uid'] = df.model + ':' + df.id + ':' + df['sample'].astype(str)
    pd.set_option('display.width', 200)
    print(df.groupby('model').agg(n=('n', 'size'), chars=('n', 'median'), s2t=('n_s2t', 'sum'), unk=('n_unk', 'sum'),
                                  copied=('copied', 'mean'), copied_gt20=('copied', lambda v: (v > .2).mean())).round(3).to_string())
    rows, arrs, off = [], [], 0
    for r, x in zip(df.itertuples(), toks):
        tok = np.concatenate([[1], x]).astype(np.uint16)
        rows.append(dict(id=r.uid, model=r.model, pid=r.id, sample=r.sample, kind=r.kind, label=r.label, target=r.target,
                         form=r.form, gclass=r.gclass, title=r.title, lo=np.nan, hi=np.nan, genre=3, n_raw=len(x),
                         n_s2t=r.n_s2t, n_unk=r.n_unk, copied=r.copied, offset=off, length=len(tok)))
        arrs.append(tok); off += len(tok)
    out = pd.DataFrame(rows)
    for f in range(5):  # same vocabulary in every fold; any fold model can score the split
        np.concatenate(arrs).tofile(f'{D}/corpus3_f{f}/aiimit.bin')
        out.to_csv(f'{D}/corpus3_f{f}/aiimit.tsv', sep='\t', index=False)
    df.drop(columns=['text']).to_csv(f'{P}/aiimit/pieces_meta.tsv', sep='\t', index=False)
    print('wrote', len(out), 'pieces')


if __name__ == '__main__':
    {'build': build, 'gptparts': gptparts, 'prepare': prepare}[sys.argv[1]]()
