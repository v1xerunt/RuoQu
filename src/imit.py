"""Limits of imitation: extract titled pieces from literary collections (別集) and flag imitations.

WYG layout: genre headings are indented by 2 full-width spaces, piece titles by 3, text starts at column 0.
Imitation = explicit title markers (擬 效 倣 仿 追和 和陶 集句 補亡 學…體 代…作 古意). The imitated
model's date is parsed from the title where possible. Controls = the same author's other pieces.
Writes corpus2_f{k}/imit.{bin,tsv}: pieces from the collections held out in fold k.
Usage: python imit.py test <id>   |   python imit.py build
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, sys, json, zipfile
import numpy as np
import pandas as pd
from extract import clean_juan, juan_key

RAW = (WORK + '/raw')
D = (WORK + '/data')
V = os.environ.get('LETTER_V', '2')
IMIT = re.compile(r'擬|效|斆|倣|仿|追和|和陶|集句|補亡|學.{0,8}體|^代.{0,10}作|古意|雜體')
# a genre heading passes its imitation marker (and target) on to its pieces only if it looks like a genre label,
# not a misparsed poem title (e.g. 讀五代史記作古樂府五首 in a collection with an unusual layout)
GENRE_OK = re.compile(r'^.{0,6}(詩|賦|詞|辭|騷|樂府|歌|行|體|古|陶)$')
COUNT = re.compile(r'[一二三四五六七八九十百千]+[首章篇]$')
MAX_SHARE = 0.3  # a collection with more than this share of 'imitations' is a parse failure: dropped and reported
# (regex on the title, approximate date of the imitated model); first match wins
TARGETS = [
    # 江淹雜體三十首 official titles of the imitated poets
    (r'李都尉|班婕妤', -40), (r'古離別|古別離|古離别', 150), (r'魏文帝|陳思王|劉文學|王侍中', 215), (r'嵇中散|阮步兵', 255),
    (r'張司空|潘黃門|陸平原|左記室|張黃門', 295), (r'劉太尉|盧郎中|郭弘農', 315), (r'孫廷尉|許徵君', 350),
    (r'殷東陽|謝僕射|陶徵君', 405), (r'謝臨川|顏特進|謝法曹|王徵君', 435), (r'袁太尉|謝光祿|鮑參軍|休上人', 455),
    (r'補亡|三百篇|詩經|風雅|國風|周頌|商頌', -700), (r'尚書|典謨|誥', -900),
    (r'騷|楚辭|九歌|九章|九辯|招魂|天問|離騷|屈原|宋玉', -290),
    (r'李都尉|李陵|蘇武|班婕妤', -50), (r'古詩|擬古|古意|十九首|古樂府|樂府', 150),
    (r'建安|曹植|曹丕|魏文帝|陳思|王粲|劉楨|徐幹|應瑒|阮瑀|公幹|仲宣', 210),
    (r'阮籍|嵇康|詠懷|叔夜|嗣宗', 255), (r'張華|潘岳|陸機|左思|太沖|安仁|士衡|張協|劉琨|郭璞|遊仙', 290),
    (r'陶|淵明|靖節|彭澤|五柳|孫綽|許詢', 405), (r'謝靈運|康樂|謝惠連|顏延之|謝莊', 430),
    (r'鮑照|鮑參軍|明遠|湯惠休', 455), (r'謝朓|玄暉|沈約|休文|永明|江淹|文通', 490),
    (r'齊梁|宮體|徐陵|庾信|子山|陰鏗|何遜|選體', 540),
    (r'王維|摩詰|孟浩然|李白|太白|青蓮', 745), (r'杜甫|少陵|工部|子美|老杜|杜陵', 760),
    (r'韋應物|韋蘇州|柳宗元|子厚', 785), (r'韓愈|昌黎|退之|孟郊|東野|賈島', 810),
    (r'白居易|樂天|香山|長慶|元稹|微之|李賀|長吉|昌谷', 820),
    (r'李商隱|義山|玉溪|溫庭筠|飛卿|西崑|韓偓|香奩', 850),
    (r'梅堯臣|宛陵|歐陽|六一|蘇舜欽', 1050), (r'東坡|蘇軾|子瞻|山谷|黃庭堅|魯直|涪翁', 1085),
    (r'陸游|放翁|楊萬里|誠齋|范成大|石湖', 1180),
]


def target_of(title):
    for pat, y in TARGETS:
        if re.search(pat, title): return y
    return np.nan


def pieces(zid):
    z = zipfile.ZipFile(f'{RAW}/krzip/{zid}.zip')
    names = sorted([n for n in z.namelist() if re.search(r'_\d+[a-z]?\.txt$', n)], key=juan_key)
    out, genre, title, buf = [], '', None, []
    parent = ''
    def flush():
        if title is not None and buf:
            txt = clean_juan('\n'.join(buf))
            if len(txt) >= 20: out.append((genre, title, txt))
    for n in names:
        if juan_key(n)[0] == 0: continue
        for l in z.read(n).decode('utf8', 'replace').split('\n'):
            if l.startswith('#') or l.startswith('*'): continue
            s = re.sub(r'<pb:[^>]*>|¶', '', l)
            if not s.strip(): continue
            lead = len(s) - len(s.lstrip('　'))
            core = re.sub(r'\([^)]*\)', '', s.strip('　 '))
            if '四庫全書' in core or re.search(r'卷[第之]?[一二三四五六七八九十百\d]+$', core): continue
            if lead == 2 and len(core) <= 12:
                flush(); genre, title, buf = core, None, []; parent = ''
            elif lead == 3 and len(core) <= 40:
                flush(); title, buf = core, []; parent = core
            elif lead == 4 and len(core) <= 24 and title is not None:
                flush(); title, buf = parent + '／' + core, []  # sub-piece of a titled group
            elif lead >= 4 and len(core) <= 30:
                continue  # author / editor credit lines
            else:
                buf.append(s)
    flush()
    return out


def classify_genre(genre, title):
    g = genre + title
    if re.search(r'詩|歌|行|吟|曲|謠|詞|樂府|絶句|絕句|律|篇$', g): return 'verse'
    if re.search(r'賦|騷|辭', g): return 'fu'
    return 'prose'


def build():
    vm = json.load(open(f'{D}/varmap{V}.json'))
    vocab = json.load(open(f'{D}/corpus{V}_f0/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    rng = np.random.default_rng(0)
    for f in range(5):
        h = pd.read_csv(f'{D}/corpus{V}_f{f}/heldout.tsv', sep='\t')
        h = h[h.cat.isin(['KR4b', 'KR4c', 'KR4d', 'KR4e', 'KR4f', 'KR4g']) & ((h.hi - h.lo) <= 120)]
        rows, arrs, off = [], [], 0
        for _, r in h.iterrows():
            try: ps = pieces(r.id)
            except Exception as e: continue
            yr = (r.lo + r.hi) / 2
            gok = lambda g: bool(GENRE_OK.search(COUNT.sub('', g)))
            is_im = lambda g, t: bool(IMIT.search(t) or (gok(g) and IMIT.search(g)))
            tgt = lambda g, t: target_of(t) if not np.isnan(target_of(t)) else (target_of(g) if gok(g) else np.nan)
            imit = [(g, t, x) for g, t, x in ps if is_im(g, t) and tgt(g, t) <= r.lo - 100]
            if not imit: continue
            if len(imit) > MAX_SHARE * len(ps):
                print(f'  dropped {r.id} {r.title}: {len(imit)}/{len(ps)} pieces flagged (parse failure?)', flush=True); continue
            ctrl = [(g, t, x) for g, t, x in ps if not is_im(g, t)]
            ctrl = [ctrl[k] for k in rng.permutation(len(ctrl))[:max(3 * len(imit), 20)]]
            for kind, lst in (('imit', imit), ('ctrl', ctrl)):
                for k, (g, t, x) in enumerate(lst):
                    tok = np.array([1] + [c2i.get(vm.get(c, c), 2) for c in x[:600]], np.uint16)
                    rows.append(dict(id=f'{r.id}#{kind}{k:04d}', work=r.id, author=r.author, lo=r.lo, hi=r.hi,
                                     genre=3, kind=kind, gclass=classify_genre(g, t), title=t,
                                     target=tgt(g, t) if kind == 'imit' else np.nan, n_raw=len(tok) - 1,
                                     offset=off, length=len(tok)))
                    arrs.append(tok); off += len(tok)
        if rows:
            np.concatenate(arrs).tofile(f'{D}/corpus{V}_f{f}/imit.bin')
            df = pd.DataFrame(rows); df.to_csv(f'{D}/corpus{V}_f{f}/imit.tsv', sep='\t', index=False)
            print(f'fold {f}: {df.work.nunique()} collections, {(df.kind == "imit").sum()} imitations '
                  f'({df[df.kind == "imit"].target.notna().sum()} with a parsed model date), {(df.kind == "ctrl").sum()} controls', flush=True)


def build_genuine(width=60, per_coll=150):
    """'Ceiling' pieces: ordinary (non-imitation) pieces of collections whose author lived near an imitated
    model's date (|mid - target| <= width), held out in fold k -> corpus{V}_f{k}/genuine.{bin,tsv}.
    How early the model dates a genuine target-era piece bounds how far any imitation could move."""
    vm = json.load(open(f'{D}/varmap{V}.json'))
    vocab = json.load(open(f'{D}/corpus{V}_f0/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    targets = np.array(sorted({y for _, y in TARGETS}), float)
    rng = np.random.default_rng(1)
    for f in range(5):
        h = pd.read_csv(f'{D}/corpus{V}_f{f}/heldout.tsv', sep='\t')
        h = h[h.cat.isin(['KR4a', 'KR4b', 'KR4c', 'KR4d', 'KR4e', 'KR4f', 'KR4g']) & ((h.hi - h.lo) <= 150)]
        mid = (h.lo + h.hi) / 2
        near = np.abs(mid.values[:, None] - targets[None]).min(1) <= width
        rows, arrs, off = [], [], 0
        for _, r in h[near].iterrows():
            try: ps = pieces(r.id)
            except Exception: continue
            ps = [(g, t, x) for g, t, x in ps if not (IMIT.search(t) or IMIT.search(g))]
            ps = [ps[k] for k in rng.permutation(len(ps))[:per_coll]]
            yr = (r.lo + r.hi) / 2
            tg = targets[np.abs(targets - yr).argmin()]
            for k, (g, t, x) in enumerate(ps):
                tok = np.array([1] + [c2i.get(vm.get(c, c), 2) for c in x[:600]], np.uint16)
                rows.append(dict(id=f'{r.id}#gen{k:04d}', work=r.id, author=r.author, lo=r.lo, hi=r.hi, genre=3,
                                 kind='genuine', gclass=classify_genre(g, t), title=t, target=tg, n_raw=len(tok) - 1,
                                 offset=off, length=len(tok)))
                arrs.append(tok); off += len(tok)
        if rows:
            np.concatenate(arrs).tofile(f'{D}/corpus{V}_f{f}/genuine.bin')
            df = pd.DataFrame(rows); df.to_csv(f'{D}/corpus{V}_f{f}/genuine.tsv', sep='\t', index=False)
            print(f'fold {f}: {df.work.nunique()} collections, {len(df)} pieces; targets {sorted(df.target.unique())}', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'test':
        ps = pieces(sys.argv[2])
        tg = lambda g, t: target_of(t) if not np.isnan(target_of(t)) else target_of(g)
        im = [(g, t, len(x)) for g, t, x in ps if (IMIT.search(t) or IMIT.search(g)) and not np.isnan(tg(g, t))]
        print(len(ps), 'pieces;', len(im), 'imitations')
        for g, t, n in im[:40]: print(f'  [{g}] {t} ({n}) target={tg(g, t)}')
    elif sys.argv[1] == 'genuine':
        build_genuine()
    else:
        build()
