"""Build the dated corpus with leakage control.

Stages
1. labels + roles (train / heldout / blind / e4 / excluded)
2. normalize characters (varmap), tokenize
3. chronological quotation masking: in each text, delete spans covered by a 12-gram first attested
   in a strictly earlier 25-year bin (the text's "own language" remains)
4. protection masking: delete from training texts every 8-gram occurring in a blind/e4 text and every
   12-gram occurring in a held-out text's residual
Output: data/corpus/{split}.bin (uint16 tokens) + {split}.tsv (per-text index) + vocab.json
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, json, zipfile, collections
import numpy as np
import pandas as pd

D = (WORK + '/data')
P = (PROJECT + '/data')
RAW = (WORK + '/raw')
VER = os.environ.get('LETTER_V', '')
SUF = os.environ.get('LETTER_CSUFFIX', '')  # variant corpora, e.g. stricter quotation masking
OUT = f'{D}/corpus{VER}{SUF}'
PAD, SEP, UNK = 0, 1, 2
N_CHRON, N_PROT = int(os.environ.get('LETTER_NCHRON', 12)), 8
FOLDS = [int(f) for f in os.environ.get('LETTER_FOLDS', '0,1,2,3,4').split(',')]
BIN = 25
K = 5
EXCL_CATS = {'KR3k', 'KR4h', 'KR2h', 'KR6s', 'KR6t', 'KR6v'}
GENRES = ['經', '史', '子', '集', '道', '佛譯', '佛撰']
# E4: archaizing works with secure dates (imitation target in note)
E4 = {'KR3a0009': '揚雄法言 擬論語', 'KR3g0001': '揚雄太玄 擬易', 'KR5f0017': '太玄集註本',
      'KR3a0014': '王通中說 擬論語', 'KR3g0004': '司馬光潛虛 擬太玄', 'KR3g0003': '衛元嵩元包 擬歸藏'}
# 尚書: chapters forged in the 4th c. (偽古文) vs. 今文 chapters
GUWEN = set('大禹謨 五子之歌 胤征 仲虺之誥 湯誥 伊訓 太甲上 太甲中 太甲下 咸有一德 說命上 說命中 說命下 '
            '泰誓上 泰誓中 泰誓下 武成 旅獒 微子之命 蔡仲之命 周官 君陳 畢命 君牙 冏命'.split())
ZHOU_GAO = set('大誥 康誥 酒誥 梓材 召誥 洛誥 多士 無逸 君奭 多方 立政 顧命 康王之誥'.split())


def genre_of(row):
    p = row['part']
    if p == 'KR1': return 0
    if p == 'KR2': return 1
    if p == 'KR3': return 2
    if p == 'KR4': return 3
    if p == 'KR5': return 4
    if p == 'KR6': return 5 if '譯' in str(row['function']) else 6
    return 2


def load_labels():
    m = pd.read_csv(f'{D}/meta.tsv', sep='\t')
    from meta import DYN
    DYN = dict(DYN, 戦國=(-475, -221), 前晉=(265, 420))
    # person span inconsistent with the catalog dynasty -> fall back to the dynasty span
    for i, r in m.iterrows():
        d = r['dyn'] if isinstance(r['dyn'], str) else ''
        if d in DYN and pd.notna(r['lo']) and r['date_src'] not in ('dyn', 'text_date'):
            a, b = DYN[d]
            if r['hi'] < a - 30 or r['lo'] > b + 30:
                m.loc[i, ['lo', 'hi', 'date_src']] = [a, b, 'dyn_fix']
        if pd.isna(r['lo']) and d in DYN:
            m.loc[i, ['lo', 'hi', 'date_src']] = [DYN[d][0], DYN[d][1], 'dyn']
    # catalog '宋' for Buddhist translators/authors of the Liu Song (420-479), found by the audit (v3 fix)
    LIU_SONG = {'求那跋摩', '求那跋陀羅', '釋寶雲', '寶雲', '法海', '畺良耶舍', '曇摩蜜多', '功德直', '先公', '沮渠京聲',
                '慧簡', '翔公', '竺道生', '智嚴', '曇無竭', '慧嚴', '僧伽跋摩', '劉義慶', '謝靈運', '鮑照', '顏延之'}
    if os.environ.get('LETTER_V', '') >= '3':
        # only for texts whose span came from the dynasty name (person dates, if any, are kept)
        sel = (m.dyn == '宋') & m.author.isin(LIU_SONG) & m.date_src.isin(['dyn', 'dyn_fix'])
        m.loc[sel, ['lo', 'hi', 'date_src']] = [420, 479, 'liu_song_fix']
    man = pd.read_csv(f'{P}/manual_dates.tsv', sep='\t').set_index('id')
    for i in man.index:
        m.loc[m.id == i, ['lo', 'hi', 'date_src']] = [man.loc[i, 'lo'], man.loc[i, 'hi'], 'manual']
    m['genre'] = m.apply(genre_of, axis=1)
    return m


def commentary_only(cat_rows):
    from meta import AUTHOR_FN, COMMENT_FN
    out = set()
    for r in cat_rows:
        fns = [p.get('function', '') for p in r['persons']]
        if fns and not any((f in AUTHOR_FN) or (f and not any(f.startswith(c) for c in COMMENT_FN)) for f in fns):
            out.add(r['id'])
    return out


def assign_roles(m, seed=0):
    cat_rows = [json.loads(l) for l in open(f'{D}/catalog_raw.jsonl')]
    comm = commentary_only(cat_rows)
    blind = pd.read_csv(f'{P}/blind_v0.tsv', sep='\t')
    bl = set(blind.id)
    man = set(pd.read_csv(f'{P}/manual_dates.tsv', sep='\t').id)
    # v3: editions of an author's collection carrying later commentary inline (e.g. 李太白集注, 九家集注杜詩):
    # labelled with the author's date but containing later language -> excluded
    annotated = set()
    if os.environ.get('LETTER_V', '') >= '3':
        pers = {r['id']: r['persons'] for r in cat_rows}
        for _, r in m.iterrows():
            if r['part'] in ('KR2', 'KR3', 'KR4') and re.search(r'註|注|箋|考異', str(r['title'])):
                if any(re.search(r'注|註|箋|釋|補|集', p.get('function', '')) for p in pers.get(r['id'], [])[1:]):
                    annotated.add(r['id'])
        annotated |= {'KR5c0139'}  # 南華真經註疏: 郭象注 with 成玄英's Tang sub-commentary inline
        print('annotated editions excluded:', len(annotated))
    role = []
    for _, r in m.iterrows():
        if r['id'] in bl: role.append('blind')
        elif r['id'] in annotated: role.append('excluded')
        elif r['id'] in E4: role.append('e4')
        elif r['foreign'] or r['cat'] in EXCL_CATS or pd.isna(r['lo']): role.append('excluded')
        elif r['id'] in comm and r['id'] not in man: role.append('excluded')
        else: role.append('train')
    m['role'] = role
    # K folds over author groups, stratified by period (fold k = held-out set of model k)
    rng = np.random.default_rng(seed)
    tr = m[m.role == 'train'].copy()
    tr['akey'] = tr.apply(lambda r: f"{r['author']}|{r['dyn']}" if isinstance(r['author'], str) and r['author'] not in ('', '闕名', '失譯', '佚名') else r['id'], axis=1)
    tr['mid'] = (tr.lo + tr.hi) / 2
    tr['period'] = pd.cut(tr.mid, [-2000, -221, 220, 589, 907, 1279, 1368, 1644, 3000], labels=False)
    fold_of = {}
    for p, g in tr.groupby('period'):
        keys = sorted(set(g.akey)); rng.shuffle(keys)
        # balance folds by characters: greedy assignment of author groups to the lightest fold
        size = g.groupby('akey').size()
        load_ = np.zeros(K)
        for k_ in keys:
            f = int(np.argmin(load_)); fold_of[k_] = f; load_[f] += size[k_]
    m['fold'] = -1
    m.loc[tr.index, 'fold'] = [fold_of[k_] for k_ in tr.akey]
    return m


def load_texts(ids):
    want = set(ids); texts = {}
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']:
        for line in open(f'{D}/extract/{part}.jsonl'):
            r = json.loads(line)
            if r['id'] in want and r.get('juans'):
                texts[r['id']] = r['juans']
    return texts


def split_shangshu():
    """尚書正文 (TLS edition) -> one text per chapter, 書序 removed."""
    from extract import clean_juan
    z = zipfile.ZipFile(f'{RAW}/krzip/KR1b0001.zip')
    out = []
    for n in sorted(x for x in z.namelist() if x.endswith('.txt')):
        raw = z.read(n).decode('utf8')
        mt = re.search(r'^\*\* \d+ 《(.+?)》', raw, re.M)
        if not mt: continue
        title = mt.group(1)
        body = raw[mt.end():]
        # drop the 書序 paragraph (ends with 作《…》) when it precedes the chapter text
        body = re.sub(r'^.*?作《[^》]*》[^\n]*\n', '', body, count=1, flags=re.S) if '作《' in body[:400] else body
        t = clean_juan(body)
        if title == '舜典':  # 姚方興's 28 added characters (5th c.)
            t = t.replace('曰若稽古帝舜曰重華協于帝濬哲文明溫恭允塞玄德升聞乃命以位', '')
        cls = 'guwen' if title in GUWEN else ('zhougao' if title in ZHOU_GAO else 'jinwen')
        out.append((f'KR1b0001_{len(out) + 1:02d}', title, cls, t))
    return out


def ngram_hash(tok, n):
    """Polynomial rolling hash of every n-gram (uint64, wraps). len = len(tok)-n+1."""
    t = tok.astype(np.uint64)
    L = len(t) - n + 1
    if L <= 0: return np.zeros(0, np.uint64)
    h = np.zeros(L, np.uint64)
    B = np.uint64(1000003)
    with np.errstate(over='ignore'):
        for k in range(n):
            h = h * B + t[k:k + L]
    return h


def cover(flag, n, L):
    """positions covered by any n-gram whose start index is flagged."""
    d = np.zeros(L + 1, np.int32)
    idx = np.flatnonzero(flag)
    np.add.at(d, idx, 1); np.add.at(d, idx + n, -1)
    return np.cumsum(d[:L]) > 0


def member(h, sorted_u):
    if len(h) == 0 or len(sorted_u) == 0: return np.zeros(len(h), bool)
    j = np.searchsorted(sorted_u, h); j[j >= len(sorted_u)] = 0
    return sorted_u[j] == h


def compact(t, k, minlen=8):
    """keep runs of k (not crossing SEP) of length >= minlen, each preceded by SEP."""
    k = k & (t != SEP)
    kk = np.concatenate([[False], k, [False]]).astype(np.int8)
    d = np.diff(kk)
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    parts = []
    for a, b in zip(st, en):
        if b - a >= minlen:
            parts.append(np.array([SEP], np.uint16)); parts.append(t[a:b])
    return np.concatenate(parts) if parts else np.zeros(0, np.uint16)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = assign_roles(load_labels())
    print(m.role.value_counts()); print(m[m.role == 'train'].fold.value_counts())
    use = m[m.role != 'excluded'].copy()
    texts = load_texts(use.id)
    vm = json.load(open(f'{D}/varmap{VER}.json'))
    # Shangshu chapters replace KR1b0001
    ss = split_shangshu()
    rows = []
    for _, r in use.iterrows():
        if r['id'] == 'KR1b0001' or r['id'] not in texts: continue
        rows.append(dict(id=r['id'], title=r['title'], role=r['role'], fold=r['fold'], lo=r['lo'], hi=r['hi'], genre=r['genre'],
                         part=r['part'], cat=r['cat'], author=r['author'], date_src=r['date_src'],
                         text=''.join(vm.get(c, c) for c in 'Ｓ'.join(texts[r['id']]))))
    for cid, title, cls, t in ss:
        rows.append(dict(id=cid, title='尚書·' + title, role='blind', fold=-1, lo=np.nan, hi=np.nan, genre=0, part='KR1',
                         cat='KR1b', author=cls, date_src='shangshu', text=''.join(vm.get(c, c) for c in t)))
    df = pd.DataFrame(rows)
    # vocab from train texts
    cnt = collections.Counter()
    for t in df[df.role == 'train'].text: cnt.update(t)
    chars = [c for c, n in cnt.most_common() if n >= 20 and c != 'Ｓ']
    vocab = ['<pad>', '<sep>', '<unk>'] + chars
    c2i = {c: i for i, c in enumerate(vocab)}
    c2i['Ｓ'] = SEP
    json.dump(vocab, open(f'{OUT}/vocab.json', 'w'), ensure_ascii=False)
    print('vocab', len(vocab), 'coverage', sum(cnt[c] for c in chars) / sum(cnt.values()))
    toks = [np.array([c2i.get(c, UNK) for c in t], np.uint16) for t in df.text]
    df = df.drop(columns='text')
    df['n_raw'] = [len(t) for t in toks]

    # --- chronological masking (texts with labels: train, heldout, e4) ---
    chron = df.role.isin(['train', 'e4']).values
    mids = ((df.lo + df.hi) / 2).values
    bins = np.where(chron, np.floor((np.nan_to_num(mids) + 2000) / BIN), 0).astype(np.int32)
    H, Bn, owner = [], [], []
    for i in np.flatnonzero(chron):
        h = ngram_hash(toks[i], N_CHRON); H.append(h); Bn.append(np.full(len(h), bins[i], np.int32))
        owner.append(np.full(len(h), i, np.int32))
    H = np.concatenate(H); Bn = np.concatenate(Bn); owner = np.concatenate(owner)
    print('chron hashes', len(H), flush=True)
    order = np.argsort(H, kind='stable')
    Hs = H[order]
    starts = np.concatenate([[0], np.flatnonzero(Hs[1:] != Hs[:-1]) + 1])
    gmin = np.minimum.reduceat(Bn[order], starts)
    gid = np.repeat(np.arange(len(starts)), np.diff(np.concatenate([starts, [len(Hs)]])))
    earliest = np.empty(len(H), np.int32); earliest[order] = gmin[gid]
    quoted_start = earliest < Bn
    Hu = Hs[starts].copy()
    del order, Hs, gid, earliest
    keep = [np.ones(len(t), bool) for t in toks]
    pos = 0
    for i in np.flatnonzero(chron):
        L = len(toks[i]); nh = max(0, L - N_CHRON + 1)
        keep[i] &= ~cover(quoted_start[pos:pos + nh], N_CHRON, L); pos += nh
    del H, Bn, owner, quoted_start

    # --- protection masking shared by all folds: blind and e4 texts ---
    prot8 = np.unique(np.concatenate([ngram_hash(toks[i], N_PROT) for i in np.flatnonzero(df.role.isin(['blind', 'e4']).values)]))
    print('protected 8-grams', len(prot8), flush=True)
    tr_idx = np.flatnonzero((df.role == 'train').values)
    base = {}
    for i in tr_idx:
        base[i] = keep[i] & ~cover(member(ngram_hash(toks[i], N_PROT), prot8), N_PROT, len(toks[i]))
    # --- residual of blind texts: remove every 12-gram attested in any dated text ---
    for i in np.flatnonzero((df.role == 'blind').values):
        h = ngram_hash(toks[i], N_CHRON)
        keep[i] &= ~cover(member(h, Hu), N_CHRON, len(toks[i]))
    del Hu
    h12 = {i: ngram_hash(toks[i], N_CHRON) for i in tr_idx}
    df['n_keep'] = [int(k.sum()) for k in keep]

    def write(split_rows, name, var, keepfn, out):
        arrs, offs, off = [], [], 0
        for i in split_rows:
            t = compact(toks[i], keepfn(i) if var == 'res' else np.ones(len(toks[i]), bool))
            offs.append((off, len(t))); arrs.append(t); off += len(t)
        a = np.concatenate(arrs) if arrs else np.zeros(0, np.uint16)
        a.tofile(f'{out}/{name}.bin')
        sub = df.iloc[split_rows].copy()
        sub['offset'] = [o for o, _ in offs]; sub['length'] = [l for _, l in offs]
        sub.to_csv(f'{out}/{name}.tsv', sep='\t', index=False)
        print(out.split('/')[-1], name, len(split_rows), 'texts', a.size / 1e6, 'M tokens', flush=True)

    folds = df.fold.values
    for f in FOLDS:
        out = f'{D}/corpus{VER}{SUF}_f{f}'
        os.makedirs(out, exist_ok=True)
        json.dump(vocab, open(f'{out}/vocab.json', 'w'), ensure_ascii=False)
        held = tr_idx[folds[tr_idx] == f]; train = tr_idx[folds[tr_idx] != f]
        prot12 = []
        for i in held:
            ok = np.convolve(keep[i].astype(np.int32), np.ones(N_CHRON, np.int32), 'valid') == N_CHRON
            prot12.append(h12[i][ok[:len(h12[i])]])
        prot12 = np.unique(np.concatenate(prot12))
        write(train, 'train', 'res', lambda i: base[i] & ~cover(member(h12[i], prot12), N_CHRON, len(toks[i])), out)
        write(held, 'heldout', 'res', lambda i: keep[i], out)
        for split in ['blind', 'e4']:
            rows_ = np.flatnonzero((df.role == split).values)
            for var in ['full', 'res']:
                write(rows_, f'{split}_{var}', var, lambda i: keep[i], out)
    print(df.groupby(['role', 'fold'])[['n_raw', 'n_keep']].sum())


if __name__ == '__main__':
    main()
