"""Baselines on the same passages as the TC-LM.

  ngram   : period-specific character n-gram LMs (interpolated absolute discounting), the standard in DH
  fw      : function-word frequencies + multinomial logistic regression (stylometry)
  lr      : character 1-3-gram hashed features + logistic regression (strong discriminative baseline)
  first   : "earliest attestation" (terminus post quem from the latest-first-attested character bigram/trigram)
Every method writes per-passage log-scores over the same year bins: runs/baselines/<method>_<split>.npz
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, argparse
import numpy as np
import pandas as pd
from score import passages

C = (WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
_CORP = __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
OUT = (WORK + '/runs/baselines') + ('' if _CORP == 'corpus_f0' else '_' + _CORP)
PAD, SEP = 0, 1
EDGES = np.array([-1100, -500, -221, -100, 0, 100, 220, 317, 420, 500, 589, 700, 800, 907, 1000, 1100, 1200,
                  1279, 1368, 1450, 1550, 1644, 1720, 1800, 1912, 1950])
MIDS = (EDGES[:-1] + EDGES[1:]) / 2
NB = len(MIDS)
FW = ('之 乎 者 也 矣 焉 哉 兮 耳 而 以 於 于 其 所 為 則 乃 且 若 或 莫 弗 勿 毋 不 未 非 亦 皆 咸 俱 悉 凡 蓋 夫 豈 寧 何 奚 '
      '曷 胡 安 惡 孰 誰 盍 與 歟 邪 耶 爾 云 然 斯 此 彼 是 茲 夫 吾 我 余 予 朕 汝 女 爾 若 乃 厥 其 之 諸 旃 '
      '將 已 既 曾 嘗 方 始 遂 因 故 是以 是故 然則 然而 雖 縱 苟 如 使 令 被 見 為 所 把 將 得 了 著 個 箇 這 那 '
      '甚 麼 什 便 都 只 祇 卻 就 才 纔 還 又 也 底 的 地 呢 嗎 罷 吧 些 們 每 自 從 向 在 到 去 來 裏 裡 上 下 中 '
      '頗 甚 極 最 太 稍 略 殊 良 頗 尤 益 愈 彌 更 復 再 亦 又 並 共 同 相 互 交 遞 迭 惟 唯 維 但 特 徒 直 '
      '乎哉 也哉 也夫 矣夫 云爾 而已 而已矣 也者 者也 之謂 所謂 不亦 何以 何為 奈何 如何 何如 若何 得無 無乃 '
      '是時 爾時 於是 於時 即時 須臾 然後 而後 以後 以前 之後 之前 如此 若此 如是 若是 云何 云云 便是 就是 只是 '
      '不曾 未曾 不得 不可 不能 可以 能夠 應當 應該 須要 必須 一切 一個 一箇 這個 那個 甚麼 什麼 怎麼 怎生 如今 而今').split()
FW = list(dict.fromkeys(FW))


def bin_of(year):
    return np.clip(np.searchsorted(EDGES, year, side='right') - 1, 0, NB - 1)


def label_bins(lo, hi):
    """fraction of the interval in each bin"""
    w = np.clip(np.minimum(EDGES[1:], hi + 1e-6) - np.maximum(EDGES[:-1], lo), 0, None)
    return w / w.sum() if w.sum() > 0 else np.eye(NB)[bin_of((lo + hi) / 2)]


def load(split):
    data = np.memmap(f'{C}/{split}.bin', dtype=np.uint16, mode='r')
    idx = pd.read_csv(f'{C}/{split}.tsv', sep='\t')
    return data, idx


def eval_passages(split, P=256, K=24):
    data, idx = load(split)
    out = []
    for _, r in idx.iterrows():
        tok = np.asarray(data[r.offset:r.offset + r.length])
        for pi, p in enumerate(passages(tok, P, K)):
            out.append((r.id, pi, p[(p != SEP) & (p != PAD)]))
    return out


def train_passages(P=256, per_bin=20000, seed=0):
    """balanced sample of training passages per bin (label = bin of a random year in the interval)"""
    data, idx = load('train')
    idx = idx[idx.length >= P].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fr = np.stack([label_bins(a, b) for a, b in zip(idx.lo, idx.hi)])  # texts x bins
    X, y = [], []
    for b in range(NB):
        w = fr[:, b] * idx.length.values
        if w.sum() == 0: continue
        ii = rng.choice(len(idx), size=per_bin, p=w / w.sum())
        for i in ii:
            s = idx.offset[i] + rng.integers(0, idx.length[i] - P + 1)
            p = np.asarray(data[s:s + P]); X.append(p[p != SEP]); y.append(b)
    return X, np.array(y)


# ---------------- character n-gram LMs per bin ----------------
HB = np.uint64(1000003)


def nhash(t, n):
    """hashes of all n-grams of t (uint64), and a mask of n-grams not containing SEP"""
    t64 = t.astype(np.uint64); L = len(t) - n + 1
    if L <= 0: return np.zeros(0, np.uint64), np.zeros(0, bool)
    h = np.zeros(L, np.uint64)
    with np.errstate(over='ignore'):
        for k in range(n): h = h * HB + t64[k:k + L] + np.uint64(1)
    ok = np.convolve((t == SEP).astype(np.int32), np.ones(n, np.int32), 'valid') == 0
    return h, ok


def lookup(u, c, h):
    j = np.searchsorted(u, h); j[j >= len(u)] = 0
    return np.where(u[j] == h, c[j], 0).astype(np.float64)


def build_ngram(toks, order):
    m = {'N': sum(int((t != SEP).sum()) for t in toks)}
    for n in range(1, order + 1):
        hs, pre = [], []
        for t in toks:
            h, ok = nhash(t, n)
            hs.append(h[ok])
            if n > 1:
                hp, _ = nhash(t[:-1], n - 1); pre.append(hp[:len(h)][ok])
        hs = np.concatenate(hs)
        u, first, c = np.unique(hs, return_index=True, return_counts=True)
        m[n] = (u, c)
        if n > 1:  # number of distinct continuations of each (n-1)-gram context
            pu, pc = np.unique(np.concatenate(pre)[first], return_counts=True)
            m[f't{n - 1}'] = (pu, pc)
    return m


def score_ngram(t, m, order, V, D=0.75):
    """log-likelihood of t (no SEP) under an interpolated absolute-discounting model"""
    h1, _ = nhash(t, 1)
    c1 = lookup(*m[1], h1)
    prob = (np.maximum(c1 - D, 0) + D * len(m[1][0]) / V) / m['N']
    for n in range(2, order + 1):
        hn, _ = nhash(t, n)
        hc, _ = nhash(t[:-1], n - 1); hc = hc[:len(hn)]
        cn = lookup(*m[n], hn); cc = lookup(*m[n - 1], hc); ty = lookup(*m[f't{n - 1}'], hc)
        low = prob[n - 1:]
        new = np.where(cc > 0, (np.maximum(cn - D, 0) + D * ty * low) / np.maximum(cc, 1), low)
        prob = np.concatenate([prob[:n - 1], new])
    return np.log(np.maximum(prob, 1e-12)).sum()


def ngram(split_list, order=4):
    data, idx = load('train')
    V = int(data.max()) + 1
    tb = np.array([bin_of((a + b) / 2) for a, b in zip(idx.lo, idx.hi)])
    models = []
    for b in range(NB):
        sel = np.flatnonzero(tb == b)
        if len(sel) == 0: models.append(None); continue
        toks = [np.asarray(data[idx.offset[i]:idx.offset[i] + idx.length[i]]) for i in sel]
        models.append(build_ngram(toks, order))
        print('bin', b, MIDS[b], 'texts', len(sel), 'chars', models[-1]['N'], flush=True)
    for split in split_list:
        ps = eval_passages(split)
        LL = np.full((len(ps), NB), -1e9, np.float32)
        for k, (_, _, p) in enumerate(ps):
            for b, m in enumerate(models):
                if m is not None and len(p) > 0: LL[k, b] = score_ngram(p.astype(np.int64), m, order, V)
        np.savez_compressed(f'{OUT}/ngram_{split}.npz', ids=np.array([a for a, _, _ in ps]),
                            pi=np.array([b for _, b, _ in ps]), ll=LL, nt=np.array([len(p) for _, _, p in ps]), mids=MIDS)
        print('ngram', split, len(ps), flush=True)


# ---------------- linear models ----------------
def featurize(ps, kind, vocab):
    from scipy.sparse import csr_matrix
    rows, cols, vals = [], [], []
    if kind == 'fw':
        fw_idx = {w: i for i, w in enumerate(FW)}
        for k, p in enumerate(ps):
            s = ''.join(vocab[c] for c in p)
            n = max(len(s), 1)
            for w, i in fw_idx.items():
                c = s.count(w)
                if c: rows.append(k); cols.append(i); vals.append(c / n * 100)
        return csr_matrix((vals, (rows, cols)), shape=(len(ps), len(FW)))
    # hashed char 1-3 grams, sublinear tf
    Hdim = 1 << 21
    for k, p in enumerate(ps):
        t = p.astype(np.int64)
        feats = [t, t[:-1] * 20011 + t[1:], (t[:-2] * 20011 + t[1:-1]) * 20011 + t[2:]]
        f = np.concatenate([(feats[0] * 3 + 0), (feats[1] * 3 + 1), (feats[2] * 3 + 2)]) % Hdim
        u, c = np.unique(f, return_counts=True)
        v = 1 + np.log(c); v = v / np.sqrt((v ** 2).sum())
        rows.extend([k] * len(u)); cols.extend(u.tolist()); vals.extend(v.tolist())
    return csr_matrix((vals, (rows, cols)), shape=(len(ps), Hdim))


def linear(kind, split_list):
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    vocab = json.load(open(f'{C}/vocab.json'))
    X, y = train_passages(per_bin=6000 if kind == 'lr' else 6000)
    Xf = featurize(X, kind, vocab)
    if kind == 'fw':
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler(with_mean=False).fit(Xf); Xf = sc.transform(Xf)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xf, y)
    else:
        clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=20, tol=None, random_state=0).fit(Xf, y)
    print(kind, 'train acc', (clf.predict(Xf) == y).mean(), flush=True)
    for split in split_list:
        ps = eval_passages(split)
        F = featurize([p for _, _, p in ps], kind, vocab)
        if kind == 'fw': F = sc.transform(F)
        lp = np.full((len(ps), NB), -1e9, np.float32)
        lp[:, clf.classes_] = clf.predict_log_proba(F)
        np.savez_compressed(f'{OUT}/{kind}_{split}.npz', ids=np.array([a for a, _, _ in ps]),
                            pi=np.array([b for _, b, _ in ps]), ll=lp, nt=np.array([len(p) for _, _, p in ps]), mids=MIDS)
        print(kind, split, len(ps), flush=True)


# ---------------- earliest attestation ----------------
def first(split_list, min_texts=2):
    """For every char bigram/trigram: earliest bin in which it occurs in >= min_texts training texts.
    Passage TPQ = latest such first-attestation bin among its n-grams (score: one-hot log-score)."""
    data, idx = load('train')
    hashes = lambda t, n: nhash(t, n)[0][nhash(t, n)[1]]
    res = {}
    for n in (2, 3):
        H, Bb = [], []
        for i in range(len(idx)):
            t = np.asarray(data[idx.offset[i]:idx.offset[i] + idx.length[i]])
            h = np.unique(hashes(t, n))
            H.append(h); Bb.append(np.full(len(h), bin_of(idx.lo[i]), np.int16))  # earliest possible year of text
        H = np.concatenate(H); Bb = np.concatenate(Bb)
        o = np.lexsort((Bb, H)); H, Bb = H[o], Bb[o]
        st = np.concatenate([[0], np.flatnonzero(H[1:] != H[:-1]) + 1])
        # earliest bin with >= min_texts texts: the (min_texts)-th smallest bin within each group
        cnt = np.diff(np.concatenate([st, [len(H)]]))
        ok = cnt >= min_texts
        kth = Bb[np.minimum(st + min_texts - 1, len(H) - 1)]
        res[n] = (H[st][ok], kth[ok])
        print('first', n, ok.sum(), flush=True)
    for split in split_list:
        ps = eval_passages(split)
        LL = np.full((len(ps), NB), -1e9, np.float32); tpq = np.zeros(len(ps))
        for k, (_, _, p) in enumerate(ps):
            fb = []
            for n in (2, 3):
                u, b = res[n]; h = hashes(p, n)
                j = np.searchsorted(u, h); j[j >= len(u)] = 0
                fb.append(b[j][u[j] == h])
            fb = np.concatenate(fb)
            # robust TPQ: 2nd latest first-attestation (a single n-gram can be a copying artifact)
            t = np.sort(fb)[-2] if len(fb) >= 2 else 0
            tpq[k] = t
            LL[k, t:] = 0  # uniform over bins not earlier than the TPQ
        np.savez_compressed(f'{OUT}/first_{split}.npz', ids=np.array([a for a, _, _ in ps]),
                            pi=np.array([b for _, b, _ in ps]), ll=LL, nt=np.array([len(p) for _, _, p in ps]),
                            mids=MIDS, tpq=tpq)
        print('first', split, len(ps), flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    method = sys.argv[1]; splits = sys.argv[2:] or ['heldout', 'blind_full', 'blind_res', 'e4_full', 'e4_res']
    {'ngram': ngram, 'first': first}.get(method, lambda s: linear(method, s))(splits)
