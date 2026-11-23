"""The model's view of 2,500 years of lexical change, separated from topic.

Each character occurrence in a period-stratified sample of held-out passages (lex_tok.sh; every text scored by the
fold model that never saw it) gives a temporal profile: log p(x_i | context, t) over the 100-year grid, centred on its
mean over t. Holding the context fixed, the profile says in which era this word is most expected *here* -- time
separated from subject matter. Profiles are averaged per character and per two-character word (sum of the two
characters' profiles at each occurrence).
Outputs (runs/lexical_timeline/): profiles.npz; fig_wave.png (top time-specific words sorted by peak era);
fig_pairs.png (replacement curves); fig_registers.png (the same function words by register); era_table.tsv.
Usage: python lexical_timeline.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from freeze import register

FONT = (WORK + '/fonts/NotoSansCJKtc-Regular.otf')
font_manager.fontManager.addfont(FONT)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False
R = (WORK + '/runs')
D = (WORK + '/data')
OUT = f'{R}/lexical_timeline'
YEARS = np.arange(-1000, 1901, 25)[::4].astype(float)
KEEP = YEARS >= -600
Y = YEARS[KEEP]
MIN_CHAR, MIN_WORD = 200, 60
import re
LAYOUT = re.compile(r'[卷第首章篇]|^[一二三四五六七八九十百]+$|注|註|集卷')  # catalogue / layout tokens, not language
PAIRS = [('吾', '我'), ('曰', '云', '道', '說'), ('之', '底', '的'), ('也', '矣', '了'), ('其', '他'), ('何', '甚', '麼'),
         ('此', '這'), ('彼', '那'), ('皆', '都'), ('乃', '便', '就')]


def load():
    vocab = json.load(open(f'{D}/corpus3_f0/vocab.json')) + ['X']
    meta = pd.concat([pd.read_csv(f'{D}/corpus3_f{f}/heldout.tsv', sep='\t', usecols=['id', 'cat', 'genre', 'lo', 'hi'])
                      for f in range(5)]).drop_duplicates('id').set_index('id')
    occ = []
    for f in range(5):
        run = 'l3' if f == 0 else f'l3_f{f}'
        p = f'{R}/{run}/scores_heldout_unk_lexsample_tok.pkl'
        if not os.path.exists(p): continue
        for e in pickle.load(open(p, 'rb')):
            t = e['tok'].astype(np.int64); lp = e['lp'][:len(YEARS)][KEEP].astype(np.float32)
            m = t > 1
            occ.append((e['id'], t[m], (lp - lp.mean(0, keepdims=True))[:, m]))
    return vocab, meta, occ


def main():
    os.makedirs(OUT, exist_ok=True)
    vocab, meta, occ = load()
    V = len(vocab); G = len(Y)
    S = np.zeros((V, G)); N = np.zeros(V)
    reg = {}
    W = {}
    for i, t, c in occ:
        np.add.at(S, t, c.T); np.add.at(N, t, 1)
        rg = register(meta.at[i, 'cat'], meta.at[i, 'genre']) if i in meta.index else 'other'
        reg.setdefault(rg, [np.zeros((V, G)), np.zeros(V)])
        np.add.at(reg[rg][0], t, c.T); np.add.at(reg[rg][1], t, 1)
        key = t[:-1] * V + t[1:]; cc = (c[:, :-1] + c[:, 1:]).T
        for k, row in zip(key, cc):  # two-character words
            a = W.get(k)
            if a is None: W[k] = [row.copy(), 1]
            else: a[0] += row; a[1] += 1
    print('occurrences', int(N.sum()), 'texts', len({o[0] for o in occ}), flush=True)
    P = S / np.maximum(N, 1)[:, None]
    np.savez_compressed(f'{OUT}/profiles.npz', years=Y, P=P, N=N, vocab=np.array(vocab))
    # two-character words: keep frequent ones whose profile is more peaked than either character alone
    words = [(k, a[0] / a[1], a[1]) for k, a in W.items() if a[1] >= MIN_WORD]
    rows = []
    for k, prof, n in words:
        a, b = divmod(k, V)
        if a < 3 or b < 3: continue
        w = vocab[a] + vocab[b]
        rows.append(dict(item=w, n=n, peak=Y[int(np.argmax(prof))], spec=float(prof.max() - np.median(prof)), prof=prof))
    for v in range(3, V):
        if N[v] >= MIN_CHAR:
            rows.append(dict(item=vocab[v], n=int(N[v]), peak=Y[int(np.argmax(P[v]))], spec=float(P[v].max() - np.median(P[v])), prof=P[v]))
    df = pd.DataFrame(rows)
    df['len'] = df.item.str.len()
    df = df[~df.item.str.contains(LAYOUT)].reset_index(drop=True)
    print(len(df), 'items (characters + two-character words)')
    # era table: most time-specific items peaking in each 100-year column
    tab = []
    for y in Y:
        d = df[df.peak == y].sort_values('spec', ascending=False).head(12)
        tab.append(dict(year=int(y), n_items=int((df.peak == y).sum()), top=' '.join(d.item)))
    pd.DataFrame(tab).to_csv(f'{OUT}/era_table.tsv', sep='\t', index=False)
    for r in tab: print(r['year'], r['n_items'], r['top'])
    # figure A: wave of the most time-specific items, sorted by peak era (up to 6 per century, words preferred)
    sel = df.sort_values('spec', ascending=False).groupby('peak').head(6).sort_values(['peak', 'spec'], ascending=[True, False])
    M = np.stack(sel.prof.values)
    pd.DataFrame(M, columns=[int(y) for y in Y]).assign(item=sel.item.values, peak=sel.peak.values, spec=sel.spec.values, n=sel.n.values).to_csv(f'{OUT}/fig_wave_data.tsv', sep='\t', index=False)
    fig, ax = plt.subplots(figsize=(9, max(6, len(sel) * 0.11)))
    vmax = np.quantile(np.abs(M), 0.98)
    ax.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[Y[0] - 50, Y[-1] + 50, len(sel), 0])
    ax.set_yticks(np.arange(len(sel)) + 0.5); ax.set_yticklabels(sel.item, fontsize=5)
    ax.set_xlabel('年代（條件年代，固定上下文）'); ax.set_title('語言變遷波浪圖：各字詞在同一上下文中最「像」哪個時代', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_wave.png', dpi=200); plt.close(fig)
    # figure B: replacement curves
    c2i = {c: k for k, c in enumerate(vocab)}
    rows_ = [dict(group='／'.join(g), char=ch, n=int(N[c2i[ch]]), **{str(int(y)): float(v) for y, v in zip(Y, P[c2i[ch]])}) for g in PAIRS for ch in g if ch in c2i]
    pd.DataFrame(rows_).to_csv(f'{OUT}/fig_pairs_data.tsv', sep='\t', index=False)
    fig, axs = plt.subplots(2, 5, figsize=(14, 5.5), sharex=True)
    for ax, grp in zip(axs.flat, PAIRS):
        for ch in grp:
            if ch in c2i and N[c2i[ch]] >= 50: ax.plot(Y, P[c2i[ch]], label=f'{ch} (n={int(N[c2i[ch]])})')
        ax.axhline(0, color='grey', lw=0.5); ax.legend(fontsize=7, frameon=False); ax.set_title('／'.join(grp), fontsize=9)
    fig.suptitle('字的傳記：同一上下文中各年代的相對對數概率（替代關係）', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_pairs.png', dpi=200); plt.close(fig)
    # figure C: the same function words by register (freezing)
    regs = [r for r in ['別集', '史部', '子部', '佛典譯經', '佛教撰述', '禪宗語錄', '小說筆記', '詞曲'] if r in reg]
    focus = ['之', '也', '矣', '其', '曰', '我', '是', '底', '了', '這']
    rr = [dict(char=ch, register=rg, n=int(reg[rg][1][c2i[ch]]), **{str(int(y)): float(v) for y, v in zip(Y, reg[rg][0][c2i[ch]] / max(reg[rg][1][c2i[ch]], 1))}) for ch in focus if ch in c2i for rg in regs]
    pd.DataFrame(rr).to_csv(f'{OUT}/fig_registers_data.tsv', sep='\t', index=False)
    fig, axs = plt.subplots(2, 5, figsize=(14, 5.5), sharex=True)
    for ax, ch in zip(axs.flat, focus):
        if ch not in c2i: continue
        for rg in regs:
            Sx, Nx = reg[rg]
            if Nx[c2i[ch]] >= 30: ax.plot(Y, Sx[c2i[ch]] / Nx[c2i[ch]], label=rg, lw=1)
        ax.axhline(0, color='grey', lw=0.5); ax.set_title(ch, fontsize=10)
    axs.flat[0].legend(fontsize=6, frameon=False)
    fig.suptitle('同一字在不同語域中的年代曲線', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_registers.png', dpi=200); plt.close(fig)
    df.drop(columns='prof').to_csv(f'{OUT}/items.tsv', sep='\t', index=False)
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
