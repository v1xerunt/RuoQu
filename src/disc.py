"""Discriminative control (Ithaca-style): the same Transformer encoder predicts a distribution over
year bins (25-year grid) directly; training target is uniform over the label interval.
Train:  python disc.py train --name disc_m --size m
Score:  python disc.py score --name disc_m --split heldout  -> scores_<split>_unk.npz with the
        log-probabilities over the grid in 'll' (same format as score.py, so evaluate.py applies).
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, math, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Block
from train import Sampler, SIZES, C
from score import passages, GRID

R = (WORK + '/runs')
PAD, SEP = 0, 1


class Disc(nn.Module):
    def __init__(self, vocab, d, L, H, ctx, nbins, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, H, drop) for _ in range(L)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, nbins)
        for b in self.blocks:  # bidirectional encoder
            b.causal = False

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        m = (x != PAD).float()[..., None]
        for b in self.blocks:
            B, T, Cd = h.shape
            q, k, v = b.qkv(b.ln1(h)).view(B, T, 3, b.h, Cd // b.h).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=(x != PAD)[:, None, None, :],
                                               dropout_p=b.drop if self.training else 0.0)
            h = h + F.dropout(b.proj(a.transpose(1, 2).reshape(B, T, Cd)), b.drop, self.training)
            h = h + F.dropout(b.fc2(F.gelu(b.fc1(b.ln2(h)))), b.drop, self.training)
        h = (self.ln(h) * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(h)


def target(lo, hi):
    t = ((GRID[None] >= lo[:, None] - 12.5) & (GRID[None] <= hi[:, None] + 12.5)).astype(np.float32)
    empty = t.sum(1) == 0
    t[empty, np.abs(GRID[None] - ((lo + hi) / 2)[empty, None]).argmin(1)] = 1
    return t / t.sum(1, keepdims=True)


def train(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = f'{R}/{args.name}'; os.makedirs(out, exist_ok=True)
    vocab = json.load(open(f'{C}/vocab.json'))
    d, L, H = SIZES[args.size]
    model = Disc(len(vocab) + 1, d, L, H, args.T, len(GRID)).to(dev)
    json.dump(dict(vars(args), corpus=C, vocab=len(vocab) + 1, d=d, L=L, H=H), open(f'{out}/config.json', 'w'))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=dev.type == 'cuda')
    steps = int(args.tokens / (args.bs * args.T)); warm = max(1, min(2000, steps // 20))
    sch = lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    tr = Sampler('train', args.T, args.alpha, args.seed, max_ep=args.max_ep, tokens=args.tokens)
    t0 = time.time()
    try:
        for step in range(steps):
            for g in opt.param_groups: g['lr'] = args.lr * sch(step)
            x, _, _ = tr.batch(args.bs)
            ii = tr.last_idx
            tgt = torch.from_numpy(target(tr.idx.lo.values[ii].astype(float), tr.idx.hi.values[ii].astype(float))).to(dev)
            x = torch.from_numpy(np.where(x == SEP, PAD, x)).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                lg = model(x)
            loss = -(tgt * F.log_softmax(lg.float(), -1)).sum(1).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % 200 == 0:
                print(f'step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
            if step % 5000 == 0 and step:
                torch.save(model.state_dict(), f'{out}/model.pt')
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    torch.save(model.state_dict(), f'{out}/model.pt')


def score(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = json.load(open(f'{R}/{args.name}/config.json'))
    model = Disc(cfg['vocab'], cfg['d'], cfg['L'], cfg['H'], cfg['T'], len(GRID)).to(dev)
    model.load_state_dict(torch.load(f'{R}/{args.name}/model.pt', map_location=dev)); model.eval()
    C2 = cfg['corpus']
    data = np.memmap(f'{C2}/{args.split}.bin', dtype=np.uint16, mode='r')
    idx = pd.read_csv(f'{C2}/{args.split}.tsv', sep='\t')
    rows, LL, NT = [], [], []
    for _, r in idx.iterrows():
        tok = np.asarray(data[r.offset:r.offset + r.length]).astype(np.int64)
        ps = passages(tok, args.P, args.K)
        if not ps: continue
        x = np.zeros((len(ps), args.P), np.int64)
        for k, p in enumerate(ps): x[k, :len(p)] = np.where(p == SEP, PAD, p)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            lp = F.log_softmax(model(torch.from_numpy(x).to(dev)).float(), -1).cpu().numpy()
        for k, p in enumerate(ps):
            rows.append((r.id, k)); NT.append(int((p > 1).sum())); LL.append(lp[k])
    # store as log p(bin|x) - log prior(uniform): evaluate.py treats it as a log-score over the grid
    np.savez_compressed(f'{R}/{args.name}/scores_{args.split}_unk.npz', ids=np.array([a for a, _ in rows]),
                        pi=np.array([b for _, b in rows]), ll=np.array(LL, np.float32), nt=np.array(NT), grid=GRID)
    print('saved', len(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('--name', required=True); ap.add_argument('--size', default='m')
    ap.add_argument('--tokens', type=float, default=1e9); ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--T', type=int, default=256); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--alpha', type=float, default=0.3); ap.add_argument('--max_ep', type=float, default=8)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--split', default='heldout')
    ap.add_argument('--P', type=int, default=256); ap.add_argument('--K', type=int, default=24)
    a = ap.parse_args()
    train(a) if a.mode == 'train' else score(a)
