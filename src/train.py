"""Train the time-conditioned char LM.

Sampling: a text is drawn with probability proportional to its length re-weighted so that each
50-year bin gets mass ∝ (bin chars)^alpha (period balancing); a window is drawn uniformly within it;
the conditioning year is drawn uniformly inside the text's date interval.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, json, math, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from model import TCLM
from views import table

C = (WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
PAD, SEP = 0, 1
SIZES = {'xs': (64, 2, 4), 's': (384, 6, 6), 'm': (512, 8, 8), 'l': (768, 12, 12), 'xl': (1024, 16, 16)}


def text_weights(idx, alpha, bin_w=50, max_ep=None, tokens=None):
    lo, hi, n = idx.lo.values, idx.hi.values, idx.length.values.astype(float)
    edges = np.arange(-1200, 2001, bin_w)
    mass = np.zeros(len(edges) - 1)
    frac = []
    for a, b, k in zip(lo, hi, n):
        w = np.clip(np.minimum(edges[1:], b + 1e-6) - np.maximum(edges[:-1], a), 0, None)
        w = w / w.sum() if w.sum() > 0 else w
        frac.append(w); mass += k * w
    target = np.where(mass > 0, mass ** alpha, 0); target /= target.sum()
    if max_ep and tokens:  # cap repetition of sparse bins, give the excess to the others
        cap = max_ep * mass / tokens
        for _ in range(50):
            over = target > cap
            if not over.any(): break
            excess = (target[over] - cap[over]).sum()
            target[over] = cap[over]
            free = (~over) & (mass > 0)
            target[free] += excess * target[free] / target[free].sum()
    scale = np.where(mass > 0, target / np.maximum(mass, 1e-9), 0)
    wt = np.array([k * (f * scale).sum() for k, f in zip(n, frac)])
    return wt / wt.sum(), edges, mass, target


class Sampler:
    def __init__(self, split, T, alpha, seed, min_len=64, max_ep=None, tokens=None):
        self.data = np.memmap(f'{C}/{split}.bin', dtype=np.uint16, mode='r')
        idx = pd.read_csv(f'{C}/{split}.tsv', sep='\t')
        idx = idx[idx.length >= min_len].reset_index(drop=True)
        self.idx = idx
        self.w, *_ = text_weights(idx, alpha, max_ep=max_ep, tokens=tokens)
        self.T = T
        self.rng = np.random.default_rng(seed)

    def batch(self, B):
        ii = self.rng.choice(len(self.idx), size=B, p=self.w)
        self.last_idx = ii
        x = np.zeros((B, self.T), np.int64)
        for b, i in enumerate(ii):
            off, n = self.idx.offset[i], self.idx.length[i]
            if n > self.T:
                s = off + self.rng.integers(0, n - self.T + 1)
                x[b] = self.data[s:s + self.T]
            else:
                x[b, :n] = self.data[off:off + n]
        lo, hi = self.idx.lo.values[ii], self.idx.hi.values[ii]
        year = lo + self.rng.random(B) * (hi - lo)
        genre = self.idx.genre.values[ii]
        return x, year.astype(np.float32), genre.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default='m')
    ap.add_argument('--name', required=True)
    ap.add_argument('--tokens', type=float, default=2e9)
    ap.add_argument('--bs', type=int, default=64)  # per GPU
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--max_ep', type=float, default=8)
    ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--p_drop_date', type=float, default=0.1)
    ap.add_argument('--p_drop_genre', type=float, default=0.1)
    ap.add_argument('--compile', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--view', default='full')
    ap.add_argument('--eval', type=int, default=1)
    args = ap.parse_args()

    ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if ddp:
        dist.init_process_group('nccl'); rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    else:
        rank, world = 0, 1
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = f'{WORK}/runs/{args.name}'
    os.makedirs(out, exist_ok=True)

    vocab = json.load(open(f'{C}/vocab.json'))
    d, L, H = SIZES[args.size]
    model = TCLM(len(vocab) + 1, d, L, H, ctx=args.T, drop=args.drop).to(dev)
    remap = table(args.view)
    n_param = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f'params {n_param / 1e6:.1f}M', flush=True)
        json.dump(dict(vars(args), corpus=C, vocab=len(vocab) + 1, d=d, L=L, H=H, n_param=n_param), open(f'{out}/config.json', 'w'))
    fwd = torch.compile(model) if args.compile else model
    if ddp:
        fwd = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[int(os.environ['LOCAL_RANK'])])
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1}, {'params': nodecay, 'weight_decay': 0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=dev.type == 'cuda')
    steps = int(args.tokens / (args.bs * args.T * world))
    warm = max(1, min(2000, steps // 20))
    sch = lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    tr = Sampler('train', args.T, args.alpha, args.seed * 100 + rank, max_ep=args.max_ep, tokens=args.tokens)
    va = Sampler('heldout', args.T, args.alpha, 12345)
    vb = [va.batch(args.bs) for _ in range(8)]
    G = 7

    def run(x, year, genre, train):
        if remap is not None: x = remap[x]
        x = torch.from_numpy(x).to(dev, non_blocking=True)
        year = torch.from_numpy(year).to(dev); genre = torch.from_numpy(genre).to(dev)
        B = x.shape[0]
        if train:
            known = torch.rand(B, device=dev) >= args.p_drop_date
            genre = torch.where(torch.rand(B, device=dev) < args.p_drop_genre, torch.full_like(genre, G), genre)
        else:
            known = torch.ones(B, dtype=torch.bool, device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            logits = fwd(x, year, known, genre)
        m = (x != PAD) & (x != SEP)
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x.view(-1), reduction='none')
        return (loss * m.view(-1)).sum() / m.sum()

    t0 = time.time(); log = open(f'{out}/log.txt', 'a') if rank == 0 else None
    try:
      for step in range(steps):
        for g in opt.param_groups: g['lr'] = args.lr * sch(step)
        model.train()
        loss = run(*tr.batch(args.bs), True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if rank == 0 and (step % 200 == 0 or step == steps - 1):
            msg = f'step {step}/{steps} loss {loss.item():.4f} lr {args.lr * sch(step):.2e} {time.time() - t0:.0f}s'
            if step % 2000 == 0 or step == steps - 1:
                model.eval()
                with torch.no_grad():
                    vl = np.mean([run(*b, False).item() for b in vb])
                msg += f' val {vl:.4f}'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if rank == 0 and (step % 5000 == 0 or step == steps - 1) and step > 0:
            torch.save(model.state_dict(), f'{out}/model.pt')
    except KeyboardInterrupt:
        if rank == 0:
            torch.save(model.state_dict(), f'{out}/model.pt'); print('interrupted at step', step, '- saved', flush=True)
    if ddp: dist.destroy_process_group()
    if rank == 0 and args.eval:
        del model, fwd, opt; torch.cuda.empty_cache()
        import subprocess
        subprocess.run(['bash', 'eval_all.sh', args.name], check=False)


if __name__ == '__main__':
    main()
