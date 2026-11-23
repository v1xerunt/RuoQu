"""Score evaluation texts under the TC-LM on a grid of years.

For each text, cut passages of P characters and compute the summed log-likelihood of each passage
under every grid year (plus the date-unknown condition). Optionally keeps per-token log-probs.
Output: runs/<name>/scores_<split>_<genre>.npz
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, json, argparse
import numpy as np
import pandas as pd
import torch
from model import TCLM
from views import table

C = (WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
R = (WORK + '/runs')
PAD, SEP = 0, 1
GRID = np.arange(-1000, 1901, 25).astype(np.float32)


def passages(tok, P, K, rng=None):
    n = len(tok) // P
    if n == 0:
        return [tok] if len(tok) >= 32 else []
    starts = np.arange(n) * P
    if n > K:
        starts = starts[np.linspace(0, n - 1, K).round().astype(int)]
    return [tok[s:s + P] for s in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--P', type=int, default=256)
    ap.add_argument('--K', type=int, default=24)
    ap.add_argument('--genre', default='unk', help="'unk' or 'true'")
    ap.add_argument('--tokens', type=int, default=0, help='save per-token logprobs at every k-th grid year (0=off)')
    ap.add_argument('--bs', type=int, default=236)
    ap.add_argument('--ids', default='', help='comma-separated subset of ids, or @file')
    ap.add_argument('--tag', default='', help='suffix for the output file name')
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = json.load(open(f'{R}/{args.name}/config.json'))
    global C
    C = cfg.get('corpus', C)  # evaluate on the held-out fold of the corpus the model was trained on
    model = TCLM(cfg['vocab'], cfg['d'], cfg['L'], cfg['H'], ctx=cfg['T']).to(dev)
    sd = torch.load(f'{R}/{args.name}/model.pt', map_location=dev)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in sd.items()})
    model.eval()
    import views; views.C = C  # the view table must use the vocabulary of the model's own corpus
    remap = table(cfg.get('view', 'full'))
    data = np.memmap(f'{C}/{args.split}.bin', dtype=np.uint16, mode='r')
    idx = pd.read_csv(f'{C}/{args.split}.tsv', sep='\t')
    if args.ids:
        ids = open(args.ids[1:]).read().strip() if args.ids.startswith('@') else args.ids
        idx = idx[idx.id.isin(ids.split(','))]
    G = len(GRID)
    years = np.concatenate([GRID, [0.0]]).astype(np.float32)  # last = unknown date
    known = np.concatenate([np.ones(G, bool), [False]])
    rows, LL, NT, TOK = [], [], [], []
    queue = []  # (id, pi, passage, genre)

    def flush():
        if not queue: return
        P_ = max(len(q[2]) for q in queue)
        nq = len(queue); ny = len(years)
        x = np.zeros((nq, P_), np.int64)
        for k, q in enumerate(queue): x[k, :len(q[2])] = q[2]
        X = torch.from_numpy(x).to(dev).repeat_interleave(ny, 0)
        Y = torch.from_numpy(np.tile(years, nq)).to(dev)
        Kn = torch.from_numpy(np.tile(known, nq)).to(dev)
        Gn = torch.from_numpy(np.repeat([q[3] for q in queue], ny)).to(dev)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            lp = torch.cat([model.token_logprob(X[s:s + args.bs], Y[s:s + args.bs], Kn[s:s + args.bs], Gn[s:s + args.bs])
                            for s in range(0, len(X), args.bs)]).float()
        lp = lp.view(nq, ny, P_)
        m = ((torch.from_numpy(x) != PAD) & (torch.from_numpy(x) != SEP)).float().to(dev)
        ll = (lp * m[:, None]).sum(2).cpu().numpy()
        for k, q in enumerate(queue):
            rows.append((q[0], q[1])); NT.append(int(m[k].sum().item())); LL.append(ll[k])
            if args.tokens:
                n = len(q[2])
                TOK.append(dict(id=q[0], pi=q[1], tok=q[2].astype(np.uint16),
                                lp=lp[k][list(range(0, G, args.tokens)) + [G], :n].cpu().numpy().astype(np.float16)))
        queue.clear()

    per_batch = max(1, args.bs // len(years))
    for _, r in idx.iterrows():
        tok = np.asarray(data[r.offset:r.offset + r.length]).astype(np.int64)
        if remap is not None: tok = remap[tok]
        g = 7 if args.genre == 'unk' else int(r.genre)
        for pi, p in enumerate(passages(tok, args.P, args.K)):
            queue.append((r.id, pi, p, g))
            if len(queue) >= per_batch: flush()
    flush()
    out = f'{R}/{args.name}/scores_{args.split}_{args.genre}{args.tag}.npz'
    np.savez_compressed(out, ids=np.array([a for a, _ in rows]), pi=np.array([b for _, b in rows]),
                        ll=np.array(LL, np.float32), nt=np.array(NT), grid=GRID)
    if args.tokens:
        import pickle
        pickle.dump(TOK, open(out.replace('.npz', '_tok.pkl'), 'wb'))
    print('saved', out, len(rows), 'passages')


if __name__ == '__main__':
    main()
