"""Time-conditioned character-level GPT.

Input: [DATE] [GENRE] x_1 ... x_n. The DATE prefix embedding is a smooth function of the year
(Fourier features -> MLP), so sparse early periods borrow strength from neighbours.
A learned "unknown" date / genre embedding is used with dropout, giving p(x | genre) as well.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Y0, Y1 = -1100.0, 1950.0  # year range for normalization


class DateEmb(nn.Module):
    def __init__(self, d, n_freq=16):
        super().__init__()
        # periods from ~3000y down to ~25y
        self.register_buffer('freqs', torch.tensor([2 * math.pi / p for p in torch.logspace(math.log10(6000), math.log10(25), n_freq).tolist()]))
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(), nn.Linear(d, d))
        self.unk = nn.Parameter(torch.zeros(d))

    def forward(self, year, known):
        x = (year - Y0)[:, None] * self.freqs[None]
        u = ((year - Y0) / (Y1 - Y0))[:, None]
        e = self.mlp(torch.cat([torch.sin(x), torch.cos(x), u], -1))
        return torch.where(known[:, None], e, self.unk[None].expand_as(e))


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        self.h, self.drop = h, drop

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, T, C)), self.drop, self.training)
        x = x + F.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))), self.drop, self.training)
        return x


class TCLM(nn.Module):
    def __init__(self, vocab, d=512, n_layer=8, n_head=8, ctx=512, n_genre=7, drop=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx + 2, d)
        self.date = DateEmb(d)
        self.genre = nn.Embedding(n_genre + 1, d)  # last = unknown
        self.n_genre = n_genre
        self.blocks = nn.ModuleList([Block(d, n_head, drop) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith('proj.weight') or n.endswith('fc2.weight'):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x, year, year_known, genre):
        """x: (B,T) tokens. Returns logits (B,T,V) predicting x[:,t] from prefix + x[:,<t]."""
        B, T = x.shape
        pre = torch.stack([self.date(year, year_known), self.genre(genre)], 1)  # (B,2,d)
        h = torch.cat([pre, self.tok(x[:, :-1])], 1) if T > 1 else pre
        h = h[:, :T + 1]
        h = h + self.pos(torch.arange(h.shape[1], device=x.device))[None]
        for b in self.blocks:
            h = b(h)
        h = self.ln(h)[:, 1:T + 1]  # position 1 (genre token) predicts x_0
        return self.head(h)

    def token_logprob(self, x, year, year_known, genre):
        logits = self.forward(x, year, year_known, genre).float()
        return -F.cross_entropy(logits.transpose(1, 2), x.long(), reduction='none')
