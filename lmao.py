#!/usr/bin/env python3
# unet_text_infer.py — exact match to 1024-dim TokenUNet for loading checkpoint

import os, argparse
from collections import Counter

import torch, torch.nn as nn
import torch.nn.functional as F

SEQ_LEN     = 64
DIFF_STEPS  = 8
MASK, PAD   = "<mask>", "<pad>"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Tokenizer (same as training) ────────────────────────────── #
class WordTokenizer:
    def __init__(self, vocab, specials=(PAD, MASK)):
        self.t2i = {tok: i for i, tok in enumerate(specials)}
        for w in vocab:
            if w not in self.t2i:
                self.t2i[w] = len(self.t2i)
        self.i2t = {i: tok for tok, i in self.t2i.items()}
        self.vsz = len(self.t2i)
        self.pad = self.t2i[PAD]
        self.mask= self.t2i[MASK]

    def encode(self, text: str):
        ids = [self.t2i.get(w, self.pad) for w in text.split()]
        ids = (ids + [self.pad] * SEQ_LEN)[:SEQ_LEN]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        return " ".join(self.i2t[i] for i in ids if i != self.pad)

# ─── U-Net building blocks (identical names/attrs) ────────────── #
class FiLM(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, channels * 2)
    def forward(self, x, t_emb):
        g, b = self.fc(t_emb).chunk(2, dim=-1)
        return x * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)

class ResBlock1D(nn.Module):
    def __init__(self, channels, emb_dim, kernel=3, groups=8):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act   = nn.GELU()
        self.film  = FiLM(channels, emb_dim)
    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.film(h, t_emb)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h

class TokenUNet(nn.Module):
    def __init__(self, vocab_size,
                 emb_dim=1024,                # ← match training
                 channels=(1024, 1536, 2048), # ← match training
                 steps=DIFF_STEPS):
        super().__init__()
        self.token_emb   = nn.Embedding(vocab_size, emb_dim)
        self.time_emb    = nn.Sequential(
            nn.Embedding(steps+1, emb_dim),
            nn.SiLU(), nn.Linear(emb_dim, emb_dim*4),
            nn.SiLU(), nn.Linear(emb_dim*4, emb_dim)
        )

        self.down_blocks = nn.ModuleList()
        prev_ch = emb_dim
        for ch in channels:
            self.down_blocks.append(nn.ModuleDict({
                "res":  ResBlock1D(prev_ch, emb_dim),
                "down": nn.Conv1d(prev_ch, ch, 4, 2, 1)
            }))
            prev_ch = ch

        self.mid = ResBlock1D(prev_ch, emb_dim)

        self.up_blocks = nn.ModuleList()
        for ch in reversed(channels):
            self.up_blocks.append(nn.ModuleDict({
                "up":   nn.ConvTranspose1d(prev_ch, ch, 4, 2, 1),
                "res":  ResBlock1D(ch, emb_dim)
            }))
            prev_ch = ch

        self.norm = nn.GroupNorm(8, emb_dim)
        self.proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, t):
        feat = self.token_emb(x).transpose(1,2)  # [B,E,L]
        temb = self.time_emb(t)                  # [B,E]

        skips = []
        for blk in self.down_blocks:
            feat = blk["res"](feat, temb)
            skips.append(feat)
            feat = blk["down"](feat)

        feat = self.mid(feat, temb)

        for blk in self.up_blocks:
            feat = blk["up"](feat)
            skip = skips.pop()
            # length align
            if skip.shape[-1] != feat.shape[-1]:
                Lmin = min(skip.shape[-1], feat.shape[-1])
                skip, feat = skip[...,:Lmin], feat[...,:Lmin]
            # channel adapt
            if skip.shape[1] != feat.shape[1]:
                adapt = nn.Conv1d(skip.shape[1], feat.shape[1], 1).to(skip.device)
                skip = adapt(skip)
            feat = blk["res"](feat + skip, temb)

        out = self.norm(feat).transpose(1,2)     # [B,L,E]
        return F.log_softmax(self.proj(out), dim=-1)

# ─── load checkpoint & tokenizer ────────────────────────────────────── #
def load_model_and_tokenizer(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    vocab = ckpt["vocab"]
    model = TokenUNet(len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tok = WordTokenizer(vocab)
    return model, tok

# ─── diffusion infill (single time-step per call) ───────────────────── #
@torch.no_grad()
def diffusion_infill(prompt, model, tok, steps=DIFF_STEPS):
    seq = tok.encode(prompt).unsqueeze(0).to(DEVICE)
    seq[seq==tok.pad] = tok.mask
    for t in reversed(range(steps)):
        t_batch = torch.full((1,), t, dtype=torch.long, device=DEVICE)
        logits  = model(seq, t_batch)[0]          # [L, V]
        masked  = (seq[0]==tok.mask).nonzero(as_tuple=True)[0]
        if len(masked)==0: break
        fill_n  = max(1, len(masked)//3)
        idxs    = masked[torch.randperm(len(masked))[:fill_n]]
        for idx in idxs:
            probs = torch.exp(logits[idx])
            seq[0,idx] = torch.multinomial(probs, 1)
    return tok.decode(seq[0].tolist())

# ─── main ─────────────────────────────────────────────────────────────── #
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True, help="path to model.pt")
    parser.add_argument("--prompt", required=True, help="text with <mask> tokens")
    args = parser.parse_args()

    model, tok = load_model_and_tokenizer(args.ckpt)
    out = diffusion_infill(args.prompt, model, tok)
    print("IN  >", args.prompt)
    print("OUT >", out)
