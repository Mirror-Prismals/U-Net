#!/usr/bin/env python3
# unet_text_diffusion.py — 1-D U-Net diffusion LM (fixed ignore_index)
# 2025-06-21 • MIT-0

import os, argparse
from collections import Counter

import torch, torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# ─── hyperparams ───────────────────────────── #
SEQ_LEN     = 64
DIFF_STEPS  = 8
BATCH_SIZE  = 4         # reduce for memory reasons!
LR          = 3e-4
EMB_DIM     = 1024      # ↑ was 384
CHANNELS    = (1024, 1536, 2048)
EPOCHS      = 1
MASK, PAD   = "<mask>", "<pad>"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_IDX  = -100
# ───────────────────────────────────────────── #


# ─── tokenizer ────────────────────────────────────────────── #
class WordTokenizer:
    def __init__(self, vocab, specials=(PAD, MASK)):
        self.t2i = {tok: i for i, tok in enumerate(specials)}
        for w in vocab:
            if w not in self.t2i:
                self.t2i[w] = len(self.t2i)
        self.i2t = {i: tok for tok, i in self.t2i.items()}
        self.vsz = len(self.t2i)
        self.pad = self.t2i[PAD]
        self.mask = self.t2i[MASK]

    def encode(self, text: str):
        ids = [self.t2i.get(w, self.pad) for w in text.split()]
        ids = (ids + [self.pad] * SEQ_LEN)[:SEQ_LEN]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        return " ".join(self.i2t[i] for i in ids if i != self.pad)

def build_vocab(src_path: str, max_size=50000):
    cnt = Counter()
    paths = ([src_path] if os.path.isfile(src_path)
             else [os.path.join(src_path, f) for f in os.listdir(src_path) if f.endswith(".txt")])
    for p in paths:
        with open(p, encoding="utf8") as f:
            for ln in f:
                cnt.update(ln.split())
    return [w for w, _ in cnt.most_common(max_size) if w not in (MASK, PAD)]
# ───────────────────────────────────────────────────────────── #

# ─── dataset ───────────────────────────────────────────────── #
class TextDataset(Dataset):
    def __init__(self, folder: str, tokenizer: WordTokenizer):
        self.samples = []
        for fn in os.listdir(folder):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(folder, fn), encoding="utf8") as f:
                words = f.read().split()
            for i in range(0, len(words), SEQ_LEN):
                chunk = words[i:i+SEQ_LEN]
                if len(chunk) < SEQ_LEN:
                    chunk += [PAD] * (SEQ_LEN - len(chunk))
                ids = [tokenizer.t2i.get(w, tokenizer.pad) for w in chunk]
                self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
# ───────────────────────────────────────────────────────────── #

# ─── U-Net building blocks ──────────────────────────────────── #
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
# ───────────────────────────────────────────────────────────── #

# ─── Token U-Net with adaptive skip handling ─────────────────── #
class TokenUNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB_DIM, channels=CHANNELS, steps=DIFF_STEPS):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.time_emb  = nn.Sequential(
            nn.Embedding(steps+1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim*4),
            nn.SiLU(),
            nn.Linear(emb_dim*4, emb_dim)
        )
        # encoder
        self.down_blocks = nn.ModuleList()
        prev_ch = emb_dim
        for ch in channels:
            self.down_blocks.append(nn.ModuleDict({
                "res":  ResBlock1D(prev_ch, emb_dim),
                "down": nn.Conv1d(prev_ch, ch, kernel_size=4, stride=2, padding=1)
            }))
            prev_ch = ch
        # bottleneck
        self.mid = ResBlock1D(prev_ch, emb_dim)
        # decoder
        self.up_blocks = nn.ModuleList()
        for ch in reversed(channels):
            self.up_blocks.append(nn.ModuleDict({
                "up":   nn.ConvTranspose1d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                "res":  ResBlock1D(ch, emb_dim)
            }))
            prev_ch = ch
        # final
        self.norm = nn.GroupNorm(8, emb_dim)
        self.proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, t):
        # x: [B, L], t: [B]
        B, L = x.shape
        # embed & time
        feat = self.token_emb(x).transpose(1, 2)         # [B, E, L]
        temb = self.time_emb(t)                         # [B, E]
        # encoder
        skips = []
        for blk in self.down_blocks:
            feat = blk["res"](feat, temb)
            skips.append(feat)
            feat = blk["down"](feat)
        # mid
        feat = self.mid(feat, temb)
        # decoder
        for blk in self.up_blocks:
            feat = blk["up"](feat)
            skip = skips.pop()
            # length align
            if skip.shape[-1] != feat.shape[-1]:
                Lmin = min(skip.shape[-1], feat.shape[-1])
                skip = skip[..., :Lmin]
                feat = feat[..., :Lmin]
            # channel adapt
            if skip.shape[1] != feat.shape[1]:
                adapt = nn.Conv1d(skip.shape[1], feat.shape[1], 1).to(skip.device)
                skip = adapt(skip)
            feat = blk["res"](feat + skip, temb)
        # final proj
        out = self.norm(feat).transpose(1, 2)            # [B, L, E]
        return F.log_softmax(self.proj(out), dim=-1)    # [B, L, V]
# ───────────────────────────────────────────────────────────── #

# ─── diffusion mask scheduler ────────────────────────────────── #
def q_sample_mask(x_start, t_vec, mask_id, steps):
    # per-token mask rate = (t+1)/steps
    rate = (t_vec.float() + 1) / steps
    m = torch.rand_like(rate) < rate
    x = x_start.clone()
    x[m] = mask_id
    return x
# ───────────────────────────────────────────────────────────── #

# ─── training loop ──────────────────────────────────────────── #
def train(args):
    # build vocabulary & tokenizer
    vocab = build_vocab(args.vocab_src)
    tok   = WordTokenizer(vocab)
    # data
    ds = TextDataset(args.corpus, tok)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # model, optimizer, loss
    model = TokenUNet(tok.vsz).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.NLLLoss(ignore_index=IGNORE_IDX)
    # train
    print(f"[i] Model has {sum(p.numel() for p in model.parameters()):,} parameters")
# here?
    for ep in range(1, EPOCHS+1):
        for step, batch in enumerate(dl, 1):
            batch = batch.to(DEVICE)                           # [B, L]
            t     = torch.randint(0, DIFF_STEPS, (batch.size(0),), device=DEVICE)
            noisy = q_sample_mask(batch, t[:,None].expand(-1, SEQ_LEN),
                                   tok.mask, DIFF_STEPS)       # [B, L]
            logp  = model(noisy, t)                            # [B, L, V]
            target = batch.clone()
            target[noisy != tok.mask] = IGNORE_IDX             # ignore non-masked
            loss = lossf(logp.view(-1, logp.size(-1)), target.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0:
                print(f"ep{ep} step{step:4d} loss {loss.item():.4f}")
        # save
        torch.save({"state_dict": model.state_dict(),
                    "vocab":      list(tok.t2i.keys())},
                   args.ckpt)
        print(f"[✓] epoch {ep} saved → {args.ckpt}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("1-D U-Net Text Diffusion")
    p.add_argument("--corpus",    required=True, help="folder of .txt files")
    p.add_argument("--vocab_src", required=True, help="file or folder for vocab")
    p.add_argument("--ckpt",      default="unet_text.pt", help="checkpoint path")
    train(p.parse_args())
