#!/usr/bin/env python3
# unet_text_diffusion_audio.py — shared 1D U-Net text+image+audio with gradient-level balancing
# 2025-06-21 • MIT-0

import os
import glob
import argparse
from collections import Counter
from PIL import Image
import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── hyperparams ─────────────────────────────────────────────── #
SEQ_LEN     = 64       # text length & # patches for image/audio
IMG_SIZE    = 64
PATCH       = 8
DIFF_STEPS  = 8
BATCH_SIZE  = 1
LR          = 3e-4
EMB_DIM     = 512
CHANNELS    = (512, 768, 1024)
EPOCHS      = 1
MASK, PAD   = "<mask>", "<pad>"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_IDX  = -100
BETA_START  = 1e-4
BETA_END    = 0.02

# audio config
SR          = 16000
N_MELS      = 64
HOP_LENGTH  = 256
WIN_LENGTH  = 1024
# ─────────────────────────────────────────────────────────────── #

# ─── diffusion schedule ─────────────────────────────────────── #
betas      = torch.linspace(BETA_START, BETA_END, DIFF_STEPS, device=DEVICE)
alphas     = 1 - betas
alphas_cum = torch.cumprod(alphas, dim=0)
# ─────────────────────────────────────────────────────────────── #

def q_sample_text(x, t, mask_id):
    rate = (t.float()+1)/DIFF_STEPS
    m = torch.rand_like(rate) < rate
    y = x.clone(); y[m] = mask_id
    return y

def q_sample_img(x, t):
    out = []
    for ti, xi in zip(t, x):
        a = alphas_cum[ti].sqrt()
        noise = torch.randn_like(xi)
        out.append(a*xi + (1-a)*noise)
    return torch.stack(out)

def q_sample_audio(x, t):
    out = []
    for ti, xi in zip(t, x):
        a = alphas_cum[ti].sqrt()
        noise = torch.randn_like(xi)
        out.append(a*xi + (1-a)*noise)
    return torch.stack(out)

# ─── TEXT tokenizer & dataset ───────────────────────────────── #
class WordTokenizer:
    def __init__(self, vocab, specials=(PAD, MASK)):
        self.t2i = {tok:i for i,tok in enumerate(specials)}
        for w in vocab:
            if w not in self.t2i:
                self.t2i[w] = len(self.t2i)
        self.i2t = {i:tok for tok,i in self.t2i.items()}
        self.vsz  = len(self.t2i)
        self.pad  = self.t2i[PAD]
        self.mask = self.t2i[MASK]
    def encode(self, txt):
        ids = [self.t2i.get(w, self.pad) for w in txt.strip().split()]
        ids = (ids + [self.pad]*SEQ_LEN)[:SEQ_LEN]
        return torch.tensor(ids, dtype=torch.long)
    def decode(self, ids):
        return " ".join(self.i2t[i] for i in ids if i != self.pad)

def build_vocab(src, max_size=50000):
    cnt = Counter()
    paths = ([src] if os.path.isfile(src)
             else [os.path.join(dp,f)
                   for dp,_,fs in os.walk(src) for f in fs if f.endswith(".txt")])
    for p in paths:
        with open(p, encoding="utf8", errors="ignore") as f:
            for ln in f:
                cnt.update(ln.split())
    return [w for w,_ in cnt.most_common(max_size) if w not in (MASK, PAD)]

class TextDataset(Dataset):
    def __init__(self, root, tok):
        self.samples = []
        for dp,_,fs in os.walk(root):
            for fn in fs:
                if not fn.endswith(".txt"): continue
                words = open(os.path.join(dp,fn), encoding="utf8").read().split()
                for i in range(0, len(words), SEQ_LEN):
                    chunk = words[i:i+SEQ_LEN]
                    if len(chunk) < SEQ_LEN:
                        chunk += [PAD]*(SEQ_LEN - len(chunk))
                    ids = [tok.t2i.get(w, tok.pad) for w in chunk]
                    self.samples.append(torch.tensor(ids, dtype=torch.long))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ─── PIL‐only image loader ────────────────────────────────────── #
def pil_transform(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2,0,1)

class AllImagesDataset(Dataset):
    def __init__(self, root):
        exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.gif")
        self.paths = sum((glob.glob(os.path.join(root,"**",e), recursive=True) for e in exts), [])
        if not self.paths:
            raise RuntimeError(f"No images in {root!r}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return pil_transform(img), 0

def get_image_loader(root):
    return DataLoader(AllImagesDataset(root),
                      batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ─── Librosa‐based audio loader ───────────────────────────────── #
class AllAudioDataset(Dataset):
    def __init__(self, root):
        exts = ("*.wav","*.mp3","*.flac")
        self.paths = sum((glob.glob(os.path.join(root,"**",e), recursive=True) for e in exts), [])
        if not self.paths:
            raise RuntimeError(f"No audio in {root!r}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        wav, sr = librosa.load(self.paths[idx], sr=SR, mono=True)
        if wav.shape[0] < WIN_LENGTH:
            wav = np.pad(wav, (0, WIN_LENGTH - wav.shape[0]), mode="constant")
        spec = librosa.feature.melspectrogram(
            y=wav, sr=SR,
            n_fft=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
        )
        spec = torch.from_numpy(spec).float()  # (n_mels, T)
        T = spec.shape[1]
        if T < SEQ_LEN:
            spec = F.pad(spec, (0, SEQ_LEN - T))
        spec = spec[:, :SEQ_LEN]
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec.T, 0  # (SEQ_LEN, n_mels)

def get_audio_loader(root):
    return DataLoader(AllAudioDataset(root),
                      batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ─── U-Net blocks ────────────────────────────────────────────── #
class FiLM(nn.Module):
    def __init__(self, ch, emb):
        super().__init__()
        self.fc = nn.Linear(emb, ch*2)
    def forward(self, x, temb):
        g,b = self.fc(temb).chunk(2, dim=-1)
        return x * (1+g.unsqueeze(-1)) + b.unsqueeze(-1)

class ResBlock1D(nn.Module):
    def __init__(self, channels, emb_dim, kernel=3, groups=8):
        super().__init__()
        p = kernel//2
        self.conv1 = nn.Conv1d(channels,channels,kernel,padding=p)
        self.conv2 = nn.Conv1d(channels,channels,kernel,padding=p)
        self.norm1 = nn.GroupNorm(groups,channels)
        self.norm2 = nn.GroupNorm(groups,channels)
        self.act   = nn.GELU()
        self.film  = FiLM(channels,emb_dim)
    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.film(h, temb)
        h = self.conv2(self.act(self.norm2(h)))
        if h.size(-1) != x.size(-1):
            L = min(h.size(-1), x.size(-1))
            h, x = h[...,:L], x[...,:L]
        return x + h

class TokenUNet(nn.Module):
    def __init__(self, emb_dim, channels, steps):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Embedding(steps+1, emb_dim),
            nn.SiLU(), nn.Linear(emb_dim, emb_dim*4),
            nn.SiLU(), nn.Linear(emb_dim*4, emb_dim)
        )
        prev = emb_dim
        self.down = nn.ModuleList()
        for ch in channels:
            self.down.append(nn.ModuleDict({
                "res": ResBlock1D(prev, emb_dim),
                "down": nn.Conv1d(prev, ch, 4, 2, 1)
            }))
            prev = ch
        self.mid = ResBlock1D(prev, emb_dim)
        self.up  = nn.ModuleList()
        for ch in reversed(channels):
            self.up.append(nn.ModuleDict({
                "up": nn.ConvTranspose1d(prev, ch, 4, 2, 1),
                "res": ResBlock1D(ch, emb_dim)
            }))
            prev = ch
        self.norm = nn.GroupNorm(8, emb_dim)
    def forward(self, x, t):
        temb = self.time_mlp(t)
        skips=[]; h=x
        for blk in self.down:
            h = blk["res"](h, temb); skips.append(h)
            h = blk["down"](h)
        h = self.mid(h, temb)
        for blk in self.up:
            h = blk["up"](h)
            skip = skips.pop()
            if skip.size(-1) != h.size(-1):
                L = min(skip.size(-1), h.size(-1))
                skip, h = skip[...,:L], h[...,:L]
            if skip.size(1) != h.size(1):
                adapt = nn.Conv1d(skip.size(1), h.size(1), 1).to(skip.device)
                skip = adapt(skip)
            h = blk["res"](h + skip, temb)
        return self.norm(h)

# ─── multimodal diffuser ─────────────────────────────────────── #
class MMDiffuser(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb   = nn.Embedding(vocab_size, EMB_DIM)
        self.patch_embed = nn.Conv2d(3, EMB_DIM, PATCH, PATCH)
        self.audio_proj  = nn.Conv1d(N_MELS, EMB_DIM, 1)
        self.unet        = TokenUNet(EMB_DIM, CHANNELS, DIFF_STEPS)
        self.text_out    = nn.Linear(EMB_DIM, vocab_size)
        self.patch_out   = nn.Conv1d(EMB_DIM, 3*PATCH*PATCH, 1)
        self.audio_out   = nn.Conv1d(EMB_DIM, N_MELS, 1)

    def forward_text(self, toks, t):
        x = self.token_emb(toks).transpose(1,2)
        h = self.unet(x,t).transpose(1,2)
        return F.log_softmax(self.text_out(h), dim=-1)

    def forward_img(self, imgs, t):
        p   = self.patch_embed(imgs)
        seq = p.view(p.size(0), p.size(1), -1)
        h   = self.unet(seq, t)
        return self.patch_out(h)

    def forward_audio(self, spec, t):
        x = spec.transpose(1,2)
        x = self.audio_proj(x)
        h = self.unet(x, t)
        return self.audio_out(h)

def train(args):
    vocab        = build_vocab(args.vocab_src)
    tok          = WordTokenizer(vocab)
    text_loader  = DataLoader(TextDataset(args.text_corpus, tok),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    img_loader   = get_image_loader(args.image_corpus)
    audio_loader = get_audio_loader(args.audio_corpus)

    model  = MMDiffuser(tok.vsz).to(DEVICE)
    shared = list(model.unet.parameters())
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    tloss  = nn.NLLLoss(ignore_index=IGNORE_IDX)
    iloss  = nn.MSELoss()
    aloss  = nn.MSELoss()

    print(f"[i] params: {sum(p.numel() for p in model.parameters()):,}")

    for ep in range(1, EPOCHS+1):
        ti, ii, ai = iter(text_loader), iter(img_loader), iter(audio_loader)
        steps = max(len(text_loader), len(img_loader), len(audio_loader))
        for step in range(1, steps+1):
            # TEXT
            try: xb = next(ti).to(DEVICE)
            except StopIteration:
                ti = iter(text_loader); xb = next(ti).to(DEVICE)
            t_txt    = torch.randint(0, DIFF_STEPS, (xb.size(0),), device=DEVICE)
            xb_noisy = q_sample_text(xb, t_txt, tok.mask)
            logits   = model.forward_text(xb_noisy, t_txt)
            tgt      = xb.clone(); tgt[xb_noisy!=tok.mask] = IGNORE_IDX
            lt       = tloss(logits.view(-1, logits.size(-1)), tgt.view(-1))

            # IMAGE
            try: imgs,_ = next(ii)
            except StopIteration:
                ii = iter(img_loader); imgs,_ = next(ii)
            imgs   = imgs.to(DEVICE)
            t_img  = torch.randint(0, DIFF_STEPS, (imgs.size(0),), device=DEVICE)
            noise  = torch.randn_like(imgs)
            imgs_n = q_sample_img(imgs, t_img)
            pred_i = model.forward_img(imgs_n, t_img)
            li     = iloss(pred_i, noise.view(noise.size(0), -1, noise.shape[-1]))

            # AUDIO
            try: spec,_ = next(ai)
            except StopIteration:
                ai   = iter(audio_loader); spec,_ = next(ai)
            spec    = spec.to(DEVICE)
            t_a     = torch.randint(0, DIFF_STEPS, (spec.size(0),), device=DEVICE)
            noise_a = torch.randn_like(spec)
            spec_n  = q_sample_audio(spec.transpose(1,2), t_a).transpose(1,2)
            pred_a  = model.forward_audio(spec_n, t_a)
            al      = aloss(pred_a, noise_a.transpose(1,2))

            # gradient norms & balance
            grads_t = torch.autograd.grad(lt, shared, retain_graph=True)
            gnorm_t = torch.sqrt(sum((g**2).sum() for g in grads_t))
            grads_i = torch.autograd.grad(li, shared, retain_graph=True)
            gnorm_i = torch.sqrt(sum((g**2).sum() for g in grads_i))
            grads_a = torch.autograd.grad(al, shared, retain_graph=True)
            gnorm_a = torch.sqrt(sum((g**2).sum() for g in grads_a))

            target = (gnorm_t + gnorm_i + gnorm_a) / 3
            w_t = (target / (gnorm_t + 1e-8)).detach()
            w_i = (target / (gnorm_i + 1e-8)).detach()
            w_a = (target / (gnorm_a + 1e-8)).detach()

            loss = w_t*lt + w_i*li + w_a*al
            opt.zero_grad(); loss.backward(); opt.step()

            if step % 50 == 0:
                print(f"ep{ep} step{step:4d}  txt={lt.item():.4f}×{w_t.item():.2f} "
                      f"img={li.item():.4f}×{w_i.item():.2f} "
                      f"aud={al.item():.4f}×{w_a.item():.2f}")

        torch.save({"state_dict": model.state_dict(), "vocab": vocab}, args.ckpt)
        print(f"[✓] ep{ep} saved → {args.ckpt}")

if __name__=="__main__":
    p = argparse.ArgumentParser("shared 1D UNet text+image+audio")
    p.add_argument("--text_corpus",  required=True)
    p.add_argument("--vocab_src",    required=True)
    p.add_argument("--image_corpus", required=True)
    p.add_argument("--audio_corpus", required=True)
    p.add_argument("--ckpt",         default="shared_mm_audio.pt")
    args = p.parse_args()
    train(args)
