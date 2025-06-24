#!/usr/bin/env python3
import argparse
import torch
import math
import numpy as np
import librosa
from librosa.feature.inverse import mel_to_audio
import soundfile as sf
from unet_text_diffusion_audio import (
    MMDiffuser,
    WordTokenizer,
    build_vocab,
    BETA_START,
    BETA_END,
    DIFF_STEPS,
    SEQ_LEN,
    DEVICE,
    N_MELS,
    SR,
    HOP_LENGTH,
    WIN_LENGTH
)

# Prepare diffusion constants
betas = torch.linspace(BETA_START, BETA_END, DIFF_STEPS, device=DEVICE)
alphas = 1 - betas
alphas_cum = torch.cumprod(alphas, dim=0)

def sample_text(model, tok, prompt=None):
    # Initialize all tokens as mask
    tokens = torch.full((1, SEQ_LEN), tok.mask, dtype=torch.long, device=DEVICE)
    if prompt:
        # Encode prompt into beginning of sequence
        ids = tok.encode(prompt).unsqueeze(0).to(DEVICE)
        tokens[:, :ids.size(1)] = ids
    # Reverse diffusion loop
    for t in reversed(range(DIFF_STEPS)):
        t_batch = torch.tensor([t], device=DEVICE)
        logits = model.forward_text(tokens, t_batch)
        probs = torch.exp(logits)
        mask_positions = tokens[0] == tok.mask
        sampled = torch.multinomial(probs[0], num_samples=1).squeeze(-1)
        tokens[0, mask_positions] = sampled[mask_positions]
    return tok.decode(tokens[0].cpu().tolist())

def sample_audio(model):
    # Unconditional audio sampling
    x = torch.randn(1, SEQ_LEN, N_MELS, device=DEVICE)
    for t in reversed(range(DIFF_STEPS)):
        t_batch = torch.tensor([t], device=DEVICE)
        pred_noise = model.forward_audio(x, t_batch)
        beta = betas[t]; alpha = alphas[t]; alpha_cum = alphas_cum[t]
        x = (1/alpha.sqrt()) * (x - (1-alpha)/math.sqrt(alpha_cum) * pred_noise.transpose(1,2)).transpose(1,2)
        if t > 0:
            x = x + beta.sqrt() * torch.randn_like(x)
    # Convert final mel-spectrogram back to waveform
    spec = x.squeeze(0).cpu().numpy().T
    audio = mel_to_audio(spec, sr=SR, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH)
    return audio

def sample_image(model):
    # Placeholder: implement similarly using model.forward_img and reverse diffusion
    raise NotImplementedError("Unconditional image sampling not implemented in this script.")

def main():
    parser = argparse.ArgumentParser("Sample from multimodal diffuser")
    parser.add_argument('--modality', choices=['text', 'audio', 'image'], required=True)
    parser.add_argument('--prompt', type=str, help='Text prompt (for text)')
    parser.add_argument('--ckpt', type=str, default='shared_mm_audio.pt')
    parser.add_argument('--vocab_src', type=str, required=True)
    args = parser.parse_args()

    # Load vocabulary and model
    vocab = build_vocab(args.vocab_src)
    tok = WordTokenizer(vocab)
    model = MMDiffuser(tok.vsz).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])

    if args.modality == 'text':
        output = sample_text(model, tok, args.prompt)
        print(output)
    elif args.modality == 'audio':
        audio = sample_audio(model)
        sf.write('sample.wav', audio, SR)
        print("Saved sample.wav")
    else:
        sample_image(model)

if __name__ == '__main__':
    main()
