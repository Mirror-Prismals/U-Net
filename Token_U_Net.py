class TokenUNet(nn.Module):
    def __init__(self, vocab, emb_dim=384, chans=(384,512,768), steps=DIFFUSION_STEPS):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, emb_dim)
        self.t_embed = nn.Sequential(
            nn.Embedding(steps+1, emb_dim),
            nn.SiLU(), nn.Linear(emb_dim, emb_dim*4),
            nn.SiLU(), nn.Linear(emb_dim*4, emb_dim)
        )

        # Down
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        prev = emb_dim
        for ch in chans:
            self.downs.append(
                nn.ModuleDict(dict(
                    res=ResBlock1D(prev, emb_dim),
                    down=nn.Conv1d(prev, ch, 4, stride=2, padding=1)
                ))
            )
            prev = ch

        # Mid
        self.mid = ResBlock1D(prev, emb_dim)

        # Up (mirror)
        for ch in reversed(chans):
            self.ups.append(
                nn.ModuleDict(dict(
                    up=nn.ConvTranspose1d(prev, ch, 4, stride=2, padding=1),
                    res=ResBlock1D(ch, emb_dim)
                ))
            )
            prev = ch

        self.norm_out = nn.GroupNorm(8, emb_dim)
        self.proj = nn.Linear(emb_dim, vocab)

    def forward(self, x_tok, t):
        B, L = x_tok.shape
        x = self.token_emb(x_tok).transpose(1, 2)       # [B,E,L] for conv
        t_emb = self.t_embed(t)                         # [B,E]

        skips = []
        for d in self.downs:
            x = d['res'](x, t_emb)
            skips.append(x)
            x = d['down'](x)

        x = self.mid(x, t_emb)

        for u in self.ups:
            x = u['up'](x)
            x = x + skips.pop()[:, :, :x.shape[-1]]     # crop if odd length
            x = u['res'](x, t_emb)

        x = self.norm_out(x).transpose(1, 2)            # back to [B,L,E]
        logits = self.proj(x)
        return nn.functional.log_softmax(logits, dim=-1)
