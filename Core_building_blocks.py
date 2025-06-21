class FiLM(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.to_gamma_beta = nn.Linear(emb_dim, ch * 2)
    def forward(self, x, t_emb):
        gamma, beta = self.to_gamma_beta(t_emb).chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

class ResBlock1D(nn.Module):
    def __init__(self, ch, emb_dim, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.film = FiLM(ch, emb_dim)
        self.act = nn.GELU()
    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, t_emb)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h
