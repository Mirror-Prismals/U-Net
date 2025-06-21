tokens ──► Embed d=E ──► [B, E, L]  ← reshape
                  │
                  ▼
            Time-embed d=E  (add/FILM)

DOWN path (stride-2 Conv1d)
  ├─ ResBlock(C)  → skip₀
  ├─ ResBlock(2C) → skip₁
  └─ ResBlock(4C) → skip₂

MID bottleneck ResBlocks(8C)

UP path (ConvTranspose1d stride-2)
  ◄── skip₂ + ResBlock(4C)
  ◄── skip₁ + ResBlock(2C)
  ◄── skip₀ + ResBlock(C)

Proj ► Linear(E→V) ► log-softmax
