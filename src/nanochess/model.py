from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Config:
    C: int  # Embedding Dim
    V: int  # Vocab Size
    L: int  # Max sequence length
    H: int  # Attention heads per block
    N: int  # Num transformer blocks


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.C)
        self.mha = nn.MultiheadAttention(
            embed_dim=config.C, num_heads=config.H, batch_first=True
        )
        self.ln2 = nn.LayerNorm(config.C)

        # C -> 4C -> C
        self.mlp = nn.Sequential(
            nn.Linear(config.C, 4 * config.C),
            nn.GELU(),
            nn.Linear(4 * config.C, config.C),
        )

    def forward(self, x):
        T = x.shape[1]

        norm = self.ln1(x)
        M = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device),diagonal=1)
        attention_out, _ = self.mha(norm, norm, norm, attn_mask=M, need_weights=False)

        x = x + attention_out
        x = x + self.mlp(self.ln2(x))

        return x


class NanoChess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.wte = nn.Embedding(config.V, config.C)
        self.wpe = nn.Embedding(config.L, config.C)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.N)])
        self.layerNorm = nn.LayerNorm(config.C)
        self.lm_head = nn.Linear(config.C, config.V, bias=False)

        self.lm_head.weight = self.wte.weight

        # nn.Embedding initializes weights from N(0, 1).
        # With tied output weights, this makes initial predictions overconfident.
        # Shrink the token and position embeddings to reduce the initial scale.
        with torch.no_grad():
            self.wte.weight.mul_(0.02)
            self.wpe.weight.mul_(0.02)

    def forward(self, x):
        T = x.shape[1]
        positions = torch.arange(T, device=x.device)
        x = self.wte(x) + self.wpe(positions)
        x = self.blocks(x)
        x = self.layerNorm(x)
        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    config = Config(C=768, V=1987, L=200, H=12, N=12)
    model = NanoChess(config)
    tokens = torch.tensor([[12, 45, 83, 21]])
    logits = model(tokens)
    print(logits.shape)
