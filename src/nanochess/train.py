import pickle
import time

import torch
from torch.nn import functional as F

from nanochess.dataset import get_loader
from nanochess.model import Config, NanoChess


# WIP
def build_config(embed_dim: int, n_attention_heads: int, n_transform_layers: int):
    with open("data/processed/meta.pkl", "rb") as f:
        meta = pickle.load(f)

        config = Config(
            C=embed_dim,
            V=meta["vocab_size"],
            L=meta["input_length_p99"],
            H=n_attention_heads,
            N=n_transform_layers,
        )

        return config


@torch.no_grad()
def validate(model, val_loader, device, max_length):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    try:
        for inputs, targets in val_loader:
            inputs = inputs[:, :max_length].to(device)
            targets = targets[:, :max_length].to(device)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += (targets != -100).sum().item()
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("Validation data contains no target tokens")
    return total_loss / total_tokens


def train_model(
    batch_size: int = 64,
    embed_dim: int = 64,
    n_attention_heads: int = 4,
    n_transform_layers: int = 2,
):
    if min(batch_size, embed_dim, n_attention_heads, n_transform_layers) < 1:
        raise ValueError("Batch size and model dimensions must be positive")
    if embed_dim % n_attention_heads != 0:
        raise ValueError("Embedding dimension must be divisible by attention heads")

    num_epochs = 1
    warmup_steps = 4000

    train_loader = get_loader("train", batch_size)
    val_loader = get_loader("val", batch_size)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = min(warmup_steps, max(1, total_steps // 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = build_config(
        embed_dim=embed_dim,
        n_attention_heads=n_attention_heads,
        n_transform_layers=n_transform_layers,
    )

    model = NanoChess(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.compile()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.001,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=1e-5,
            ),
        ],
        milestones=[warmup_steps],
    )

    model.train()
    start_time = time.perf_counter()
    tokens_seen = 0

    for epoch in range(num_epochs):
        for step, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train[:, : config.L].to(device)
            y_train = y_train[:, : config.L].to(device)
            tokens_seen += (y_train != -100).sum().item()

            optimizer.zero_grad(set_to_none=True)

            logits = model(x_train)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                y_train.flatten(),
                ignore_index=-100,
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 200 == 0:
                tok_per_sec = tokens_seen / (time.perf_counter() - start_time)
                print(f"epoch {epoch + 1} | step {step} | loss {loss.item():.4f} | {tok_per_sec:.0f} tok/s")

        val_loss = validate(model, val_loader, device, config.L)
        print(f"epoch {epoch + 1} | val loss {val_loss:.4f}")
