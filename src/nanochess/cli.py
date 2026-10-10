from pathlib import Path

import typer

from nanochess.fetch import fetch_raw
from nanochess.parse import parse_games
from nanochess.train import train_model

app = typer.Typer()


@app.command()
def fetch():
    """Download chess games from Lichess"""
    try:
        fetch_raw()
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def parse(max_valid_games: int = 10_000_000):
    """Parse downloaded games into training data."""
    try:
        vocab = Path("data/vocab.txt")
        dump = Path("data/raw/lichess_db_standard_rated_2026-09.pgn.zst")
        parse_games(vocab, dump, max_valid_games)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def train(
    batch_size: int = 16,
    embed_dim: int = 64,
    n_atn: int = 4,
    n_transform: int = 2,
):
    """Train the NanoChess transformer model"""
    try:
        train_model(batch_size, embed_dim, n_atn, n_transform)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def evaluate():
    """Evaluate the model"""
    print("Called evaluate()")


@app.command()
def web():
    """Start the chess web app"""
    print("Called web()")
