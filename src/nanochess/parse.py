import pickle
import random
import tempfile
from pathlib import Path

import bulletchess.pgn
import numpy as np
import zstandard as zstd
from tqdm import tqdm

from nanochess.tokenizer import build_stoi, create_examples


def iterate_games(dump):
    # Games always start with header:
    # [Event ...
    # We parse each line until we reach a new [Event line
    # then yield each game to caller, when resumed clear for the next game
    game = ""
    with zstd.open(dump, "rt") as file:
        for line in file:
            if line.startswith('[Event "') and game:
                yield game
                game = ""

            game += line
        if game:
            yield game


def parse_games(vocab: Path, dump: Path, max_games: int = 10_000_000):
    if not vocab.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab}")
    if not dump.is_file():
        raise FileNotFoundError(f"Dump file not found: {dump}. Run 'nc fetch' first.")
    if max_games < 1:
        raise ValueError("max_games must be positive")

    # Ensure output directory exists
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    stoi = build_stoi(vocab)  # Map : move -> token idx
    rng = random.Random(42)  # For repoducible splits


    if any(not 0 <= idx <= 65535 for idx in stoi.values()):
        raise ValueError("Token IDs must fit in uint16")

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        open(output_dir / "train.bin", "wb") as train,
        open(output_dir / "val.bin", "wb") as val,
        tqdm(total=max_games, desc="Parsing", unit="game") as progress,
    ):
        temp = Path(tmpdir) / "game.pgn"
        for game in iterate_games(dump):
            # Bulletchess requires a file path
            temp.write_text(game, encoding="utf-8")

            with bulletchess.pgn.PGNFile.open(str(temp)) as pgn:
                parsed = pgn.next_game()

                if parsed is None:
                    continue

                if parsed["Result"] not in ("1-0", "0-1", "1/2-1/2"):
                    continue
                
                

                # Split by game to keep both perspectives together
                destination = val if rng.random() < 0.1 else train

                try:
                    examples = create_examples(parsed, stoi)
                except (KeyError, ValueError):
                    continue

                # Generate white and black perspectives
                for ids in examples:
                    destination.write(np.asarray(ids, dtype=np.uint16).tobytes())

            progress.update(1)

            if progress.n >= max_games:
                break

    meta = {
        "vocab_size": len(stoi),
        "stoi": stoi,
        "itos": {idx: token for token, idx in stoi.items()},
    }

    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
