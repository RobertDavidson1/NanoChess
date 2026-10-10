import pickle
import random
import tempfile
from pathlib import Path

import bulletchess.pgn
import numpy as np
import zstandard as zstd
from tqdm import tqdm

from nanochess.tokenize import build_stoi, create_examples


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
        open(output_dir / "train_offsets.bin", "wb") as train_offsets, 
        open(output_dir / "val_offsets.bin", "wb") as val_offsets, 
        tqdm(total=max_games, desc="Parsing", unit="game") as progress,
    ):
        temp = Path(tmpdir) / "game.pgn"


        # Store initial starting index so what when we need length of game i we have:
        # length(i) =  offset[i +1] - offset[i]
        train_offsets.write(np.asarray([0], dtype=np.uint64).tobytes())
        val_offsets.write(np.asarray([0], dtype=np.uint64).tobytes())

        for game in iterate_games(dump):
            # Bulletchess requires a file path
            temp.write_text(game, encoding="utf-8")

            with bulletchess.pgn.PGNFile.open(str(temp)) as pgn:
                parsed = pgn.next_game()

                if parsed is None:
                    continue

                if parsed["Result"] not in ("1-0", "0-1", "1/2-1/2"):
                    continue

                
                # Randomly split : 10% Val, 90% Train
                destination = val if rng.random() < 0.1 else train
                offsets = val_offsets if destination is val else train_offsets

                try:
                    examples = create_examples(parsed, stoi)
                except (KeyError, ValueError):
                    continue

                # Generate white and black perspectives
                for ids in examples:
                    destination.write(np.asarray(ids, dtype=np.uint16).tobytes())


                    # Store the offsets of each games so that for game i:
                    # length(i) =  offset[i +1] - offset[i]
                    pos_bytes = destination.tell()  # Get the file position in bytes
                    pos_tokens = pos_bytes // 2     # 2 bytes per token -> num_tokens = num_bytes // 2
                    offsets.write(np.asarray(pos_tokens, dtype=np.uint64).tobytes()) # Write the positions as uint64 to the offsets

            progress.update(1)

            if progress.n >= max_games:
                break
    

    offsets = np.fromfile(output_dir / "train_offsets.bin", dtype=np.uint64)
    lengths = np.diff(offsets, n=1)
    meta = {
        "vocab_size": len(stoi),
        "stoi": stoi,
        "itos": {idx: token for token, idx in stoi.items()},
        "input_legnth_p99" : int(np.percentile(lengths, 99, method="higher"))
    }

    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
