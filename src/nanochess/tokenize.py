from pathlib import Path


def build_stoi(vocab: Path):
    stoi = {}
    with open(vocab, "r") as f:
        for idx, line in enumerate(f):
            stoi[line.strip("\n")] = idx
    return stoi


def elo_category(elo: int) -> str:
    if elo < 1200:
        return "<BEGINNER>"
    if elo < 1600:
        return "<INTERMEDIATE>"
    if elo < 2000:
        return "<ADVANCED>"
    if elo < 2400:
        return "<EXPERT>"
    return "<MASTER>"


def outcome(result, color):
    OUTCOMES = {
        "1-0": {"White": "<WIN>", "Black": "<LOSS>"},
        "0-1": {"White": "<LOSS>", "Black": "<WIN>"},
        "1/2-1/2": {"White": "<DRAW>", "Black": "<DRAW>"},
    }
    return OUTCOMES[result][color]


def create_examples(game, stoi):
    game_mode = game["Event"].removeprefix("Rated ").split()[0]
    result = game["Result"]
    moves = [stoi[move.uci()] for move in game.moves]

    examples = []

    for color in ("White", "Black"):
        elo = int(game[f"{color}Elo"])

        meta = [
            stoi[f"<{game_mode.upper()}>"],
            stoi[elo_category(elo)],
            stoi[f"<{color.upper()}>"],
            stoi[outcome(result, color)],
        ]

        tokens = [stoi["<BOS>"]] + meta + moves + [stoi["<EOS>"]]

        examples.append(tokens)

    return examples
