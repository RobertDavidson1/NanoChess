<h1 align="center">
  <a href="https://robertdavidson1.github.io/NanoChess/play/">
    <img src="img/Icon.png" alt="NanoChess" width="800" />
  </a>
</h1>
<div align="center">
  <img src="img/Web.png" alt="NanoChess chessboard" width="400" />
</div>

NanoChess is an experiment in teaching a language model to play chess. Instead of feeding it natural language text, it learns directly from millions of Lichess games, representing moves as tokens and using metadata such as rating, colour, time control, and result to predict what comes next. The goal is a compact chess model that can eventually run privately and entirely in the browser, with no server required.

(WIP) Once trained, the model will be exported to ONNX so the browser can run inference entirely client-side. You can try the chessboard [here](https://robertdavidson1.github.io/NanoChess/play/).

### Contents

- [Getting started](#getting-started)
- [Project structure](#project-structure)

### Getting started

```Bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Fetch data from lichess (~30GB dump)
nc fetch

# 5. Parse and tokenize up to 10 million games
nc parse

# WIP - Only prints a batch atm
# 6. Train the model
# nc train


# WIP - not implemented
# 7. Evaluate the model
# nc eval

# WIP - not implemented
# 8. Export the model to onnx
# nc export

```

### Project structure

```text
NanoChess/
├── play/                         # Static browser chessboard
│   └── index.html                # Board UI, styling, and chess rules
├── src/nanochess/                # Python package
│   ├── cli.py                    # `nc fetch`, `nc parse`, and `nc train` commands
│   ├── dataset.py                # PyTorch dataset, offsets, and DataLoaders
│   ├── fetch.py                  # Downloads Lichess game dump
│   ├── model.py                  # GPT-style transformer model
│   ├── parse.py                  # Parses PGN games into token data
│   ├── tokenize.py               # Maps vocabulary tokens and creates game examples
│   └── train.py                  # Training entry point (currently WIP)
├── data/
│   ├── vocab.txt                 # Move and metadata vocabulary
│   ├── raw/                      # Downloaded PGN dumps (ignored by Git)
│   └── processed/                # Token files, offsets, and metadata
├── img/                          # README and web artwork
│   ├── Icon.png                  # NanoChess logo
│   └── Web.png                   # Chessboard preview
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Locked dependency versions
└── README.md                     # Project documentation
```
