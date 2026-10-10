# WIP - NanoChess

<div align="center">
  <img src="img/image.png" alt="NanoChess" width="420" />
</div>

NanoChess is a small GPT-style chess model trained on chess tokens rather than English text. Given a sequence, it predicts the most likely next token.

(WIP) Once trained, the model will be exported to ONNX so the browser can run inference entirely client-side. You can try the chessboard [here](https://robertdavidson1.github.io/NanoChess/play/).

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
