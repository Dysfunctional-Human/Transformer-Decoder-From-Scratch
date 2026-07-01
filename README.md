# Transformer-Decoder-From-Scratch

This repository contains a set of from-scratch PyTorch models, data utilities, and training scripts to experiment with small transformer-style decoder models and simpler baselines (e.g. bigram). It is intended as a research/learning playground where models, tokenizers and datasets can be swapped in and out quickly.

## Quick start

1. Create and activate a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell on Windows
pip install -r requirements.txt
```

1. Train a model using the default config:

```bash
python -m scripts.trainer
```

1. Check saved checkpoints and results under `trained_models/`.

## Project status (high level)

- Data utilities: implemented (`data/data_preparation.py`) including shared character tokenizers and a BPE training/loading path.
- Training loop and engine: implemented (`scripts/trainer.py`) with model saving, plotting and sample generation during training.
- Models: several model implementations live in `models/` (see below). Models can be selected from the central config.
- Generation helper: `scripts/generateFromModel.py` is a placeholder you can extend to load a checkpoint and sample from a trained model.

## Repository layout

- `configs/config.py` — central configuration (model selection, tokenizers, paths, hyperparams).
- `data/` — data loader, tokenizer builders and dataset wrapper (`data/data_preparation.py`).
- `models/` — model implementations:
  - `bigramModel.py` — simple bigram baseline
  - `decoderModel.py` / `decoderModelv2.py` — decoder architectures
  - `selfAttentionModel.py` / `multiHeadAttentionModel.py` — attention-based variants
- `scripts/trainer.py` — training entrypoint and training engine.
- `scripts/generateFromModel.py` — generation script (extendable; currently empty by design).
- `trained_models/` — output directory used by the trainer to save checkpoints, results.json and plots.
- `corpus/` — tokenizer artifacts (e.g. `corpus/bpe/tokenizer.json`).
- `dataset/` — prepared text datasets (TinyStories files included).

## Models and development stages

The repository contains models in different stages of maturity. Use the mapping below to understand which files correspond to which family and their intended use:

- Bigram baseline: `models/bigramModel.py`
  - Extremely small, deterministic baseline useful for sanity checks and fast iteration.

- Decoder family:
  - `models/decoderModel.py` — initial decoder implementation (attention blocks, feedforward, etc.).
  - `models/decoderModelv2.py` — improved decoder variant; currently the default model in the provided `configs/config.py`.

- Self-Attention / Transformer-inspired variants:
  - `models/selfAttentionModel.py` — single-head self-attention experiments.
  - `models/multiHeadAttentionModel.py` — multi-head attention variant with configurable heads.

Status notes:

- `bigramModel` and training pipeline are stable and useful for debugging.
- `decoderModelv2` is the repository default and exercises the BPE tokenizer path.
- Attention and multi-head variants are available for experimentation; performance and stability vary with hyperparameters.

## Tokenizers and data

This project supports two tokenizer modes configured in `configs/config.py`:

- Shared character tokenizer (default path used for smaller models): built by `build_shared_tokenizer()` in `data/data_preparation.py`. Artifacts (vocab/stoi/itos/meta) are saved under `corpus/<tokenizer_type>/`.
- BPE tokenizer: trained/loaded via the `tokenizers` library. When `USE_BPE` is enabled in `configs/config.py` the trainer will look for `corpus/bpe/tokenizer.json` and will train a new tokenizer automatically if it is not found (using `build_bpe_tokenizer`). The BPE tokenizer keeps a mapping of token string ⇄ id and supports special tokens like `<|endoftext|>` and `<|unk|>`.

Files of interest:

- Tokenizer artifacts: `corpus/<tokenizer_type>/vocab.json`, `stoi.json`, `itos.json`, `meta.json` and (for BPE) `tokenizer.json`.
- Example BPE file: `corpus/bpe/tokenizer.json` (if present — otherwise the trainer will create it).
- Dataset text files: `dataset/TinyStories_train_100k.txt`, `dataset/TinyStories_valid_5k.txt`, etc.

## How tokenization is used

- When `USE_BPE` is True the dataset uses the BPE `Tokenizer` to encode/decode text. The BPE tokenizer produces token ids; the `Dataset` ensures the `<|endoftext|>` token is appended when encoding stories.
- When `USE_SHARED_TOKENIZER` is True a character-level vocabulary is built and saved/loaded using `save_tokenizer_artifacts()` / `load_tokenizer_artifacts()`.

## Plug-and-play models

The training entrypoint (`scripts/trainer.py`) reads the configuration from `configs/config.py`. Model selection is driven by the `MODEL_NAME` variable in that file. To try a different model:

1. Edit `configs/config.py` and set `MODEL_NAME` to one of: `bigram`, `self_attention`, `multi_head_attention`, `decoder`, `decoderv2`.
2. Optionally adjust hyperparameters in `configs/config.py` (context length, batch size, learning rate, BPE flags, etc.).
3. Run the trainer:

```bash
python -m scripts.trainer
```

## Notes for adding a new model

1. Create a new file under `models/` implementing an nn.Module with the same training/generation interface as the other models (implement `forward()` and `generate()`/`forward()` that returns logits/loss as expected by the engine).
2. Update `configs/config.py` to import and map your model name to the class.
3. Train/test using the trainer.

## Saving, checkpoints and naming convention

- Checkpoints and results are saved under `trained_models/<model_name>/<timestamp>_<model_name>.pt` and accompany a `results.json` (loss curves) and `plot.png`.
- The trainer uses a timestamp prefix formatted like `dd-mm-yy-HH-MM-SS` for directories and names.

## Examples

Train with default config (decoderv2 by default):

```bash
python -m scripts.trainer
```

Switch to bigram model:

1. Edit `configs/config.py` and set `MODEL_NAME = "bigram"` (or use the provided mapping).
2. Run the trainer (same command).

## Generate from a checkpoint

`scripts/generateFromModel.py` is intended as a convenience script for loading a checkpoint and sampling a model. The file is currently a minimal placeholder — you can implement a CLI that:

- Loads `configs/config.py` (or a small specialized config)
- Instantiates the right model class using the same kwargs used in training
- Loads a saved state dict from a file in `trained_models/`
- Calls `model.generate()` or a similar sampling wrapper

If you'd like, I can add a ready-to-run `generateFromModel.py` CLI — tell me the preferred arguments and sampling options (temperature, max tokens, checkpoint path) and I'll implement it.

## Development notes and tips

- The trainer handles tokenizer creation: if `USE_BPE` is enabled and `corpus/bpe/tokenizer.json` is missing it will train and save a BPE tokenizer automatically.
- The `Dataset` class supports both BPE and shared tokenizers; when using BPE the dataset routes encoding/decoding calls through the tokenizers library.
- Keep `trained_models/` in `.gitignore` to avoid committing large checkpoints (the repo already contains sample trained_models artifacts for reference).

## Contributing

- Add models under `models/` and expose them by updating `configs/config.py`.
- Keep changes to the training interface minimal: the trainer expects models that support a `forward(X, y)` returning logits/loss and a `generate()` method used for sampling during training.

## Requirements & license

- See `requirements.txt` for Python package dependencies. The project uses PyTorch and the `tokenizers` library (Hugging Face tokenizers) for BPE support.
- This project includes a `LICENSE` file in the repository root.
