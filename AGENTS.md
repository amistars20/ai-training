# ai-train — Friend AI Chat Clone

Single-package Python project. LoRA fine-tunes a small LLM (Qwen/Llama/Phi) on WhatsApp chat dumps to mimic a friend's texting style.

## Quick start (Colab — recommended)

1. Open `colab_train.ipynb` in Google Colab
2. Upload your `_chat.txt` file
3. Enter your name and friend's name
4. Run all cells (~30 min on T4 GPU)
5. Download the GGUF model
6. Quantize + chat locally: `llama-quantize friend-model-f16.gguf friend-model-Q4_K_M.gguf Q4_K_M` then `bash chat.sh friend-model-Q4_K_M.gguf "Name"`

## Local training (CPU — slow)

Use `train.sh` only for small tests. Full dataset takes ~17h on CPU.

## Entrypoints

- **`train.sh`** — full pipeline: parse → prepare → train → GGUF. Usage: `bash train.sh <_chat.txt> <your_name> <friend_name> [0.5b|1b|1.5b|3b]`
- **`chat.sh`** — interactive chat with trained model: `bash chat.sh <model.gguf> <friend_name>`
- **`chat.py`** — Python inference (no GGUF): `source .venv/bin/activate && python chat.py working/merged-model "Name"`
- **`parse_chat.py`** — standalone parser: `python parse_chat.py <_chat.txt> <you> <friend>`
- **`prepare_data.py`** — pairs → Qwen-format JSONL: `python prepare_data.py <pairs.json> --model qwen --context 2`
- **`colab_train.ipynb`** — Colab notebook for GPU training (repo: `amistars20/ai-training`)
- **Colab URL**: `https://colab.research.google.com/github/amistars20/ai-training/blob/master/colab_train.ipynb`

## Setup

- **System deps**: `sudo pacman -S llama-cpp` (provides llama-cli, llama-quantize)
- **Python venv**: auto-created at `.venv/` by `train.sh`
- **HF_TOKEN**: set as env var for rate-limited downloads (`export HF_TOKEN="hf_..."`). Get from https://huggingface.co/settings/tokens

## Training

- **Colab (GPU)**: 1-3 epochs, batch_size=8, grad_accum=1, bf16, `packing=False`, `response_template` for loss masking. ~2.5-3h/epoch.
- **Local (CPU)**: 3 epochs, batch_size=2, grad_accum=4, fp32. 8-36h depending on model size.
- **Context**: `--context 2` includes 2 previous turns as conversation context
- Model sizes: 0.5b (recommended), 1b, 1.5b, 3b
- Resume from checkpoint: `RESUME=1 bash train.sh ...`

## Inference

| Method | Command |
|--------|---------|
| **llama-cli** (fastest) | `bash chat.sh working/friend-model-Q4_K_M.gguf "Name"` |
| **Python** (no GGUF) | `source .venv/bin/activate && python chat.py working/merged-model "Name"` |

## File layout

- `working/` — all training artifacts (ignored by git)
- `.venv/` — Python venv (recreatable)
- `working/llama.cpp-source/` — cloned temporarily for GGUF conversion, auto-removed
