#!/usr/bin/env bash
set -euo pipefail

# ─── Friend AI — Training Pipeline ───────────────────────────────────────────
# Usage: bash train.sh <chat.txt> <your_name> <friend_name> [model_size]
#
# model_size: 0.5b (default), 1b, 1.5b, 3b
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHAT_FILE="${1:?Usage: $0 <chat.txt> <your_name> <friend_name> [model_size]}"
YOUR_NAME="${2:?}"
FRIEND_NAME="${3:?}"
MODEL_SIZE="${4:-0.5b}"

WORK_DIR="$SCRIPT_DIR/working"
VENV_DIR="$SCRIPT_DIR/.venv"
mkdir -p "$WORK_DIR"

# ─── Step 0: Check tools ────────────────────────────────────────────────────

echo "=== Step 0: Checking tools ==="
if ! command -v llama-cli &>/dev/null; then
    echo "Installing llama.cpp..."
    sudo pacman -S --noconfirm llama-cpp
fi
if ! command -v llama-quantize &>/dev/null; then
    echo "llama-quantize not found. Install: sudo pacman -S llama-cpp"
    exit 1
fi

# ─── Set up Python virtual environment ───────────────────────────────────────

echo "=== Step 0b: Setting up Python environment ==="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install dependencies (quietly, only if needed)
pip install -q torch --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null
pip install -q transformers peft datasets accelerate scipy gguf 2>/dev/null
pip install -q trl 2>/dev/null

echo "Python environment ready."

# ─── Step 1: Parse chat ─────────────────────────────────────────────────────

echo "=== Step 1: Parsing chat ==="
python3 "$SCRIPT_DIR/parse_chat.py" "$CHAT_FILE" "$YOUR_NAME" "$FRIEND_NAME"
PAIRS_FILE="${CHAT_FILE%.*}.pairs.json"
echo "Pairs file: $PAIRS_FILE"

# ─── Step 2: Prepare training data ──────────────────────────────────────────

echo "=== Step 2: Preparing training data ==="
python3 "$SCRIPT_DIR/prepare_data.py" "$PAIRS_FILE" --model qwen

# Move data files to working directory
cp "$SCRIPT_DIR/train.jsonl" "$WORK_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/valid.jsonl" "$WORK_DIR/" 2>/dev/null || true

TRAIN_FILE="$WORK_DIR/train.jsonl"
VALID_FILE="$WORK_DIR/valid.jsonl"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: training data not found at $TRAIN_FILE"
    exit 1
fi

# ─── Step 3: Train LoRA (Python + transformers + peft) ──────────────────────

echo "=== Step 3: Training LoRA adapters ==="
echo "Using model: $MODEL_SIZE"
echo "This will take a while. Training on CPU with 8 threads..."
echo "Estimated time: 8-36 hours depending on model size."
echo ""

export MODEL_SIZE
export TRAIN_FILE
export VALID_FILE
export WORK_DIR
export OMP_NUM_THREADS=8

time python3 "$SCRIPT_DIR/train.py" 2>&1 | tee "$WORK_DIR/training.log"

echo "Training complete."

# ─── Step 4: Convert to GGUF ────────────────────────────────────────────────

echo "=== Step 4: Converting to GGUF ==="
export MODEL_DIR="$WORK_DIR/merged-model"
export WORK_DIR
export MODEL_NAME="friend-model"

# Make sure we use the right Python for subprocess calls
export VENV_PYTHON="$VENV_DIR/bin/python3"

python3 "$SCRIPT_DIR/convert_to_gguf.py" 2>&1 | tee -a "$WORK_DIR/training.log"

# ─── Find the output model ──────────────────────────────────────────────────

FINAL_MODEL="$WORK_DIR/friend-model-Q4_K_M.gguf"
if [ ! -f "$FINAL_MODEL" ]; then
    FINAL_MODEL="$WORK_DIR/friend-model-f16.gguf"
fi

# ─── Done ──────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "  ✅ TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "Model: $FINAL_MODEL"
echo "Size: $(du -h "$FINAL_MODEL" | cut -f1)"
echo ""
echo "To chat with your friend AI:"
echo "  bash $SCRIPT_DIR/chat.sh $FINAL_MODEL \"$FRIEND_NAME\""
echo ""
echo "Or directly:"
echo "  llama-cli -m \"$FINAL_MODEL\" --temp 0.8 --repeat-penalty 1.1 --ctx-size 4096 --jinja"
echo ""
