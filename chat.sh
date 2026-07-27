#!/usr/bin/env bash
set -euo pipefail

# ─── Friend AI — Interactive Chat ───────────────────────────────────────────
# Usage: bash chat.sh <model.gguf> <friend_name>
# ─────────────────────────────────────────────────────────────────────────────

MODEL="${1:?Usage: $0 <model.gguf> <friend_name>}"
FRIEND_NAME="${2:-Friend}"

if ! command -v llama-cli &>/dev/null; then
    echo "llama-cli not found. Install: sudo pacman -S llama-cpp"
    exit 1
fi

echo "=== Friend AI Chat ==="
echo "You are chatting with $FRIEND_NAME (AI clone)"
echo "Type 'exit' or Ctrl+C to quit"
echo ""

# Try to use --jinja (chat mode) if available, otherwise fallback to manual prompt
HAS_JINJA=false
llama-cli --help 2>&1 | grep -q jinja && HAS_JINJA=true

if $HAS_JINJA; then
    # Modern llama.cpp with chat template support
    llama-cli \
        -m "$MODEL" \
        --temp 0.8 \
        --top-p 0.9 \
        --repeat-penalty 1.1 \
        --ctx-size 4096 \
        --jinja \
        --system "You are $FRIEND_NAME. You text exactly like them — same slang, same tone, same typos, same punctuation (or lack thereof). Respond naturally and concisely." \
        --chat-mode chatml
else
    # Fallback: manual prompt format
    while true; do
        printf "You: "
        IFS= read -r user_msg
        if [ "$user_msg" = "exit" ] || [ "$user_msg" = "quit" ]; then
            break
        fi
        if [ -z "$user_msg" ]; then
            continue
        fi

        PROMPT="<|im_start|>system
You are $FRIEND_NAME. You text exactly like them — same slang, same tone, same typos, same punctuation (or lack thereof). Respond naturally and concisely.<|im_end|>
<|im_start|>user
$user_msg<|im_end|>
<|im_start|>assistant
"

        echo "--- $FRIEND_NAME ---"
        llama-cli \
            -m "$MODEL" \
            -p "$PROMPT" \
            --temp 0.8 \
            --top-p 0.9 \
            --repeat-penalty 1.1 \
            --ctx-size 4096 \
            --threads 8 \
            --no-display-prompt \
            2>/dev/null
        echo ""
    done
fi
