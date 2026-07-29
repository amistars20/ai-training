# ai-train — Friend AI Chat Clone

LoRA fine-tunes a small LLM on WhatsApp chat dumps to mimic a friend's texting style.

## Entrypoints

- **`colab_train.ipynb`** — primary training path (T4 GPU, ~2h/epoch for 1.5B). Open at `https://colab.research.google.com/github/amistars20/ai-training/blob/master/colab_train.ipynb`
- **`train.sh`** — full local pipeline (parse → prepare → train → GGUF). CPU only, very slow (8-36h).
- **`chat.sh <model.gguf> <friend_name>`** — interactive chat via `llama-cli`. Auto-detects `--jinja` support.
- **`chat.py <model_path> <friend_name>`** — Python inference, no GGUF needed.

## Training

### Critical: trl 1.x API
- `response_template` and `DataCollatorForCompletionOnlyLM` were **removed** in trl 1.x
- `packing=False` — no FA2, trains on all tokens in the `text` field
- Data must not contain system prompts; it's just user→assistant turns

### Model mapping (train.py:33-37)
| Size | Model | Notes |
|------|-------|-------|
| 0.5b | `Qwen/Qwen2.5-0.5B-Instruct` | Recommended for CPU training |
| 1b | `meta-llama/Llama-3.2-1B-Instruct` | **Gated** — requires HF token & license acceptance |
| **1.5b** | `Qwen/Qwen2.5-1.5B-Instruct` | Current default, non-gated |
| 3b | `microsoft/Phi-3-mini-4k-instruct` | |

### Command: `prepare_data.py --model <qwen|llama> --context <N>`
- `--model qwen` → `<|im_start|>` format (Qwen, SmolLM2, Phi)
- `--model llama` → `<|start_header_id|>` format (Llama, Gemma)
- Default context=2 (includes 2 previous turns)

### GPU/CPU config (hardcoded in train.py)
- **GPU**: batch_size=4, grad_accum=2, fp16, adamw_torch_fused, cosine LR
- **CPU**: batch_size=2, grad_accum=4, fp32, adamw_torch, cosine LR

### Colab
- Set `NUM_EPOCHS` and `RESUME` env vars in the training cell
- `working/` is symlinked to Drive (`MyDrive/ai-train-working`) — checkpoints persist across sessions
- Set `HF_TOKEN` Colab secret for faster downloads (not required for Qwen models)
- Step 1 auto-clones from public repo `github.com/amistars20/ai-training`

## Inference

- `chat.sh` uses `llama-cli --jinja --chat-mode chatml` if available, else manual `<|im_start|>` prompt
- `chat.py` uses `tokenizer.apply_chat_template()` — works with any model that has a chat template
- System prompt: "You are {friend_name}. Same slang, tone, typos, punctuation."

## Setup

- **Arch**: `sudo pacman -S llama-cpp` (provides `llama-cli`, `llama-quantize`)
- **Python venv**: auto-created at `.venv/` by `train.sh`
- **HF_TOKEN**: env var for rate-limited downloads (`export HF_TOKEN="hf_..."`)
- No CI, no tests, no lint, no typecheck, no pyproject.toml

## File layout

- `working/` — all training artifacts (gitignored). Contains `checkpoints/`, `merged-model/`, `*.gguf`
- `chats/` — raw chat + parsed data (gitignored)
- `.venv/` — Python venv (recreatable)
- `chat.json` — OpenCode session log (gitignored)
- All Python scripts sit flat in root (Colab copies them via `glob('/tmp/ai-train/*.py')`)

## Architecture notes

- **parse_chat.py**: auto-detects Android/iOS/EU WhatsApp export formats. Deduplicates near-identical pairs. `--stats-only` to inspect without saving.
- **convert_to_gguf.py**: clones `llama.cpp` source temporarily (auto-removed) for `convert_hf_to_gguf.py`, then quantizes with `llama-quantize`.
- **colab_api_server.py**: threaded HTTP server for remote agent control via Cloudflare Tunnel. Not used in manual mode.
- GGUF conversion can be done locally after downloading merged-model from Colab.
