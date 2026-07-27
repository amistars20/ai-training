#!/usr/bin/env python3
import json
import sys
import random
from pathlib import Path


QWEN_USER = "<|im_start|>user\n{msg}<|im_end|>\n"
QWEN_ASST = "<|im_start|>assistant\n{msg}<|im_end|>\n"

LLAMA_USER = "<|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|>\n"
LLAMA_ASST = "<|start_header_id|>assistant<|end_header_id|>\n\n{msg}<|eot_id|>\n"


def format_multi_turn(
    pairs: list[dict],
    friend_name: str,
    template: str = "qwen",
    context_size: int = 2,
) -> list[str]:
    if context_size < 0:
        context_size = 0

    user_tpl = QWEN_USER if template == "qwen" else LLAMA_USER
    asst_tpl = QWEN_ASST if template == "qwen" else LLAMA_ASST

    texts = []
    for i in range(len(pairs)):
        start = max(0, i - context_size)
        window = pairs[start:i + 1]

        text = ""
        for p in window[:-1]:
            text += user_tpl.format(msg=p["user"])
            text += asst_tpl.format(msg=p["assistant"])
        text += user_tpl.format(msg=window[-1]["user"])
        text += asst_tpl.format(msg=window[-1]["assistant"])

        texts.append(text)

    return texts


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_data.py <pairs.json> [--model qwen|llama] [--context N]")
        sys.exit(1)

    pairs_path = Path(sys.argv[1])
    model = "qwen"
    context_size = 2

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model = sys.argv[idx + 1]
    if "--context" in sys.argv:
        idx = sys.argv.index("--context")
        context_size = int(sys.argv[idx + 1])

    with open(pairs_path, encoding="utf-8") as f:
        data = json.load(f)

    pairs = data["pairs"]
    friend_name = data["friend_name"]

    print(f"Loaded {len(pairs)} pairs")
    print(f"Friend: {friend_name}")
    print(f"Context size: {context_size}")

    texts = format_multi_turn(pairs, friend_name, model, context_size)

    random.shuffle(texts)

    split = int(len(texts) * 0.95)
    train_texts = texts[:split]
    valid_texts = texts[split:]

    print(f"Train: {len(train_texts)}, Validation: {len(valid_texts)}")

    out_dir = pairs_path.parent
    train_file = out_dir / "train.jsonl"
    valid_file = out_dir / "valid.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for t in train_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for t in valid_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    print(f"Train data: {train_file}")
    print(f"Valid data: {valid_file}")

    total_chars = sum(len(t) for t in train_texts)
    est_tokens = total_chars // 3
    print(f"Estimated training tokens: ~{est_tokens:,}")

    sample = train_texts[0]
    print(f"\n--- Sample (context={context_size}) ---")
    print(sample[:400])
    print(f"...\nEstimated tokens: ~{estimate_tokens(sample)}")


if __name__ == "__main__":
    main()
