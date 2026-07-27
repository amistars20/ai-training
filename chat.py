#!/usr/bin/env python3
"""
Friend AI — Python inference (no GGUF needed).
Loads the merged model and runs an interactive chat session.
"""
import sys
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "working/merged-model"
    friend_name = sys.argv[2] if len(sys.argv) > 2 else "Friend"

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Usage: python chat.py <model_path> [friend_name]")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print(f"Loaded. Chatting with {friend_name} (type 'exit' to quit)\n")

    system_msg = {
        "role": "system",
        "content": (
            f"You are {friend_name}. You text exactly like them — "
            f"same slang, same tone, same typos, same punctuation "
            f"(or lack thereof). Respond naturally and concisely."
        ),
    }

    messages = [system_msg]

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_msg.lower() in ("exit", "quit"):
            break
        if not user_msg:
            continue

        messages.append({"role": "user", "content": user_msg})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        reply = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        print(f"--- {friend_name} ---")
        print(reply)
        print()

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
