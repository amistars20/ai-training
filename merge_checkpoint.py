#!/usr/bin/env python3
"""Merge the last LoRA checkpoint back into the base model on CPU."""

import os
import sys
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def find_last_checkpoint(work_dir: str) -> str | None:
    ckpt_dir = Path(work_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    dirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not dirs:
        return None
    return str(max(dirs, key=lambda d: int(d.name.split("-")[-1])))


def main():
    work_dir = os.environ.get("WORK_DIR", "")
    if not work_dir:
        work_dir = os.path.abspath("working")
    if not os.path.exists(work_dir):
        work_dir = "/content/drive/MyDrive/ai-train-working"

    print(f"Work dir: {work_dir}")
    if not os.path.exists(work_dir):
        print("ERROR: working directory not found. Mount Drive first.")
        sys.exit(1)

    checkpoint = find_last_checkpoint(work_dir)
    if not checkpoint:
        print(f"ERROR: no checkpoints in {work_dir}/checkpoints/")
        sys.exit(1)
    print(f"Checkpoint: {checkpoint}")

    adapter_cfg = os.path.join(checkpoint, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            model_name = json.load(f).get("base_model_name_or_path")
    if not model_name:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Base model: {model_name}")

    print("Loading base model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, checkpoint)
    print("Merging...")
    model = model.merge_and_unload()

    merged_path = os.path.join(work_dir, "merged-model")
    print(f"Saving to {merged_path}...")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    zip_path = os.path.join(work_dir, "merged-model")
    print("Creating zip...")
    shutil.make_archive(zip_path, 'zip', merged_path)

    size_gb = sum(f.stat().st_size for f in Path(merged_path).rglob("*")) / (1024**3)
    print(f"\n=== Done ===  ({size_gb:.1f} GB)")
    print(f"Zip: {zip_path}.zip (download from drive.google.com)")
    print(f"Or use: bash chat.sh working/merged-model \"Sucrose Jar\"")


if __name__ == "__main__":
    main()
