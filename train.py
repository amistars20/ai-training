#!/usr/bin/env python3
import json, sys, os, math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset


def find_checkpoint(work_dir: str) -> str | None:
    ckpt_dir = Path(work_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    dirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not dirs:
        return None
    return str(max(dirs, key=lambda d: int(d.name.split("-")[-1])))


def main():
    model_size = os.environ.get("MODEL_SIZE", "0.5b")
    train_file = os.environ.get("TRAIN_FILE", "train.jsonl")
    valid_file = os.environ.get("VALID_FILE", "valid.jsonl")
    work_dir = os.environ.get("WORK_DIR", "working")
    resume = os.environ.get("RESUME", "").lower() in ("1", "true", "yes")

    os.makedirs(work_dir, exist_ok=True)
    ckpt_dir = os.path.join(work_dir, "checkpoints")

    models = {
        "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "1b":   "meta-llama/Llama-3.2-1B-Instruct",
        "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "3b":   "microsoft/Phi-3-mini-4k-instruct",
    }
    if model_size not in models:
        print(f"Unknown model size: {model_size}")
        sys.exit(1)

    model_name = models[model_size]
    print(f"Model: {model_name}")
    print(f"Train data: {train_file}")
    print(f"Valid data: {valid_file}")

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    fp16 = has_cuda and torch.cuda.get_device_capability()[0] >= 7
    dtype = torch.float16 if fp16 else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": train_file, "validation": valid_file})

    if has_cuda:
        training_args = SFTConfig(
            output_dir=ckpt_dir,
            num_train_epochs=int(os.environ.get("NUM_EPOCHS", 1)),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_steps=50,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_torch_fused",
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=fp16,
            packing=False,
            max_length=1024,
            dataset_text_field="text",
        )
    else:
        training_args = TrainingArguments(
            output_dir=ckpt_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=50,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_torch",
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    if resume:
        checkpoint = find_checkpoint(work_dir)
        if checkpoint:
            print(f"Resuming from checkpoint: {checkpoint}")
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()
    else:
        trainer.train()

    adapter_path = os.path.join(work_dir, "lora-adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved: {adapter_path}")

    merged_path = os.path.join(work_dir, "merged-model")
    print(f"Merging LoRA into base model...")
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, dtype=dtype, trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved: {merged_path}")

    eval_results = trainer.evaluate()
    print(f"\nFinal eval loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print("\n=== Done ===")
    print("Next: run convert_to_gguf.py to convert to GGUF format")


if __name__ == "__main__":
    main()
