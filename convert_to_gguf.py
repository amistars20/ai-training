#!/usr/bin/env python3
"""
Convert a Hugging Face merged model to GGUF format.
Clones llama.cpp temporarily for the conversion script, then removes it.
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path


def main():
    model_dir = os.environ.get("MODEL_DIR", "")
    work_dir = os.environ.get("WORK_DIR", "working")
    model_name = os.environ.get("MODEL_NAME", "friend-model")
    keep_llama = os.environ.get("KEEP_LLAMA_CPP", "").lower() in ("1", "true")

    if not model_dir:
        model_dir = os.path.join(work_dir, "merged-model")
    model_dir = os.path.abspath(model_dir)
    work_dir = os.path.abspath(work_dir)

    if not Path(model_dir).exists():
        print(f"Model directory not found: {model_dir}")
        print("Set MODEL_DIR env var or ensure working/merged-model exists.")
        sys.exit(1)

    llama_dir = os.path.join(work_dir, "llama.cpp-source")
    converter = os.path.join(llama_dir, "convert_hf_to_gguf.py")

    if not Path(converter).exists():
        print("Cloning llama.cpp for conversion script...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp.git",
             llama_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Clone failed: {result.stderr}")
            sys.exit(1)
        print("Cloned.")
    else:
        print("llama.cpp source already exists.")

    f16_path = os.path.join(work_dir, f"{model_name}-f16.gguf")
    print(f"Converting {model_dir} to GGUF (F16)...")

    result = subprocess.run(
        [sys.executable, converter, str(model_dir),
         "--outfile", str(f16_path), "--outtype", "f16"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Conversion failed: {result.stderr}")
        if not keep_llama and Path(llama_dir).exists():
            shutil.rmtree(llama_dir)
        sys.exit(1)
    print(f"F16 GGUF: {f16_path}")

    q4_path = os.path.join(work_dir, f"{model_name}-Q4_K_M.gguf")
    print(f"Quantizing to Q4_K_M...")

    result = subprocess.run(
        ["llama-quantize", str(f16_path), str(q4_path), "Q4_K_M"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Quantization failed: {result.stderr}")
        if not keep_llama and Path(llama_dir).exists():
            shutil.rmtree(llama_dir)
        sys.exit(1)

    file_size = Path(q4_path).stat().st_size / (1024 * 1024)
    print(f"Q4_K_M GGUF: {q4_path} ({file_size:.1f} MB)")

    if not keep_llama and Path(llama_dir).exists():
        print("Cleaning up llama.cpp source...")
        shutil.rmtree(llama_dir)

    print("\n=== Done ===")
    print(f"Final quantized model: {q4_path}")
    print(f"To chat: bash chat.sh \"{q4_path}\" \"FriendName\"")


if __name__ == "__main__":
    main()
