#!/usr/bin/env python3
"""
Extract Video-LLaVA's LM head weights and save standalone.

The LM head is NOT quantized in 4-bit models (stays float16), so we can
extract it once and reuse for offline token-level metrics without loading
the full model each time.

Output: ~256MB file with {'lm_head_weight': [32000, 4096], 'vocab_size', 'hidden_dim'}

Usage:
    python extract_vl_lm_head.py [--output_path ./vl_lm_head.pt]
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def extract_lm_head(
    model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
    output_path: str = "./feasible/feature_alignment/vl_lm_head.pt",
):
    from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Video-LLaVA ({model_id}) in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.eval()
    print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB")

    # Extract LM head (not quantized -- stays float16)
    lm_head_weight = model.language_model.lm_head.weight.data.cpu().float()
    vocab_size, hidden_dim = lm_head_weight.shape
    print(f"  LM head shape: [{vocab_size}, {hidden_dim}]")
    print(f"  LM head dtype: {lm_head_weight.dtype}")

    payload = {
        "lm_head_weight": lm_head_weight,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "model_id": model_id,
    }
    torch.save(payload, output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    # Free GPU
    del model
    torch.cuda.empty_cache()

    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Video-LLaVA LM head weights")
    parser.add_argument("--model_id", default="LanguageBind/Video-LLaVA-7B-hf")
    parser.add_argument("--output_path", default="./feasible/feature_alignment/vl_lm_head.pt")
    args = parser.parse_args()

    extract_lm_head(model_id=args.model_id, output_path=args.output_path)
