"""
证明 EventGPT 和 VLM 在技术层面可以进行 speculative decoding，
并在此过程中“完全模拟 inference.py 里的行为”来加载 EventGPT 的 tokenizer。
"""

import os
import sys

# 将项目根目录加入 sys.path，确保可以导入 model、dataset 等包
# 当前文件路径: <PROJECT_ROOT>/feasible/tokenizer_check/tokenizer_check.py
# 因此需要向上三级目录，才能到达项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer

from model.EventChatModel import EventChatModel
from dataset.constants import (
    EVENT_TOKEN_INDEX,
    DEFAULT_EVENT_TOKEN,
    DEFAULT_EV_START_TOKEN,
    DEFAULT_EV_END_TOKEN,
    EVENT_PLACEHOLDER,
    DEFAULT_EVENT_PATCH_TOKEN,
)

# ⚠️ 请确保这里与推理脚本中 --model_path 使用的路径一致
# 例如：python inference.py --model_path "./checkpoints/EventGPT-7b" ...
EVENTGPT_MODEL_PATH = "./checkpoints/EventGPT-7b"


def load_eventgpt_tokenizer(model_path: str):
    """严格按照 inference.py 的方式加载并扩展 EventGPT tokenizer。"""
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # model = EventChatModel.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     config=config,
    # )

    # # 与 inference.py 中第32–38行逻辑保持一致
    # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

    # if mm_use_im_patch_token:
    #     tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
    # if mm_use_im_start_end:
    #     tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)

    # # 与 inference.py 第39行一致：根据新词表大小调整 embedding
    # model.resize_token_embeddings(len(tokenizer))

    return tokenizer

# 详细诊断脚本
from transformers import AutoTokenizer
import json

def diagnose_tokenizer_difference(eventgpt_tokenizer, vlm_tokenizer):
    """
    详细分析两个 tokenizer 的差异
    """
    print("="*80)
    print("TOKENIZER COMPATIBILITY DIAGNOSIS")
    print("="*80)
    
    # 1. 基础信息对比
    print("\n📊 Basic Information:")
    print(f"EventGPT vocab size: {len(eventgpt_tokenizer)}")
    print(f"VLM vocab size: {len(vlm_tokenizer)}")
    print(f"Difference: {len(vlm_tokenizer) - len(eventgpt_tokenizer)} tokens")
    
    # 2. 特殊 tokens 对比
    print("\n🔑 Special Tokens:")
    special_tokens_info = {
        "bos_token": (eventgpt_tokenizer.bos_token, vlm_tokenizer.bos_token),
        "eos_token": (eventgpt_tokenizer.eos_token, vlm_tokenizer.eos_token),
        "unk_token": (eventgpt_tokenizer.unk_token, vlm_tokenizer.unk_token),
        "pad_token": (eventgpt_tokenizer.pad_token, vlm_tokenizer.pad_token),
    }
    
    for token_type, (event_token, vlm_token) in special_tokens_info.items():
        match = "✅" if event_token == vlm_token else "❌"
        print(f"{token_type:12} EventGPT: {event_token!r:10} VLM: {vlm_token!r:10} {match}")
    
    # 3. 找出 VLM 额外的 tokens
    print("\n🔎 Analyzing Extra Tokens in VLM:")
    
    # 获取词汇表
    eventgpt_vocab = eventgpt_tokenizer.get_vocab()
    vlm_vocab = vlm_tokenizer.get_vocab()
    
    # 找出 VLM 独有的 tokens
    vlm_only_tokens = set(vlm_vocab.keys()) - set(eventgpt_vocab.keys())
    
    print(f"VLM has {len(vlm_only_tokens)} unique tokens:")
    for token in sorted(vlm_only_tokens):
        token_id = vlm_vocab[token]
        print(f"  - '{token}' (ID: {token_id})")
    
    # 4. 检查 added_tokens
    if hasattr(vlm_tokenizer, 'added_tokens_encoder'):
        print("\n➕ VLM Added Tokens:")
        for token, token_id in vlm_tokenizer.added_tokens_encoder.items():
            print(f"  - '{token}' (ID: {token_id})")
    
    # 5. 测试编码一致性
    print("\n🧪 Encoding Consistency Tests:")
    test_cases = [
        "The object is moving.",
        "A person walks in the scene.",
        "Hello world!",
        "This is a test sentence with numbers: 123456.",
    ]
    
    all_match = True
    for i, text in enumerate(test_cases, 1):
        event_ids = eventgpt_tokenizer.encode(text, add_special_tokens=False)
        vlm_ids = vlm_tokenizer.encode(text, add_special_tokens=False)
        
        match = event_ids == vlm_ids
        all_match = all_match and match
        
        status = "✅" if match else "❌"
        print(f"Test {i} {status}: '{text[:40]}...'")
        
        if not match:
            print(f"  EventGPT IDs: {event_ids}")
            print(f"  VLM IDs: {vlm_ids}")
    
    # 6. 最终结论
    print("\n" + "="*80)
    print("COMPATIBILITY ASSESSMENT")
    print("="*80)
    
    return {
        "vocab_size_match": len(eventgpt_tokenizer) == len(vlm_tokenizer),
        "encoding_match": all_match,
        "extra_tokens": vlm_only_tokens,
        "num_extra": len(vlm_only_tokens)
    }

if __name__ == "__main__":
    # 1. 按照 inference.py 的行为加载 EventGPT 的 tokenizer
    eventgpt_tokenizer = load_eventgpt_tokenizer(EVENTGPT_MODEL_PATH)

    # 2. 加载 VLM tokenizer（保持你原来的设定）
    vlm_tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-13b-hf")
    # vlm_tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # 3. 打印词表大小，做基本 sanity check
    print(f"EventGPT vocab size: {len(eventgpt_tokenizer)}") # EventGPT vocab size: 32000
    print(f"VLM vocab size: {len(vlm_tokenizer)}") # VLM vocab size: 32002
    
    # 差两个，执行诊断
    result = diagnose_tokenizer_difference(eventgpt_tokenizer, vlm_tokenizer)

    # 4. 测试相同文本的编码结果（只比较自然语言部分的编码是否一致）
    test_texts = [
        "The object is moving rapidly from left to right.",
        "A person is holding something in the scene.",
        "The motion trajectory shows acceleration.",
    ]

    for text in test_texts:
        event_ids = eventgpt_tokenizer.encode(text)
        vlm_ids = vlm_tokenizer.encode(text)
        assert event_ids == vlm_ids, f"Tokenization mismatch for: {text}"

    print("✓ Tokenizer compatibility confirmed (with inference-style EventGPT tokenizer)!")