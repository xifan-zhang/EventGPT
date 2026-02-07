#!/usr/bin/env python3
"""Debug token shapes for EventGPT."""

import os
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer, BitsAndBytesConfig
from common.common import load_image
from model.EventChatModel import EventChatModel
from conversation import conv_templates, SeparatorStyle
from data.data_process import tokenizer_event_token, EVENT_TOKEN_INDEX, prepare_event_prompt
from model.EventChatModel import get_spatio_temporal_features

# Load model
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

eventgpt_path = "./checkpoints/EventGPT-7b"
model = EventChatModel.from_pretrained(
    eventgpt_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
processor = model.get_visual_tower().event_processor

# Load sample
dataset_dir = Path("/home/ps/Documents/code/EventGPT/data/my_egpt_dsec_train/my_egpt_dsec_train_1s")
import json
with open(dataset_dir / "EventGPT_Instruction_Subset.json") as f:
    dataset = json.load(f)
sample = dataset[0]
print(f"Sample ID: {sample['id']}")

# Process
event_image_path = dataset_dir / "event_image" / sample['id'].replace('/', '_') + ".png"
img_array = load_image(str(event_image_path))
event_image_size = [[img_array.shape[0], img_array.shape[1]]]
device = model.device

event = processor(img_array, return_tensors='pt')['pixel_values'][0]
event = event.to(device, dtype=torch.bfloat16)

query = "What are the key elements in this scene?"
conv_mode = 'eventgpt_v1'
prompt = prepare_event_prompt(query, conv_mode)
input_ids = tokenizer_event_token(
    prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to(device)

print(f"\ninput_ids shape: {input_ids.shape}")
print(f"input_ids[0][:20]: {input_ids[0][:20].tolist()}")

# Get visual features
with torch.inference_mode():
    feature = model.visval_encode(event.unsqueeze(0))
    feature = model.get_model().feature_adaptor(feature)
    feature = feature.squeeze(0)
    event_features = get_spatio_temporal_features([feature])
    event_features = event_features.unsqueeze(0)

# Generate
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        event_features=event_features,
        event_image_sizes=event_image_size,
        do_sample=False,
        max_new_tokens=50,
        use_cache=True,
    )

print(f"\noutput_ids shape: {output_ids.shape}")
print(f"output_ids[0][:20]: {output_ids[0][:20].tolist()}")
print(f"output_ids[0][-10:]: {output_ids[0][-10:].tolist()}")
print(f"\nFull output text: {tokenizer.decode(output_ids[0], skip_special_tokens=True)[:300]}")
print(f"\nSliced (from input_ids.shape[1]={input_ids.shape[1]}): len={len(output_ids[0][input_ids.shape[1]:])}")
print(f"Sliced tokens: {output_ids[0][input_ids.shape[1]:].tolist()}")
