#!/usr/bin/env python3
"""
Shared-Decoder Speculative Decoding: EventGPT + Video-LLaVA
===========================================================

Architecture:
- EventGPT visual encoder → Adapter → Video-LLaVA decoder (shared)
- Video-LLaVA visual encoder → Video-LLaVA decoder (shared)

This enables true speculative decoding with high token acceptance rate
by using the same LLM decoder for both draft and target.

Pipeline:
1. Extract visual features from both models (training set)
2. Train alignment adapter on training set
3. Benchmark speculative decoding on test set

Usage:
    conda activate egpt
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python shared_decoder_speculative_S1.py \
        --dataset_dir ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
        --max_samples 100
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from dataclasses import dataclass

# Add paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'feasible', 'feature_alignment'))
sys.path.insert(0, os.path.join(ROOT, 'feasible'))

from transformers import (
    AutoConfig,
    AutoTokenizer,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
    CLIPImageProcessor,
)

from model.EventChatModel import EventChatModel
from dataset.constants import (
    DEFAULT_EVENT_PATCH_TOKEN,
    DEFAULT_EV_START_TOKEN,
    DEFAULT_EV_END_TOKEN,
)


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class AdapterConfig:
    """Configuration for feature alignment adapter."""
    input_dim: int = 4096  # EventGPT projected features
    output_dim: int = 4096  # Video-LLaVA projected features
    hidden_dim: int = 2048
    num_layers: int = 2
    dropout: float = 0.1


class FeatureAlignmentAdapter(nn.Module):
    """
    Adapter to align EventGPT visual features to Video-LLaVA space.

    Input: EventGPT projected features [batch, seq_len, 4096]
    Output: Aligned features [batch, seq_len, 4096]
    """

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.input_dim

        for i in range(config.num_layers):
            out_dim = config.hidden_dim if i < config.num_layers - 1 else config.output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < config.num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.dropout))
            in_dim = out_dim

        self.adapter = nn.Sequential(*layers)

        # Residual connection if dimensions match
        self.use_residual = config.input_dim == config.output_dim

    def forward(self, x):
        out = self.adapter(x)
        if self.use_residual:
            out = out + x
        return out


class SharedDecoderPipeline:
    """
    Pipeline for shared-decoder speculative decoding.

    EventGPT visual features → Adapter → Video-LLaVA decoder
    """

    def __init__(
        self,
        eventgpt_path: str,
        videollava_path: str,
        device: str = 'cuda',
    ):
        self.device = device
        self.eventgpt_path = eventgpt_path
        self.videollava_path = videollava_path

        # Will be loaded on demand
        self.eventgpt = None
        self.videollava = None
        self.adapter = None

    def load_eventgpt(self):
        """Load EventGPT model."""
        if self.eventgpt is not None:
            return

        print("Loading EventGPT...")
        config = AutoConfig.from_pretrained(self.eventgpt_path)

        # Fix CLIP path
        if hasattr(config, 'mm_visual_tower'):
            visual_tower = config.mm_visual_tower
            if not os.path.exists(visual_tower):
                local_clip = os.path.join(ROOT, "checkpoints", "clip-vit-large-patch14-336")
                if os.path.exists(local_clip):
                    config.mm_visual_tower = local_clip
                else:
                    config.mm_visual_tower = "openai/clip-vit-large-patch14-336"

        self.eventgpt_tokenizer = AutoTokenizer.from_pretrained(self.eventgpt_path, use_fast=False)
        self.eventgpt = EventChatModel.from_pretrained(
            self.eventgpt_path,
            torch_dtype=torch.bfloat16,
            config=config,
        )

        # Add special tokens
        if getattr(self.eventgpt.config, "mm_use_im_patch_token", True):
            self.eventgpt_tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
        if getattr(self.eventgpt.config, "mm_use_im_start_end", False):
            self.eventgpt_tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)

        self.eventgpt.resize_token_embeddings(len(self.eventgpt_tokenizer))

        vision_tower = self.eventgpt.get_visual_tower()
        self.event_processor = vision_tower.event_processor

        self.eventgpt.to(self.device)
        self.eventgpt.eval()
        print("EventGPT loaded.")

    def unload_eventgpt(self):
        """Unload EventGPT to free memory."""
        if self.eventgpt is not None:
            del self.eventgpt
            del self.eventgpt_tokenizer
            self.eventgpt = None
            cleanup_gpu()
            print("EventGPT unloaded.")

    def load_videollava(self):
        """Load Video-LLaVA model."""
        if self.videollava is not None:
            return

        print("Loading Video-LLaVA...")
        self.videollava_processor = VideoLlavaProcessor.from_pretrained(self.videollava_path)
        self.videollava = VideoLlavaForConditionalGeneration.from_pretrained(
            self.videollava_path,
            torch_dtype=torch.float16,
        )
        self.videollava.to(self.device)
        self.videollava.eval()
        print("Video-LLaVA loaded.")

    def unload_videollava(self):
        """Unload Video-LLaVA to free memory."""
        if self.videollava is not None:
            del self.videollava
            del self.videollava_processor
            self.videollava = None
            cleanup_gpu()
            print("Video-LLaVA unloaded.")

    def extract_eventgpt_features(self, event_image_paths: list, target_len: int = 257) -> torch.Tensor:
        """
        Extract projected visual features from EventGPT.

        Args:
            event_image_paths: List of paths to event images
            target_len: Target sequence length (default 257 to match Video-LLaVA)

        Returns: [target_len, 4096] tensor of projected features
        """
        self.load_eventgpt()

        # Process event images through CLIP
        event_tensors = []
        for img_path in event_image_paths:
            img = np.array(Image.open(img_path).convert('RGB'))
            processed = self.event_processor(img, return_tensors='pt')['pixel_values'][0]
            processed = processed.to(self.device, dtype=torch.bfloat16)
            event_tensors.append(processed)

        with torch.no_grad():
            # Get visual tower
            vision_tower = self.eventgpt.get_visual_tower()

            # Extract CLIP features for each event image
            features_list = []
            for ev_tensor in event_tensors:
                ev_tensor = ev_tensor.unsqueeze(0)  # [1, C, H, W]

                # Through CLIP vision model
                clip_output = vision_tower.visual_tower.vision_model(ev_tensor)
                clip_features = clip_output.last_hidden_state  # [1, seq_len, 1024]

                # Through visual projector (MLP)
                model = self.eventgpt.get_model()
                projected = model.visual_projector(clip_features)  # [1, seq_len, 4096]

                # Through feature adaptor if exists
                if hasattr(model, 'feature_adaptor'):
                    projected = model.feature_adaptor(projected)

                features_list.append(projected.squeeze(0))  # [seq_len, 4096]

            # Combine features (e.g., mean pool across images)
            if len(features_list) > 1:
                # Stack and mean pool
                stacked = torch.stack(features_list)  # [num_images, seq_len, 4096]
                features = stacked.mean(dim=0)  # [seq_len, 4096]
            else:
                features = features_list[0]

            # Resize to target length using interpolation
            current_len = features.shape[0]
            if current_len != target_len:
                # Interpolate to target length
                features = features.unsqueeze(0).transpose(1, 2)  # [1, 4096, seq_len]
                features = torch.nn.functional.interpolate(
                    features,
                    size=target_len,
                    mode='linear',
                    align_corners=False
                )
                features = features.transpose(1, 2).squeeze(0)  # [target_len, 4096]

        return features.float()

    def extract_videollava_features(self, video_frame_paths: list, target_len: int = 257) -> torch.Tensor:
        """
        Extract projected visual features from Video-LLaVA.

        Args:
            video_frame_paths: List of paths to video frames
            target_len: Target sequence length (default 257)

        Returns: [target_len, 4096] tensor of projected features
        """
        self.load_videollava()

        # Load video frames as PIL images
        frames = []
        for img_path in video_frame_paths[:8]:
            img = Image.open(img_path).convert('RGB')
            frames.append(img)

        with torch.no_grad():
            # Process frames - use images list directly
            inputs = self.videollava_processor(
                text="",  # Empty text for vision-only processing
                images=frames,
                videos=frames,
                return_tensors="pt",
            )

            # Try multiple keys for pixel values
            pixel_values = inputs.get('pixel_values_videos')
            if pixel_values is None:
                pixel_values = inputs.get('pixel_values')
            if pixel_values is None:
                # Try to find any tensor in the inputs
                for key, val in inputs.items():
                    if isinstance(val, torch.Tensor) and 'pixel' in key.lower():
                        pixel_values = val
                        break

            if pixel_values is None:
                raise ValueError(f"Could not get pixel values from processor. Keys: {list(inputs.keys())}")

            pixel_values = pixel_values.to(self.device, torch.float16)

            # Through video tower
            video_tower = self.videollava.video_tower

            # Reshape if needed: [batch, num_frames, C, H, W] -> [batch*num_frames, C, H, W]
            if pixel_values.dim() == 5:
                b, t, c, h, w = pixel_values.shape
                pixel_values = pixel_values.view(b * t, c, h, w)
            elif pixel_values.dim() == 4:
                pass  # Already in [batch, C, H, W] format

            vision_outputs = video_tower(pixel_values, output_hidden_states=True)
            vision_features = vision_outputs.last_hidden_state  # [num_frames, seq_len, hidden]

            # Through multi-modal projector
            projected = self.videollava.multi_modal_projector(vision_features)  # [num_frames, seq_len, 4096]

            # Mean pool across frames
            features = projected.mean(dim=0)  # [seq_len, 4096]

            # Resize to target length if needed
            current_len = features.shape[0]
            if current_len != target_len:
                features = features.unsqueeze(0).transpose(1, 2)  # [1, 4096, seq_len]
                features = torch.nn.functional.interpolate(
                    features,
                    size=target_len,
                    mode='linear',
                    align_corners=False
                )
                features = features.transpose(1, 2).squeeze(0)  # [target_len, 4096]

        return features.float()

    def train_adapter(
        self,
        eventgpt_features: torch.Tensor,
        videollava_features: torch.Tensor,
        num_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 16,
    ):
        """
        Train alignment adapter on paired features.

        Args:
            eventgpt_features: [num_samples, seq_len, 4096]
            videollava_features: [num_samples, seq_len, 4096]
        """
        print(f"\nTraining adapter on {len(eventgpt_features)} samples...")

        config = AdapterConfig(
            input_dim=eventgpt_features.shape[-1],
            output_dim=videollava_features.shape[-1],
            hidden_dim=512,  # Smaller hidden dim for stability
            num_layers=2,
        )
        self.adapter = FeatureAlignmentAdapter(config).to(self.device)

        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

        # Flatten features: [num_samples, seq_len, dim] -> [num_samples * seq_len, dim]
        src = eventgpt_features.view(-1, eventgpt_features.shape[-1]).to(self.device)
        tgt = videollava_features.view(-1, videollava_features.shape[-1]).to(self.device)

        dataset_size = src.shape[0]

        for epoch in range(num_epochs):
            self.adapter.train()
            total_loss = 0
            num_batches = 0

            # Shuffle
            perm = torch.randperm(dataset_size)
            src_shuffled = src[perm]
            tgt_shuffled = tgt[perm]

            for i in range(0, dataset_size, batch_size):
                batch_src = src_shuffled[i:i+batch_size]
                batch_tgt = tgt_shuffled[i:i+batch_size]

                optimizer.zero_grad()

                aligned = self.adapter(batch_src)

                # MSE + Cosine similarity loss
                mse_loss = F.mse_loss(aligned, batch_tgt)
                cos_loss = 1 - F.cosine_similarity(aligned, batch_tgt, dim=-1).mean()
                loss = mse_loss + 0.5 * cos_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / num_batches

            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")

        self.adapter.eval()
        print("Adapter training complete.")

        return self.adapter

    def generate_with_shared_decoder(
        self,
        visual_features: torch.Tensor,
        query: str,
        max_new_tokens: int = 50,
        use_adapter: bool = False,
    ) -> tuple:
        """
        Generate text using Video-LLaVA's decoder with custom visual features.

        Args:
            visual_features: [seq_len, 4096] projected visual features
            query: Text query
            use_adapter: Whether to pass features through alignment adapter

        Returns:
            (output_text, token_ids, generation_time)
        """
        self.load_videollava()

        # Move features to device first
        visual_features = visual_features.to(self.device)

        # Apply adapter if requested
        if use_adapter and self.adapter is not None:
            with torch.no_grad():
                visual_features = self.adapter(visual_features.unsqueeze(0)).squeeze(0)

        visual_features = visual_features.to(torch.float16)

        # Prepare text inputs
        prompt = f"USER: <video>\n{query} ASSISTANT:"

        # Tokenize
        text_inputs = self.videollava_processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']

        # Get text embeddings
        text_embeds = self.videollava.get_input_embeddings()(input_ids)

        # Find <video> token position and inject visual features
        video_token_id = self.videollava_processor.tokenizer.convert_tokens_to_ids('<video>')

        # Find position of video token
        video_positions = (input_ids == video_token_id).nonzero(as_tuple=True)

        if len(video_positions[0]) > 0:
            batch_idx = video_positions[0][0]
            pos = video_positions[1][0]

            # Insert visual features at video token position
            # visual_features: [seq_len, 4096]
            # text_embeds: [1, text_len, 4096]

            before = text_embeds[:, :pos, :]
            after = text_embeds[:, pos+1:, :]
            visual = visual_features.unsqueeze(0)  # [1, vis_len, 4096]

            inputs_embeds = torch.cat([before, visual, after], dim=1)

            # Update attention mask
            vis_len = visual_features.shape[0]
            vis_mask = torch.ones(1, vis_len, device=self.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([
                attention_mask[:, :pos],
                vis_mask,
                attention_mask[:, pos+1:]
            ], dim=1)
        else:
            # No video token found, prepend visual features
            visual = visual_features.unsqueeze(0)
            inputs_embeds = torch.cat([visual, text_embeds], dim=1)
            vis_mask = torch.ones(1, visual_features.shape[0], device=self.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # Generate
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
            outputs = self.videollava.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.videollava_processor.tokenizer.pad_token_id,
                eos_token_id=self.videollava_processor.tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        gen_time = time.time() - start_time

        # Decode
        output_text = self.videollava_processor.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        token_ids = outputs[0].tolist()

        return output_text, token_ids, gen_time


def extract_all_features(
    pipeline: SharedDecoderPipeline,
    samples: list,
    dataset_dir: str,
    model_type: str,  # 'eventgpt' or 'videollava'
) -> list:
    """Extract features from all samples for one model type."""
    features_list = []

    for sample in tqdm(samples, desc=f"Extracting {model_type}"):
        try:
            if model_type == 'eventgpt':
                event_images = sample.get('event_image', [])
                if not event_images:
                    continue
                paths = [os.path.join(dataset_dir, 'event_image', p) for p in event_images]
                features = pipeline.extract_eventgpt_features(paths)
            else:
                video_data = sample.get('video_data', '')
                video_dir = os.path.join(dataset_dir, 'video', video_data)
                paths = sorted(Path(video_dir).glob('*.png'))[:8]
                if not paths:
                    continue
                features = pipeline.extract_videollava_features([str(p) for p in paths])

            features_list.append({
                'id': sample.get('id'),
                'features': features.cpu(),
            })
        except Exception as e:
            print(f"\nError on {sample.get('id')}: {e}")
            continue

    return features_list


def main():
    parser = argparse.ArgumentParser(description='Shared-Decoder Speculative Decoding')
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/my_egpt_dsec_train/my_egpt_dsec_train_1s')
    parser.add_argument('--eventgpt_path', type=str,
                        default='./checkpoints/EventGPT-7b')
    parser.add_argument('--videollava_path', type=str,
                        default='LanguageBind/Video-LLaVA-7B-hf')
    parser.add_argument('--output_dir', type=str,
                        default='./feasible/benchmark_inference')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Max samples for feature extraction')
    parser.add_argument('--test_samples', type=int, default=20,
                        help='Number of test samples for speculative decoding')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Adapter training epochs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--query', type=str,
                        default='What are the key elements in this scene?')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SHARED-DECODER SPECULATIVE DECODING")
    print("="*70)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Test samples: {args.test_samples}")
    print("="*70)

    # Load dataset
    json_path = os.path.join(args.dataset_dir, "EventGPT_Instruction_Subset.json")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    # Split into train/test (80/20)
    torch.manual_seed(42)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_samples = [dataset[i] for i in train_indices]
    test_samples = [dataset[i] for i in test_indices[:args.test_samples]]

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Initialize pipeline
    pipeline = SharedDecoderPipeline(
        eventgpt_path=args.eventgpt_path,
        videollava_path=args.videollava_path,
        device=args.device,
    )

    # =========================================================================
    # PHASE 1: Extract EventGPT features (training set)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Extract EventGPT Features (Training Set)")
    print("="*60)

    pipeline.load_eventgpt()
    eventgpt_train_features = extract_all_features(
        pipeline, train_samples, args.dataset_dir, 'eventgpt'
    )
    pipeline.unload_eventgpt()

    # =========================================================================
    # PHASE 2: Extract Video-LLaVA features (training set)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Extract Video-LLaVA Features (Training Set)")
    print("="*60)

    pipeline.load_videollava()
    videollava_train_features = extract_all_features(
        pipeline, train_samples, args.dataset_dir, 'videollava'
    )

    # Match features by ID
    egpt_dict = {f['id']: f['features'] for f in eventgpt_train_features}
    vllava_dict = {f['id']: f['features'] for f in videollava_train_features}

    common_ids = set(egpt_dict.keys()) & set(vllava_dict.keys())
    print(f"\nMatched training samples: {len(common_ids)}")

    # Stack features
    egpt_features = torch.stack([egpt_dict[id] for id in common_ids])
    vllava_features = torch.stack([vllava_dict[id] for id in common_ids])

    print(f"EventGPT features shape: {egpt_features.shape}")
    print(f"Video-LLaVA features shape: {vllava_features.shape}")

    # =========================================================================
    # PHASE 3: Train Adapter
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: Train Feature Alignment Adapter")
    print("="*60)

    # Unload Video-LLaVA during adapter training to save memory
    pipeline.unload_videollava()

    pipeline.train_adapter(
        egpt_features,
        vllava_features,
        num_epochs=args.num_epochs,
    )

    # Save adapter
    adapter_path = os.path.join(args.output_dir, 'shared_decoder_adapter_S1.pt')
    torch.save({
        'config': pipeline.adapter.config,
        'state_dict': pipeline.adapter.state_dict(),
    }, adapter_path)
    print(f"Adapter saved to {adapter_path}")

    # =========================================================================
    # PHASE 4: Benchmark Speculative Decoding on Test Set
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 4: Benchmark Speculative Decoding (Test Set)")
    print("="*60)

    # Extract test features
    pipeline.load_eventgpt()
    eventgpt_test_features = extract_all_features(
        pipeline, test_samples, args.dataset_dir, 'eventgpt'
    )
    pipeline.unload_eventgpt()

    pipeline.load_videollava()
    videollava_test_features = extract_all_features(
        pipeline, test_samples, args.dataset_dir, 'videollava'
    )

    # Match test features
    egpt_test_dict = {f['id']: f['features'] for f in eventgpt_test_features}
    vllava_test_dict = {f['id']: f['features'] for f in videollava_test_features}
    test_ids = list(set(egpt_test_dict.keys()) & set(vllava_test_dict.keys()))

    print(f"\nMatched test samples: {len(test_ids)}")

    # Run speculative decoding comparison
    results = []
    total_matched = 0
    total_tokens = 0

    for sample_id in tqdm(test_ids, desc="Speculative Decoding"):
        try:
            egpt_feat = egpt_test_dict[sample_id]
            vllava_feat = vllava_test_dict[sample_id]

            # Draft: EventGPT features + adapter → Video-LLaVA decoder
            draft_output, draft_tokens, draft_time = pipeline.generate_with_shared_decoder(
                egpt_feat,
                args.query,
                max_new_tokens=50,
                use_adapter=True,
            )

            # Target: Video-LLaVA features → Video-LLaVA decoder
            target_output, target_tokens, target_time = pipeline.generate_with_shared_decoder(
                vllava_feat,
                args.query,
                max_new_tokens=50,
                use_adapter=False,
            )

            # Compute token-level acceptance
            min_len = min(len(draft_tokens), len(target_tokens))
            if min_len > 0:
                matches = sum(1 for i in range(min_len) if draft_tokens[i] == target_tokens[i])
                acceptance = matches / min_len
            else:
                matches = 0
                acceptance = 0.0

            total_matched += matches
            total_tokens += min_len

            results.append({
                'id': sample_id,
                'draft_output': draft_output,
                'target_output': target_output,
                'draft_tokens': draft_tokens,
                'target_tokens': target_tokens,
                'draft_time': draft_time,
                'target_time': target_time,
                'matched_tokens': matches,
                'total_tokens': min_len,
                'acceptance_rate': acceptance,
            })

        except Exception as e:
            print(f"\nError on {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute summary
    overall_acceptance = total_matched / total_tokens if total_tokens > 0 else 0
    avg_draft_time = np.mean([r['draft_time'] for r in results]) if results else 0
    avg_target_time = np.mean([r['target_time'] for r in results]) if results else 0

    summary = {
        'config': {
            'train_samples': len(train_samples),
            'test_samples': len(test_ids),
            'num_epochs': args.num_epochs,
            'query': args.query,
        },
        'results': {
            'overall_token_acceptance_rate': overall_acceptance,
            'total_matched_tokens': total_matched,
            'total_tokens_compared': total_tokens,
            'avg_draft_time': avg_draft_time,
            'avg_target_time': avg_target_time,
            'c_ratio': avg_draft_time / avg_target_time if avg_target_time > 0 else 0,
        },
        'samples': results,
    }

    # Save results
    output_path = os.path.join(args.output_dir, 'shared_decoder_results_S1.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("SHARED-DECODER SPECULATIVE DECODING RESULTS")
    print("="*70)

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_ids)}")

    print(f"\n{'─'*70}")
    print("TOKEN-LEVEL ACCEPTANCE RATE (Shared Decoder)")
    print(f"{'─'*70}")
    print(f"  Overall: {overall_acceptance:.2%}")
    print(f"  Matched: {total_matched}/{total_tokens} tokens")

    print(f"\n{'─'*70}")
    print("TIMING")
    print(f"{'─'*70}")
    print(f"  Avg draft time:  {avg_draft_time:.3f}s")
    print(f"  Avg target time: {avg_target_time:.3f}s")
    print(f"  c ratio:         {summary['results']['c_ratio']:.3f}")

    if overall_acceptance > 0.5:
        print(f"\n  ✓ High acceptance rate with shared decoder!")
        print(f"    → Speculative decoding is beneficial")
    elif overall_acceptance > 0.1:
        print(f"\n  ~ Moderate acceptance - may benefit from more training")
    else:
        print(f"\n  ⚠ Low acceptance - features may not be well aligned")

    print("="*70)

    pipeline.unload_videollava()


if __name__ == '__main__':
    main()
