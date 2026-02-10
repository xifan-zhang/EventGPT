#!/usr/bin/env python3
"""
L6: LoRA finetune EventGPT decoder to align hidden states with Video-LLaVA.

Author: Alice Zhang
Date: 2026-02-07

Unlike L1-L5 (separate adapter on pre-extracted hidden states), L6 modifies
the drafter model itself via QLoRA so its decoder hidden states natively
align with Video-LLaVA's hidden state space.

Setup:
    - EventGPT loaded in 4-bit (frozen base weights)
    - LoRA applied to decoder q,k,v,o projections
    - Vision encoder fully frozen (no grad)
    - Gradient checkpointing enabled
    - Target: pre-extracted VL hidden states from existing chunks

Training:
    For each sample:
    1. Load event image from disk + question from metadata
    2. Encode event image through frozen vision encoder
    3. Teacher-forced forward pass through decoder with LoRA
       (using original EGPT tokens so positions align)
    4. Extract last-layer hidden states at generation positions
    5. Loss = MSE + cosine vs VL hidden states
    6. Backprop through LoRA params only

Memory: ~8-10 GB on 4090 24GB (4-bit model + LoRA + grad checkpointing)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from concurrent.futures import ThreadPoolExecutor, Future
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


class LoRATrainer:
    """Train EventGPT decoder with LoRA to align hidden states with VL."""

    def __init__(
        self,
        existing_chunks_dir: str,
        dataset_dir: str,
        questions_file: str,
        max_questions: int = 10,
        event_image_key: str = 'event_image',
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        gradient_accumulation: int = 4,
        num_epochs: int = 20,
        early_stopping: int = 5,
        output_dir: str = 'pipeline/adapter_train/tasks/L6',
    ):
        self.device = "cuda"
        self.existing_chunks_dir = Path(existing_chunks_dir)
        self.dataset_dir = Path(dataset_dir)
        self.event_image_key = event_image_key
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.output_dir = Path(output_dir)

        # Load dataset JSON
        json_path = self.dataset_dir / "EventGPT_Instruction_Subset.json"
        with open(json_path) as f:
            self.dataset = json.load(f)

        # Load questions
        questions_file = Path(questions_file)
        if questions_file.exists():
            with open(questions_file) as f:
                questions_data = json.load(f)
            self.questions = [q['question'] for q in questions_data][:max_questions]
        else:
            self.questions = ["What are the key elements in this scene?"]

        # Load existing chunk index
        index_path = self.existing_chunks_dir / "index.json"
        with open(index_path) as f:
            self.chunk_index = json.load(f)

        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self):
        """Load EventGPT in 4-bit with LoRA adapters."""
        from model.EventChatModel import EventChatModel

        print("\n[1/3] Loading EventGPT (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        eventgpt_path = "/home/ps/Documents/code/EventGPT/checkpoints/EventGPT-7b"
        self.model = EventChatModel.from_pretrained(
            eventgpt_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(eventgpt_path, use_fast=True)
        self.processor = self.model.get_visual_tower().event_processor
        print(f"  Base model loaded. GPU: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

        # [2/3] Freeze everything first
        print("\n[2/3] Freezing base model + vision encoder...")
        for param in self.model.parameters():
            param.requires_grad = False

        # [3/3] Add LoRA to decoder layers
        print(f"\n[3/3] Adding LoRA (rank={self.lora_rank}, alpha={self.lora_alpha})...")
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

        # Enable gradient checkpointing (use_reentrant=False required for
        # frozen base + LoRA: inputs_embeds don't have requires_grad, but
        # non-reentrant checkpointing still propagates grads to LoRA params)
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        trainable, total = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)")
        print(f"  GPU after LoRA: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

    def prepare_input(self, sample: Dict, query: str) -> Optional[Dict]:
        """Prepare model input from event image + question.

        Returns dict with:
            - input_ids: prompt token IDs [1, prompt_len]
            - event_features: encoded event features [1, n_patches, hidden]
            - event_image_size: [H, W]
        """
        from model.EventChatModel import get_spatio_temporal_features
        from dataset.conversation import prepare_event_prompt
        from dataset.constants import EVENT_TOKEN_INDEX
        from common.common import tokenizer_event_token, load_image

        try:
            if self.event_image_key not in sample or not sample[self.event_image_key]:
                return None

            event_image_paths = sample[self.event_image_key]
            img_path = self.dataset_dir / self.event_image_key / event_image_paths[0]
            if not img_path.exists():
                return None

            img = load_image(str(img_path))
            img_array = np.array(img)
            event_image_size = list(img_array.shape[:2])

            event = self.processor(img_array, return_tensors='pt')['pixel_values'][0]
            event = event.to(self.device, dtype=torch.bfloat16)

            # Vision encoding (frozen, no grad)
            with torch.no_grad():
                feature = self.model.base_model.model.visval_encode(event.unsqueeze(0))
                feature = self.model.base_model.model.get_model().feature_adaptor(feature)
                feature = feature.squeeze(0)
                event_features = get_spatio_temporal_features([feature])
                event_features = event_features.unsqueeze(0)

            # Prepare prompt
            conv_mode = 'eventgpt_v1'
            prompt = prepare_event_prompt(query, conv_mode)
            input_ids = tokenizer_event_token(
                prompt, self.tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.device)

            return {
                'input_ids': input_ids,
                'event_features': event_features.detach(),
                'event_image_size': event_image_size,
            }

        except Exception as e:
            print(f"  Input prep error: {e}")
            return None

    def teacher_forced_forward(
        self,
        input_data: Dict,
        gen_token_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Teacher-forced forward pass to get decoder hidden states.

        Args:
            input_data: From prepare_input()
            gen_token_ids: Token IDs for teacher forcing [seq_len]

        Returns:
            hidden_states at generation positions [seq_len, 4096]
        """
        try:
            input_ids = input_data['input_ids']  # [1, prompt_len]
            event_features = input_data['event_features']
            event_image_size = input_data['event_image_size']

            # Append generation tokens to prompt
            gen_ids = gen_token_ids.unsqueeze(0).to(self.device)  # [1, gen_len]
            full_input_ids = torch.cat([input_ids, gen_ids], dim=1)  # [1, prompt_len + gen_len]
            prompt_len = input_ids.shape[1]

            # Prepare multimodal inputs (replaces event token with visual features)
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.model.base_model.model.prepare_inputs_labels_for_multimodal(
                full_input_ids,
                None,  # position_ids
                None,  # attention_mask
                None,  # past_key_values
                None,  # labels
                event_features=event_features,
                event_image_sizes=[event_image_size],
            )

            # Enable grad on inputs so gradient checkpointing can propagate
            inputs_embeds = inputs_embeds.requires_grad_(True)

            # Forward pass through decoder with LoRA (with gradients)
            outputs = self.model.base_model.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )

            # Get last layer hidden states
            last_hidden = outputs.last_hidden_state  # [1, total_len, 4096]

            # The visual features change the effective length, so we take
            # the last gen_len positions as generation hidden states
            gen_len = gen_ids.shape[1]
            gen_hidden = last_hidden[0, -gen_len:, :]  # [gen_len, 4096]

            return gen_hidden

        except Exception as e:
            print(f"  Forward error: {e}")
            return None

    def train(self):
        """Main training loop."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.output_dir / f"L6_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            'task': 'L6_LoRA_finetune',
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'gradient_accumulation': self.gradient_accumulation,
            'num_epochs': self.num_epochs,
            'early_stopping': self.early_stopping,
            'event_image_key': self.event_image_key,
            'existing_chunks': str(self.existing_chunks_dir),
            'dataset_dir': str(self.dataset_dir),
            'timestamp': timestamp,
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Optimizer (only LoRA params)
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=1e-6
        )

        # Loss functions
        mse_loss_fn = nn.MSELoss()
        cos_loss_fn = nn.CosineEmbeddingLoss()

        # Load chunk list
        chunks_dir = self.existing_chunks_dir / "chunks"
        chunk_files = self.chunk_index['chunks']

        # Split chunks into train/val (last chunk = val)
        if len(chunk_files) > 1:
            train_chunks = chunk_files[:-1]
            val_chunks = chunk_files[-1:]
        else:
            train_chunks = chunk_files
            val_chunks = []

        print(f"\nTraining: {len(train_chunks)} chunks, Validation: {len(val_chunks)} chunks")
        print(f"Output: {save_dir}")

        best_val_loss = float('inf')
        patience = 0
        history = []

        for epoch in range(self.num_epochs):
            # === TRAIN ===
            self.model.train()
            epoch_loss = 0
            epoch_cos_sim = 0
            epoch_samples = 0
            optimizer.zero_grad()

            # Prefetch helper: prepare input for sample i in background
            def _prepare_sample(meta_i, chunk_meta, chunk_seq_lens):
                """CPU-bound: load image, encode vision, tokenize. No GPU grad."""
                meta = chunk_meta[meta_i]
                sample_idx = meta['sample_idx']
                question_idx = meta['question_idx']
                target_len = chunk_seq_lens[meta_i].item()

                if sample_idx >= len(self.dataset) or question_idx >= len(self.questions):
                    return None

                egpt_text = meta.get('egpt_text', '')
                if not egpt_text:
                    return None

                gen_tokens = self.tokenizer.encode(egpt_text, add_special_tokens=False)
                gen_tokens = gen_tokens[:target_len]
                if len(gen_tokens) < 5:
                    return None

                sample = self.dataset[sample_idx]
                query = self.questions[question_idx]
                input_data = self.prepare_input(sample, query)
                if input_data is None:
                    return None

                return {
                    'input_data': input_data,
                    'gen_token_ids': torch.tensor(gen_tokens, dtype=torch.long),
                    'target_len': target_len,
                    'meta_i': meta_i,
                }

            prefetch_pool = ThreadPoolExecutor(max_workers=1)

            for chunk_idx, chunk_info in enumerate(train_chunks):
                chunk_path = chunks_dir / chunk_info['path']
                chunk_data = torch.load(chunk_path, map_location='cpu')
                vl_hidden = chunk_data['vl_hidden']
                seq_lens = chunk_data['seq_lens']
                metadata = chunk_data['metadata']
                n_samples = len(metadata)

                pbar = tqdm(range(n_samples), desc=f"Epoch {epoch+1} chunk {chunk_idx+1}",
                            leave=False)

                # Submit first prefetch
                pending_future: Optional[Future] = None
                if n_samples > 0:
                    pending_future = prefetch_pool.submit(
                        _prepare_sample, 0, metadata, seq_lens)

                for i in pbar:
                    # Wait for prefetched result
                    prepared = pending_future.result() if pending_future else None

                    # Submit next prefetch (overlaps with GPU work below)
                    if i + 1 < n_samples:
                        pending_future = prefetch_pool.submit(
                            _prepare_sample, i + 1, metadata, seq_lens)
                    else:
                        pending_future = None

                    if prepared is None:
                        continue

                    input_data = prepared['input_data']
                    gen_token_ids = prepared['gen_token_ids']
                    target_len = prepared['target_len']
                    meta_i = prepared['meta_i']

                    # Teacher-forced forward pass
                    gen_hidden = self.teacher_forced_forward(input_data, gen_token_ids)
                    if gen_hidden is None:
                        del input_data
                        continue

                    # Align lengths
                    actual_len = min(len(gen_hidden), target_len, len(gen_token_ids))
                    if actual_len < 5:
                        del input_data, gen_hidden
                        continue

                    pred = gen_hidden[:actual_len].float()
                    target = vl_hidden[meta_i, :actual_len].to(self.device).float()

                    # Loss
                    mse = mse_loss_fn(pred, target)
                    cos_target = torch.ones(actual_len, device=self.device)
                    cos = cos_loss_fn(pred, target, cos_target)
                    loss = mse + 0.5 * cos

                    # Scale for gradient accumulation
                    loss = loss / self.gradient_accumulation
                    loss.backward()

                    epoch_loss += loss.item() * self.gradient_accumulation
                    with torch.no_grad():
                        cos_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
                    epoch_cos_sim += cos_sim
                    epoch_samples += 1

                    # Optimizer step
                    if epoch_samples % self.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_norm=1.0
                        )
                        optimizer.step()
                        optimizer.zero_grad()

                    pbar.set_postfix({
                        'loss': f'{epoch_loss/max(epoch_samples,1):.4f}',
                        'cos': f'{epoch_cos_sim/max(epoch_samples,1):.4f}',
                    })

                    # Free memory
                    del input_data, gen_hidden, pred, target, loss
                    if epoch_samples % 100 == 0:
                        torch.cuda.empty_cache()

                del chunk_data, vl_hidden
                gc.collect()

            prefetch_pool.shutdown(wait=False)

            # Final optimizer step for remaining samples
            if epoch_samples % self.gradient_accumulation != 0:
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            avg_train_loss = epoch_loss / max(epoch_samples, 1)
            avg_train_cos = epoch_cos_sim / max(epoch_samples, 1)

            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train loss: {avg_train_loss:.4f}, cos_sim: {avg_train_cos:.4f}")
            print(f"  Samples: {epoch_samples}, LR: {scheduler.get_last_lr()[0]:.2e}")

            # === VALIDATE ===
            val_loss = 0
            val_cos_sim = 0
            val_accept_90 = 0
            val_samples = 0

            if val_chunks:
                self.model.eval()
                val_pool = ThreadPoolExecutor(max_workers=1)
                with torch.no_grad():
                    for chunk_info in val_chunks:
                        chunk_path = chunks_dir / chunk_info['path']
                        chunk_data = torch.load(chunk_path, map_location='cpu')
                        vl_hidden = chunk_data['vl_hidden']
                        seq_lens = chunk_data['seq_lens']
                        metadata = chunk_data['metadata']
                        n_val = len(metadata)

                        # Submit first prefetch
                        val_future: Optional[Future] = None
                        if n_val > 0:
                            val_future = val_pool.submit(
                                _prepare_sample, 0, metadata, seq_lens)

                        for i in tqdm(range(n_val), desc="  Validating", leave=False):
                            prepared = val_future.result() if val_future else None
                            if i + 1 < n_val:
                                val_future = val_pool.submit(
                                    _prepare_sample, i + 1, metadata, seq_lens)
                            else:
                                val_future = None

                            if prepared is None:
                                continue

                            input_data = prepared['input_data']
                            gen_token_ids = prepared['gen_token_ids']
                            target_len = prepared['target_len']
                            meta_i = prepared['meta_i']

                            gen_hidden = self.teacher_forced_forward(input_data, gen_token_ids)
                            if gen_hidden is None:
                                del input_data
                                continue

                            actual_len = min(len(gen_hidden), target_len, len(gen_token_ids))
                            if actual_len < 5:
                                continue

                            pred = gen_hidden[:actual_len].float()
                            target = vl_hidden[meta_i, :actual_len].to(self.device).float()

                            mse = mse_loss_fn(pred, target)
                            cos_target = torch.ones(actual_len, device=self.device)
                            cos = cos_loss_fn(pred, target, cos_target)

                            val_loss += (mse + 0.5 * cos).item()
                            cos_sims = nn.functional.cosine_similarity(pred, target, dim=-1)
                            val_cos_sim += cos_sims.mean().item()
                            val_accept_90 += (cos_sims > 0.90).float().mean().item()
                            val_samples += 1

                            del input_data, gen_hidden, pred, target

                        del chunk_data, vl_hidden
                        gc.collect()
                        torch.cuda.empty_cache()
                val_pool.shutdown(wait=False)

                avg_val_loss = val_loss / max(val_samples, 1)
                avg_val_cos = val_cos_sim / max(val_samples, 1)
                avg_val_accept = val_accept_90 / max(val_samples, 1)

                print(f"  Val loss: {avg_val_loss:.4f}, cos_sim: {avg_val_cos:.4f}")
                print(f"  Accept@0.90: {avg_val_accept:.2%}")

                # Save best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience = 0
                    self.model.save_pretrained(str(save_dir / 'best_model'))
                    print(f"  ** Saved best model **")
                else:
                    patience += 1

                if patience >= self.early_stopping:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                # No val set — save every epoch
                avg_val_loss = avg_train_loss
                avg_val_cos = avg_train_cos
                avg_val_accept = 0

            # History
            history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_cos_sim': avg_train_cos,
                'val_loss': avg_val_loss,
                'val_cos_sim': avg_val_cos,
                'val_accept_90': avg_val_accept,
                'lr': scheduler.get_last_lr()[0],
            })
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

        # Save final model
        self.model.save_pretrained(str(save_dir / 'final_model'))
        print(f"\nTraining complete. Output: {save_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="L6: LoRA finetune EventGPT for VL alignment")

    # Data
    parser.add_argument('--existing_chunks', type=str, required=True,
                        help='Path to existing chunked data with VL hidden states')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to DSEC dataset with event images')
    parser.add_argument('--questions_file', type=str,
                        default=str(ROOT / 'feasible/token_alignment/top50_questions.json'))
    parser.add_argument('--max_questions', type=int, default=10)
    parser.add_argument('--event_image_key', type=str, default='event_image',
                        help='JSON key for event images (event_image or event_image_1f)')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Training
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Samples per forward pass (use gradient_accumulation for effective batch)')
    parser.add_argument('--gradient_accumulation', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--early_stopping', type=int, default=5)

    # Output
    parser.add_argument('--output_dir', type=str,
                        default=str(Path(__file__).parent / 'tasks' / 'L6'))

    args = parser.parse_args()

    trainer = LoRATrainer(
        existing_chunks_dir=args.existing_chunks,
        dataset_dir=args.dataset_dir,
        questions_file=args.questions_file,
        max_questions=args.max_questions,
        event_image_key=args.event_image_key,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_epochs=args.num_epochs,
        early_stopping=args.early_stopping,
        output_dir=args.output_dir,
    )

    trainer.load_model()
    trainer.train()


if __name__ == "__main__":
    main()
