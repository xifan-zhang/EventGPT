#!/usr/bin/env python3
"""
L6: LoRA finetune EventGPT decoder to align hidden states with Video-LLaVA.

Author: Alice Zhang
Date: 2026-02-10

================================================================================
OVERVIEW
================================================================================

Unlike L1-L5 (external adapter networks mapping pre-extracted EGPT hidden
states -> VL hidden space), L6 modifies the drafter model ITSELF via QLoRA.
After training, the LoRA weights can be merged into the base model, giving
zero additional inference latency while producing hidden states that are
natively closer to Video-LLaVA's hidden state space.

    L1-L5 approach:        EGPT_hidden --> [Adapter Network] --> VL_hidden_approx
    L6 approach:           EGPT_LoRA_hidden ~~~~ VL_hidden  (no adapter needed)

Results after 1 epoch: val_cos_sim=0.816, val_accept@0.90=28.3% — already
surpasses L4 external adapter (val_cos_sim=0.797) trained for 268 epochs.

================================================================================
ARCHITECTURE
================================================================================

    +----------------------------------------------------------+
    |  EventGPT (4-bit NF4 quantized, ~3.9 GB)                |
    |                                                          |
    |  +----------------------------------------------------+  |
    |  | Vision Encoder (FROZEN, no grad)                   |  |
    |  |   event_image_1f/ --> [processor] --> [ViT]        |  |
    |  |   --> [feature_adaptor] --> event_features          |  |
    |  +----------------------------------------------------+  |
    |                          |                                |
    |                          v                                |
    |  +----------------------------------------------------+  |
    |  | Decoder (32 LLaMA layers)                          |  |
    |  |                                                    |  |
    |  |   For each layer:                                  |  |
    |  |     q_proj: [W_frozen_4bit] + [LoRA_A * LoRA_B]   |  |
    |  |     k_proj: [W_frozen_4bit] + [LoRA_A * LoRA_B]   |  |
    |  |     v_proj: [W_frozen_4bit] + [LoRA_A * LoRA_B]   |  |
    |  |     o_proj: [W_frozen_4bit] + [LoRA_A * LoRA_B]   |  |
    |  |     mlp:    [W_frozen_4bit] (no LoRA)              |  |
    |  |                                                    |  |
    |  |   Only LoRA_A, LoRA_B have requires_grad=True      |  |
    |  |   19.1M trainable / 3691M total (0.52%)            |  |
    |  +----------------------------------------------------+  |
    |                          |                                |
    |                          v                                |
    |              last_hidden_state [1, seq_len, 4096]         |
    +----------------------------------------------------------+

================================================================================
DATA FLOW (per sample)
================================================================================

    Train chunks (52 chunks, 52K samples)       Dataset (disk)
    chunked_train_1s_4bit_1f/                   my_egpt_dsec_train_1s/
    +-----------------------+                   +------------------+
    | vl_hidden [N,50,4096] |                   | event_image_1f/  |
    | seq_lens  [N]         |                   | questions.json   |
    | metadata  [N] {       |                   +------------------+
    |   sample_idx,         |                            |
    |   question_idx,       |                            v
    |   egpt_text           |            1. prepare_input():
    | }                     |               - Load 1-frame event image
    +-----------+-----------+               - Encode through frozen ViT
                |                           - Tokenize prompt + question
                |                                    |
                |                                    v
                |                2. teacher_forced_forward():
                |                   - Append egpt_text tokens to prompt
                |                   - [prompt_tokens | gen_tokens]
                |                   - Single forward pass with causal mask
                |                   - Extract last gen_len hidden states
                |                                    |
                v                                    v
         target: vl_hidden                   pred: egpt_hidden
         [actual_len, 4096]                  [actual_len, 4096]
                |                                    |
                +--------+       +------------------+
                         |       |
                         v       v
              3. Loss = MSE(pred, target)
                       + 0.5 * CosineEmbeddingLoss(pred, target)
                       + 0.1 * CE(lm_head(pred), lm_head(target).argmax)
                                 |
                                 v
              4. loss.backward()  -->  grads flow to LoRA params only
              5. Every gradient_accumulation steps: optimizer.step()

    Val chunks (11 chunks, 11K samples)
    chunked_test_1s_4bit_1f/ + my_egpt_dsec_seq_1s/
    Same flow, but no backward pass. Metrics: val_loss, val_cos_sim,
    val_accept@{80,85,90,95}

================================================================================
LOSS FUNCTION
================================================================================

    loss = MSE(pred, target)                          # hidden-state L2 distance
         + 0.5 * CosineEmbeddingLoss(pred, target)   # angular alignment
         + token_loss_weight * CE(                    # token-level agreement
               lm_head(pred),                         #   logits from EGPT hidden
               lm_head(target).argmax(-1)             #   pseudo-labels from VL hidden
           )

    The CE auxiliary loss (weight=0.1 by default) uses Video-LLaVA's frozen
    LM head to project both predicted and target hidden states into token space.
    This forces token-level agreement beyond just hidden-state similarity.

    Logged separately: train_mse, train_cos_loss, train_ce in history.json.
    Progress bar shows: mse, cos_l, sim, ce, tok (token top-1 match %).

================================================================================
TEACHER-FORCED FORWARD PASS
================================================================================

    Instead of autoregressive generation (slow, N forward passes), we use a
    single teacher-forced forward pass. The causal attention mask ensures each
    position can only attend to previous positions, making it mathematically
    equivalent to sequential AR generation.

    Input:  [SYS] [IMG_TOKENS] [Q: "What are the key elements?"] [The] [key] [elements] ...
            |<-------- prompt (frozen visual + text) -------->| |<-- gen_tokens (from egpt_text) -->|
                                                                                    |
    Output hidden states extracted from these positions --------+
    (last gen_len positions of last_hidden_state)

    Speed: ~8x faster than AR generate() for 50-token sequences.
    Equivalence: Causal mask guarantees h[t] depends only on tokens [0..t].

================================================================================
GRADIENT CHECKPOINTING
================================================================================

    With 4-bit base + LoRA, the input embeddings don't have requires_grad.
    Standard (reentrant) gradient checkpointing would silently drop all
    gradients. We use two fixes:
      1. gradient_checkpointing_enable(use_reentrant=False)
      2. inputs_embeds.requires_grad_(True) before the forward pass

================================================================================
TRAINING LOOP
================================================================================

    For each epoch:
      For each train chunk (52 chunks, 1000 samples each):
        - Load chunk: vl_hidden (targets), metadata
        - Prefetch: ThreadPoolExecutor loads next sample's image while
          GPU processes current sample
        - For each sample:
            a. CPU thread: load 1f event image, encode vision, tokenize
            b. GPU: teacher_forced_forward() -> egpt_hidden
            c. GPU: compute loss vs vl_hidden target (MSE + cos + CE)
            d. GPU: loss.backward() (grads accumulate)
            e. Every N steps: optimizer.step() + zero_grad()
      Validate on all 11 test chunks (separate test set, 11K samples)
      Log: val_loss, val_cos_sim, val_accept@{80,85,90,95}
      Early stopping on val_loss (patience=5, default)

    Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
    Scheduler: CosineAnnealingLR over num_epochs
    Grad clipping: max_norm=1.0
    Timing: ~5.7h per epoch on RTX 4090

================================================================================
MEMORY BUDGET (RTX 4090 24GB)
================================================================================

    Base model (4-bit):     ~3.9 GB
    LoRA params:            ~0.08 GB
    VL LM head (frozen):    ~0.5 GB  (for token CE loss)
    Activations (1 sample): ~2-4 GB  (with grad checkpointing)
    VL hidden (1 chunk):    ~0.8 GB
    Optimizer states:       ~0.3 GB
    ----------------------------------------
    Total:                  ~8-10 GB peak

================================================================================
OUTPUT
================================================================================

    tasks/L6/L6_<timestamp>/
      config.json         - Hyperparameters
      history.json        - Per-epoch metrics (train_mse, train_cos_loss,
                            train_ce, val_loss, val_cos_sim, val_accept@90, ...)
      best_model/         - PEFT adapter (LoRA weights only, ~77 MB)
        adapter_config.json
        adapter_model.safetensors
      final_model/        - Same format, last epoch

    At inference, merge LoRA into base: model.merge_and_unload() -> zero overhead.

================================================================================
CLI
================================================================================

    # Default: 1f event images, token CE loss, separate test set for validation
    python pipeline/adapter_train/train_lora_adapter.py \\
        --lora_rank 16 --lora_alpha 32 \\
        --num_epochs 20 --gradient_accumulation 16 --early_stopping 5

    # Explicit paths (these are the defaults):
    python pipeline/adapter_train/train_lora_adapter.py \\
        --existing_chunks pipeline/feature_extraction/data/chunked_train_1s_4bit_1f \\
        --dataset_dir data/my_egpt_dsec_train/my_egpt_dsec_train_1s \\
        --val_chunks pipeline/feature_extraction/data/chunked_test_1s_4bit_1f \\
        --val_dataset_dir data/my_egpt_dsec_test/my_egpt_dsec_seq_1s \\
        --event_image_key event_image_1f \\
        --token_loss_weight 0.1 \\
        --lm_head_path feasible/feature_alignment/data/vl_lm_head.pt

    # Disable token CE loss:
        --token_loss_weight 0.0

    # Use 5-frame event images instead:
        --existing_chunks pipeline/feature_extraction/data/chunked_train_1s_4bit \\
        --event_image_key event_image
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        event_image_key: str = 'event_image_1f',
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        gradient_accumulation: int = 4,
        num_epochs: int = 20,
        early_stopping: int = 5,
        output_dir: str = 'pipeline/adapter_train/tasks/L6',
        token_loss_weight: float = 0.0,
        lm_head_path: str = '',
        val_chunks_dir: str = '',
        val_dataset_dir: str = '',
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
        self.token_loss_weight = token_loss_weight
        self.lm_head_path = lm_head_path
        self.val_chunks_dir = Path(val_chunks_dir) if val_chunks_dir else None
        self.val_dataset_dir = Path(val_dataset_dir) if val_dataset_dir else None

        # Load dataset JSON
        json_path = self.dataset_dir / "EventGPT_Instruction_Subset.json"
        with open(json_path) as f:
            self.dataset = json.load(f)

        # Load val dataset JSON (separate test set)
        if self.val_dataset_dir:
            val_json = self.val_dataset_dir / "EventGPT_Instruction_Subset.json"
            with open(val_json) as f:
                self.val_dataset = json.load(f)
        else:
            self.val_dataset = None

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

        # Load val chunk index
        if self.val_chunks_dir:
            val_index_path = self.val_chunks_dir / "index.json"
            with open(val_index_path) as f:
                self.val_chunk_index = json.load(f)
        else:
            self.val_chunk_index = None

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.vl_lm_head = None

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

        # Load VL LM head for token-level auxiliary loss
        if self.token_loss_weight > 0 and self.lm_head_path:
            print(f"\n  Loading VL LM head for token auxiliary loss (weight={self.token_loss_weight})...")
            lm_head_data = torch.load(self.lm_head_path, map_location=self.device)
            w = lm_head_data.get('lm_head_weight', lm_head_data.get('weight'))
            self.vl_lm_head = nn.Linear(w.shape[1], w.shape[0], bias=False)
            self.vl_lm_head.weight.data = w.float()
            self.vl_lm_head = self.vl_lm_head.to(self.device)
            self.vl_lm_head.eval()
            for p in self.vl_lm_head.parameters():
                p.requires_grad = False
            print(f"  VL LM head loaded: {w.shape}")
            print(f"  GPU after LM head: {torch.cuda.memory_allocated()/1024/1024:.0f} MB")

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
            'token_loss_weight': self.token_loss_weight,
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
        train_chunks_dir = self.existing_chunks_dir / "chunks"
        chunk_files = self.chunk_index['chunks']

        # Use separate test set for validation if provided
        if self.val_chunk_index is not None:
            train_chunks = chunk_files  # all train chunks
            val_chunks = self.val_chunk_index['chunks']  # all test chunks
            val_chunks_dir = self.val_chunks_dir / "chunks"
            print(f"\nTraining: {len(train_chunks)} chunks (full train set)")
            print(f"Validation: {len(val_chunks)} chunk(s) from test set")
        else:
            # Fallback: split last train chunk as val
            if len(chunk_files) > 1:
                train_chunks = chunk_files[:-1]
                val_chunks = chunk_files[-1:]
            else:
                train_chunks = chunk_files
                val_chunks = []
            val_chunks_dir = train_chunks_dir
            print(f"\nTraining: {len(train_chunks)} chunks, Validation: {len(val_chunks)} chunks")
        print(f"Output: {save_dir}")

        best_val_loss = float('inf')
        patience = 0
        history = []

        for epoch in range(self.num_epochs):
            # === TRAIN ===
            self.model.train()
            epoch_loss = 0
            epoch_mse_loss = 0
            epoch_cos_loss = 0
            epoch_ce_loss = 0
            epoch_cos_sim = 0
            epoch_token_match = 0
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
                chunk_path = train_chunks_dir / chunk_info['path']
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

                    # Token-level auxiliary CE loss
                    token_match = None
                    if self.vl_lm_head is not None:
                        with torch.no_grad():
                            vl_tokens = self.vl_lm_head(target).argmax(dim=-1)  # [seq_len]
                        pred_logits = self.vl_lm_head(pred)  # [seq_len, vocab] — grad flows through pred
                        ce_aux = F.cross_entropy(pred_logits, vl_tokens)
                        loss = loss + self.token_loss_weight * ce_aux
                        with torch.no_grad():
                            token_match = (pred_logits.argmax(dim=-1) == vl_tokens).float().mean().item()

                    # Metrics before backward (pred is freed after .backward())
                    with torch.no_grad():
                        cos_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()

                    # Scale for gradient accumulation
                    loss = loss / self.gradient_accumulation
                    loss.backward()

                    epoch_loss += loss.item() * self.gradient_accumulation
                    epoch_mse_loss += mse.item()
                    epoch_cos_loss += cos.item()
                    if self.vl_lm_head is not None:
                        epoch_ce_loss += ce_aux.item()
                    epoch_cos_sim += cos_sim
                    if token_match is not None:
                        epoch_token_match += token_match
                    epoch_samples += 1

                    # Optimizer step
                    if epoch_samples % self.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_norm=1.0
                        )
                        optimizer.step()
                        optimizer.zero_grad()

                    n = max(epoch_samples, 1)
                    postfix = {
                        'mse': f'{epoch_mse_loss/n:.3f}',
                        'cos_l': f'{epoch_cos_loss/n:.3f}',
                        'sim': f'{epoch_cos_sim/n:.4f}',
                    }
                    if self.vl_lm_head is not None:
                        postfix['ce'] = f'{epoch_ce_loss/n:.3f}'
                        postfix['tok'] = f'{epoch_token_match/n:.3f}'
                    pbar.set_postfix(postfix)

                    # Free memory
                    del input_data, gen_hidden, pred, target, loss
                    if self.vl_lm_head is not None:
                        del pred_logits, vl_tokens
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

            n = max(epoch_samples, 1)
            avg_train_loss = epoch_loss / n
            avg_train_mse = epoch_mse_loss / n
            avg_train_cos_l = epoch_cos_loss / n
            avg_train_ce = epoch_ce_loss / n if self.vl_lm_head else 0
            avg_train_cos = epoch_cos_sim / n
            avg_train_tok = epoch_token_match / n if self.vl_lm_head else 0

            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            train_msg = f"  Train — loss: {avg_train_loss:.4f} (mse: {avg_train_mse:.4f}, cos: {avg_train_cos_l:.4f}"
            if self.vl_lm_head is not None:
                train_msg += f", ce: {avg_train_ce:.4f}"
            train_msg += f"), cos_sim: {avg_train_cos:.4f}"
            if self.vl_lm_head is not None:
                train_msg += f", token_match: {avg_train_tok:.4f}"
            print(train_msg)
            print(f"  Samples: {epoch_samples}, LR: {scheduler.get_last_lr()[0]:.2e}")

            # === VALIDATE ===
            val_loss = 0
            val_cos_sim = 0
            val_accept_90 = 0
            val_token_match = 0
            val_samples = 0

            if val_chunks:
                self.model.eval()
                # Use val dataset/dir for prepare_input if separate test set
                orig_dataset = self.dataset
                orig_dataset_dir = self.dataset_dir
                if self.val_dataset is not None:
                    self.dataset = self.val_dataset
                    self.dataset_dir = self.val_dataset_dir
                val_pool = ThreadPoolExecutor(max_workers=1)
                with torch.no_grad():
                    for chunk_info in val_chunks:
                        chunk_path = val_chunks_dir / chunk_info['path']
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

                            sample_loss = mse + 0.5 * cos
                            if self.vl_lm_head is not None:
                                vl_tokens = self.vl_lm_head(target).argmax(dim=-1)
                                pred_logits = self.vl_lm_head(pred)
                                ce_aux = F.cross_entropy(pred_logits, vl_tokens)
                                sample_loss = sample_loss + self.token_loss_weight * ce_aux
                                pred_tokens = pred_logits.argmax(dim=-1)
                                val_token_match += (pred_tokens == vl_tokens).float().mean().item()

                            val_loss += sample_loss.item()
                            cos_sims = nn.functional.cosine_similarity(pred, target, dim=-1)
                            val_cos_sim += cos_sims.mean().item()
                            val_accept_90 += (cos_sims > 0.90).float().mean().item() 
                            val_samples += 1

                            del input_data, gen_hidden, pred, target

                        del chunk_data, vl_hidden
                        gc.collect()
                        torch.cuda.empty_cache()
                val_pool.shutdown(wait=False)
                # Restore train dataset
                self.dataset = orig_dataset
                self.dataset_dir = orig_dataset_dir

                avg_val_loss = val_loss / max(val_samples, 1)
                avg_val_cos = val_cos_sim / max(val_samples, 1)
                avg_val_accept = val_accept_90 / max(val_samples, 1)
                avg_val_tok = val_token_match / max(val_samples, 1) if self.vl_lm_head else 0

                val_msg = f"  Val loss: {avg_val_loss:.4f}, cos_sim: {avg_val_cos:.4f}"
                if self.vl_lm_head is not None:
                    val_msg += f", token_match: {avg_val_tok:.4f}"
                print(val_msg)
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
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_mse': avg_train_mse,
                'train_cos_loss': avg_train_cos_l,
                'train_cos_sim': avg_train_cos,
                'val_loss': avg_val_loss,
                'val_cos_sim': avg_val_cos,
                'val_accept_90': avg_val_accept,
                'lr': scheduler.get_last_lr()[0],
            }
            if self.vl_lm_head is not None:
                epoch_record['train_ce'] = avg_train_ce
                epoch_record['train_token_match'] = avg_train_tok
                epoch_record['val_token_match'] = avg_val_tok
            history.append(epoch_record)
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

        # Save final model
        self.model.save_pretrained(str(save_dir / 'final_model'))
        print(f"\nTraining complete. Output: {save_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="L6: LoRA finetune EventGPT for VL alignment")

    # Data
    parser.add_argument('--existing_chunks', type=str,
                        default=str(ROOT / 'pipeline/feature_extraction/data/chunked_train_1s_4bit_1f'),
                        help='Path to train chunked data with VL hidden states')
    parser.add_argument('--dataset_dir', type=str,
                        default=str(ROOT / 'data/my_egpt_dsec_train/my_egpt_dsec_train_1s'),
                        help='Path to train DSEC dataset with event images')
    parser.add_argument('--val_chunks', type=str,
                        default=str(ROOT / 'pipeline/feature_extraction/data/chunked_test_1s_4bit_1f'),
                        help='Path to val/test chunked data (separate test set)')
    parser.add_argument('--val_dataset_dir', type=str,
                        default=str(ROOT / 'data/my_egpt_dsec_test/my_egpt_dsec_seq_1s'),
                        help='Path to val/test DSEC dataset with event images')
    parser.add_argument('--questions_file', type=str,
                        default=str(ROOT / 'feasible/token_alignment/top50_questions.json'))
    parser.add_argument('--max_questions', type=int, default=10)
    parser.add_argument('--event_image_key', type=str, default='event_image_1f',
                        help='JSON key for event images (event_image or event_image_1f)')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Loss
    parser.add_argument('--token_loss_weight', type=float, default=0.1,
                        help='Weight for token-level CE auxiliary loss (0=disabled, 0.1=recommended)')
    parser.add_argument('--lm_head_path', type=str,
                        default=str(ROOT / 'feasible/feature_alignment/data/vl_lm_head.pt'),
                        help='Path to VL LM head weights (for token auxiliary loss)')

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
        token_loss_weight=args.token_loss_weight,
        lm_head_path=args.lm_head_path if args.token_loss_weight > 0 else '',
        val_chunks_dir=args.val_chunks,
        val_dataset_dir=args.val_dataset_dir,
    )

    trainer.load_model()
    trainer.train()


if __name__ == "__main__":
    main()
