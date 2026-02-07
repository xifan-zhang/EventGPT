"""
Feature Extraction for Token Alignment
======================================

Extracts hidden states, logits, and tokens from both EventGPT and Video-LLaVA
for training token alignment modules.

Usage:
    python feasible/token_alignment/extract_features.py \
        --dataset_dir ./data/my_egpt_dsec_train/my_egpt_dsec_train_1s \
        --output_dir ./feasible/token_alignment/cached_outputs_1s \
        --max_samples 1000 \
        --extract_hidden_states \
        --extract_logits
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import numpy as np

# Fix protobuf issue
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""
    dataset_dir: str
    output_dir: str
    max_samples: int = -1
    max_new_tokens: int = 50
    extract_hidden_states: bool = True
    extract_logits: bool = False  # Memory intensive
    device: str = 'cuda'
    batch_size: int = 1  # Process one at a time for memory
    use_4bit: bool = True


class FeatureExtractor:
    """Extracts features from EventGPT and Video-LLaVA."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Will be loaded lazily
        self.egpt_model = None
        self.egpt_tokenizer = None
        self.vl_model = None
        self.vl_processor = None

    def load_eventgpt(self):
        """Load EventGPT model."""
        if self.egpt_model is not None:
            return

        print("Loading EventGPT...")
        from model.EventChatModel import EventChatModel
        from transformers import AutoTokenizer

        self.egpt_tokenizer = AutoTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.5",
            use_fast=False,
            trust_remote_code=True,
        )

        # Load with 4-bit quantization
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.egpt_model = EventChatModel.from_pretrained(
                str(project_root / "checkpoints" / "EventGPT"),
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.egpt_model = EventChatModel.from_pretrained(
                str(project_root / "checkpoints" / "EventGPT"),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.egpt_model.eval()
        print("EventGPT loaded.")

    def load_videollava(self):
        """Load Video-LLaVA model."""
        if self.vl_model is not None:
            return

        print("Loading Video-LLaVA...")
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

        model_id = "LanguageBind/Video-LLaVA-7B-hf"

        self.vl_processor = VideoLlavaProcessor.from_pretrained(model_id)

        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.vl_model = VideoLlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.vl_model.eval()
        print("Video-LLaVA loaded.")

    def unload_models(self):
        """Unload models to free memory."""
        if self.egpt_model is not None:
            del self.egpt_model
            self.egpt_model = None

        if self.vl_model is not None:
            del self.vl_model
            self.vl_model = None

        torch.cuda.empty_cache()

    def extract_egpt_features(
        self,
        event_data: Dict,
        prompt: str,
    ) -> Dict:
        """
        Extract features from EventGPT.

        Returns:
            Dict with 'tokens', 'hidden_states', 'logits'
        """
        self.load_eventgpt()

        # Prepare input
        conversation = [{"from": "human", "value": f"<event>\n{prompt}"}]
        from model.EventChatModel import tokenizer_event_token
        input_ids, attention_mask = tokenizer_event_token(
            conversation,
            self.egpt_tokenizer,
            has_event=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Load event frames
        event_frames = event_data['event_images']  # [num_frames, H, W, C]
        if isinstance(event_frames, np.ndarray):
            event_frames = torch.from_numpy(event_frames)
        event_frames = event_frames.to(self.device, dtype=torch.bfloat16)

        # Generate with output_hidden_states
        with torch.no_grad():
            outputs = self.egpt_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                event_frames=event_frames.unsqueeze(0),  # [1, num_frames, H, W, C]
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                output_hidden_states=self.config.extract_hidden_states,
                return_dict_in_generate=True,
                output_scores=self.config.extract_logits,
            )

        # Extract tokens
        generated_ids = outputs.sequences[0, input_ids.size(1):]
        tokens = generated_ids.cpu().tolist()

        result = {
            'tokens': tokens,
            'text': self.egpt_tokenizer.decode(tokens, skip_special_tokens=True),
        }

        # Extract hidden states (from the last layer)
        if self.config.extract_hidden_states and hasattr(outputs, 'hidden_states'):
            # hidden_states is tuple of (step, layer, batch, seq, hidden)
            # We want the last layer hidden states for each generated token
            hidden_list = []
            for step_hidden in outputs.hidden_states:
                if step_hidden is not None and len(step_hidden) > 0:
                    # Get last layer, last position
                    last_layer = step_hidden[-1]  # [batch, seq, hidden]
                    hidden_list.append(last_layer[0, -1, :].cpu())  # [hidden]

            if hidden_list:
                result['hidden_states'] = torch.stack(hidden_list)  # [seq, hidden]

        # Extract logits
        if self.config.extract_logits and hasattr(outputs, 'scores'):
            logits_list = [score[0].cpu() for score in outputs.scores]
            result['logits'] = torch.stack(logits_list)  # [seq, vocab]

        return result

    def extract_vl_features(
        self,
        video_frames: List,
        prompt: str,
    ) -> Dict:
        """
        Extract features from Video-LLaVA.

        Returns:
            Dict with 'tokens', 'hidden_states', 'logits'
        """
        self.load_videollava()

        # Prepare input
        full_prompt = f"USER: <video>\n{prompt} ASSISTANT:"

        inputs = self.vl_processor(
            text=full_prompt,
            videos=video_frames,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_length = inputs['input_ids'].size(1)

        # Generate with output_hidden_states
        with torch.no_grad():
            outputs = self.vl_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                output_hidden_states=self.config.extract_hidden_states,
                return_dict_in_generate=True,
                output_scores=self.config.extract_logits,
            )

        # Extract tokens
        generated_ids = outputs.sequences[0, input_length:]
        tokens = generated_ids.cpu().tolist()

        result = {
            'tokens': tokens,
            'text': self.vl_processor.tokenizer.decode(tokens, skip_special_tokens=True),
        }

        # Extract hidden states
        if self.config.extract_hidden_states and hasattr(outputs, 'hidden_states'):
            hidden_list = []
            for step_hidden in outputs.hidden_states:
                if step_hidden is not None and len(step_hidden) > 0:
                    last_layer = step_hidden[-1]
                    hidden_list.append(last_layer[0, -1, :].cpu())

            if hidden_list:
                result['hidden_states'] = torch.stack(hidden_list)

        # Extract logits
        if self.config.extract_logits and hasattr(outputs, 'scores'):
            logits_list = [score[0].cpu() for score in outputs.scores]
            result['logits'] = torch.stack(logits_list)

        return result

    def load_sample(self, sample_dir: Path) -> Optional[Dict]:
        """Load a sample from the dataset."""
        meta_path = sample_dir / "meta.json"
        if not meta_path.exists():
            return None

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Load event images
        event_images = []
        event_dir = sample_dir / "event_images"
        if event_dir.exists():
            from PIL import Image
            for img_file in sorted(event_dir.glob("*.png")):
                img = Image.open(img_file).convert('RGB')
                event_images.append(np.array(img))

        # Load video frames
        video_frames = []
        frame_dir = sample_dir / "frames"
        if frame_dir.exists():
            from PIL import Image
            for img_file in sorted(frame_dir.glob("*.png"))[:8]:  # Max 8 frames
                img = Image.open(img_file).convert('RGB')
                video_frames.append(img)

        # Single event image if no directory
        event_image_path = sample_dir / "event_image.png"
        if event_image_path.exists() and not event_images:
            from PIL import Image
            img = Image.open(event_image_path).convert('RGB')
            event_images = [np.array(img)]

        return {
            'meta': meta,
            'event_images': np.stack(event_images) if event_images else None,
            'video_frames': video_frames,
            'sample_id': sample_dir.name,
        }

    def extract_all(self):
        """Extract features from all samples."""
        dataset_path = Path(self.config.dataset_dir)
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all samples
        sample_dirs = sorted([
            d for d in dataset_path.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        ])

        if self.config.max_samples > 0:
            sample_dirs = sample_dirs[:self.config.max_samples]

        print(f"Found {len(sample_dirs)} samples")

        # Storage for extracted features
        all_draft_tokens = []
        all_target_tokens = []
        all_draft_hidden = []
        all_target_hidden = []
        all_draft_logits = []
        all_target_logits = []
        all_sample_ids = []

        prompt = "Describe this scene in detail."

        # Extract features
        pbar = tqdm(sample_dirs, desc="Extracting features")
        for sample_dir in pbar:
            try:
                sample = self.load_sample(sample_dir)
                if sample is None:
                    continue

                # Extract EventGPT features
                egpt_result = self.extract_egpt_features(
                    {'event_images': sample['event_images']},
                    prompt,
                )

                # Free EventGPT memory before loading Video-LLaVA
                self.unload_models()
                torch.cuda.empty_cache()

                # Extract Video-LLaVA features
                vl_result = self.extract_vl_features(
                    sample['video_frames'],
                    prompt,
                )

                # Store results
                all_draft_tokens.append(egpt_result['tokens'])
                all_target_tokens.append(vl_result['tokens'])
                all_sample_ids.append(sample['sample_id'])

                if 'hidden_states' in egpt_result:
                    all_draft_hidden.append(egpt_result['hidden_states'])
                if 'hidden_states' in vl_result:
                    all_target_hidden.append(vl_result['hidden_states'])
                if 'logits' in egpt_result:
                    all_draft_logits.append(egpt_result['logits'])
                if 'logits' in vl_result:
                    all_target_logits.append(vl_result['logits'])

                pbar.set_postfix({
                    'egpt_len': len(egpt_result['tokens']),
                    'vl_len': len(vl_result['tokens']),
                })

            except Exception as e:
                print(f"Error processing {sample_dir}: {e}")
                continue

        # Free memory
        self.unload_models()

        # Pad and save
        self._save_features(
            output_path,
            all_draft_tokens,
            all_target_tokens,
            all_draft_hidden,
            all_target_hidden,
            all_draft_logits,
            all_target_logits,
            all_sample_ids,
        )

    def _save_features(
        self,
        output_path: Path,
        draft_tokens: List,
        target_tokens: List,
        draft_hidden: List,
        target_hidden: List,
        draft_logits: List,
        target_logits: List,
        sample_ids: List,
    ):
        """Pad sequences and save to disk."""
        print("Saving features...")

        # Find max sequence length
        max_len = max(
            max(len(t) for t in draft_tokens),
            max(len(t) for t in target_tokens),
        )
        max_len = min(max_len, 128)  # Cap at 128

        # Pad tokens
        def pad_tokens(tokens_list, max_len, pad_value=0):
            padded = []
            for tokens in tokens_list:
                tokens = tokens[:max_len]
                padded.append(tokens + [pad_value] * (max_len - len(tokens)))
            return torch.tensor(padded, dtype=torch.long)

        draft_tokens_tensor = pad_tokens(draft_tokens, max_len)
        target_tokens_tensor = pad_tokens(target_tokens, max_len)

        torch.save(draft_tokens_tensor, output_path / 'draft_tokens.pt')
        torch.save(target_tokens_tensor, output_path / 'target_tokens.pt')

        # Pad and save hidden states
        if draft_hidden:
            hidden_dim = draft_hidden[0].size(-1)
            draft_hidden_padded = self._pad_hidden(draft_hidden, max_len, hidden_dim)
            torch.save(draft_hidden_padded, output_path / 'draft_hidden.pt')

        if target_hidden:
            hidden_dim = target_hidden[0].size(-1)
            target_hidden_padded = self._pad_hidden(target_hidden, max_len, hidden_dim)
            torch.save(target_hidden_padded, output_path / 'target_hidden.pt')

        # Pad and save logits (memory intensive!)
        if draft_logits:
            vocab_size = draft_logits[0].size(-1)
            draft_logits_padded = self._pad_logits(draft_logits, max_len, vocab_size)
            torch.save(draft_logits_padded, output_path / 'draft_logits.pt')

        if target_logits:
            vocab_size = target_logits[0].size(-1)
            target_logits_padded = self._pad_logits(target_logits, max_len, vocab_size)
            torch.save(target_logits_padded, output_path / 'target_logits.pt')

        # Save sample IDs
        with open(output_path / 'sample_ids.json', 'w') as f:
            json.dump(sample_ids, f)

        # Save metadata
        meta = {
            'num_samples': len(sample_ids),
            'max_seq_len': max_len,
            'has_hidden_states': bool(draft_hidden),
            'has_logits': bool(draft_logits),
            'config': asdict(self.config),
        }
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {len(sample_ids)} samples to {output_path}")

    def _pad_hidden(self, hidden_list: List[torch.Tensor], max_len: int, hidden_dim: int) -> torch.Tensor:
        """Pad hidden states to uniform length."""
        padded = []
        for h in hidden_list:
            seq_len = h.size(0)
            if seq_len < max_len:
                pad = torch.zeros(max_len - seq_len, hidden_dim)
                h = torch.cat([h, pad], dim=0)
            else:
                h = h[:max_len]
            padded.append(h)
        return torch.stack(padded)

    def _pad_logits(self, logits_list: List[torch.Tensor], max_len: int, vocab_size: int) -> torch.Tensor:
        """Pad logits to uniform length."""
        padded = []
        for l in logits_list:
            seq_len = l.size(0)
            if seq_len < max_len:
                pad = torch.zeros(max_len - seq_len, vocab_size)
                l = torch.cat([l, pad], dim=0)
            else:
                l = l[:max_len]
            padded.append(l)
        return torch.stack(padded)


def main():
    parser = argparse.ArgumentParser(description="Extract features for token alignment")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to output directory")
    parser.add_argument('--max_samples', type=int, default=-1,
                        help="Maximum samples to process (-1 for all)")
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument('--extract_hidden_states', action='store_true',
                        help="Extract hidden states (recommended)")
    parser.add_argument('--extract_logits', action='store_true',
                        help="Extract logits (memory intensive)")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use")
    parser.add_argument('--no_4bit', action='store_true',
                        help="Disable 4-bit quantization")

    args = parser.parse_args()

    config = ExtractionConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        extract_hidden_states=args.extract_hidden_states,
        extract_logits=args.extract_logits,
        device=args.device,
        use_4bit=not args.no_4bit,
    )

    extractor = FeatureExtractor(config)
    extractor.extract_all()


if __name__ == "__main__":
    main()
