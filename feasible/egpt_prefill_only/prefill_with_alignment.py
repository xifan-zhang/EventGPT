#!/usr/bin/env python3
"""
Prefill-Only with Token Alignment Integration

Enhanced prefill-only approach that uses the trained TokenAdapter to:
1. Generate EventGPT tokens during parallel prefill
2. Convert to Video-LLaVA token predictions using TokenAdapter
3. Optionally use predictions to prime/guide Video-LLaVA decode

This combines the benefits of:
- Parallel prefill (guaranteed prefill time savings)
- Token alignment (higher quality draft predictions)
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Fix protobuf compatibility
if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in os.environ:
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Default trained TokenAdapter checkpoint (27.9% acceptance, 51.6% top-5)
DEFAULT_ADAPTER_PATH = "./feasible/token_alignment/task/starred/1q_20260128_151847/best_model.pt"


@dataclass
class AlignedPrefillConfig:
    """Configuration for aligned prefill inference."""
    eventgpt_path: str = "./checkpoints/EventGPT-7b"
    videollava_model_id: str = "LanguageBind/Video-LLaVA-7B-hf"
    token_adapter_path: str = DEFAULT_ADAPTER_PATH  # Path to trained TokenAdapter
    device: str = "cuda"
    use_4bit: bool = True
    max_new_tokens: int = 50
    # Alignment settings
    use_alignment: bool = True  # Use TokenAdapter to convert drafts
    alignment_top_k: int = 5    # Use top-k predictions from adapter


@dataclass
class AlignedPrefillResult:
    """Result from aligned prefill inference."""
    # Final output
    output_text: str
    output_tokens: List[int]

    # Preview (EventGPT raw)
    preview_text: str
    preview_tokens: List[int]

    # Aligned predictions (from TokenAdapter)
    aligned_text: Optional[str]
    aligned_tokens: Optional[List[int]]
    alignment_confidence: Optional[float]

    # Timing
    egpt_prefill_time: float
    egpt_decode_time: float
    adapter_time: float  # Time for TokenAdapter inference
    vl_prefill_time: float
    vl_decode_time: float
    wall_clock_time: float

    # Analysis
    free_tokens_count: int
    overlap_window: float
    acceptance_rate: float  # How many aligned tokens matched VL output


class TokenAdapterPredictor:
    """Wrapper for TokenAdapter inference."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        """Load the TokenAdapter model."""
        if self._model is not None:
            return

        # Import TokenAdapter from token_alignment
        sys.path.insert(0, str(ROOT / "feasible" / "token_alignment"))
        from train_and_evaluate import TokenAdapter

        checkpoint = torch.load(self.model_path, map_location=self.device)

        self._model = TokenAdapter(
            vocab_size=32010,
            embed_dim=512,
            num_layers=4,
            num_heads=8,
            max_seq_len=128,
        )
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model = self._model.to(self.device)
        self._model.eval()

    def predict(
        self,
        draft_tokens: List[int],
        top_k: int = 1,
    ) -> Tuple[List[int], List[float]]:
        """
        Predict Video-LLaVA tokens from EventGPT draft tokens.

        Args:
            draft_tokens: List of EventGPT token IDs
            top_k: Return top-k predictions per position

        Returns:
            Tuple of (predicted_tokens, confidence_scores)
        """
        self._load_model()

        with torch.inference_mode():
            # Prepare input
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=self.device)

            # Pad if needed
            if draft_tensor.shape[1] < 128:
                pad_len = 128 - draft_tensor.shape[1]
                draft_tensor = torch.nn.functional.pad(draft_tensor, (0, pad_len))

            # Forward pass
            logits = self._model(draft_tensor[:, :-1])

            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k, dim=-1)

            # Use top-1 predictions
            predicted_tokens = top_indices[:, :, 0].squeeze(0).tolist()
            confidences = top_probs[:, :, 0].squeeze(0).tolist()

            # Truncate to original length
            predicted_tokens = predicted_tokens[:len(draft_tokens)]
            confidences = confidences[:len(draft_tokens)]

            return predicted_tokens, confidences


class AlignedPrefillInference:
    """
    Prefill-only inference with TokenAdapter alignment.

    Combines:
    - Parallel prefill for time savings
    - TokenAdapter for better draft predictions
    """

    def __init__(
        self,
        config: AlignedPrefillConfig,
    ):
        self.config = config
        self._prefill_inference = None
        self._adapter = None

    def _load_components(self):
        """Load inference components."""
        if self._prefill_inference is None:
            from prefill_only import PrefillOnlyInference

            self._prefill_inference = PrefillOnlyInference(
                eventgpt_path=self.config.eventgpt_path,
                videollava_model_id=self.config.videollava_model_id,
                device=self.config.device,
                use_4bit=self.config.use_4bit,
            )
            self._prefill_inference._load_models()

        if self.config.use_alignment and self.config.token_adapter_path and self._adapter is None:
            self._adapter = TokenAdapterPredictor(
                model_path=self.config.token_adapter_path,
                device=self.config.device,
            )

    def generate(
        self,
        event_image_path: str,
        video_frames: List,
        query: str,
        max_new_tokens: int = None,
    ) -> AlignedPrefillResult:
        """
        Run aligned prefill inference.

        Args:
            event_image_path: Path to event image
            video_frames: List of PIL images for Video-LLaVA
            query: Text query
            max_new_tokens: Maximum tokens to generate

        Returns:
            AlignedPrefillResult with outputs and timing
        """
        self._load_components()

        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        wall_clock_start = time.time()

        # Run parallel prefill (this gets us EventGPT preview + VL output)
        result = self._prefill_inference.generate(
            event_image_path=event_image_path,
            video_frames=video_frames,
            query=query,
            max_new_tokens=max_new_tokens,
            return_preview=True,
        )

        # Apply TokenAdapter alignment if available
        aligned_text = None
        aligned_tokens = None
        alignment_confidence = None
        adapter_time = 0

        if self._adapter and result.preview_tokens:
            adapter_start = time.time()

            aligned_tokens, confidences = self._adapter.predict(
                result.preview_tokens,
                top_k=self.config.alignment_top_k,
            )
            aligned_text = self._prefill_inference._videollava_processor.tokenizer.decode(
                aligned_tokens, skip_special_tokens=True
            )
            alignment_confidence = float(np.mean(confidences))

            adapter_time = time.time() - adapter_start

        # Compute acceptance rate (aligned vs actual VL output)
        acceptance_rate = 0.0
        if aligned_tokens and result.output_tokens:
            min_len = min(len(aligned_tokens), len(result.output_tokens))
            matches = sum(1 for i in range(min_len)
                         if aligned_tokens[i] == result.output_tokens[i])
            acceptance_rate = matches / max(min_len, 1)

        wall_clock_time = time.time() - wall_clock_start

        return AlignedPrefillResult(
            output_text=result.output_text,
            output_tokens=result.output_tokens,
            preview_text=result.preview_text or "",
            preview_tokens=result.preview_tokens or [],
            aligned_text=aligned_text,
            aligned_tokens=aligned_tokens,
            alignment_confidence=alignment_confidence,
            egpt_prefill_time=result.egpt_prefill_time,
            egpt_decode_time=result.egpt_decode_time,
            adapter_time=adapter_time,
            vl_prefill_time=result.vl_prefill_time,
            vl_decode_time=result.vl_decode_time,
            wall_clock_time=wall_clock_time,
            free_tokens_count=result.free_tokens_count,
            overlap_window=result.overlap_window,
            acceptance_rate=acceptance_rate,
        )


def find_best_adapter_model() -> Optional[str]:
    """Find the best trained TokenAdapter model."""
    # Check default starred model first
    default_path = ROOT / "feasible" / "token_alignment" / "task" / "starred" / "1q_20260128_151847" / "best_model.pt"
    if default_path.exists():
        return str(default_path)

    # Fall back to searching other directories
    adapter_dirs = [
        ROOT / "feasible" / "token_alignment" / "task" / "starred",
        ROOT / "feasible" / "token_alignment" / "checkpoints_1s",
        ROOT / "feasible" / "token_alignment" / "checkpoints_token_adapter",
        ROOT / "feasible" / "token_alignment" / "task",
    ]

    for adapter_dir in adapter_dirs:
        if adapter_dir.exists():
            # Look for best_model.pt
            for model_path in adapter_dir.rglob("best_model.pt"):
                return str(model_path)

    return None


def main():
    """Test aligned prefill inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Aligned prefill inference test")
    parser.add_argument("--event_image", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--query", type=str, default="What are the key elements in this scene?")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to trained TokenAdapter. Auto-detected if not specified.")

    args = parser.parse_args()

    # Find adapter model
    adapter_path = args.adapter_path or find_best_adapter_model()
    if adapter_path:
        print(f"Using TokenAdapter: {adapter_path}")
    else:
        print("No TokenAdapter found. Running without alignment.")

    # Load video frames
    try:
        import av
        container = av.open(args.video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
            if len(frames) >= 8:
                break
        container.close()
    except Exception as e:
        print(f"Error loading video: {e}")
        return

    # Run inference
    config = AlignedPrefillConfig(
        token_adapter_path=adapter_path,
        use_alignment=adapter_path is not None,
    )

    inference = AlignedPrefillInference(config)
    result = inference.generate(
        event_image_path=args.event_image,
        video_frames=frames,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "="*70)
    print("ALIGNED PREFILL INFERENCE RESULT")
    print("="*70)
    print(f"\nQuery: {args.query}")

    print(f"\n{'─'*70}")
    print("OUTPUTS:")
    print(f"\n1. EventGPT Preview (raw):")
    print(f"   {result.preview_text[:200]}...")

    if result.aligned_text:
        print(f"\n2. Aligned Prediction (TokenAdapter):")
        print(f"   {result.aligned_text[:200]}...")
        print(f"   Confidence: {result.alignment_confidence*100:.1f}%")
        print(f"   Acceptance Rate: {result.acceptance_rate*100:.1f}%")

    print(f"\n3. Final Output (Video-LLaVA):")
    print(f"   {result.output_text[:200]}...")

    print(f"\n{'─'*70}")
    print("TIMING ANALYSIS:")
    print(f"  EGPT Prefill:     {result.egpt_prefill_time*1000:.1f} ms")
    print(f"  EGPT Decode:      {result.egpt_decode_time*1000:.1f} ms")
    if result.adapter_time > 0:
        print(f"  Adapter:          {result.adapter_time*1000:.1f} ms")
    print(f"  VL Prefill:       {result.vl_prefill_time*1000:.1f} ms")
    print(f"  VL Decode:        {result.vl_decode_time*1000:.1f} ms")
    print(f"  Overlap Window:   {result.overlap_window*1000:.1f} ms")
    print(f"  Free Tokens:      {result.free_tokens_count}")
    print(f"  Wall-Clock Time:  {result.wall_clock_time*1000:.1f} ms")

    print("="*70)


if __name__ == "__main__":
    main()
