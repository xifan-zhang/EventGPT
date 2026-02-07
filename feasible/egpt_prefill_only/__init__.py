"""
EventGPT Prefill Only: Parallel prefill benefit without speculative decoding.

This module implements Option 2 of the speculative decoding research:
- Run EventGPT and Video-LLaVA prefill in parallel
- Get "free" tokens during the parallel prefill window
- Use standard AR decode with Video-LLaVA (no SD verification)
- Saves ~232ms prefill overhead (315ms - 83ms)

Benefits:
- Simpler than full speculative decoding
- No acceptance rate concerns
- Still benefits from EventGPT's faster prefill

Token Alignment Model:
- Default adapter: task/starred/1q_20260128_151847/best_model.pt
- Acceptance rate: 27.9% (up from 1.58% baseline)
- Top-5 accuracy: 51.64%
"""

from .prefill_only import (
    PrefillOnlyInference,
    compute_prefill_benefit,
)

from .prefill_with_alignment import (
    AlignedPrefillInference,
    AlignedPrefillConfig,
    AlignedPrefillResult,
    TokenAdapterPredictor,
    DEFAULT_ADAPTER_PATH,
)

from .prefill_then_verify import (
    PrefillThenVerifyInference,
    PrefillThenVerifyResult,
)

__all__ = [
    # Basic prefill-only (no verification)
    "PrefillOnlyInference",
    "compute_prefill_benefit",
    # With token alignment
    "AlignedPrefillInference",
    "AlignedPrefillConfig",
    "AlignedPrefillResult",
    "TokenAdapterPredictor",
    "DEFAULT_ADAPTER_PATH",
    # Full speculative decoding with batch verification
    "PrefillThenVerifyInference",
    "PrefillThenVerifyResult",
]
