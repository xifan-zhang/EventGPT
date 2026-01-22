"""
Tokenizer Alignment for EventGPT and Video-LLaVA

Purpose: Align tokenizers to improve acceptance rate (α) in speculative decoding.

Approach (Option 1):
1. Analyze both tokenizers (vocabularies, special tokens, merge rules)
2. Create mapping between token IDs
3. Implement re-tokenization to align outputs
4. Test improvement on benchmark samples
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer

class TokenizerAligner:
    """Align tokenizers between EventGPT (draft) and Video-LLaVA (target)."""

    def __init__(
        self,
        eventgpt_model_path: str = "./checkpoints/EventGPT-7b",
        videollava_model_path: str = "LanguageBind/Video-LLaVA-7B-hf"
    ):
        print("Loading tokenizers...")
        self.eventgpt_tokenizer = AutoTokenizer.from_pretrained(eventgpt_model_path)
        self.videollava_tokenizer = AutoTokenizer.from_pretrained(videollava_model_path)

        print(f"EventGPT tokenizer: {type(self.eventgpt_tokenizer).__name__}")
        print(f"  Vocab size: {len(self.eventgpt_tokenizer)}")
        print(f"Video-LLaVA tokenizer: {type(self.videollava_tokenizer).__name__}")
        print(f"  Vocab size: {len(self.videollava_tokenizer)}")

        # Build vocabulary mappings
        self._build_vocab_mappings()

    def _build_vocab_mappings(self):
        """Build mappings between tokenizers."""
        # Get vocabularies (token_id -> token_string)
        self.egpt_vocab = {i: s for s, i in self.eventgpt_tokenizer.get_vocab().items()}
        self.llava_vocab = {i: s for s, i in self.videollava_tokenizer.get_vocab().items()}

        # Reverse mappings (token_string -> token_id)
        self.egpt_vocab_reverse = {s: i for i, s in self.egpt_vocab.items()}
        self.llava_vocab_reverse = {s: i for i, s in self.llava_vocab.items()}

        # Find shared tokens
        egpt_tokens = set(self.egpt_vocab.values())
        llava_tokens = set(self.llava_vocab.values())

        self.shared_tokens = egpt_tokens & llava_tokens
        self.egpt_only_tokens = egpt_tokens - llava_tokens
        self.llava_only_tokens = llava_tokens - egpt_tokens

        print(f"\nVocabulary Analysis:")
        print(f"  Shared tokens: {len(self.shared_tokens):,}")
        print(f"  EventGPT-only: {len(self.egpt_only_tokens):,}")
        print(f"  LLaVA-only: {len(self.llava_only_tokens):,}")
        print(f"  Overlap: {len(self.shared_tokens) / max(len(egpt_tokens), len(llava_tokens)) * 100:.1f}%")

    def compare_token_sequences(
        self,
        text: str,
        max_length: int = 100
    ) -> Dict[str, List[int]]:
        """Compare tokenization of the same text between both tokenizers."""
        # Tokenize with both
        egpt_tokens = self.eventgpt_tokenizer.encode(text, add_special_tokens=False)[:max_length]
        llava_tokens = self.videollava_tokenizer.encode(text, add_special_tokens=False)[:max_length]

        # Decode back to text to see what each token represents
        egpt_decoded = [self.eventgpt_tokenizer.decode([t]) for t in egpt_tokens]
        llava_decoded = [self.videollava_tokenizer.decode([t]) for t in llava_tokens]

        return {
            'eventgpt_tokens': egpt_tokens,
            'videollava_tokens': llava_tokens,
            'eventgpt_decoded': egpt_decoded,
            'videollava_decoded': llava_decoded,
            'token_match_rate': sum(1 for e, l in zip(egpt_tokens, llava_tokens) if e == l) / min(len(egpt_tokens), len(llava_tokens))
        }

    def create_translation_map(self) -> Dict[int, int]:
        """
        Create a mapping from EventGPT token IDs to Video-LLaVA token IDs.

        For shared tokens: direct 1:1 mapping
        For EventGPT-only tokens: find closest Video-LLaVA token (by text similarity or subword)
        """
        translation_map = {}

        print("\nCreating token ID translation map...")

        # Direct mapping for shared tokens
        for token_str in self.shared_tokens:
            egpt_id = self.egpt_vocab_reverse[token_str]
            llava_id = self.llava_vocab_reverse[token_str]
            translation_map[egpt_id] = llava_id

        print(f"  Direct mappings: {len(translation_map)}")

        # For EventGPT-only tokens, find best match in Video-LLaVA
        # This is a simplified approach - can be improved with semantic similarity
        mapped_count = 0
        for egpt_id, token_str in self.egpt_vocab.items():
            if egpt_id in translation_map:
                continue

            # Try to find substring match in LLaVA vocabulary
            best_match = None
            best_match_len = 0

            for llava_str, llava_id in self.llava_vocab_reverse.items():
                # Check if EventGPT token is substring of LLaVA token or vice versa
                if token_str in llava_str and len(llava_str) > best_match_len:
                    best_match = llava_id
                    best_match_len = len(llava_str)
                elif llava_str in token_str and len(token_str) > best_match_len:
                    best_match = llava_id
                    best_match_len = len(token_str)

            if best_match is not None:
                translation_map[egpt_id] = best_match
                mapped_count += 1

        print(f"  Substring mappings: {mapped_count}")
        print(f"  Total mappings: {len(translation_map)}/{len(self.egpt_vocab)}")

        return translation_map

    def translate_tokens(
        self,
        egpt_token_ids: List[int],
        translation_map: Dict[int, int]
    ) -> List[int]:
        """Translate EventGPT token IDs to Video-LLaVA token IDs."""
        translated = []
        unmapped = []

        for token_id in egpt_token_ids:
            if token_id in translation_map:
                translated.append(translation_map[token_id])
            else:
                # Use special token or keep original (will likely cause verification failure)
                translated.append(token_id)
                unmapped.append(token_id)

        return translated, unmapped

    def compute_acceptance_rate_with_alignment(
        self,
        draft_tokens: List[int],
        target_tokens: List[int],
        translation_map: Dict[int, int]
    ) -> Tuple[float, int, int]:
        """
        Compute acceptance rate with token translation.

        Returns: (acceptance_rate, num_accepted, num_draft_tokens)
        """
        # Translate draft tokens to target tokenizer space
        translated_draft, unmapped = self.translate_tokens(draft_tokens, translation_map)

        # Count matches in the overlapping range
        min_len = min(len(translated_draft), len(target_tokens))
        accepted = sum(1 for i in range(min_len) if translated_draft[i] == target_tokens[i])

        return accepted / len(draft_tokens), accepted, len(draft_tokens)


def main():
    """Test tokenizer alignment."""
    import argparse

    parser = argparse.ArgumentParser(description="Test tokenizer alignment")
    parser.add_argument("--eventgpt_path", type=str, default="./checkpoints/EventGPT-7b")
    parser.add_argument("--videollava_path", type=str, default="LanguageBind/Video-LLaVA-7B-hf")
    parser.add_argument("--test_texts", type=int, default=5)
    args = parser.parse_args()

    # Initialize aligner
    aligner = TokenizerAligner(args.eventgpt_path, args.videollava_path)

    # Create translation map
    translation_map = aligner.create_translation_map()

    # Save translation map
    output_path = "feasible/tokenizer_alignment/translation_map.json"
    os.makedirs("feasible/tokenizer_alignment", exist_ok=True)

    # Convert int keys to strings for JSON serialization
    json_map = {str(k): v for k, v in translation_map.items()}
    with open(output_path, 'w') as f:
        json.dump(json_map, f, indent=2)

    print(f"\nTranslation map saved to {output_path}")

    # Test on some sample texts
    test_texts = [
        "Describe what you see in this driving scene.",
        "The image shows a car driving on a road.",
        "A vehicle is moving through an intersection.",
        "The scene contains a truck and several cars.",
        "Weather conditions appear to be clear and sunny."
    ]

    print("\n" + "="*70)
    print("Testing token alignment on sample texts:")
    print("="*70)

    for i, text in enumerate(test_texts[:args.test_texts]):
        print(f"\n--- Test {i+1}: {text} ---")
        result = aligner.compare_token_sequences(text)

        print(f"EventGPT tokens ({len(result['eventgpt_tokens'])}):")
        print(f"  IDs: {result['eventgpt_tokens'][:15]}")
        print(f"  Text: {' '.join(result['eventgpt_decoded'][:15])}")

        print(f"Video-LLaVA tokens ({len(result['videollava_tokens'])}):")
        print(f"  IDs: {result['videollava_tokens'][:15]}")
        print(f"  Text: {' '.join(result['videollava_decoded'][:15])}")

        print(f"Token match rate: {result['token_match_rate']:.1%}")


if __name__ == "__main__":
    main()
