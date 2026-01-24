# Tokenizer Alignment for EventGPT and Video-LLaVA

**Created:** 2026-01-22
**Purpose:** Improve acceptance rate (α) for speculative decoding by aligning tokenizers

## Problem

Current benchmark results show:
- **Acceptance rate α ≈ 5-7%** (only 5-7% of draft tokens accepted by target)
- This low acceptance rate prevents speculative decoding from providing speedup

## Key Discovery (2026-01-22)

**Tokenizers are 100% IDENTICAL!**

```
EventGPT tokenizer: LlamaTokenizerFast, Vocab size: 32000
Video-LLaVA tokenizer: LlamaTokenizerFast, Vocab size: 32003
Shared tokens: 32,000 / 32,000 (100.0% overlap)
```

**Test Results on Sample Texts:**
- All 5 test texts: **100% token match rate**
- Same input text → identical token IDs

**Benchmark Results (20s dataset):**
| Target Model | Samples | α (Acceptance Rate) | Draft Tokens | Accepted |
|--------------|---------|---------------------|--------------|----------|
| LLaVA 1.5-7B | 38 | 6.8% | 1,702 | 115 |
| Video-LLaVA-7B | 10 | **3.9%** | 357 | 14 |

**Conclusion:** The low acceptance rate (α ≈ 4-7%) is **NOT** due to tokenizer differences.
It is due to **different model outputs** during generation! Video-LLaVA has even LOWER α than LLaVA 1.5.

## Solution: Option 1 - Tokenizer Alignment

### Approach

1. **Analyze Tokenizers**
   - Compare vocabularies of both models
   - Identify shared vs unique tokens
   - Calculate overlap percentage

2. **Create Translation Map**
   - Build 1:1 mapping for shared tokens
   - Create substring mappings for similar tokens
   - Handle special tokens (BOS, EOS, PAD, etc.)

3. **Translate Draft Tokens**
   - Convert EventGPT token IDs to Video-LLaVA token IDs
   - Verify improved acceptance rate on benchmark samples

4. **Test & Validate**
   - Re-run benchmarks with translated tokens
   - Measure improvement in α
   - Target: α > 30% for effective speculative decoding

## Implementation

### Files

- `align_tokenizers.py` - Main alignment script
  - `TokenizerAligner` class for comparing and aligning tokenizers
  - Creates translation map from EventGPT → Video-LLaVA
  - Computes improved acceptance rate with alignment

- `test_alignment.py` - Test script for benchmark samples
  - Loads existing benchmark results
  - Applies token translation
  - Measures acceptance rate improvement

- `translation_map.json` - Generated token ID mapping

## Usage

### Basic Analysis

```bash
cd feasible/tokenizer_alignment
python align_tokenizers.py \
    --eventgpt_path ../../checkpoints/EventGPT-7b \
    --videollava_path LanguageBind/Video-LLaVA-7B-hf \
    --test_texts 10
```

### Expected Output

```
Vocabulary Analysis:
  Shared tokens: 48,234
  EventGPT-only: 12,456
  LLaVA-only: 8,901
  Overlap: 95.2%

Translation map saved to translation_map.json
```

## Updated Roadmap (After Discovery)

### Phase 1: Analysis ✅ COMPLETE
- [x] Create tokenizer alignment framework
- [x] Load and compare tokenizers
- [x] Generate vocabulary statistics
- [x] **DISCOVERY: Tokenizers are 100% identical**

### Phase 2: Analyze Model Outputs (Current)
- [ ] Analyze actual benchmark outputs to understand why models generate different text
- [ ] Compare EventGPT vs Video-LLaVA outputs on same input
- [ ] Identify patterns in output differences

### Phase 3: Explore Solutions
- [x] **Option C**: Match sampling parameters exactly
- [ ] **Option A**: Use Video-LLaVA as draft (same tokenizer, 100% compatible)
- [ ] **Option B**: Fine-tune models to generate similar text

### Phase 4: Implementation & Validation
- [ ] Implement chosen solution
- [ ] Re-run benchmarks with new approach
- [ ] Measure acceptance rate improvement

## Why Low Acceptance Rate?

Since tokenizers are identical, α ≈ 5-7% means:
1. **Models generate different text** despite same input
2. **Different sampling strategies or temperature** → **TESTED: No improvement**
3. **Model weights trained on different data distributions**

### Test: Matching Sampling Parameters (Option C)

Tested if matching temperature improves α:

| Configuration | Temperature | α (Acceptance) | Result |
|---------------|-------------|----------------|--------|
| Baseline | 0.2 | 3.9% | - |
| Greedy | 0.0 | **0.0%** | Worse! |
| EventGPT default | 0.6 | - | - |

**Finding:** Greedy decoding made α **worse** (0% vs 3.9%). EventGPT produces identical output with temp=0 and temp=0.2, but Video-LLaVA's output changes. The root cause is **different model training**, not sampling parameters.

## Alternative Options

If Option 1 doesn't provide sufficient improvement:

**Option 2: Fine-tune for Alignment**
- Train both models on same caption-style data
- Make them use similar phrasing patterns
- Higher accuracy but more complex

**Option 3: Use Video-LLaVA as Draft**
- Use quantized/smaller Video-LLaVA as draft
- Same tokenizer = 100% token compatibility
- May be simplest solution

## Notes

**Key Finding (2026-01-22):**
- Both models use LlamaTokenizerFast with 100% identical vocabularies
- Video-LLaVA has 3 additional special tokens (32003 vs 32000)
- **Translation is NOT needed** - tokenizers are already aligned
- The problem is different model outputs, not tokenizer mismatch

**Next Steps:**
- Analyze actual model outputs to understand why they differ
- Consider using Video-LLaVA as both draft AND target (with different sizes)
- Or fine-tune EventGPT to match Video-LLaVA's output patterns
