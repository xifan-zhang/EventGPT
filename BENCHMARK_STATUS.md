# Comprehensive Benchmark Status Report

**Last Updated:** 2026-01-27 16:50 UTC
**Status:** All Remaining Datasets Running

---

## ✅ Completed Benchmarks

### 1s Dataset (1,100 samples) - COMPLETE
- **Timestamp:** 2026-01-27 16:40:40
- **Duration:** ~22 minutes
- **Token-Level Acceptance Rate:** 3.42% ± 1.82%
- **Results File:** `parallel_prefill_5stages_20260127_160820.json`

**Key Results:**
- Hidden tokens (avg): 35.3 tokens
- Overlap window (avg): 360.5 ms
- Parallel speedup (avg): 1.48x
- Model speedup (EGPT vs VL): 2.07x

---

## 🏃 Currently Running

### 500ms Dataset (2,220 samples)
- **Start Time:** 2026-01-27 16:47 UTC
- **Current Progress:** 17/2,220 samples (~0.8%)
- **Rate:** ~1.76 seconds per sample
- **ETA:** ~65 minutes (completion around 17:52 UTC)
- **Log File:** `/tmp/all_benchmarks.log`

**Expected Results:** ~4,600 KB JSON file

---

## ⏳ Remaining Datasets (Queued)

| Dataset | Samples | Est. Time | Total Cumulative |
|---------|---------|-----------|-----------------|
| **500ms** | 2,220 | ~65 min | ~65 min |
| 2s | 540 | ~16 min | ~81 min |
| 4s | 260 | ~8 min | ~89 min |
| 5s | 193 | ~6 min | ~95 min |
| 8s | 117 | ~3 min | ~98 min |
| 10s | 93 | ~3 min | ~101 min |
| 16s | 23 | <1 min | ~102 min |
| 20s | 38 | ~1 min | ~103 min |

**Total Remaining:** 3,484 samples
**Total ETA:** ~103 minutes (~1h 43min from now)
**Expected Completion:** ~18:33 UTC

---

## 📊 Unified Settings (All Datasets)

```
max_new_tokens:    50
Quantization:      4-bit NF4 + double quant
Acceptance Rate:   Token-level (target tokenizer)
GPU Memory:        ~7.95 GB
Output Format:     JSON + Markdown report
```

### Why max_new_tokens=50?

- **Benchmark Speed:** ~22-30 minutes per 1,100 samples
- **Output Quality:** ~30-40 words per description
- **Alignment:** Hidden tokens (~35) fit within limit
- **Consistency:** Same limit across all models and datasets
- **Flexibility:** Adjustable via command-line flag if needed

---

## 📁 Output Files

All results saved to: `feasible/benchmark_parallel_prefill/results/`

**For Each Dataset:**
- JSON: `parallel_prefill_5stages_[TIMESTAMP].json` (detailed results)
- Markdown: `parallel_prefill_5stages_[TIMESTAMP].md` (summary report)

---

## 🔍 Monitoring Progress

### Check Current Progress
```bash
# Show latest dataset processing
tail -10 /tmp/all_benchmarks.log

# Show only benchmark progress
tail -20 /tmp/all_benchmarks.log | grep "5-Stage"

# Count completed datasets
ls feasible/benchmark_parallel_prefill/results/parallel_prefill_5stages_*.json | wc -l
```

### Wait for Completion
```bash
# Monitor in real-time
watch 'tail -5 /tmp/all_benchmarks.log'

# Check if process is running
ps aux | grep run_all_remaining_benchmarks
```

---

## 📈 Combined Results Preview (When Complete)

**Total Samples Benchmarked:**
- 1s: 1,100 ✅
- 500ms: 2,220 🏃
- 2s: 540 ⏳
- 4s: 260 ⏳
- 5s: 193 ⏳
- 8s: 117 ⏳
- 10s: 93 ⏳
- 16s: 23 ⏳
- 20s: 38 ⏳

**Grand Total: 4,584 samples** (will match original analysis with corrected metrics)

---

## 🎯 Expected Insights

After all benchmarks complete:

1. **Token-Level Acceptance vs Video Duration**
   - Does acceptance change with longer sequences?
   - Trend across 500ms to 20s range

2. **Parallel Prefill Consistency**
   - Hidden tokens stable across durations?
   - Overlap window variation

3. **Speedup Scaling**
   - Does 1.48x speedup scale with sequence length?
   - Peak performance duration

4. **Model Behavior Divergence**
   - Acceptance distribution changes
   - Output quality by duration

---

## 📝 Next Actions

1. **Monitor Progress:** Watch `/tmp/all_benchmarks.log`
2. **When Complete:** Analyze combined results across all durations
3. **Generate Summary:** Unified report with token-level acceptance by dataset
4. **Compare Metrics:** Old semantic vs new token-level acceptance rates

---

## 🚀 Summary

- ✅ Token-level acceptance calculation **FIXED**
- ✅ Protobuf compatibility **RESOLVED**
- ✅ 1s dataset benchmark **COMPLETE** (3.42% token acceptance)
- 🏃 500ms dataset currently running (17/2220)
- ⏳ 7 more datasets queued (~1h 40min remaining)
- 📊 **4,584 total samples** with corrected metrics

**Status: All systems running normally. Benchmarks will complete autonomously.**

---

*Script: `/home/ps/Documents/code/EventGPT/run_all_remaining_benchmarks.sh`*
*Log: `/tmp/all_benchmarks.log`*
