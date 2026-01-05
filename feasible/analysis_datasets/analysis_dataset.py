"""
Analysis Script for EventGPT Dataset

Author: Alice Zhang
Date: 2025-06-24

Description:
    This script analyzes the EventGPT dataset, providing statistics and visualizations for:
    1. Dataset structure and feature schema.
    2. Label distributions (if applicable).
    3. Conversation length (number of turns).
    4. Event metrics derived from raw .npy files:
       - Figure 1: Event Duration Distribution (ms)
       - Figure 2: Event Count Distribution (number of events per sample)
       - Figure 3: Event Density Distribution (events/ms)
    
    The results are saved as figures in the `figures/` directory and a summary text file.
"""

import os
import sys
import json
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, ClassLabel, Value, Features
from tqdm import tqdm


# Ensure project root is on sys.path so we can import `data.load_dataset`
# ROOT should be the project root (EventGPT/)
# __file__ -> feasible/egpt_dataset/analysis_dataset.py
# dirname -> feasible/egpt_dataset
# dirname -> feasible
# dirname -> EventGPT
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load dataset directly from local directory (downloaded via snapshot_download)
DATA_DIR = "/mnt/hdd/data/EventGPT-datasets"
JSON_FILE = os.path.join(DATA_DIR, "EventGPT_Instruction_Subset.json")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

try:
    # Try loading as a Hugging Face dataset from disk (if it was saved with save_to_disk)
    ds = Dataset.load_from_disk(DATA_DIR)
except (FileNotFoundError, ValueError):
    print("⚠️ Could not load from disk (not a saved Arrow dataset).")
    print(f"🔄 Attempting to load from JSON: {JSON_FILE}")
    if os.path.exists(JSON_FILE):
        # Load from JSON file
        from datasets import load_dataset
        ds = load_dataset("json", data_files=JSON_FILE, split="train")
    else:
        raise FileNotFoundError(f"Could not find dataset at {DATA_DIR} or {JSON_FILE}")

print(f"✅ Loaded {len(ds)} examples")


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")
    print("🔍 Starting analysis... ✨")


def _summarize_features(features: Features) -> None:
    print("\nFeature schema:")
    for name, feature in features.items():
        dtype = type(feature).__name__
        extra: Optional[str] = None
        if isinstance(feature, ClassLabel):
            extra = f"num_classes={feature.num_classes}, names={feature.names[:10]}"
        elif isinstance(feature, Value):
            extra = f"dtype={feature.dtype}"

        if extra:
            print(f"  - {name}: {dtype} ({extra})")
        else:
            print(f"  - {name}: {dtype}")


def _show_examples(dataset: Dataset, num_rows: int = 3) -> None:
    num_rows = min(num_rows, len(dataset))
    print(f"\nShowing {num_rows} example rows:")
    for i in range(num_rows):
        row = dataset[i]
        print(f"  [#{i}]")
        for k, v in row.items():
            # Truncate long strings / lists for readability
            text = str(v)
            if len(text) > 200:
                text = text[:197] + "..."
            print(f"    - {k}: {text}")


def _analyze_split(name: str, split_ds: Dataset) -> None:
    _print_header(f"📊 Split: {name}")
    print(f"📈 Number of rows: {len(split_ds)}")

    # Feature / schema summary
    with tqdm(total=1, desc="🔬 Analyzing features", unit="step") as pbar:
        _summarize_features(split_ds.features)
        pbar.update(1)

    # Try to print basic label distribution if a likely label column exists
    label_candidates = ["label", "labels", "category", "event_type"]
    label_col = next((c for c in label_candidates if c in split_ds.column_names), None)
    if label_col is not None:
        print(f"\n🏷️ Label distribution for column '{label_col}':")
        try:
            # `split_ds[label_col]` is a list-like of labels
            from collections import Counter

            with tqdm(total=2, desc="📊 Computing label stats", unit="step") as pbar:
                counter = Counter(split_ds[label_col])
                pbar.update(1)
                total = sum(counter.values())
                pbar.update(1)

            for lbl, cnt in counter.most_common(20):
                frac = cnt / total * 100
                print(f"  🎯 {lbl}: {cnt} ({frac:.2f}%)")
            
            # Save label distribution pie chart
            plt.figure(figsize=(8, 8))
            labels, values = zip(*counter.most_common(10)) # Top 10
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(f"Top 10 Labels ({name})")
            plt.axis("equal")
            fname = os.path.join(FIG_DIR, f"label_dist_{name}.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
            print(f"Saved label distribution pie chart to: {fname}")

        except Exception as exc:  # pragma: no cover - best-effort reporting
            print(f"  (Could not compute label distribution: {exc})")
    else:
        print("\n🤷 No obvious label column found (looked for: "
              f"{', '.join(label_candidates)}).")

    # Check for conversations and analyze length
    if "conversations" in split_ds.column_names:
        print(f"\n💬 Analyzing conversations for split '{name}'...")
        try:
             # Calculate number of turns per conversation
            conv_lengths = [len(c) for c in split_ds["conversations"]]
            
            plt.figure(figsize=(10, 6))
            plt.hist(conv_lengths, bins=20, edgecolor='black')
            plt.title(f"Distribution of Conversation Turns ({name})")
            plt.xlabel("Number of Turns")
            plt.ylabel("Count")
            fname = os.path.join(FIG_DIR, f"conv_turns_dist_{name}.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
            print(f"Saved conversation turns histogram to: {fname}")
            
            avg_len = sum(conv_lengths) / len(conv_lengths)
            print(f"  - Average conversation turns: {avg_len:.2f}")
            print(f"  - Max conversation turns: {max(conv_lengths)}")
            print(f"  - Min conversation turns: {min(conv_lengths)}")

        except Exception as exc:
            print(f"  (Could not analyze conversations: {exc})")

    # Show a few concrete examples
    _show_examples(split_ds)


def _compute_duration_stats(durations: Sequence[float]) -> None:
    arr = np.asarray(durations, dtype=float)
    _print_header("Duration statistics")
    print(f"Count: {arr.size}")
    print(f"Min:   {arr.min():.4f}")
    print(f"Max:   {arr.max():.4f}")
    print(f"Mean:  {arr.mean():.4f}")
    print(f"Median:{np.median(arr):.4f}")
    for q in (0.25, 0.5, 0.75, 0.9, 0.95):
        print(f"Quantile {q:.2f}: {np.quantile(arr, q):.4f}")


def _analyze_event_metrics(split_name: str, split_ds: Dataset) -> None:
    """Analyze and visualize event durations, counts, and densities."""
    print(f"\n⏱️ Analyzing event metrics for split '{split_name}'...")

    if "event_data" not in split_ds.column_names:
        print(f"❌ No 'event_data' column found. Cannot calculate event metrics.")
        return
        
    print("📂 Calculating event durations, counts, and densities from .npy files...")
    
    durations = []
    event_counts = []
    event_densities = [] # events per ms
    detailed_metrics = [] # List of dicts for JSON output
    
    missing_files = 0
    event_base_dir = os.path.join(DATA_DIR, "event_npy")
    
    with tqdm(total=len(split_ds), desc="⏳ Reading .npy data", unit="file") as pbar:
        for row in split_ds:
            rel_path = row["event_data"]
            full_path = os.path.join(event_base_dir, rel_path)
            
            if os.path.exists(full_path):
                try:
                    data = np.load(full_path, allow_pickle=True)
                    if data.shape == ():
                        data_dict = data.item()
                        t = data_dict.get('t')
                        # 'x' or 'y' length is event count
                        x = data_dict.get('x')
                        count = len(x) if x is not None else 0
                    else:
                        data_dict = data.item() if data.ndim == 0 and data.dtype == 'O' else data
                        t = data_dict.get('t') if isinstance(data_dict, dict) else None
                        x = data_dict.get('x') if isinstance(data_dict, dict) else None
                        count = len(x) if x is not None else 0

                    if t is not None and len(t) > 0:
                        t_min, t_max = t.min(), t.max()
                        dur_us = float(t_max - t_min)
                        dur_ms = dur_us / 1000.0
                        
                        durations.append(dur_ms)
                        event_counts.append(count)
                        
                        if dur_ms > 0:
                            density = count / dur_ms
                            event_densities.append(density)
                        else:
                            density = 0
                            event_densities.append(0)
                        
                        detailed_metrics.append({
                            "filename": rel_path,
                            "duration_ms": dur_ms,
                            "event_count": count,
                            "density_ev_per_ms": density
                        })
                    else:
                        pass 
                except Exception:
                    pass
            else:
                missing_files += 1
            pbar.update(1)
    
    if missing_files > 0:
        print(f"⚠️ Warning: {missing_files} .npy files were missing.")

    if not durations:
        print("❌ No valid event data found.")
        return

    # --- Save raw stats to text files ---
    durations_file = os.path.join(FIG_DIR, f"durations_{split_name}.txt")
    counts_file = os.path.join(FIG_DIR, f"counts_{split_name}.txt")
    densities_file = os.path.join(FIG_DIR, f"densities_{split_name}.txt")
    detailed_json_file = os.path.join(FIG_DIR, f"detailed_metrics_{split_name}.json")

    # Save raw lists line by line
    with open(durations_file, 'w') as f:
        f.write("# Event Durations (ms)\n")
        for d in durations:
            f.write(f"{d:.4f}\n")
    print(f"Saved raw durations to: {durations_file}")
    
    with open(counts_file, 'w') as f:
        f.write("# Event Counts\n")
        for c in event_counts:
            f.write(f"{c}\n")
    print(f"Saved raw counts to: {counts_file}")
    
    with open(densities_file, 'w') as f:
        f.write("# Event Densities (events/ms)\n")
        for d in event_densities:
            f.write(f"{d:.4f}\n")
    print(f"Saved raw densities to: {densities_file}")
    
    with open(detailed_json_file, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"Saved detailed metrics JSON to: {detailed_json_file}")

    # --- Figure 1: Durations ---
    _print_header(f"Duration Stats (ms) - {split_name}")
    _compute_duration_stats(durations) # Reuse helper
    
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, edgecolor='black')
    plt.title(f"Event Duration Distribution ({split_name})")
    plt.xlabel("Duration (ms)")
    plt.ylabel("Count")
    fname = os.path.join(FIG_DIR, f"figure1_duration_{split_name}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 (Duration) to: {fname}")
    
    # --- Figure 2: Event Numbers (Counts) ---
    arr_counts = np.asarray(event_counts)
    _print_header(f"Event Counts Stats - {split_name}")
    print(f"Mean: {arr_counts.mean():.2f}, Max: {arr_counts.max()}, Min: {arr_counts.min()}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(event_counts, bins=50, edgecolor='black')
    plt.title(f"Event Counts Distribution ({split_name})")
    plt.xlabel("Number of Events")
    plt.ylabel("Count")
    fname = os.path.join(FIG_DIR, f"figure2_counts_{split_name}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 (Counts) to: {fname}")

    # --- Figure 3: Event Density (events/ms) ---
    arr_dens = np.asarray(event_densities)
    _print_header(f"Event Density Stats (ev/ms) - {split_name}")
    print(f"Mean: {arr_dens.mean():.2f}, Max: {arr_dens.max():.2f}, Min: {arr_dens.min():.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(event_densities, bins=50, edgecolor='black')
    plt.title(f"Event Density Distribution ({split_name})")
    plt.xlabel("Density (events/ms)")
    plt.ylabel("Count")
    fname = os.path.join(FIG_DIR, f"figure3_density_{split_name}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 (Density) to: {fname}")

    print(f"🎉 Metrics analysis complete for '{split_name}' ✨")


def main() -> None:
    """Analyze the EventGPT dataset loaded via `data/load_dataset.py`."""
    
    # Redirect stdout to a file + console
    log_file = os.path.join(FIG_DIR, "analysis_summary.txt")
    class Tee(object):
        def __init__(self, name, mode):
            self.file = open(name, mode)
            self.stdout = sys.stdout
            sys.stdout = self
        def __del__(self):
            sys.stdout = self.stdout
            self.file.close()
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
            self.stdout.flush()

    # Only tee if we haven't already (to avoid double wrapping in interactive sessions)
    if not isinstance(sys.stdout, Tee):
         sys.stdout = Tee(log_file, "w")

    _print_header("🎯 EventGPT Dataset Overview")
    print(f"Analysis results saved to: {log_file}")
    print(ds)
    
    # Count total files in event_npy directory
    print("\n📂 Checking actual files on disk in 'event_npy'...")
    event_npy_dir = os.path.join(DATA_DIR, "event_npy")
    file_count = 0
    if os.path.exists(event_npy_dir):
        for root, _, files in os.walk(event_npy_dir):
            file_count += sum(1 for f in files if f.endswith('.npy'))
    print(f"   - Total .npy files found on disk: {file_count}")
    
    # Count instances in dataset
    total_dataset_instances = 0
    if isinstance(ds, DatasetDict):
        for split in ds.values():
            total_dataset_instances += len(split)
    elif isinstance(ds, Dataset):
        total_dataset_instances = len(ds)
    
    print(f"   - Total instances in dataset object: {total_dataset_instances}")
    
    if file_count != total_dataset_instances:
        print(f"⚠️ Mismatch: Dataset has {total_dataset_instances} entries, but found {file_count} .npy files.")
    else:
        print("✅ Counts match!")

    if isinstance(ds, DatasetDict):
        print("\n📂 Available splits:", list(ds.keys()))
        for split_name, split_ds in tqdm(ds.items(), desc="🚀 Processing splits", unit="split"):
            _analyze_split(split_name, split_ds)
            _analyze_event_metrics(split_name, split_ds)
        print("\n🎊 All splits analyzed successfully! ✨")
    elif isinstance(ds, Dataset):
        print("\n📄 Single Dataset (no explicit splits).")
        with tqdm(total=2, desc="🔥 Analyzing dataset", unit="step") as pbar:
            _analyze_split("all", ds)
            pbar.update(1)
            _analyze_event_metrics("all", ds)
            pbar.update(1)
        print("\n🎊 Dataset analysis complete! ✨")
    else:
        raise TypeError(f"Unexpected dataset type: {type(ds)}")


if __name__ == "__main__":
    main()


