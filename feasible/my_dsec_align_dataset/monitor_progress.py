#!/usr/bin/env python3
"""Monitor progress of dataset building."""

import json
import time
import os

DATASET_DIR = "/home/ps/Documents/code/EventGPT/feasible/my_dsec_align_dataset"
TOTAL_SAMPLES = 5208
TOTAL_QUESTIONS = 50
TOTAL_TARGET = TOTAL_SAMPLES * TOTAL_QUESTIONS

def get_progress():
    """Get current progress from JSON file."""
    json_path = os.path.join(DATASET_DIR, "my_dsec_align_train_1s.json")
    if not os.path.exists(json_path):
        return 0, 0
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return len(data), len(data) / TOTAL_TARGET * 100

def main():
    print("="*60)
    print("Dataset Build Progress Monitor")
    print("="*60)
    print(f"Target: {TOTAL_SAMPLES} samples × {TOTAL_QUESTIONS} questions = {TOTAL_TARGET} entries")
    print()
    
    last_count = 0
    while True:
        count, percent = get_progress()
        elapsed = count - last_count
        
        print(f"\rEntries: {count:6d} / {TOTAL_TARGET} ({percent:5.2f}%) | +{elapsed:4d} since last check", end="", flush=True)
        last_count = count
        
        if count >= TOTAL_TARGET:
            print("\n✓ Complete!")
            break
        
        time.sleep(10)

if __name__ == "__main__":
    main()
