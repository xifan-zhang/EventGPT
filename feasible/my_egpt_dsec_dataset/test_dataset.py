#!/usr/bin/env python3
"""
Test script for my-egpt-dsec-dataset.

Usage: python test_dataset.py
"""

import json
import numpy as np
import os

def test_dataset():
    """Test the dataset loading and validation."""
    dataset_path = 'my_egpt_dsec_instruction_subset.json'
    event_npy_dir = 'event_npy'

    # Load dataset
    print("Loading dataset...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Dataset loaded: {len(dataset)} entries")

    # Validate structure
    print("\nValidating structure...")
    valid_entries = 0
    for i, entry in enumerate(dataset):
        # Check required fields
        if not all(key in entry for key in ['id', 'event_data', 'conversations']):
            print(f"Entry {i}: Missing required fields")
            continue

        # Check conversations
        conv = entry['conversations']
        if len(conv) != 2:
            print(f"Entry {i}: Expected 2 conversations, got {len(conv)}")
            continue

        # Check conversation format
        if conv[0]['from'] != 'human' or conv[1]['from'] != 'gpt':
            print(f"Entry {i}: Invalid conversation format")
            continue

        # Check event file exists
        event_file = entry['event_data']
        event_path = os.path.join(event_npy_dir, event_file)
        if not os.path.exists(event_path):
            print(f"Entry {i}: Event file {event_file} not found")
            continue

        valid_entries += 1

        # Load and validate event data (first few entries)
        if i < 5:
            try:
                event_data = np.load(event_path, allow_pickle=True).item()
                required_keys = ['p', 't', 'x', 'y']
                if not all(key in event_data for key in required_keys):
                    print(f"Entry {i}: Event data missing required keys")
                    continue

                num_events = len(event_data['t'])
                time_range = event_data['t'].max() - event_data['t'].min()
                print(f"  Entry {i}: {num_events} events, time range: {time_range} µs")

            except Exception as e:
                print(f"Entry {i}: Error loading event data: {e}")
                continue

    print(f"\nValidation complete: {valid_entries}/{len(dataset)} valid entries")

    # Show sample entries
    print("\nSample entries:")
    for i in range(min(3, len(dataset))):
        entry = dataset[i]
        print(f"\nEntry {i}:")
        print(f"  ID: {entry['id']}")
        print(f"  Event file: {entry['event_data']}")
        question_clean = entry['conversations'][0]['value'].replace('<event>\n', '')
        print(f"  Question: {question_clean}")
        print(f"  Answer: {entry['conversations'][1]['value']}")

    print("\nDataset test completed successfully!")

if __name__ == "__main__":
    test_dataset()
