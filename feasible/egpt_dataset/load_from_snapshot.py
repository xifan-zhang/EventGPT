"""
Example: How to load a dataset that was downloaded using snapshot_download

When you use snapshot_download, it downloads raw files from Hugging Face,
but doesn't create a format that Dataset.load_from_disk() can read.

Here are several ways to load the dataset:
"""

from datasets import load_dataset, Dataset
import json
import os

DATA_DIR = "/mnt/hdd/data/EventGPT-datasets"
JSON_FILE = os.path.join(DATA_DIR, "dataset_info.json")

# Method 1: Load directly from JSON file using load_dataset
# This is the recommended approach
print("Method 1: Using load_dataset with JSON file")
ds1 = load_dataset('json', data_files=JSON_FILE)
# Returns DatasetDict with 'train' split
if isinstance(ds1, dict):
    ds1 = ds1['train']
print(f"✅ Loaded {len(ds1)} examples")
print(f"Columns: {ds1.column_names}\n")

# Method 2: Load JSON manually and convert to Dataset
print("Method 2: Manual JSON loading")
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
ds2 = Dataset.from_list(data)
print(f"✅ Loaded {len(ds2)} examples")
print(f"Columns: {ds2.column_names}\n")

# Method 3: Use load_dataset with local_files_only (if dataset was cached)
# This works if the dataset was previously loaded and cached
print("Method 3: Using load_dataset with local_files_only")
try:
    ds3 = load_dataset("XduSyL/EventGPT-datasets", local_files_only=True)
    print(f"✅ Loaded from cache")
except Exception as e:
    print(f"❌ Could not load from cache: {e}\n")

# Method 4: If you want to save it in Arrow format for faster loading later
print("Method 4: Saving to Arrow format for faster future loading")
if not os.path.exists(os.path.join(DATA_DIR, "state.json")):
    # Save in Arrow format
    save_path = os.path.join(DATA_DIR, "arrow_format")
    ds1.save_to_disk(save_path)
    print(f"✅ Saved to {save_path}")
    print(f"   Next time, use: Dataset.load_from_disk('{save_path}')")
else:
    print("Dataset already in Arrow format")

