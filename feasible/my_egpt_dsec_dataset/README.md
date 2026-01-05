# My-EventGPT-DSEC-Dataset

Custom built EventGPT DSEC dataset for event description tasks.

## Project Structure

```
my_egpt_dsec_dataset/
├── build_my_egpt_dsec_dataset.py    # Main script to build the dataset
├── my_egpt_dsec_instruction_subset.json  # Dataset JSON file
├── event_npy/                       # Event data file directory
│   ├── synthetic_000000.npy        # Event slice data
│   ├── synthetic_000001.npy
│   └── ...
└── README.md                        # Project documentation
```

## Dataset Format

The dataset uses a format compatible with EventGPT:

```json
{
  "id": "my_dsec_00000000",
  "event_data": "synthetic_000000.npy",
  "conversations": [
    {
      "from": "human",
      "value": "<event>\nWhat are the key elements in this scene?"
    },
    {
      "from": "gpt",
      "value": "Based on the event data, the scene shows key elements in this scene. Confidence: 0.984"
    }
  ]
}
```

## Event Data Format

Each `.npy` file contains a dictionary with the following keys:
- `p`: Polarity array (0 or 1)
- `t`: Timestamp array (microseconds)
- `x`: x-coordinate array (0-345)
- `y`: y-coordinate array (0-259)

## Build Steps

### 1. Prepare Dependencies
```bash
pip install numpy h5py torch transformers qwen-vl-utils pillow tqdm
```

### 2. Run Build Script (Build Event Data)
```bash
cd /home/ps/Documents/code/EventGPT/feasible/my_egpt_dsec_dataset
python build_my_egpt_dsec_dataset.py
```

### 3. Generate Question-Answer Pairs (Using Qwen)
```bash
python generate_answers_qwen.py --dataset_dir /path/to/dataset
```

## Generating Answers using Qwen

The `generate_answers_qwen.py` script uses Qwen-VL models to generate high-quality question-answer pairs for the DSEC dataset.

### Features
- Supports multiple Qwen models (Qwen2-VL, Qwen3-VL)
- Automatically selects the question with the highest confidence
- Filters to keep only high-confidence samples (>= 90%)
- Supports resuming and processing specific sequences

### Usage

```bash
# Use default model (Qwen3-VL-8B)
python generate_answers_qwen.py --dataset_dir /mnt/hdd/data/my_egpt_dsec_dataset

# Use Qwen2-VL-7B (via alias)
python generate_answers_qwen.py --model_id qwen2-7b --dataset_dir /mnt/hdd/data/my_egpt_dsec_dataset

# Use specific model ID
python generate_answers_qwen.py --model_id Qwen/Qwen2-VL-2B-Instruct

# Process specific sequence (e.g., thun_01_a)
python generate_answers_qwen.py --sequence thun_01_a --dsec_root /mnt/hdd/data/DSEC/test

# Limit number of samples for testing
python generate_answers_qwen.py --max_samples 100 --dataset_dir /mnt/hdd/data/my_egpt_dsec_dataset
```

### Supported Model Aliases

| Alias | Model ID | Type |
|-------|----------|------|
| `qwen3-8b` | Qwen/Qwen3-VL-8B-Instruct | Qwen3-VL |
| `qwen2-7b` | Qwen/Qwen2-VL-7B-Instruct | Qwen2-VL |
| `qwen2-2b` | Qwen/Qwen2-VL-2B-Instruct | Qwen2-VL |
| `qwen2-72b` | Qwen/Qwen2-VL-72B-Instruct | Qwen2-VL |

## Dataset Features

- **Slicing Method**: Continuous 50ms time window (50,000 µs)
- **Question Source**: Top 50 frequent questions from DSEC subset
- **Answer Generation**: Simulated Qwen-70B response (or actual if using script), includes confidence
- **Compatibility**: Fully compatible with EventGPT dataset format

## Usage Example

```python
import json

# Load dataset
with open('my_egpt_dsec_instruction_subset.json', 'r') as f:
    dataset = json.load(f)

# Access first entry
entry = dataset[0]
print(f"ID: {entry['id']}")
print(f"Question: {entry['conversations'][0]['value']}")
print(f"Answer: {entry['conversations'][1]['value']}")
```

## Notes

- The current version contains 100 synthetic entries for demonstration.
- To process real DSEC data, HDF5 compression plugin issues need to be resolved.
- Answer generation in the build script is simulated; for actual generation use `generate_answers_qwen.py`.
