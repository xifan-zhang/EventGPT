# from huggingface_hub import login
# login()

from datasets import load_dataset
import os
import time
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("XduSyL/EventGPT-datasets", cache_dir=".")


def download_progress_callback(t):
    """Progress callback for snapshot_download"""
    # This function will be called with download progress information
    # t is a tqdm progress bar object that gets updated automatically
    pass


print("🚀 Starting dataset download...")
print("📦 Repository: XduSyL/EventGPT-datasets")
print("📁 Target directory: /mnt/hdd/data/EventGPT-datasets")

start_time = time.time()

try:
    # snapshot_download has built-in progress bars with tqdm
    snapshot_download(
        repo_id="XduSyL/EventGPT-datasets",
        repo_type="dataset",
        local_dir="/mnt/hdd/data/EventGPT-datasets",
        # ignore_patterns=["*.arrow"],
    )
except Exception as e:
    print(f"❌ Download failed: {e}")
    raise

end_time = time.time()
elapsed_time = end_time - start_time

# Get the size of downloaded data
total_size = 0
for root, dirs, files in os.walk("/mnt/hdd/data/EventGPT-datasets"):
    for file in files:
        try:
            total_size += os.path.getsize(os.path.join(root, file))
        except OSError:
            pass

print(f"\n✅ Dataset downloaded successfully!")
print(f"📊 Total size: {total_size / (1024*1024*1024):.2f} GB")
print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
print(f"📈 Average speed: {total_size / (1024*1024) / elapsed_time:.2f} MB/s" if elapsed_time > 0 else "📈 Speed: N/A")
print("🎉 Download complete! Dataset ready for use ✨")