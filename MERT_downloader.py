from huggingface_hub import snapshot_download
import os

# 1. Define where to save

# MODEL_ID = "m-a-p/MERT-v1-330M"
MODEL_ID = "m-a-p/MERT-v1-95M"

local_folder = str(MODEL_ID)
os.makedirs(local_folder, exist_ok=True)

print(f"Downloading MERT model to {local_folder}...")

# 2. Download the ENTIRE repository (weights + custom code)
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=local_folder,
    local_dir_use_symlinks=False,  # Important: ensures actual files are downloaded, not links
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional: skip TensorFlow/Flax weights to save space
)

print("Download complete!")