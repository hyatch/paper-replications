import os
import json
import glob
from typing import Tuple
import torch

from safetensors import safe_open
from transformers import AutoTokenizer
# NEW: Import the downloader function
from huggingface_hub import snapshot_download

# Assuming your custom model class is in gemma.py
from gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Loads a custom PaliGemma model by automatically downloading files from a
    Hugging Face Hub ID to a local cache, then loading them manually.
    """
    # --- AUTOMATIC DOWNLOAD STEP ---
    # This will download the repo to a local cache directory and return its path.
    # If the files are already cached, it will just return the path instantly.
    print(f"Downloading/finding model files for '{model_path}'...")
    local_model_path = snapshot_download(
        repo_id=model_path,
        local_dir_use_symlinks=False # This line MUST be present
    )
    print(f"Model files located at: {local_model_path}")

    # --- MANUAL LOADING STEP (using the new local_model_path) ---

    # 1. Load the tokenizer from the local path
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="right")
    
    # 2. Find and load all weight files
    safetensors_files = glob.glob(os.path.join(local_model_path, "*.safetensors"))
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # 3. Load the model's configuration
    with open(os.path.join(local_model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # 4. Create the model architecture
    model = PaliGemmaForConditionalGeneration(config)

    # 5. Load the weights into the model
    model.load_state_dict(tensors, strict=False)

    # 6. Move the model to the correct device and tie weights
    model.to(dtype=torch.bfloat16, device = device)
    model.tie_weights()

    return (model, tokenizer)