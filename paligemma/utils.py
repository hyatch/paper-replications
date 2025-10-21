from gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import torch
import glob
from safetensors import safe_open
from typing import Tuple
import os
# Add this import
from huggingface_hub import snapshot_download


# loads the paligemma model weights from huggingface api
def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # downloads the files from the designated repository onto the local machine
    local_model_path = snapshot_download(repo_id=model_path)

    # Load the tokenizer from the local cache
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files in the local cache
    safetensors_files = glob.glob(os.path.join(local_model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's configuration onto our cache
    # this includes layer dimensions, attention head declarations, etc
    with open(os.path.join(local_model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config)

    # Load the a state dictionary of our tensors onto our model for the weights
    model.load_state_dict(tensors, strict=False)
    
    # compress our model weights for better run time and send to cuda
    model = model.to(torch.bfloat16).to(device)

    # Tie weights between the embedding layer and the output layer for efficiency
    model.tie_weights()

    return (model, tokenizer)