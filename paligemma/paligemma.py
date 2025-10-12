from typing import List, Optional, Tuple, Union, Iterable, Dict
import torch
from PIL import Image
import numpy as np

# R, G, B
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(image: Image, size: Tuple[int, int], resample: Image.Resampling = None,
           reducing_gap: Optional[int] = None) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample = resample, reducing_gap = reducing_gap
    )
    return resized_image

def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], 
              std: Union[float, Iterable[float]])-> np.ndarray:
    mean = np.array(mean, dtype = image.dtype)
    std = np.array(std, dtype = image.dtype)
    image = (image-mean) / std
    return image

def rescale(image: np.ndarray, scale: float, dtype: np.dtype,) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    
    height, width = size[0], size[1]
    images = [
        resize(image = image, size = (height, width), resample = resample) for image in images
    ]
    
    images = [np.array(image) for image in images]
    # Pixel values between 0 and 1
    images = [rescale(image, scale = rescale_factor, dtype=np.float32) for image in images]
    # calibrates the mean to 0 and std to 1
    images = [normalize(image, mean = image_mean, std = image_std) for image in images]
    
    # [channels, height, width]
    images = [image.transpose(2,0,1) for image in images]
    
    return images


class PaliGemmaProcessor:
    
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        
        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        
        # loading some bounding box and image segementation tokens
        # this is to simulate the real PaLI-Gemma although will be unused
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # will add beginning and end tokens later
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
    
    def __call__(self, text: List[str], images: List[Image.Image],
                 padding: str = "longest", truncation: bool = True) -> dict:
        
        assert len(images) == 1 and len(text) == 1, f"Wrong format"
        
        # preprocessing into np arrays that are resized and normalized and reshaped
        pixel_values = process_images(
            images,
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,
        )
        
        # stack the values to shape [batch, channel, height, width] from a list
        pixel_values = np.stack(pixel_values, axis = 0)
        pixel_values = torch.tensor(pixel_values)
        
        # adding image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN
            )
            for prompt in text
        ]
        
        # input_ids and attention mask 
        inputs = self.tokenizer(
            input_strings,
            return_tensors = "pt",
            padding = padding,
            truncation = truncation
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}
        
        return return_data
        