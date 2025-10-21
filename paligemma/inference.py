# some nice imports for the inference
from PIL import Image
import torch
import fire

from paligemma import PaliGemmaProcessor
from gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    # load the image
    image = Image.open(image_file_path)
    # make it a list
    images = [image]
    prompts = [prompt]
    # calls the paligemmaprocesser class with the prompts and images
    # we get back a dictionary of pixel values, token ids, and the attention mask
    model_inputs = processor(text=prompts, images=images)
    
    # sends each tensor to the cuda
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    
    # should be bfloat.16
    pixel_values = model_inputs["pixel_values"].to(next(model.parameters()).dtype)

    # initializes a kvcache class
    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    # sequence of a single token generation
    for _ in range(max_tokens_to_generate):
        
        # call the gemma LLM 
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        
        # Alternatively just take the most likely token
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        # update the attention mask to have one more token
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode from token ids to tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # FINAL LINE YAY
    print(prompt + decoded)


def _sample_top_p(probs: torch.Tensor, p: float):
    # (B, vocab_size)
    # sorts the probability scores and their indexes
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    # cumulatively adds up the probability [0.4, 0.3, 0.25, 0.03, 0.02]
    # becomes [0.4, 0.7, 0.95, 0.98, 1.0]
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    #[0, 0.4, 0.7, 0.95, 0.98]
    # [F, F, F, T, T]
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    # [0.4, 0.3, 0.25, 0.0, 0.0]
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    # divide each one by the sum of the tensor
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"

    # use cuda if linux/windows, mps for macOS
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model")
    
    # load our model with pre-trained weights
    model, tokenizer = load_hf_model(model_path, device)
    
    # send to cuda in evaluation mode, not training
    model = model.to(device).eval()


    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    
    # initalize our paligemma processor that concatenates the tokens for our gemma model
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)


    print("Running inference")
    # no grad because we aren't training
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )

# we read in the arguments passed from the sh file
if __name__ == "__main__":
    fire.Fire(main)