import torch
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from configs import generationConfig, config
import torch.nn.functional as F
from scripts.trainer import get_tokenizer_artifacts
from tokenizers import Tokenizer
from typing import List
from utils.utils import load_model
from tokenizers.decoders import ByteLevel

def batch_generate(
    indices: torch.Tensor,
    model: torch.nn.Module,
    max_new_tokens: int
) -> torch.Tensor:
    """_summary_

    Args:
        indices (torch.Tensor): _description_
        model (torch.nn.Module): _description_

    Returns:
        torch.Tensor: _description_
    """
    
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            idx_cropped = indices
            if generationConfig.MODEL_NAME != "bigram":
                idx_cropped = indices[:, -model.context_window_len:]
            logits, _ = model.forward(idx_cropped, targets = None)
            logits = logits[:, -1, :]
            if generationConfig.MODEL_NAME == "decoderv2":
                logits = logits / max(model.temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            indices = torch.cat((indices, idx_next), dim=1)
            
            if model.endoftext_token_id is not None and idx_next[0, 0].item() == model.endoftext_token_id:
                break
    return indices

def decode(
    num_list: List[int],
    bpe_tokenizer: Tokenizer
) -> str:
    """Decodes a list of numbers into their original story.

    Args:
        num_list (list[int]): list of token IDs to decode.

    Returns:
        str: Decoded story with BPE markers cleaned.
    """
    token_ids = [t for t in num_list if t != -100 and 0 <= t < bpe_tokenizer.get_vocab_size()]
    # decoded = bpe_tokenizer.decode(token_ids, skip_special_tokens=True)
    # cleaned = decoded.replace("Ġ", " ")
    # cleaned = cleaned.replace("Ċ", "")
    decoded = bpe_tokenizer.decode(token_ids, skip_special_tokens=True)
    return decoded

def encode(
    text: str,
    bpe_tokenizer: Tokenizer
) -> torch.Tensor:
    encoding = bpe_tokenizer.encode(text, add_special_tokens=False)
    token_ids = encoding.ids
    if bpe_tokenizer.token_to_id("<|endoftext|>") is not None:
        token_ids.append(bpe_tokenizer.token_to_id("<|endoftext|>"))
    return token_ids

if __name__ == "__main__":
    bpe_tokenizer, _, _, _ = get_tokenizer_artifacts()
    
    if bpe_tokenizer is None:
        raise FileNotFoundError("No BPE tokenizer found")
    bpe_tokenizer.decoder = ByteLevel() 
    
    model_class = generationConfig.MODEL
    
    model = model_class(
        vocab_size=bpe_tokenizer.get_vocab_size(),
        EMBED_SIZE=generationConfig.EMBED_SIZE,
        CONTEXT_WINDOW_LEN=generationConfig.CONTEXT_WINDOW_LEN,
        NUM_HEADS=generationConfig.NUM_HEADS,
        NUM_BLOCKS=generationConfig.NUM_BLOCKS,
        DROP_PROB=generationConfig.DROP_PROB,
        SAMPLE_TEMPERATURE=generationConfig.SAMPLE_TEMPERATURE,
        DEVICE=generationConfig.DEVICE,
        endoftext_token_id=bpe_tokenizer.token_to_id("<|endoftext|>")
    )
    
    model = load_model(model=model, target_model_path=generationConfig.MODEL_PATH)
    print("Model Loaded")
    
    user_input = input("Starting text for the story (press enter to leave blank)\n")
    start_tokens = encode(text=user_input, bpe_tokenizer=bpe_tokenizer)
    start_tokens = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0)
    
    generated_ids = batch_generate(indices=start_tokens, model=model, max_new_tokens=100)
    output_text = decode(num_list=generated_ids[0].tolist(), bpe_tokenizer=bpe_tokenizer)
    print("\nGenerated story:\n", output_text)