import re
import unicodedata
import numpy as np  
from pprint import pprint
import torch
from utils.utils import get_batch
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path

def save_tokenizer_artifacts(
    target_dir: str,
    vocab: List[str],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    sep: str = "<|endoftext|>",
    unk: str = "<|unk|>"
) -> None:
    """Save important tokenizer artifacts - vocab, stoi, itos for future use

    Args:
        target_dir (str): Location to store artifacts
        vocab (List[str]): Vocabulary of the dataset
        stoi (Dict[str, int]): string-to-int dictionary
        itos (Dict[int, str]): int-to-string dictionary
        sep (str, optional): Separator token (token used to end a story). Defaults to "<|endoftext|>".
        unk (str, optional): Unknown token (for words not in vocab). Defaults to "<|unk|>".
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    
    with open(target / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=True)
    
    with open(target / "stoi.json", "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=True)
    
    # In a JSON object, keys are always strings    
    itos_json = {str(k): v for k, v in itos.items()}
    with open(target / "itos.json", "w", encoding="utf-8") as f:
        json.dump(itos_json, f, ensure_ascii=True)
        
    with open(target / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sep": sep,
                "unk_token": unk
            }, f, ensure_ascii=True
        )
    print("Tokenizer artifacts successfully saved at: ", target_dir)
        
def load_tokenizer_artifacts(
    target_dir: str
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Loads Tokenizer artifacts - vocab, stoi, itos

    Args:
        target_dir (str): Directory where artifacts are stored

    Returns:
        Tuple[List[str], Dict[str, int], Dict[int, str]]: vocab, stoi, itos
    """
    target = Path(target_dir)
    
    with open(target / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    with open(target / "stoi.json", "r", encoding="utf-8") as f:
        stoi = json.load(f)
    
    with open(target / "itos.json", "r", encoding="utf-8") as f:
        itos_json = json.load(f)
    
    itos = {int(k): v for k, v in itos_json.items()}
    print("Tokenizer artifacts successfully loaded")
    return vocab, stoi, itos

def build_shared_tokenizer(
    dataset_paths: List[str],
    sep: str = "<|endoftext|>",
    unk_token: str = "<|unk|>"
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Building a shared tokenizer from multiple datasets (mainly so train and val datasets can have common vocabulary, and stoi-itos mappings)

    Args:
        dataset_paths (List[str]): List of dataset paths
        sep (str, optional): Separator token (token used to end a story). Defaults to "<|endoftext|>".
        unk_token (str, optional): Unknown token (for words not in vocab). Defaults to "<|unk|>".

    Returns:
        Tuple[List[str], Dict[str, int], Dict[int, str]]: vocab, stoi, itos
    """
    all_tokens: List[str] = []
    for p in dataset_paths:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read().lower()
        _, _, _, tokens = Dataset.clean_data_from_raw(raw_text=raw, sep=sep)
        all_tokens.extend(tokens)

    vocab = sorted(set(all_tokens) | {sep, unk_token}) 
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    return vocab, stoi, itos
    
class Dataset():
    """Dataset class for the Decoder model
    """
    def __init__(
        self, 
        data_path: str, 
        device = "cuda", 
        debug: bool = False,
        vocab: Optional[List[str]] = None,
        stoi: Optional[Dict[str, int]] = None,
        itos: Optional[Dict[int, str]] = None,
        unk_token: str = "<|unk|>"
    ):
        """Initialize the Dataset class

        Args:
            data_path (str): Data storage location
            device (str, optional): Device to load the data on ('cpu' or 'cuda'). Defaults to "cuda".
            debug (bool, optional): Whether to print extra debug statements. Defaults to False.
            vocab (Optional[List[str]], optional): Shared vocabulary to use (if any). Defaults to None.
            stoi (Optional[Dict[str, int]], optional): Shared stoi dictionary to use (if any). Defaults to None.
            itos (Optional[Dict[int, str]], optional): Shared itos dictionary to use (if any). Defaults to None.
            unk_token (str, optional): Unknown token (for words not in vocab). Defaults to "<|unk|>".
        """
        self.data_path = data_path
        self.raw_text = None
        try:
            with open(data_path, "r", encoding='utf-8') as f:
                self.raw_text = f.read().lower()
        except Exception as e:
            raise ValueError(f"Error loading data from the given file: {self.data_path}. " 
                             f"Ensure the data path is correct and data is not corrupted. "
                             f"Logs: {e}"
                             ) from e
        if device not in ("cuda", "cpu"):
            raise ValueError("device must either be 'cpu' or 'cuda'")
        
        self.sep = "<|endoftext|>"
        self.unk_token = unk_token
        self.clean_text, self.stories, self.encoded_stories, self.all_tokens = self.clean_data()

        
        provided = [vocab is not None, stoi is not None, itos is not None]
        if any(provided) and not all (provided):
            raise ValueError("Provide vocab, itos and stoi together, or provide none of them.")
        
        if all(provided):
            self.vocab = vocab
            self.itos = itos
            self.stoi = stoi
            
            if self.sep not in self.stoi:
                raise ValueError("Shared tokenizer is missing '<|endoftext|>' token.")
            if self.unk_token not in self.stoi:
                raise ValueError(f"Shared tokenizer is missing {self.unk_token} token.")
        else:
            self.vocab = sorted(set(self.all_tokens) | {self.sep, self.unk_token})
            self.stoi = {ch:i for i, ch in enumerate(self.vocab)}
            self.itos = {i:ch for i, ch in enumerate(self.vocab)}
        
        if debug:
            if self.decode_story(self.encode_story(self.clean_text)) != self.clean_text:
                raise ValueError("Encode/Decode round‑trip validation failed. Check the data and the encode-decode functions")
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        print("Dataset being initialized on device: ", self.device)
        
    @staticmethod   # Making this static so it can be used outside the class without object initialization    
    def clean_data_from_raw(
        raw_text: str, 
        sep: str = "<|endoftext|>"
    ) -> Tuple[str, List[str], List[List[str]], List[str]]:
        """Data cleaning like removing unnecessary characters
        Args:
            raw_text (str): raw text to be cleaned
            sep (str, optional): The separator used between stories. Defaults to "<|endoftext|>"

        Returns:
            text_clean (str): Cleaned final text
            stories (list[str]): list of cleaned stories
            encoded_stories (list[list[str]]):  list of cleaned formatted stories
            total_tokens (list[str]): All the tokens from cleaned final text
        """
        text_clean = unicodedata.normalize("NFKC", raw_text)
        replace_map = str.maketrans({
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
            "―": "-",
            "…": "...",
            "\xa0": " ",   # NBSP
            "\u2009": " ", # thin space
            "\u200a": " ", # hair space
            "\u200b": "",  # zero-width space
        })
        
        text_clean = text_clean.translate(replace_map)
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'\"-()/\n\t")
        
        stories = []
        for chunk in text_clean.split(sep=sep):
            chunk = "".join(ch for ch in chunk if ch in allowed)
            chunk = re.sub(r"[ ]{2,}", " ", chunk)  # collapse repeated spaces
            chunk = chunk.strip()
            if chunk:
                stories.append(chunk)
        
        text_clean = f"{sep}".join(stories) + sep
        
        encoded_stories = [list(s) + [sep] for s in stories]
        total_tokens = [ch for s in stories for ch in s] + [sep] * len(stories)
        
        return text_clean, stories, encoded_stories, total_tokens    
    
    def clean_data(self) -> Tuple[str, List[str], List[List[str]], List[str]]:
        """Wrapper over clean_data_from_raw. Kept for backwards compatibility

        Returns:
            Tuple[str, List[str], List[List[str]], List[str]]: _description_
        """
        return Dataset.clean_data_from_raw(
            raw_text=self.raw_text,
            sep=self.sep
        )

    
    def encode_story(
        self, 
        text: str
    ) -> List[int]:
        """Encodes a story string into a list of numbers 

        Args:
            text (str): story to be encoded

        Returns:
            list[int]: encoded story
        """
        num_list = []
        c = 0
        while c < len(text):
            if text.startswith(self.sep, c):
                num_list.append(self.stoi[self.sep])
                c += len(self.sep) - 1
            else:
                num_list.append(self.stoi.get(text[c], self.stoi[self.unk_token]))
            c += 1
        return num_list
    
    def decode_story(
        self, 
        num_list: List[int]
    ) -> str:
        """Decodes a list of numbers into their original story

        Args:
            num_list (list[int]): list of numbers to be decoded

        Returns:
            str: decoded story
        """
        return ''.join([self.itos[n] for n in num_list])
    
    def info(self):
        """Displays important dataset stats
        """
        print(f"length of raw text dataset (in characters): {len(self.raw_text)}")
        print("Sample story from raw text: ")
        pprint(self.raw_text[:2000].split(sep=self.sep)[0])
        print("-"*25, 'data cleaning', '-'*25)
        print(f"length of clean text after preprocessing (in characters): {len(self.clean_text)}")
        print("Total number of stories: ", len(self.encoded_stories))
        print("Average number of characters per story: ", np.mean([len(story) for story in self.encoded_stories]))
        print("Length of vocabulary: ", len(self.vocab))
        print("Vocabulary:\n", self.vocab)
        print("Character to Number mapping:\n", self.stoi)
        print("Number to Character mapping:\n", self.itos)
        
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a sample batch of input(x) and target(y) variables

        Returns:
            x (torch.Tensor): Input variable
            y (torch.Tensor): Target variable
        """
        # Encode the full cleaned text into token IDs first to avoid slicing in the middle of a separator token.
        encoded_full = self.encode_story(self.clean_text)
        # Take a slice of token IDs (e.g., first 1000 tokens) for a manageable sample.
        sample_tokens = encoded_full[:1000]
        sample_data = torch.tensor(sample_tokens, dtype=torch.long)
        sample_data = sample_data.to(device=self.device)
        x, y = get_batch(split="train", train_data=sample_data, val_data=None, context_window_len=8, batch_size=4)
        print("Shape of x (input variables): ", x.shape)
        print("Shape of y (target variables): ", y.shape)
        print("Device of x/y batch: ", x.device)
        return x, y
    
    def view(self):
        """View the data as it would be provided to the model
        """
        x, y = self.sample()
        
        for i in range(x.shape[0]): # batch dimension
            for j in range(x.shape[1]): # time dimension
                context = x[i][:j+1]
                target = y[i][j]    # since y is shifted to the right by 1
                print(f"When input is: {context.tolist()} the target is: {target}")
                print(f"Context: {self.decode_story(context.tolist())} Target: {self.decode_story([target.tolist()])}")
            print("----------x----------")     

if __name__ == "__main__":
    mock_data = Dataset(data_path="dataset/TinyStories_train_100k.txt", device="cuda", debug=True)   
    mock_data.info()
    mock_data.view()