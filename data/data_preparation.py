import re
import unicodedata
import numpy as np  
from pprint import pprint
import torch
from utils.utils import get_batch
from typing import List, Tuple

class Dataset():
    """Dataset class for the Decoder model
    """
    def __init__(
        self, 
        data_path: str, 
        device = "cuda", 
        debug: bool = False
    ):
        """Initializes the dataset class

        Args:
            data_path (str): File path for the dataset

        Raises:
            ValueError: Incorrect file path/ Data corrupted
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
        self.clean_text, self.stories, self.encoded_stories, self.all_tokens = self.clean_data()
        self.vocab = sorted(set(self.all_tokens))
        self.stoi = {ch:i for i, ch in enumerate(self.vocab)}
        self.itos = {i:ch for i, ch in enumerate(self.vocab)}
        if debug:
            if self.decode_story(self.encode_story(self.clean_text)) != self.clean_text:
                raise ValueError("Encode/Decode round‑trip validation failed. Check the data and the encode-decode functions")
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        print("Dataset being initialized on device: ", self.device)
        
    def clean_data(self) -> Tuple[str, List[str], List[List[str]], List[str]]:
        """Data cleaning like removing unnecessary characters

        Returns:
            text_clean (str): Cleaned final text
            stories (list[str]): list of cleaned stories
            encoded_stories (list[list[str]]):  list of cleaned formatted stories
            total_tokens (list[str]): All the tokens from cleaned final text
        """
        text_clean = unicodedata.normalize("NFKC", self.raw_text)
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
        for chunk in text_clean.split(sep=self.sep):
            chunk = "".join(ch for ch in chunk if ch in allowed)
            chunk = re.sub(r"[ ]{2,}", " ", chunk)  # collapse repeated spaces
            chunk = chunk.strip()
            if chunk:
                stories.append(chunk)
        
        text_clean = f"{self.sep}".join(stories) + self.sep
        
        encoded_stories = [list(s) + [self.sep] for s in stories]
        total_tokens = [ch for s in stories for ch in s] + [self.sep] * len(stories)
        
        return text_clean, stories, encoded_stories, total_tokens
    
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
                num_list.append(self.stoi[text[c]])
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