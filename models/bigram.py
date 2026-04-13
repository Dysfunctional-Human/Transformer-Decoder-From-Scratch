import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple

"""
Training sketch for BigramLanguageModel:

    1. Start with a batch of token indices x and targets y from the dataset. Each row in
       x is a short context of token IDs, and the matching row in y is the same sequence 
       shifted by one token.

    2. Pass x through the embedding table. For this model, each token ID looks up one
       row from a learned matrix of size [V×V], so the output is a tensor of logits 
       with shape roughly [B,T,V].

    3. Compare those logits to the true next tokens y using cross-entropy loss. That 
       loss measures how wrong the model’s next-token predictions are.

    4. Call backward on the loss. PyTorch computes gradients for the embedding matrix 
       entries that were used in the forward pass.

    5. Call the optimizer step. That updates the embedding weights so the next time those
       token IDs appear, their looked-up rows produce better next-token logits.

    So the learning is not happening in a hidden stack of layers. It is happening in the
    embedding weight matrix itself, which is being optimized to become a 
    token-to-next-token transition table.

    A useful mental model is this: if the model sees token 42 often followed by token 17,
    the row for token 42 will gradually shift to assign a higher logit to token 17 than
    to other tokens. That is the “knowledge” the model learns.
"""

class BigramLanguageModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        endoftext_token_id: int | None = None
    ):
        """Initializes the Bigram Language model. One of the most basic language models

        Args:
            vocab_size (int): Vocabulary size of the dataset
            endoftext_token_id (int | None): Token id for "<|endoftext|>"
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # (num_embeddings, embed_size)
        self.endoftext_token_id = endoftext_token_id
        # Each token directly reads off the logits for the next token from a lookup table
        # For this model, embed_size == vocab_size
        # Note that Embedding isn't just a lookup table. This is a learnable weight matrix that gets updated via back propagation in the training loop
        
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """A single forward pass in the model for a batch of data

        Args:
            idx (torch.Tensor): matrix of indices of input tokens
            targets (torch.Tensor | None): matrix of indices of target tokens

        Returns:
            logits (torch.Tensor): Output logits of the model
            loss (torch.Tensor | None): Loss function value for the given input and targets
        """

        # idx and targets -> [batch_size, context_window_len]
        logits = self.token_embedding_table(idx)
        # logits -> [batch_size, context_window_len, embed_size]
        
        if targets is None:
            loss = None
        else:
            batch, context, embed = logits.shape
            
            logits = logits.view(batch*context, embed)  # logits -> [batch_size*context_window_len, embed_size]
            targets = targets.view(batch*context)   # targets -> [batch_size, context_window_len]

            loss = F.cross_entropy(logits, targets) # loss -> [1] === single floating point number
        
        return logits, loss

    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int
    ) -> torch.Tensor:
        """Generates response from the model based on an initial input. The model keeps generating until either it encounter <|endoftext|> or until
        it hits number of tokens generated equal to max_new_tokens

        Args:
            idx (torch.Tensor): Index of input token(s)
            max_new_tokens (int): Maximum number of new tokens model should generate
        
        Returns:
            idx (torch.Tensor): Indices of all input + newly generated tokens
        """
        # idx -> [batch_size, context_window_len]
        # Ensure batch size is 1 for generation
        assert idx.shape[0] == 1, "generate method only supports batch_size=1"
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx, targets=None)
            # logits -> [batch_size, context_window_len, embed_size] Here logits is 3 dimensional since target is None in the forward method
            logits = logits[:, -1, :]   # Focus only on the previous token (not the entire context window len, only the last time step)
            # logits -> [batch_size, embed_size]
            probs = F.softmax(logits, dim=1)    # probs -> [batch_size, embed_size]
            idx_next = torch.multinomial(probs, num_samples=1)  # idx_next -> [batch_size, 1]
            # Rather than picking the most probable, sampling from multinomial distribution
            idx = torch.cat((idx, idx_next), dim=1)
            # Stop generation when <|endoftext|> token is produced
            if self.endoftext_token_id is not None and idx_next[0, 0].item() == self.endoftext_token_id:
                break
        return idx
        
if __name__ == "__main__":
    from utils.utils import save_model, load_model
    import shutil
    from pathlib import Path
    import tempfile
    
    torch.manual_seed(1337)
        
    bigram = BigramLanguageModel(vocab_size=52)
    print("Bigram model: ", bigram)
    print("State Dictionary of model:", bigram.state_dict())
    
    idx = torch.randint(high=52, size=(4,8))
    targets = torch.randint(high=52, size=(4,8))
    logits, loss = bigram.forward(idx=idx, targets=targets)
    print(f"Sample forward pass result: \nOutput logits: {logits}, \nLoss: {loss.item(): .2f}")
    
    idx = torch.randint(high=52, size=(1,1))
    idx = bigram.generate(idx=idx, max_new_tokens=32)
    print("Text generation sample output: ", idx)
    
    # temporary directory for demo 
    temp_base = Path(tempfile.mkdtemp(prefix="bigram_demo_"))
    demo_dir = temp_base / "bigram"
    
    # Saving model and results in the temporary demo directory
    save_model(
        model=bigram,
        model_name="sample_bigram_model.pt",
        target_dir=str(demo_dir),
        results={"dummy": [2.4]},
    )

    loaded_model, _ = load_model(
        model=bigram,
        target_model_path=str(demo_dir / "sample_bigram_model" / "sample_bigram_model.pt"),
        model_results_path=str(demo_dir / "sample_bigram_model" / "results.pt"),
    )

    print("Loaded model's state dict: ", loaded_model.state_dict())

    # Clean up only the temporary demo directory we created
    if temp_base.exists():
        shutil.rmtree(temp_base)

