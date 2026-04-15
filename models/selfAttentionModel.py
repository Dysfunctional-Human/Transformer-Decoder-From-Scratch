import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple

class SelfAttentionLanguageModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        EMBED_SIZE: int,
        endoftext_token_id: int | None = None,
        **kwargs
    ):
        """Initializes the Self Attention Model

        Args:
            vocab_size (int): Size of dataset vocabulary
            n_embed (int): Embedding dimension
            endoftext_token_id (int | None, optional): Token_id for endoftext token to stop generation. Defaults to None.
        """
        super().__init__()
        self.n_embed = EMBED_SIZE
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embed) # (num_embeddings, embed_size)
        self.endoftext_token_id = endoftext_token_id
        self.lm_head = nn.Linear(in_features=self.n_embed, out_features=vocab_size)
          
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
        tok_emb = self.token_embedding_table(idx)
        # tok_emb -> [batch_size, context_window_len, embed_size]
        logits = self.lm_head(tok_emb)
        # logits -> [batch_size, context_window_len, vocab_size]
        
        if targets is None:
            loss = None
        else:
            batch, context, vocab_size = logits.shape
            
            logits = logits.view(batch*context, vocab_size)  # logits -> [batch_size*context_window_len, vocab_size]
            targets = targets.view(batch*context)   # targets -> [batch_size, context_window_len]

            loss = F.cross_entropy(logits, targets) # loss -> [1] === single floating point number
        
        return logits, loss

    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int
    ) -> torch.Tensor:
        """Generates response from the model based on an initial input. The model keeps generating until either it encounters <|endoftext|> or until
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
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits, _ = self.forward(idx, targets=None)
                # logits -> [batch_size, context_window_len, vocab_size] Here logits is 3 dimensional since target is None in the forward method
                logits = logits[:, -1, :]   # Focus only on the previous token (not the entire context window len, only the last time step)
                # logits -> [batch_size, vocab_size]
                probs = F.softmax(logits, dim=1)    # probs -> [batch_size, vocab_size]
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
        
    sa = SelfAttentionLanguageModel(vocab_size=52, n_embed=32)
    print("Self Attention model: ", sa)
    print("State Dictionary of model:", sa.state_dict())
    
    idx = torch.randint(high=52, size=(4,8))
    targets = torch.randint(high=52, size=(4,8))
    logits, loss = sa.forward(idx=idx, targets=targets)
    print(f"Sample forward pass result: \nOutput logits: {logits}, \nLoss: {loss.item(): .2f}")
    
    idx = torch.randint(high=52, size=(1,1))
    idx = sa.generate(idx=idx, max_new_tokens=32)
    print("Text generation sample output: ", idx)
    
    # temporary directory for demo 
    temp_base = Path(tempfile.mkdtemp(prefix="self_attention_demo_"))
    demo_dir = temp_base / "eslf_attention"
    
    # Saving model in the temporary demo directory
    save_model(
        model=sa,
        model_name="sample_self_attention_model.pt",
        target_dir=str(demo_dir),
    )

    loaded_model = load_model(
        model=sa,
        target_model_path=str(demo_dir / "sample_self_attention_model" / "sample_self_attention_model.pt"),
    )

    print("Loaded model's state dict: ", loaded_model.state_dict())

    # Clean up only the temporary demo directory we created
    if temp_base.exists():
        shutil.rmtree(temp_base)

