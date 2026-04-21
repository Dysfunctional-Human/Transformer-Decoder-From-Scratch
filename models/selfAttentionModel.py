import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple

class Head(nn.Module):
    def __init__(
        self, 
        context_window_len: int, 
        n_embed: int, 
        head_size: int
    ):
        """_summary_

        Args:
            context_window_len (int): _description_
            n_embed (int): _description_
            head_size (int): _description_
        """
        super().__init__()
        self.key = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.query = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_window_len, context_window_len)))
        self.n_embed = n_embed
        self.context_window_len = context_window_len
    
    def forward(
        self, 
        x: torch.Tensor
    ):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        B,T,C = x.shape
        # x -> [batch_size, context_window_len, n_embed]
        q = self.query(x)  
        k = self.key(x)
        # q, k -> [batch_size, context_window_len, head_size]
        wei = q @ k.transpose(-2, -1) # transposing k to [batch_size, n_embed, context_window_len] for dot product
        wei = wei * self.n_embed**-0.5 # For numerical stability
        # wei -> [batch_size, context_window_len, context_window_len] => Attention scores of each word against each word in the context window
        # Basically tells us how much weightage should the work at wei[batch][i][j] have in deciding the new embeddings of the word at ith position in the context window
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Prevents the model from cheating by looking at words into the future. Assigns negative infinity weights to the wei[batch][i][j] tokens where j > i.
        # This helps the model by not letting the the words in the future deciding the embedding of the current word, since task is next word prediction - the model can "cheat" by assigning highest weightage to the token just after the current one and thus being able to perfectly predict the next token but stil not actually learn anything valuable.
        wei = F.softmax(wei, dim=-1) # wei -> [batch_size, context_window_len, context_window_len]
        # Making all the weights add upto 1
        
        v = self.value(x)   # v -> [batch_size, context_window_len, n_embed]
        out = wei @ v   # out -> [batch_size, context_window_len, n_embed]
        # new updated embeddings from self attention
        return out
        

class SelfAttentionLanguageModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        EMBED_SIZE: int,
        HEAD_SIZE: int,
        CONTEXT_WINDOW_LEN: int,
        endoftext_token_id: int | None = None,
        **kwargs
    ):
        """Initializes the Self Attention Model

        Args:
            vocab_size (int): Size of dataset vocabulary
            EMBED_SIZE (int): Embedding dimension
            endoftext_token_id (int | None, optional): Token_id for endoftext token to stop generation. Defaults to None.
        """
        super().__init__()
        self.n_embed = EMBED_SIZE
        self.head_size = HEAD_SIZE
        self.context_window_len = CONTEXT_WINDOW_LEN
        
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embed) # (num_embeddings, embed_size)
        self.endoftext_token_id = endoftext_token_id
        
        self.lm_head = nn.Linear(in_features=self.n_embed, out_features=vocab_size)
        
        self.sa_head = Head(context_window_len=self.context_window_len, n_embed=self.n_embed, head_size=self.head_size)
          
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
        x = self.sa_head(tok_emb)
        # x -> [batch_size, context_window_len, embed_size]
        logits = self.lm_head(x)
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
                idx_cropped = idx[:, -self.context_window_len:]
                logits, _ = self.forward(idx_cropped, targets=None)
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
        
    sa = SelfAttentionLanguageModel(vocab_size=52, EMBED_SIZE=32)
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
    demo_dir = temp_base / "self_attention"
    
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

