import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple

class FeedForward(nn.Module):
    def __init__(
        self, 
        n_embed: int,
        dropout_prob: float
    ):
        """A simple linear layer followed by non-linearity

        Args:
            n_embed: Embedding dimension for input tokens
            dropout_prob: Dropout probability
        """
        super().__init__()
        self.n_embed = n_embed
        
        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=self.n_embed,
                out_features=4*self.n_embed
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=4*self.n_embed,
                out_features=self.n_embed
            ),
            nn.Dropout(p=dropout_prob)
        )
        
    def forward(
        self, 
        x: torch.Tensor
    ):
        """A single forward step through the FeedForward network

        Args:
            x (torch.Tensor): Input indices

        Returns:
            torch.Tensor: Output logits
        """
        return self.ffn(x)
        
class Block(nn.Module):
    """Transformer block: communication followed by computation
    """
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        head_size: int,
        context_window_len: int,
        dropout_prob: float
    ):
        """Initializing a single attention block

        Args:
            n_embed (int): Embedding dimension for input tokens
            n_head (int): Number of attention heads
            head_size(int): Embedding dimension for each head
            context_window_len(int): Length of context window
            dropout_prob: Dropout probability
            
        """
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, n_embed=n_embed, head_size=head_size, context_window_len=context_window_len, dropout_prob=dropout_prob)
        self.ffwd  = FeedForward(n_embed=n_embed, dropout_prob=dropout_prob)
        self.ln1 = nn.LayerNorm(normalized_shape=n_embed)
        self.ln2 = nn.LayerNorm(normalized_shape=n_embed)
        
    def forward(
        self,
        x: int
    ):
        """A single forward step for attention block

        Args:
            x (int): Input data

        Returns:
            torch.Tensor: output logits
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        context_window_len: int,
        n_embed: int,
        dropout_prob: float
    ):
        """A block of Multi Head Attention

        Args:
            num_heads (int): Number of self attention heads
            head_size (int): Embedding dimension for each head
            context_window_len (int): Length of context window
            n_embed (int): Embedding dimension of input data
            dropout_prob(float): Dropout probability
        """
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.n_embed = n_embed
        self.context_window_len = context_window_len
        
        # Validate that concatenated head dimensions match the embedding size
        if num_heads * head_size != n_embed:
            raise ValueError(
                f"num_heads ({num_heads}) * head_size ({head_size}) must equal n_embed ({n_embed})"
            )
            
        self.heads = nn.ModuleList([    # A list of self attention heads
            Head(
                context_window_len=self.context_window_len,
                n_embed=self.n_embed,
                head_size=self.head_size,
                dropout_prob=dropout_prob
            ) for _ in range(num_heads)
        ])
        
        self.proj = nn.Linear(in_features=self.head_size*self.num_heads, out_features=self.n_embed)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self, 
        x: torch.Tensor
    ):
        """Single forward pass for the multi head attention block

        Args:
            x (torch.Tensor): Tensor containing input data

        Returns:
            torch.Tensor: New attention based updated embeddings made by concatenating output embeddings from each attention head
        """
        # x -> [batch_size, context_window_len, n_embed]
        out = torch.cat([att_head(x) for att_head in self.heads], dim=-1)  # att_head(x) -> [batch_size, context_window_len, n_embed//num_heads]
        # concatenates output from each attention head -> (n_embed//num_heads)*num_heads => [batch_size, context_window_len, n_embed]
        out = self.proj(out)    # projection layer
        # x -> [batch_size, context_window_len, n_embed] 
        out = self.dropout(out) # dropout layer
        
        return out
    

class Head(nn.Module):
    def __init__(
        self, 
        context_window_len: int, 
        n_embed: int, 
        head_size: int,
        dropout_prob: float
    ):
        """A single head of self attention

        Args:
            context_window_len (int): Length of model's context window
            n_embed (int): Embedding dimension
            head_size (int): Attention embedding dimensions
            dropout_prob(float): Dropout probability
        """
        super().__init__()
        self.key = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.query = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_window_len, context_window_len)))
        self.dropout =nn.Dropout(dropout_prob)
        self.n_embed = n_embed
        self.context_window_len = context_window_len
    
    def forward(
        self, 
        x: torch.Tensor
    ):
        """Single forward pass for the attention head

        Args:
            x (torch.Tensor): Tensor containing input data

        Returns:
            out (torch.Tensor): New attention based updated embeddings
        """
        _, T, _ = x.shape
        # x -> [batch_size, context_window_len, n_embed]
        q = self.query(x)  
        k = self.key(x)
        # q, k -> [batch_size, context_window_len, head_size]
        wei = q @ k.transpose(-2, -1) # transposing k to [batch_size, head_size, context_window_len] for dot product
        wei = wei * (q.size(-1))**-0.5 # For numerical stability
        # wei -> [batch_size, context_window_len, context_window_len] => Attention scores of each word against each word in the context window
        # Basically tells us how much weightage should the word at wei[batch][i][j] have in deciding the new embeddings of the word at ith position in the context window
        if T > self.context_window_len:
            raise ValueError(
                f"T ({T}) exceeds context_window_len ({self.context_window_len})"
            )
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Prevents the model from cheating by looking at words into the future. Assigns negative infinity weights to the wei[batch][i][j] tokens where j > i.
        # This helps the model by not letting the the words in the future deciding the embedding of the current word, since task is next word prediction - the model can "cheat" by assigning highest weightage to the token just after the current one and thus being able to perfectly predict the next token but still not actually learn anything valuable.
        wei = F.softmax(wei, dim=-1) # wei -> [batch_size, context_window_len, context_window_len]
        # Making all the weights add upto 1
        wei = self.dropout(wei)
        v = self.value(x)   # v -> [batch_size, context_window_len, head_size]
        out = wei @ v   # out -> [batch_size, context_window_len, head_size]
        # new updated embeddings from self attention
        return out
        

class DecoderModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        EMBED_SIZE: int,
        CONTEXT_WINDOW_LEN: int,
        NUM_HEADS: int,
        NUM_BLOCKS: int,
        DROP_PROB: float,
        DEVICE: str = "cpu",
        # Optional token ID for the end-of-text token; used to stop generation when encountered
        endoftext_token_id: int | None = None,
        **kwargs
    ):
        """Initializes the Decoder Model


        Args:
            vocab_size (int): Size of dataset vocabulary
            EMBED_SIZE (int): Embedding dimension
            CONTEXT_WINDOW_LEN (int): Length of context window
            NUM_HEADS (int): Number of self attention heads inside a single multihead attention block
            NUM_BLOCKS (int): Number of multihead attention blocks
            DEVICE (str, optional): Model and data device. Defaults to "cpu".
            endoftext_token_id (int | None, optional): Token_id for endoftext token to stop generation. Defaults to None.
            DROP_PROB (float): Dropout probability

        Raises:
            ValueError: When embedding dimension isn't a multiple of number of attention head (output of each head needs to be concatenated)
        """
        super().__init__()
        self.n_embed = EMBED_SIZE
        self.context_window_len = CONTEXT_WINDOW_LEN
        self.device = DEVICE
        self.num_heads = NUM_HEADS
        self.num_blocks = NUM_BLOCKS
        self.drop_probs = DROP_PROB
        
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embed) # (num_embeddings, embed_size)
        self.position_embedding_table = nn.Embedding(self.context_window_len, self.n_embed) # (number of tokens given to the model, embed_size)
        self.endoftext_token_id = endoftext_token_id
        
        # Ensure embedding dimension is divisible by number of heads
        if self.n_embed % self.num_heads != 0:
            raise ValueError(f"Embedding dimension n_embed ({self.n_embed}) must be divisible by num_heads ({self.num_heads})")
        head_dim = self.n_embed // self.num_heads
        self.blocks = nn.Sequential(*[Block(n_embed=self.n_embed, n_head=self.num_heads, head_size=head_dim, context_window_len=self.context_window_len, dropout_prob=self.drop_probs)
                                      for _ in range(self.num_blocks)])
        self.ln_f = nn.LayerNorm(self.n_embed) # final layer norm
        self.lm_head = nn.Linear(in_features=self.n_embed, out_features=vocab_size)
        
        # Better weight initialization for faster convergence
        self.apply(self._init_weights)
        
    
    def _init_weights(
        self, 
        module: nn.Module
    ):
        """Initialize weights for model layers

        Args:
            module (nn.Module): Language Model
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
          
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

        B, T = idx.shape
        if T > self.context_window_len:
            raise ValueError(f"T ({T}) exceeds context_window_len ({self.context_window_len})")
        # idx and targets -> [batch_size, context_window_len]
        
        tok_emb = self.token_embedding_table(idx)
        # tok_emb -> [batch_size, context_window_len, n_embed]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        # pos_emb -> [context_window_len, n_embed]
        x = tok_emb + pos_emb
        # x -> [batch_size, context_window_len, n_embed]
        x = self.blocks(x)
        # x -> [batch_size, context_window_len, n_embed]
        x = self.ln_f(x)
        # x -> [batch_size, context_window_len, n_embed]
        logits = self.lm_head(x)
        # logits -> [batch_size, context_window_len, vocab_size]
        
        if targets is None:
            loss = None
        else:
            batch, context, vocab_size = logits.shape
            
            logits_2d = logits.reshape(batch*context, vocab_size)  # logits -> [batch_size*context_window_len, vocab_size]
            targets = targets.reshape(batch*context)   # targets -> [batch_size, context_window_len]

            loss = F.cross_entropy(logits_2d, targets) # loss -> [1] === single floating point number
        
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
        if idx.shape[0] != 1:
            raise ValueError("generate method only supports batch_size=1")
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                idx_cropped = idx[:, -self.context_window_len:]
                logits, _ = self(idx_cropped, targets=None)
                # logits -> [batch_size, context_window_len, vocab_size] Here logits is 3 dimensional since target is None in the forward method
                logits = logits[:, -1, :]   # taking the output at the last position, which encodes the whole available context.
                # logits -> [batch_size, vocab_size]
                probs = F.softmax(logits, dim=-1)    # probs -> [batch_size, vocab_size]
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
        
    decoder = DecoderModel(vocab_size=52, EMBED_SIZE=32, CONTEXT_WINDOW_LEN=8, endoftext_token_id=0, NUM_HEADS=4, NUM_BLOCKS=6, DROP_PROB=0.5)
    print("Decoder model: ", decoder)
    print("State Dictionary of model:", decoder.state_dict())
    
    idx = torch.randint(high=52, size=(4,8))
    targets = torch.randint(high=52, size=(4,8))
    logits, loss = decoder(idx=idx, targets=targets)
    print(f"Sample forward pass result: \nOutput logits: {logits}, \nLoss: {loss.item(): .2f}")
    
    idx = torch.randint(high=52, size=(1,1))
    idx = decoder.generate(idx=idx, max_new_tokens=32)
    print("Text generation sample output: ", idx)
    
    # temporary directory for demo 
    temp_base = Path(tempfile.mkdtemp(prefix="decoder_demo_"))
    demo_dir = temp_base / "decoder"
    
    # Saving model in the temporary demo directory
    save_model(
        model=decoder,
        model_name="sample_decoder_model.pt",
        target_dir=str(demo_dir),
    )

    loaded_model = load_model(
        model=decoder,
        target_model_path=str(demo_dir / "sample_decoder_model" / "sample_decoder_model.pt"),
    )

    print("Loaded model's state dict: ", loaded_model.state_dict())

    # Clean up only the temporary demo directory we created
    if temp_base.exists():
        shutil.rmtree(temp_base)