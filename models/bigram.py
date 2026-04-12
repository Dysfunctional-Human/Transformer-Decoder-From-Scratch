import torch
import torch.nn as nn
from torch.nn import functional as F

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
    def __init__(self, vocab_size, batch_size, context_window_len, device = "cuda"):
        """_summary_

        Args:
            vocab_size (_type_): _description_
            batch_size (_type_): _description_
            context_window_len (_type_): _description_
            device (str, optional): _description_. Defaults to "cuda".
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # (num_embeddings, embed_size)
        # Each token directly reads off the logits for the next token from a lookup table
        # For this model, embed_size == vocab_size
        # Note that Embedding isn't just a lookup table. This is a learnable weight matrix that gets updated via back propagation in the training loop
        
    def forward(self, idx: torch.Tensor, targets = None):
        """_summary_

        Args:
            idx (torch.Tensor): _description_
            targets (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
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

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """_summary_

        Args:
            idx (torch.Tensor): _description_
            max_new_tokens (int): _description_
        """
        # idx -> [batch_size, context_window_len]
        # Here, batch_size = 1
        for _ in range(max_new_tokens):
            # ToDo: Stop generation on encountering "<|endoftext|>"
            
            logits, _ = self.forward(idx)
            # logits -> [batch_size, context_window_len, embed_size] Here logits is 3 dimensional since target is None in the forward method
            logits = logits[:, -1, :]   # Focus only on the previous token (not the entire context window len, only the last time step)
            # logits -> [batch_size, embed_size]
            probs = F.softmax(logits, dim=1)    # probs -> [batch_size, embed_size]
            idx_next = torch.multinomial(probs, num_samples=1)  # idx_next -> [batch_size]
            # Rather than picking the most probable, sampling from multinomial distribution
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        
if __name__ == "__main__":
    torch.manual_seed(1337)
