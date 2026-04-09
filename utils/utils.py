import torch

def get_batch(
    split: str, 
    train_data: torch.Tensor, 
    val_data: torch.Tensor, 
    context_window_len: int, 
    batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns x and y tensors from training/validation tensors. Both x and y are of shape (batch_size, context_window_len)

    Args:
        split (str): The split to get data from (training or validation)
        train_data (torch.Tensor): Training data tensor
        val_data (torch.Tensor): Validation data tensor
        context_window_len (int): Length of model's context window
        batch_size (int): Length of mini batch

    Raises:
        ValueError: When split gets a value that is neither train nor val

    Returns:
        tuple[torch.Tensor, torch.Tensor]: input and target tensors
    """
    split = split.lower()
    if split not in ("train", "val"):
        raise ValueError("split must be one of train or val")
    
    data = train_data if split == "train" else val_data
    # List of indices of data to select model input and output
    ix = torch.randint(low = 0, high = len(data) - context_window_len, size = (batch_size,))    # ix -> [batch_size]
    # token numbers according to stoi that are given as input to the model
    x = torch.stack([data[i:i + context_window_len] for i in ix])   # x -> [batch_size, context_window_len]
    # same as x but everything shifted to the right by one (target is predicting the next token)
    y = torch.stack([data[i + 1: i + context_window_len + 1] for i in ix])  # y -> [batch_size, context_window_len]
    return (x, y)