import torch

def get_batch(
    split: str, 
    train_data: torch.Tensor, 
    val_data: torch.Tensor | None, 
    context_window_len: int, 
    batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns x and y tensors from train/val tensors. Both x and y are of shape (batch_size, context_window_len)

    Args:
        split (str): The split to get data from (train or val)
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
    
    if split == 'train':
        data = train_data
    else:
        if val_data is None:
            raise ValueError("val_data must be provided when split is val")
        data = val_data
    
    if len(data) <= context_window_len:
        raise ValueError(
            f"data length: ({len(data)}) must be greater than length of the context window: "
            f"{context_window_len} to build x/y batches"
        )
        
    # Random starting indices for our context window for selecting input and output
    ix = torch.randint(
        low=0,
        high=len(data)-context_window_len,
        size=(batch_size,),
        device=data.device
    )   # ix -> [batch_size] === Number of indices in a batch
    
    ## Building the index grid
    # Gives us a single row of length context_window_size containing numbers in ascending order starting from 1 (1, 2, 3 ...)
    t = torch.arange(context_window_len, device=data.device)    # t -> [context_window_len]
    # Broadcasts ix from [batch_size] to [batch_size,1] and t from [context_window_len] to [1,context_window_len] (Essentially adds the numbers until context_window_len from t into ix which contains starting indices)
    idx = ix[:, None] + t[None, :]  # idx -> [batch_size, context_window_len]
    
    # Gathering x and shifted y === x, y -> [batch_size, context_window_len]
    x = data[idx]   # Goes row by row and picks out the tokens at that index of data[row][j]
    y = data[idx+1]  # Same for y but add 1 to all the values of idx because target token should be the next token

    return (x, y)