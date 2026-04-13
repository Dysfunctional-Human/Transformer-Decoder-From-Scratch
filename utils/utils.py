import torch
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

def get_batch(
    split: str, 
    context_window_len: int, 
    batch_size: int,
    train_data: torch.Tensor, 
    val_data: torch.Tensor | None = None, 
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
def save_model(
    model: torch.nn.Module, 
    target_dir: str, 
    model_name: str, 
    results: Dict[str, List[float]]
):
    """Saves the model

    Args:
        model (torch.nn.Module): The trained model
        target_dir (str): The parent directory where the model needs to be saved
        model_name (str): Name for saving the model
        results (Dict[str, List[float]]): Model's results dictionary
    """
    dir_path = Path(target_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    target_dir_path = dir_path / Path(model_name).stem
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(('.pth', '.pt')), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    
    print(f"Saving model and training results to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    
    results_save_path = target_dir_path / "results.pt"
    torch.save(results, results_save_path)
    
def load_model(model: torch.nn.Module,
               target_model_path: str,
               model_results_path: str | None = None
) -> Tuple[torch.nn.Module, Dict[str, List[float]] | None]:
    """Load the model and results (optional) dictionary 

    Args:
        model (torch.nn.Module): The base class for the model to be loaded into
        target_model_path (str): Path to the model
        model_results_path (str | None): Path to the results dictionary

    Returns:
        torch.nn.Module, Dict[str, List[float] | None]: Model and optional results dictionary
    """
    model_path = Path(target_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict=state_dict)
    
    results = None
    if model_results_path:
        results_path = Path(model_results_path)
        if not results_path.is_file():
            raise FileNotFoundError(f"Results file not found: {results_path}")
    
        results: Dict[str, List[float]] = torch.load(results_path)
    
    return model, results 
    
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int,
    context_window_len: int,
    batch_size: int
) -> Dict[str, float]:    
    """Train and Test loss calculations

    Args:
        model (torch.nn.Module): Model to be used
        train_data (torch.Tensor): Training data
        val_data (torch.Tensor): Validation data
        eval_iters (int): Number of iterations of evaluation
        context_window_len (int): Length of context window of the model
        batch_size (int): Batch size for the data

    Returns:
        Dict[str, float]: Dictionary containing loss according to split
    """
    with torch.inference_mode():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                if split == "train":
                    X, Y = get_batch(split=split, train_data=train_data, context_window_len=context_window_len, batch_size=batch_size)
                elif split == "val":
                    X, Y = get_batch(split=split, train_data=train_data, context_window_len=context_window_len, batch_size=batch_size, val_data=val_data)
                _, loss = model.forward(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
        

def plot_model_curves(
    results: Dict[str, List[float]],
    save_path: str | None = None,
) -> None:
    """Plot training and test loss curves.

    If ``save_path`` is provided, the plot is saved to that location using
    ``matplotlib.pyplot.savefig``; otherwise the plot is displayed interactively.
    
    Args:
        results (Dict[str, List[float]]): Results dictionary to be used for plotting
        save_path (str | None, optional): Path for saving the plots. Defaults to None.
    """
    loss = results['train_loss']
    test_loss = results['test_loss']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(7,7))
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved loss curve to {save_path}")
    else:
        plt.show()
    