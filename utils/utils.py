import torch
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import json

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
):
    """Saves the model's state dictionary (device:cpu)

    Args:
        model (torch.nn.Module): The trained model
        target_dir (str): The parent directory where the model needs to be saved
        model_name (str): Name for saving the model
    """
    dir_path = Path(target_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    target_dir_path = dir_path / Path(model_name).stem
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(('.pth', '.pt')), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    
    print(f"Saving model to: {model_save_path}")
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state_dict, model_save_path)
        
def load_model(model: torch.nn.Module,
               target_model_path: str,
) -> torch.nn.Module:
    """Loads the Language Model on cpu

    Args:
        model (torch.nn.Module): Base class to use for loading model's state dictionary
        target_model_path (str): Model weights' location

    Raises:
        FileNotFoundError: Model state dictionary not found at given path

    Returns:
        torch.nn.Module: loaded language model
    """
    model_path = Path(target_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.to('cpu').load_state_dict(state_dict=state_dict)

    return model 

def save_results(
    target_dir: str, 
    model_name: str, 
    results: Dict[str, List[float]]
):
    """Saves the model results

    Args:
        target_dir (str): The parent directory where the model needs to be saved
        model_name (str): Name for saving the model
        results (Dict[str, List[float]]): Model's results dictionary
    """
    dir_path = Path(target_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    target_dir_path = dir_path / Path(model_name).stem
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    results_save_path = target_dir_path / "results.json"
    with open(results_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True)
    print("Results saved successfully at: ", target_dir_path)
    
def load_results(
    results_path: str
) -> Dict[str, List[float]]:
    """Loads model results from training time

    Args:
        results_path (str): Path for results json file location

    Returns:
        Dict[str, List[float]]: Results dictionary for the model
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print("Results loaded successfully")
    return results
    
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
            losses = None
            for k in range(eval_iters):
                if split == "train":
                    X, Y = get_batch(split=split, train_data=train_data, context_window_len=context_window_len, batch_size=batch_size)
                elif split == "val":
                    X, Y = get_batch(split=split, train_data=train_data, context_window_len=context_window_len, batch_size=batch_size, val_data=val_data)
                _, loss = model.forward(X, Y)
                if losses is None:
                    losses = torch.zeros(eval_iters, device=loss.device, dtype=loss.dtype)
                losses[k] = loss.detach()
            out[split] = losses.mean().item()
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

    plt.figure(figsize=(15,15))
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if save_path:
        save_path = Path(save_path)
        if save_path.parent != Path():
                save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved loss curve to {save_path}")
    else:
        plt.show()
    plt.close()
    