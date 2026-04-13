import torch
from tqdm.auto import tqdm
from data.data_preparation import Dataset
from utils.utils import get_batch, save_model, plot_model_curves, save_results
from typing import Dict
from models.bigram import BigramLanguageModel
from datetime import datetime
from pathlib import Path

def train_step(
    model: BigramLanguageModel,
    train_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    context_window_len: int,
    batch_size: int
) -> float:
    """_summary_

    Args:
        model (BigramLanguageModel): _description_
        train_data (torch.Tensor): _description_
        loss (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        context_window_len (int): _description_
        batch_size (int): _description_

    Returns:
        float: _description_
    """
    model.train()
    
    X, y = get_batch(split="train", context_window_len=context_window_len, batch_size=batch_size, train_data=train_data)
    _, train_loss = model.forward(X, y)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    return train_loss.item()

def test_step(
    model: BigramLanguageModel,
    train_data: torch.Tensor, 
    val_data: torch.Tensor,
    context_window_len: int,
    batch_size: int
) -> float:
    """_summary_

    Args:
        model (BigramLanguageModel): _description_
        train_data (Dataset): _description_
        val_data (Dataset): _description_
        context_window_len (int): _description_
        batch_size (int): _description_

    Returns:
        float: _description_
    """
    model.eval()
    with torch.inference_mode():
        X, y = get_batch(split="val", context_window_len=context_window_len, batch_size=batch_size, train_data=train_data, val_data=val_data)
        _, test_loss = model.forward(X, y)
        
    return test_loss.item()
        

def engine(
    train_data: Dataset,
    device: torch.device,
    val_data: Dataset,
    model: BigramLanguageModel,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    context_window_len: int, 
    batch_size: int
) -> Dict[str, float]:
    """_summary_

    Args:
        train_data (Dataset): _description_
        device (torch.Device): _description_
        val_data (Dataset): _description_
        model (BigramLanguageModel): _description_
        optimizer (torch.optim.Optimizer): _description_
        loss (torch.nn.Module): _description_
        epochs (int): _description_
        context_window_len (int): _description_
        batch_size (int): _description_

    Returns:
        Dict[str, float]: _description_
    """
    training_data = torch.tensor(train_data.encode_story(train_data.clean_text), dtype=torch.long).to(device)
    validation_data = torch.tensor(train_data.encode_story(val_data.clean_text), dtype=torch.long).to(device)
    model.to(device)
    
    results = {
        "train_loss": [],
        "test_loss": [],
    }
    
    max_test_loss = 99999999
    prefix = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    model_name = prefix + "_bigram.pt"
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                train_data=training_data,
                                optimizer=optimizer,
                                context_window_len=context_window_len,
                                batch_size=batch_size)
        test_loss = test_step(model=model,
                              train_data=training_data,
                              val_data=validation_data,
                              context_window_len=context_window_len,
                              batch_size=batch_size)
        
        if ((epoch+1) % (epochs/100)) == 0:
            print(
                f"Epoch: {epoch+1} / {epochs}| "
                f"Train Loss: {train_loss: .4f} | "
                f"Test Loss: {test_loss: .4f}"
            )
            
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        
        if ((epoch+1) % (epochs/10)) == 0:
            print("-"*90)
            print(
                f"Epoch: {epoch+1} / {epochs}"
                f"Let's see how well the model generates: {train_data.decode_story(
                    model.generate(
                        torch.zeros((1,1), dtype=torch.long, device=device), 
                        max_new_tokens=100
                    )[0].tolist()
                )}"
            )
            print("-"*90)

        if max_test_loss > test_loss:
            save_model(model=model, target_dir="trained_models/BigramLanguageModel", model_name=model_name)
            max_test_loss = test_loss
            
    save_results(target_dir="trained_models/BigramLanguageModel", model_name=model_name, results=results)
    plot_model_curves(results=results, save_path=(Path("trained_models/BigramLanguageModel") / Path(model_name).stem / "plot"))
    
    return results

if __name__ == "__main__":
    # ToDo
    # Vocab should be that which includes both train and validation tokens - build a shared vocab for train and valid data
    # Store the results as json instead of torch.save . Create different save_results and load_results functions
    train_data = Dataset(data_path="dataset/TinyStories_train_100k.txt", device="cuda", debug=True)
    val_data = Dataset(data_path="dataset/TinyStories_valid_5k.txt", device="cuda", debug=True)
    
    model = BigramLanguageModel(vocab_size=len(train_data.vocab), endoftext_token_id=train_data.stoi["<|endoftext|>"])
    
    learning_rate = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    epochs = 10000
    
    context_window_len = 8
    
    batch_size = 4
    
    engine(
        train_data=train_data,
        device=train_data.device,
        val_data=val_data,
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        context_window_len=context_window_len,
        batch_size=batch_size,
    )