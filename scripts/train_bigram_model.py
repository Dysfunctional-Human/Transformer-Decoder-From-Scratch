import torch
from tqdm.auto import tqdm
from data.data_preparation import Dataset, build_shared_tokenizer, save_tokenizer_artifacts, load_tokenizer_artifacts
from utils.utils import get_batch, save_model, plot_model_curves, save_results
from typing import Dict, Tuple
from models.bigram import BigramLanguageModel
from datetime import datetime
from pathlib import Path
from typing import List
from configs import bigram_config

def get_tokenizer_artifacts() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Gets tokenizer artifacts based on config 

    Returns:
        Tuple[List[str], Dict[str, int], Dict[int, str]]: vocab, stoi, itos for the dataset
    """
    vocab, stoi, itos = None, None, None
    if bigram_config.USE_SHARED_TOKENIZER:
        if bigram_config.REBUILD_SHARED_TOKENIZER:
            vocab, stoi, itos = build_shared_tokenizer(
                dataset_paths=[
                    bigram_config.TRAIN_PATH, 
                    bigram_config.VAL_PATH
                ]
            )
            save_tokenizer_artifacts(
                target_dir=bigram_config.TOKENIZER_DIR,
                vocab=vocab, 
                stoi=stoi,
                itos=itos
            )
        else:
            try:
                vocab, stoi, itos = load_tokenizer_artifacts(
                    target_dir=bigram_config.TOKENIZER_DIR
                )
            except FileNotFoundError:
                vocab, stoi, itos = build_shared_tokenizer(
                    dataset_paths=[
                        bigram_config.TRAIN_PATH, 
                        bigram_config.VAL_PATH
                    ]
                )
                save_tokenizer_artifacts(
                    target_dir=bigram_config.TOKENIZER_DIR,
                    vocab=vocab,
                    stoi=stoi,
                    itos=itos
                )
    return vocab, stoi, itos

def prepare_data(
    vocab: List[str] | None,
    stoi: Dict[str, int] | None,
    itos: Dict[int, str] | None
) -> Tuple[Dataset, Dataset]:
    """Prepares datasets for model training based on config

    Args:
        vocab (List[str] | None): Shared vocabulary
        stoi (Dict[str, int] | None): string-to-int mapping
        itos (Dict[int, str] | None): int-to-string mapping

    Returns:
        Tuple[Dataset, Dataset]: Training and Testing/Validation datasets
    """
    if bigram_config.USE_SHARED_TOKENIZER:
        TRAIN_DATA = Dataset(
            data_path=bigram_config.TRAIN_PATH, 
            device=bigram_config.DEVICE, 
            debug=bigram_config.DEBUG,
            vocab=vocab,
            stoi=stoi,
            itos=itos
        )
        VAL_DATA = Dataset(
            data_path=bigram_config.VAL_PATH, 
            device=bigram_config.DEVICE, 
            debug=bigram_config.DEBUG,
            vocab=vocab,
            stoi=stoi,
            itos=itos
        )
    else:
        TRAIN_DATA = Dataset(
            data_path=bigram_config.TRAIN_PATH, 
            device=bigram_config.DEVICE, 
            debug=bigram_config.DEBUG,
        )
        VAL_DATA = Dataset(
            data_path=bigram_config.VAL_PATH, 
            device=bigram_config.DEVICE, 
            debug=bigram_config.DEBUG,
        )
    return TRAIN_DATA, VAL_DATA

def train_step(
    model: BigramLanguageModel,
    train_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    context_window_len: int,
    batch_size: int
) -> float:
    """A single training step

    Args:
        model (BigramLanguageModel): Model under training
        train_data (torch.Tensor): Training data
        optimizer (torch.optim.Optimizer): Optimizer to be used
        context_window_len (int): Length of context window to be considered
        batch_size (int): Number of training samples in a single model input

    Returns:
        float: Training loss value
    """
    model.train()
    
    X, y = get_batch(
        split="train", 
        context_window_len=context_window_len, 
        batch_size=batch_size, 
        train_data=train_data
    )
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
    """A single testing step

    Args:
        model (BigramLanguageModel): Model under training
        train_data (torch.Tensor): Training data
        val_data (torch.Tensor): Validation data
        context_window_len (int): Length of context window to be considered
        batch_size (int): Number of training samples in a single model input

    Returns:
        float: Testing loss value
    """
    model.eval()
    with torch.inference_mode():
        X, y = get_batch(
            split="val", 
            context_window_len=context_window_len, 
            batch_size=batch_size, 
            train_data=train_data, 
            val_data=val_data
        )
        _, test_loss = model.forward(X, y)
        
    return test_loss.item()
        

def engine(
    train_data: Dataset,
    device: str,
    val_data: Dataset,
    model: BigramLanguageModel,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    context_window_len: int, 
    batch_size: int
) -> Dict[str, List[float]]:
    """Training loop for the model

    Args:
        train_data (Dataset): Dataset for training
        device (str): Device to conduct training on ('cpu' or 'cuda')
        val_data (Dataset): Dataset for testing/validation
        model (BigramLanguageModel): Model to be trained
        optimizer (torch.optim.Optimizer): Optimizer to be used to update model params
        epochs (int): Number of training and testing iterations
        context_window_len (int): Length of context window to be considered by the model
        batch_size (int): Number of training samples in a single model input

    Returns:
        Dict[str, List[float]]: Results dictionary containing training and testing losses for the model at every epoch
    """
    training_data = torch.tensor(
        train_data.encode_story(
            train_data.clean_text
        ), 
        dtype=torch.long).to(device)
    validation_data = torch.tensor(
        train_data.encode_story(
            val_data.clean_text
        ), 
        dtype=torch.long).to(device)
    model.to(device)
    
    print("Engine model device:", next(model.parameters()).device)
    print("Training data device:", training_data.device)
    print("Validation data device:", validation_data.device)
    
    results = {
        "train_loss": [],
        "test_loss": [],
    }
    
    min_test_loss = float("inf")
    prefix = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    model_name = prefix + "_bigram.pt"
    
    log_every = max(1, epochs // 100)
    sample_every = max(1, epochs // 10)
    
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
        
        if (epoch+1) % log_every == 0:
            print(
                f"Epoch: {epoch+1} / {epochs}| "
                f"Train Loss: {train_loss: .4f} | "
                f"Test Loss: {test_loss: .4f}"
            )
            
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        
        if (epoch+1) % sample_every == 0:
            print("-"*90)
            print(
                f"Epoch: {epoch+1} / {epochs}"
                f" Let's see how well the model generates: {train_data.decode_story(
                    model.generate(
                        torch.zeros((1,1), dtype=torch.long, device=device), 
                        max_new_tokens=100
                    )[0].tolist()
                )}"
            )
            print("-"*90)

        if min_test_loss > test_loss:
            save_model(
                model=model, 
                target_dir="trained_models/BigramLanguageModel", 
                model_name=model_name
            )
            min_test_loss = test_loss
            
    save_results(
        target_dir="trained_models/BigramLanguageModel", 
        model_name=model_name, 
        results=results
    )
    plot_model_curves(
        results=results, 
        save_path=(
            Path("trained_models/BigramLanguageModel") / Path(model_name).stem / "plot.png"
        )
    )
    print(f"Lowest Test Loss achieved during training: {min_test_loss: .4f}")
    
    return results

if __name__ == "__main__":
    
    vocab, stoi, itos = get_tokenizer_artifacts()
    TRAIN_DATA, VAL_DATA = prepare_data(
        vocab=vocab,
        stoi=stoi,
        itos=itos
    )
    MODEL = BigramLanguageModel(
        vocab_size=len(TRAIN_DATA.vocab), 
        endoftext_token_id=TRAIN_DATA.stoi["<|endoftext|>"]
    )
    OPTIMIZER = torch.optim.AdamW(
        params=MODEL.parameters(), 
        lr=bigram_config.LEARNING_RATE
    )
    print(
        "Training starting with the following config: "
        f"{bigram_config}"
        f"{TRAIN_DATA, VAL_DATA}"
        f"{MODEL, OPTIMIZER}"
    )
    
    engine(
        train_data=TRAIN_DATA,
        device=TRAIN_DATA.device,
        val_data=VAL_DATA,
        model=MODEL,
        optimizer=OPTIMIZER,
        epochs=bigram_config.EPOCHS,
        context_window_len=bigram_config.CONTEXT_WINDOW_LEN,
        batch_size=bigram_config.BATCH_SIZE,
    )