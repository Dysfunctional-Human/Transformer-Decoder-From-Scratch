import torch
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from tqdm.auto import tqdm
from data.data_preparation import Dataset, build_shared_tokenizer, save_tokenizer_artifacts, load_tokenizer_artifacts, build_bpe_tokenizer, load_bpe_tokenizer
from utils.utils import get_batch, save_model, plot_model_curves, save_results
from typing import Dict, Tuple
from datetime import datetime
from pathlib import Path
from typing import List
from configs import config
from tokenizers import Tokenizer

def get_tokenizer_artifacts() -> Tuple[Tokenizer | None, List[str], Dict[str, int], Dict[int, str]]:
    """Gets tokenizer artifacts based on config 

    Returns:
        Tuple[List[str], Dict[str, int], Dict[int, str]]: vocab, stoi, itos for the dataset
    """
    bpe_tokenizer, vocab, stoi, itos = None, None, None, None
    if config.USE_BPE:
        if not os.path.exists(f"{config.TOKENIZER_DIR}/{config.TOKENIZER_TYPE}/tokenizer.json"):
            print("Training new BPE Tokenizer")
            bpe_tokenizer, vocab, stoi, itos = build_bpe_tokenizer(
                dataset_paths=[config.TRAIN_PATH, config.VAL_PATH],
                vocab_size=config.BPE_VOCAB_SIZE,
                min_frequency=config.MIN_FREQUENCY,
                tokenizer_type=config.TOKENIZER_TYPE,
                save_dir=config.TOKENIZER_DIR
            )
        else:
            print("Loading pre-trained BPE tokenizer")
            bpe_tokenizer, vocab, stoi, itos = load_bpe_tokenizer(
                tokenizer_dir=config.TOKENIZER_DIR,
                tokenizer_type=config.TOKENIZER_TYPE
            )
    
    if config.USE_SHARED_TOKENIZER:
        if config.REBUILD_SHARED_TOKENIZER:
            vocab, stoi, itos = build_shared_tokenizer(
                dataset_paths=[
                    config.TRAIN_PATH, 
                    config.VAL_PATH
                ]
            )
            save_tokenizer_artifacts(
                target_dir=config.TOKENIZER_DIR,
                vocab=vocab, 
                stoi=stoi,
                itos=itos,
                tokenizer_type=config.TOKENIZER_TYPE
            )
        else:
            try:
                vocab, stoi, itos = load_tokenizer_artifacts(
                    target_dir=config.TOKENIZER_DIR,
                    tokenizer_type=config.TOKENIZER_TYPE
                )
            except FileNotFoundError:
                vocab, stoi, itos = build_shared_tokenizer(
                    dataset_paths=[
                        config.TRAIN_PATH, 
                        config.VAL_PATH
                    ]
                )
                save_tokenizer_artifacts(
                    target_dir=config.TOKENIZER_DIR,
                    vocab=vocab,
                    stoi=stoi,
                    itos=itos,
                    tokenizer_type=config.TOKENIZER_TYPE
                )
    return bpe_tokenizer, vocab, stoi, itos

def prepare_data(
    vocab: List[str] | None,
    stoi: Dict[str, int] | None,
    itos: Dict[int, str] | None,
    bpe_tokenizer: Tokenizer | None
) -> Tuple[Dataset, Dataset]:
    """Prepares datasets for model training based on config

    Args:
        vocab (List[str] | None): Shared vocabulary
        stoi (Dict[str, int] | None): string-to-int mapping
        itos (Dict[int, str] | None): int-to-string mapping

    Returns:
        Tuple[Dataset, Dataset]: Training and Testing/Validation datasets
    """
    if config.USE_SHARED_TOKENIZER:
        TRAIN_DATA = Dataset(
            data_path=config.TRAIN_PATH, 
            device=config.DEVICE, 
            debug=config.DEBUG,
            vocab=vocab,
            stoi=stoi,
            itos=itos,
            bpe_tokenizer=bpe_tokenizer if bpe_tokenizer is not None else None
        )
        VAL_DATA = Dataset(
            data_path=config.VAL_PATH, 
            device=config.DEVICE, 
            debug=config.DEBUG,
            vocab=vocab,
            stoi=stoi,
            itos=itos,
            bpe_tokenizer=bpe_tokenizer if bpe_tokenizer is not None else None
        )
    else:
        TRAIN_DATA = Dataset(
            data_path=config.TRAIN_PATH, 
            device=config.DEVICE, 
            debug=config.DEBUG,
        )
        VAL_DATA = Dataset(
            data_path=config.VAL_PATH, 
            device=config.DEVICE, 
            debug=config.DEBUG,
        )
    return TRAIN_DATA, VAL_DATA

def train_step(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    context_window_len: int,
    batch_size: int
) -> float:
    """A single training step

    Args:
        model (torch.nn.Module): Model under training
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return train_loss.item()

def test_step(
    model: torch.nn.Module,
    train_data: torch.Tensor, 
    val_data: torch.Tensor,
    context_window_len: int,
    batch_size: int
) -> float:
    """A single testing step

    Args:
        model (torch.nn.Module): Model under training
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
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    context_window_len: int, 
    batch_size: int,
    use_bpe: bool
) -> Dict[str, List[float]]:
    """Training loop for the model

    Args:
        train_data (Dataset): Dataset for training
        device (str): Device to conduct training on ('cpu', 'cuda' or 'mps')
        val_data (Dataset): Dataset for testing/validation
        model (nn.Module): Model to be trained
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
    
    if use_bpe:
        test_text = "Once upon a time, a little cat saw a big dog"
        
        encoded = train_data.encode_story(test_text)
        print(f"Original: {test_text}")
        print(f"Encoded IDs: {encoded[:20]}")
        
        decoded = train_data.decode_story(encoded)
        
        print(f"Decoded: {decoded}")
    
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())/1e6: .2f}M parameters")
    print("Engine model device:", next(model.parameters()).device)
    print("Training data device:", training_data.device)
    print("Validation data device:", validation_data.device)
    
    results = {
        "train_loss": [],
        "test_loss": [],
    }
    
    min_test_loss = float("inf")
    prefix = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    model_name = prefix + f"_{config.MODEL_NAME.lower()}.pt"
    
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
                target_dir=f"trained_models/{config.MODEL_NAME.lower()}", 
                model_name=model_name
            )
            min_test_loss = test_loss
            
    save_results(
        target_dir=f"trained_models/{config.MODEL_NAME.lower()}", 
        model_name=model_name, 
        results=results
    )
    plot_model_curves(
        results=results, 
        save_path=(
            Path(f"trained_models/{config.MODEL_NAME.lower()}") / Path(model_name).stem / "plot.png"
        )
    )
    print(f"Lowest Test Loss achieved during training: {min_test_loss: .4f}")
    
    return results

if __name__ == "__main__":
    
    bpe_tokenizer, vocab, stoi, itos = get_tokenizer_artifacts()
    bpe_tokenizer, _, _, _ = load_bpe_tokenizer(
        tokenizer_dir=config.TOKENIZER_DIR,
        tokenizer_type=config.TOKENIZER_TYPE
    )
    TRAIN_DATA, VAL_DATA = prepare_data(
        vocab=vocab,
        stoi=stoi,
        itos=itos,
        bpe_tokenizer=bpe_tokenizer
    )
    kwargs = {k: v for k, v in vars(config).items() if not k.startswith("__")}
    kwargs["endoftext_token_id"]= TRAIN_DATA.stoi["<|endoftext|>"]
    kwargs["vocab_size"]= len(TRAIN_DATA.vocab)
    
    MODEL = config.MODEL(**kwargs)
    
    OPTIMIZER = torch.optim.AdamW(
        params=MODEL.parameters(), 
        lr=config.LEARNING_RATE
    )
    print(
        "Training starting with the following config: "
        f"{kwargs}"
        f"{TRAIN_DATA, VAL_DATA}"
        f"{MODEL, OPTIMIZER}"
    )
    
    engine(
        train_data=TRAIN_DATA,
        device=TRAIN_DATA.device,
        val_data=VAL_DATA,
        model=MODEL,
        optimizer=OPTIMIZER,
        epochs=config.EPOCHS,
        context_window_len=config.CONTEXT_WINDOW_LEN,
        batch_size=config.BATCH_SIZE,
        use_bpe=True
    )