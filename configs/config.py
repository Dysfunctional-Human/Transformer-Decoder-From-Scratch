import torch
from models import bigramModel, selfAttentionModel

MODEL_NAME = "Self_Attention"
MODEL_NAME = MODEL_NAME.lower()
if MODEL_NAME == "bigram":
    MODEL = bigramModel.BigramLanguageModel
elif MODEL_NAME == "self_attention":
    MODEL = selfAttentionModel.SelfAttentionLanguageModel
else:
    raise ValueError(
        f"Unsupported MODEL_NAME '{MODEL_NAME}'"
    )

USE_SHARED_TOKENIZER = True
REBUILD_SHARED_TOKENIZER = False
TOKENIZER_DIR = "corpus"
TRAIN_PATH = "dataset/TinyStories_train_100k.txt"
VAL_PATH = "dataset/TinyStories_valid_5k.txt"
DEBUG = True

LEARNING_RATE = 1e-2
EPOCHS = 5000

CONTEXT_WINDOW_LEN = 8
BATCH_SIZE = 4
EMBED_SIZE = 32
HEAD_SIZE = 32

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    
if MODEL_NAME == "self_attention":
    LEARNING_RATE = 1e-3
