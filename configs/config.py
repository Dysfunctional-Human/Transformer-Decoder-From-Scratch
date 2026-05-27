import torch
from models import bigramModel, selfAttentionModel, multiHeadAttentionModel, decoderModel

MODEL_NAME = "Decoder"
MODEL_NAME = MODEL_NAME.lower()
if MODEL_NAME == "bigram":
    MODEL = bigramModel.BigramLanguageModel
elif MODEL_NAME == "self_attention":
    MODEL = selfAttentionModel.SelfAttentionLanguageModel
elif MODEL_NAME == "multi_head_attention":
    MODEL = multiHeadAttentionModel.MultiHeadAttentionLanguageModel
elif MODEL_NAME == "decoder":
    MODEL = decoderModel.DecoderModel
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
NUM_HEADS = 4

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    
if MODEL_NAME != "bigram":
    LEARNING_RATE = 1e-3

if MODEL_NAME == "decoder":
    BATCH_SIZE = 64
    CONTEXT_WINDOW_LEN = 256
    LEARNING_RATE = 3e-4
    EMBED_SIZE = 384
    NUM_HEADS = 6
    NUM_BLOCKS = 6
    DROP_PROB = 0.2
