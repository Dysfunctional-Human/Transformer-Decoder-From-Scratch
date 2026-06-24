import torch
from models import bigramModel, selfAttentionModel, multiHeadAttentionModel, decoderModel, decoderModelv2

MODEL_NAME = "DecoderV2"
MODEL_NAME = MODEL_NAME.lower()
if MODEL_NAME == "bigram":
    MODEL = bigramModel.BigramLanguageModel
elif MODEL_NAME == "self_attention":
    MODEL = selfAttentionModel.SelfAttentionLanguageModel
elif MODEL_NAME == "multi_head_attention":
    MODEL = multiHeadAttentionModel.MultiHeadAttentionLanguageModel
elif MODEL_NAME == "decoder":
    MODEL = decoderModel.DecoderModel
elif MODEL_NAME == "decoderv2":
    MODEL = decoderModelv2.DecoderModelv2
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
EPOCHS = 5

CONTEXT_WINDOW_LEN = 8
BATCH_SIZE = 4
EMBED_SIZE = 32
HEAD_SIZE = 32
NUM_HEADS = 4

TOKENIZER_TYPE  = "character"

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
    
if MODEL_NAME == "decoderv2":
    BATCH_SIZE = 64
    CONTEXT_WINDOW_LEN = 256
    LEARNING_RATE = 3e-4
    EMBED_SIZE = 384
    NUM_HEADS = 6
    NUM_BLOCKS = 6
    DROP_PROB = 0.2
    USE_BPE = True
    TOKENIZER_TYPE = "bpe"
    BPE_VOCAB_SIZE = 2000
    MIN_FREQUENCY = 2
    SAMPLE_TEMPERATURE = 0.85