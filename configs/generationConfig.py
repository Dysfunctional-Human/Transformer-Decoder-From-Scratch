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

TOKENIZER_DIR = "corpus"
MODEL_PATH = f"trained_models/{MODEL_NAME}/20-06-26-22-20-20_decoderv2/20-06-26-22-20-20_decoder.pt"

TOKENIZER_TYPE  = "character"

EMBED_SIZE = 384
CONTEXT_WINDOW_LEN = 256
NUM_HEADS = 6
NUM_BLOCKS = 6
DROP_PROB = 0.2
SAMPLE_TEMPERATURE = 0.85

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    
if MODEL_NAME == "decoderv2":
    USE_BPE = True
    TOKENIZER_TYPE = "bpe"