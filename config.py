# config.py
import transformers

# Old: "m-a-p/MERT-v1-330M"
MODEL_ID = "m-a-p/MERT-v1-95M"

SAMPLE_RATE = 24000
CHUNK_SEC = 5
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
MAX_DURATION = 10

# We fetch the config from Hugging Face to see if it's 768 or 1024
try:
    _model_config = transformers.AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    HIDDEN_DIM = _model_config.hidden_size
    print(f"--> Auto-detected Hidden Dimension for {MODEL_ID}: {HIDDEN_DIM}")
except Exception as e:
    print(f"Warning: Could not auto-detect dimension. Defaulting to 768. Error: {e}")
    HIDDEN_DIM = 768 # Safe default for 95M models