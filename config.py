import transformers

# Old: "m-a-p/MERT-v1-330M"
MODEL_ID = "m-a-p/MERT-v1-95M"
HIDDEN_DIM = 768
SAMPLE_RATE = 24000
CHUNK_SEC = 5
BATCH_SIZE = 32
MAX_DURATION = 180 # 3 min
DEFAULT_DURATION=10
EXTRACT_DURATION=30
LEARNING_RATE = 1e-4
EPOCHS = 20

# Task Configuration Registry
# Type: 'multiclass' (one winner), 'multilabel' (many winners), 'regression' (score)
TASKS = {
    "gtzan":      {"type": "multiclass", "classes": 10,  "folder": "./features/gtzan"},
    "mtt":        {"type": "multilabel", "classes": 50,  "folder": "./features/mtt", "csv": "./meta/mtt_tags.csv"},
    "mtg_jamendo":{"type": "multilabel", "classes": 50,  "folder": "./features/mtg", "csv": "./meta/mtg_top50.tsv"},
    "mtg_genre":  {"type": "multilabel", "classes": 87,  "folder": "./features/mtg_genre"},
    "mtg_mood":   {"type": "multilabel", "classes": 56,  "folder": "./features/mtg_mood"},
    "mtg_inst":   {"type": "multilabel", "classes": 40,  "folder": "./features/mtg_inst"},
    "giantsteps": {"type": "multiclass", "classes": 24,  "folder": "./features/giantsteps"}, # 12 Major + 12 Minor
    "emomusic":   {"type": "regression", "classes": 2,   "folder": "./features/emomusic", "csv": "./meta/emo.csv"}, # Arousal, Valence
    "vocalset":   {"type": "multiclass", "classes": 20,  "folder": "./features/vocalset", "csv": "./features/vocalset/vocalset.csv"},
}