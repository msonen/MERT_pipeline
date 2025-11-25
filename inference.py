import torch
import torchaudio
import argparse
import os
import sys
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from model import DownstreamHead
import config

# GTZAN Mapping (Must match the alphabetical sort used in training)
GTZAN_LABELS = [
    "Blues", "Classical", "Country", "Disco", "Hiphop", 
    "Jazz", "Metal", "Pop", "Reggae", "Rock"
]

class MERTInference:
    def __init__(self, head_weights_path=None):
        print(f"Loading MERT model: {config.MODEL_ID}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        self.base_model.eval()
        
        self.head = None
        if head_weights_path and os.path.exists(head_weights_path):
            print(f"Loading trained head: {head_weights_path}")
            # GTZAN has 10 classes
            self.head = DownstreamHead(num_classes=10) 
            self.head.load_state_dict(torch.load(head_weights_path))
            self.head.eval()
        else:
            print("WARNING: No trained head found. Output will be raw features only.")


    def process_file(self, file_path, task_name):
        conf = config.TASKS[task_name]
        
        # ... (Run MERT to get final_embedding) ...
        
        # Run Head
        logits = self.head(final_embedding)
        
        results = {}
        
        if conf['type'] == "multiclass":
            # Single Winner (Genre)
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs).item()
            results["ClassID"] = idx
            results["Score"] = probs[0][idx].item()
            
        elif conf['type'] == "multilabel":
            # Multiple Winners (Tags)
            probs = torch.sigmoid(logits) # Sigmoid for multi-label
            # Get top 5 tags
            top_probs, top_indices = torch.topk(probs, 5)
            results["Top_Tags"] = top_indices.tolist()
            results["Scores"] = top_probs.tolist()
            
        elif conf['type'] == "regression":
            # Raw values (Emotion)
            vals = logits[0].tolist()
            results["Values"] = vals # e.g. [0.8, -0.2] (Arousal, Valence)

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="Path to mp3/wav file")
    # Default to the file saved by train_gtzan.py
    parser.add_argument("--head", type=str, default="./features/trained_genre_head.pth")
    parser.add_argument("--limit", type=float, default=config.DEFAULT_DURATION)
    
    args = parser.parse_args()

    engine = MERTInference(head_weights_path=args.head)
    results = engine.process_file(args.filepath, max_duration=args.limit)
    
    print("\n--- 🎸 Prediction Results 🎸 ---")
    for k, v in results.items():
        print(f"{k}: {v}")