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

    def process_file(self, file_path, max_duration=config.MAX_DURATION):
        if not os.path.exists(file_path):
            sys.exit(f"Error: File not found at {file_path}")

        # 1. Load & Resample
        waveform, sr = torchaudio.load(file_path)
        if sr != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Limit Duration
        max_frames = int(max_duration * config.SAMPLE_RATE)
        if waveform.shape[1] > max_frames:
            waveform = waveform[:, :max_frames]

        # 2. Run MERT
        inputs = self.processor(waveform.squeeze(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            outputs = self.base_model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state
            
            # --- FIX STARTS HERE ---
            # ERROR WAS HERE: Do NOT pool manually. The head does this.
            # final_embedding = embeddings.mean(dim=1)  <-- DELETE THIS LINE

            results = {}

            # 3. Run Prediction Head
            if self.head:
                # Pass 'embeddings' (3D) directly, not 'final_embedding' (2D)
                logits = self.head(embeddings) 
                probs = torch.softmax(logits, dim=1)
                
                # Get Top Prediction
                pred_idx = torch.argmax(probs).item()
                confidence = probs[0][pred_idx].item()
                
                results["Genre"] = GTZAN_LABELS[pred_idx]
                results["Confidence"] = f"{confidence:.2%}"
            # --- FIX ENDS HERE ---
                
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="Path to mp3/wav file")
    # Default to the file saved by train_gtzan.py
    parser.add_argument("--head", type=str, default="./features/trained_genre_head.pth")
    parser.add_argument("--limit", type=float, default=config.MAX_DURATION)
    
    args = parser.parse_args()

    engine = MERTInference(head_weights_path=args.head)
    results = engine.process_file(args.filepath, max_duration=args.limit)
    
    print("\n--- 🎸 Prediction Results 🎸 ---")
    for k, v in results.items():
        print(f"{k}: {v}")