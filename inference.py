import torch
import torchaudio
import argparse
import os
import sys
import config
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from model import DownstreamHead

# --- LABEL DEFINITIONS ---
# 1. GTZAN Genres
GTZAN_LABELS = [
    "blues", "classical", "country", "disco", "hiphop", 
    "jazz", "metal", "pop", "reggae", "rock"
]

# 2. GiantSteps Keys (0-11 Major, 12-23 Minor)
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
GIANTSTEPS_LABELS = [f"{p} Major" for p in PITCHES] + [f"{p} Minor" for p in PITCHES]

# 3. EmoMusic Dimensions
EMO_LABELS = ["Arousal", "Valence"]

class MERTInference:
    def __init__(self, task_name, head_weights_path):
        print(f"Loading MERT model: {config.MODEL_ID}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        self.base_model.eval()
        
        # Load Task Config
        if task_name not in config.TASKS:
            sys.exit(f"Error: Task '{task_name}' not found in config.py")
            
        self.task_conf = config.TASKS[task_name]
        self.task_name = task_name
        
        print(f"Initializing Head for {task_name} ({self.task_conf['type']})...")
        
        # Initialize Head
        self.head = None
        if head_weights_path and os.path.exists(head_weights_path):
            self.head = DownstreamHead(num_classes=self.task_conf['classes'])
            # Load weights (strict=False allows flexibility if layers match)
            self.head.load_state_dict(torch.load(head_weights_path))
            self.head.eval()
        else:
            print(f"WARNING: Head weights not found at {head_weights_path}")

    def process_file(self, file_path, max_duration=config.MAX_DURATION):
        if not os.path.exists(file_path):
            sys.exit(f"Error: File not found at {file_path}")

        # 1. Load & Resample
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            return {"error": str(e)}

        if sr != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Limit Duration
        max_frames = int(max_duration * config.SAMPLE_RATE)
        if waveform.shape[1] > max_frames:
            waveform = waveform[:, :max_frames]

        # 2. Run MERT (Foundation Model)
        inputs = self.processor(waveform.squeeze(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            outputs = self.base_model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state
            
            # Pooling (Squash time dimension to match trained head input)
            final_embedding = embeddings.mean(dim=1) # Shape: (1, 768)

            results = {}

            # 3. Run Downstream Head
            if self.head:
                logits = self.head(final_embedding)
                
                # --- OUTPUT FORMATTING LOGIC ---
                
                # CASE A: Multiclass (Genre, Key, Singer)
                if self.task_conf['type'] == "multiclass":
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = torch.argmax(probs).item()
                    confidence = probs[0][pred_idx].item()
                    
                    results["Class ID"] = pred_idx
                    results["Confidence"] = f"{confidence:.2%}"
                    
                    # Attach readable labels if available
                    if self.task_name == "gtzan":
                        results["Prediction"] = GTZAN_LABELS[pred_idx]
                    elif self.task_name == "giantsteps":
                        results["Prediction"] = GIANTSTEPS_LABELS[pred_idx]

                # CASE B: Multilabel (Tagging)
                elif self.task_conf['type'] == "multilabel":
                    probs = torch.sigmoid(logits) # Independent probabilities
                    # Get top 5 tags
                    top_probs, top_indices = torch.topk(probs, 5)
                    
                    results["Top Tags (Indices)"] = top_indices[0].tolist()
                    results["Scores"] = [f"{p:.2f}" for p in top_probs[0].tolist()]

                # CASE C: Regression (Emotion)
                elif self.task_conf['type'] == "regression":
                    vals = logits[0].tolist()
                    if self.task_name == "emomusic":
                        results["Arousal"] = f"{vals[0]:.4f}"
                        results["Valence"] = f"{vals[1]:.4f}"
                    else:
                        results["Values"] = vals

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MERT inference on audio.")
    parser.add_argument("filepath", type=str, help="Path to audio file")
    parser.add_argument("--task", type=str, required=True, help="Task name from config.py (gtzan, giantsteps, etc)")
    parser.add_argument("--head", type=str, default=None, help="Path to .pth file. Defaults to 'trained_{task}.pth'")
    parser.add_argument("--limit", type=float, default=config.MAX_DURATION)
    
    args = parser.parse_args()

    # Default head path logic
    head_path = args.head if args.head else f"./features/trained_{args.task}.pth"

    engine = MERTInference(task_name=args.task, head_weights_path=head_path)
    results = engine.process_file(args.filepath, max_duration=args.limit)
    
    print(f"\n--- Results for {args.task.upper()} ---")
    for k, v in results.items():
        print(f"{k}: {v}")