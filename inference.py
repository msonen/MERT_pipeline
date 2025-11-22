import torch
import torchaudio
import argparse
import os
import sys
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from model import DownstreamHead
import config  # Importing your config settings

class MERTInference:
    def __init__(self, head_weights_path=None):
        print(f"Loading MERT model: {config.MODEL_ID}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        self.base_model.eval()
        
        self.head = None
        if head_weights_path and os.path.exists(head_weights_path):
            self.head = DownstreamHead(num_classes=2) 
            self.head.load_state_dict(torch.load(head_weights_path))
            self.head.eval()

    def process_file(self, file_path, window_sec=5.0, overlap_sec=1.0, max_duration=config.MAX_DURATION):
        """
        max_duration: Defaults to config.MAX_DURATION (10s) if not specified.
        """
        if not os.path.exists(file_path):
            sys.exit(f"Error: File not found at {file_path}")

        # 1. Load Audio
        waveform, sr = torchaudio.load(file_path)

        # 2. Resample
        if sr != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # 3. Mix to Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze()

        # --- LIMIT INPUT LENGTH ---
        # Uses the default from config.py unless overridden
        if max_duration is not None:
            max_frames = int(max_duration * config.SAMPLE_RATE)
            if waveform.shape[0] > max_frames:
                print(f"Limiting audio to first {max_duration} seconds (Default from config).")
                waveform = waveform[:max_frames]
        # --------------------------

        # 4. Sliding Window Logic
        window_size = int(window_sec * config.SAMPLE_RATE)
        stride = int((window_sec - overlap_sec) * config.SAMPLE_RATE)
        
        chunks = []
        if waveform.shape[0] < window_size:
            padding = window_size - waveform.shape[0]
            chunks.append(torch.nn.functional.pad(waveform, (0, padding)))
        else:
            chunks = waveform.unfold(0, window_size, stride)

        print(f"Generated {len(chunks)} windows from audio.")

        # 5. Batch Processing
        all_embeddings = []
        batch_size = 4 
        
        for i in range(0, len(chunks), batch_size):
            batch_audio = chunks[i : i + batch_size]
            inputs = self.processor(batch_audio.numpy(), 
                                  sampling_rate=config.SAMPLE_RATE, 
                                  return_tensors="pt", 
                                  padding=True)
            
            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                batch_embeds = outputs.last_hidden_state
                pooled = batch_embeds.mean(dim=1) 
                all_embeddings.append(pooled)

        # 6. Aggregate
        if not all_embeddings:
            return {"error": "Audio too short"}

        full_song_features = torch.cat(all_embeddings, dim=0)
        final_embedding = full_song_features.mean(dim=0, keepdim=True)

        results = {
            "processed_duration_approx": f"{len(chunks) * (window_sec - overlap_sec):.2f}s",
            "windows_count": len(chunks)
        }

        if self.head:
            logits = self.head(final_embedding.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs).item()
            results["prediction_class_id"] = pred_class
            results["confidence"] = probs[0][pred_class].item()
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str)
    parser.add_argument("--head", type=str, default="trained_genre_head.pth")
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=1.0)
    
    # UPDATED: Default value now comes from config.MAX_DURATION
    parser.add_argument("--limit", type=float, default=config.MAX_DURATION, 
                        help=f"Limit input to X seconds (Default: {config.MAX_DURATION})")
    
    args = parser.parse_args()

    engine = MERTInference(head_weights_path=args.head)
    
    results = engine.process_file(
        args.filepath, 
        window_sec=args.window, 
        overlap_sec=args.overlap, 
        max_duration=args.limit
    )
    
    print("\n--- Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")