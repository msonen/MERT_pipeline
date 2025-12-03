import torch
import torchaudio
import argparse
import os
import sys
import config
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from model import DownstreamHead

# --- LABEL DEFINITIONS ---
GTZAN_LABELS = [
    "blues", "classical", "country", "disco", "hiphop", 
    "jazz", "metal", "pop", "reggae", "rock"
]

VOCAL_SINGER_LABELS = [ 'female1', 'female2', 'female3', 'female4', 'female5', 'female6', 'female7', 'female8', 'female9',
                        'male1', 'male2', 'male3', 'male4', 'male5', 'male6', 'male7', 'male8', 'male9', 'male10', 'male11' ]

VOCAL_TECH_LABELS = [
    'arpeggio', 'fast_forte', 'inhaled', 'other', 'scale', 'spoken', 'straight', 'trill', 'vibrato',
    'belt', 'breathy', 'lip_trill', 'vocal_fry', 'vocal_fry', 'fast_forte', 'fast_piano', 
    'slow_forte', 'slow_piano', 'fast_piano', 'glissando', 'runs',
]

PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
GIANTSTEPS_LABELS = [f"{p} Major" for p in PITCHES] + [f"{p} Minor" for p in PITCHES]

# --- DISPLAY MAPPING (Internal Key -> Paper Name) ---
TASK_DISPLAY_NAMES = {
    "mtt": "Music Tagging",
    "giantsteps": "Key Detection",
    "gtzan": "Genre Classification",
    "emomusic": "Emotion Recognition",
    "nsynth_inst": "Instrument Classification",
    "nsynth_pitch": "Pitch Classification",
    "vocal_tech": "Vocal Technique",
    "vocal_singer": "Singer Identification",
    "mtg_jamendo": "Music Tagging (MTG)",
    "mtg_genre": "Genre Classification (MTG)",
    "mtg_mood": "Mood Detection",
    "mtg_inst": "Instrument Tagging (MTG)"
}

class MERTInference:
    def __init__(self, task_name, head_weights_path, base_model=None, processor=None):
        self.task_name = task_name
        
        # Load Task Config
        if task_name not in config.TASKS:
            # print(f"Warning: Task '{task_name}' not found in config.py. Skipping.")
            self.valid = False
            return
            
        self.task_conf = config.TASKS[task_name]
        self.valid = True

        # 1. Load Foundation Model (Only if not provided)
        if base_model and processor:
            self.processor = processor
            self.base_model = base_model
        else:
            model_actual_id = config.MODEL_ID
            if os.path.exists("./" + model_actual_id):
                model_actual_id = "./" + model_actual_id
            
            print(f"Loading MERT model: {model_actual_id}...")
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_actual_id, trust_remote_code=True)
            self.base_model = AutoModel.from_pretrained(model_actual_id, trust_remote_code=True)
            self.base_model.eval()

        # 2. Initialize Head
        self.head = None
        if head_weights_path and os.path.exists(head_weights_path):
            self.head = DownstreamHead(num_classes=self.task_conf['classes'])
            try:
                self.head.load_state_dict(torch.load(head_weights_path, map_location='cpu'))
                self.head.eval()
            except Exception as e:
                # Silently fail here so we can mark as N/A later
                self.head = None
        else:
            # Head not found
            self.head = None

    def process_file(self, file_path, max_duration=config.MAX_DURATION):
        if not self.valid or not self.head:
            return None

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
            
            # Pooling
            final_embedding = embeddings.mean(dim=1) # Shape: (1, 768)

            results = {}

            # 3. Run Downstream Head
            logits = self.head(final_embedding)
            
            # --- OUTPUT FORMATTING LOGIC ---
            if self.task_conf['type'] == "multiclass":
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[0][pred_idx].item()
                
                results["prediction"] = pred_idx
                results["confidence"] = f"{confidence:.2%}"
                
                # Attach readable labels
                if self.task_name == "gtzan":
                    results["label"] = GTZAN_LABELS[pred_idx] if pred_idx < len(GTZAN_LABELS) else "Unknown"
                elif self.task_name == "giantsteps":
                    results["label"] = GIANTSTEPS_LABELS[pred_idx] if pred_idx < len(GIANTSTEPS_LABELS) else "Unknown"
                elif self.task_name == "vocal_singer":
                    results["label"] = VOCAL_SINGER_LABELS[pred_idx] if pred_idx < len(VOCAL_SINGER_LABELS) else "Unknown"
                elif self.task_name == "vocal_tech":
                    results["label"] = VOCAL_TECH_LABELS[pred_idx] if pred_idx < len(VOCAL_TECH_LABELS) else "Unknown"
                elif "nsynth_inst" in self.task_name:
                    results["label"] = f"Class {pred_idx}"
                elif "nsynth_pitch" in self.task_name:
                     results["label"] = f"MIDI {pred_idx}"

            elif self.task_conf['type'] == "multilabel":
                probs = torch.sigmoid(logits)
                top_probs, top_indices = torch.topk(probs, 3) # Top 3 for table conciseness
                
                # We just return the indices and scores, formatting happens in the printer
                results["top_indices"] = top_indices[0].tolist()
                results["top_scores"] = [f"{p:.2f}" for p in top_probs[0].tolist()]

            elif self.task_conf['type'] == "regression":
                vals = logits[0].tolist()
                if self.task_name == "emomusic":
                    results["arousal"] = f"{vals[0]:.2f}"
                    results["valence"] = f"{vals[1]:.2f}"
                else:
                    results["values"] = vals

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MERT inference on audio.")
    parser.add_argument("filepath", type=str, help="Path to audio file")
    parser.add_argument("--task", type=str, required=True, help="Task name from config.py or 'all'")
    parser.add_argument("--head", type=str, default=None, help="Path to .pth file. (Ignored if task='all')")
    parser.add_argument("--limit", type=float, default=config.MAX_DURATION)
    
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        sys.exit(f"Error: File not found at {args.filepath}")

    # 1. Determine which tasks to run
    tasks_to_run = []
    if args.task.lower() == 'all':
        tasks_to_run = list(config.TASKS.keys())
        print(f"Analyzing {os.path.basename(args.filepath)} on all available tasks...")
    else:
        tasks_to_run = [args.task]

    # 2. Load Foundation Model ONCE
    model_actual_id = config.MODEL_ID
    if os.path.exists("./" + model_actual_id):
        model_actual_id = "./" + model_actual_id
    
    # Suppress verbose loading logs if running 'all'
    if args.task != 'all':
        print(f"Loading Base MERT Model: {model_actual_id}...")
        
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_actual_id, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(model_actual_id, trust_remote_code=True)
        base_model.eval()
    except Exception as e:
        sys.exit(f"CRITICAL: Failed to load base MERT model. {e}")

    # 3. Iterate and Run
    aggregated_results = {}

    for task in tasks_to_run:
        if args.task.lower() == 'all':
            head_path = f"./features/trained_{task}.pth"
        else:
            head_path = args.head if args.head else f"./features/trained_{task}.pth"

        engine = MERTInference(
            task_name=task, 
            head_weights_path=head_path, 
            base_model=base_model, 
            processor=processor
        )
        
        res = engine.process_file(args.filepath, max_duration=args.limit)
        aggregated_results[task] = res if res else {"status": "NA"}

    # 4. Print Formatted Table
    print("\n" + "="*85)
    print(f"{'MIR Task':<30} | {'Prediction / Value':<30} | {'Detail / Score':<20}")
    print("-" * 85)

    for task_key in tasks_to_run:
        res = aggregated_results.get(task_key, {"status": "NA"})
        display_name = TASK_DISPLAY_NAMES.get(task_key, task_key) # Fallback to key if not in map

        # Case 1: N/A (Head missing or Failed)
        if "status" in res and res["status"] == "NA":
            print(f"{display_name:<30} | {'N/A':<30} | {'N/A':<20}")
            continue
            
        # Case 2: Error
        if "error" in res:
            print(f"{display_name:<30} | {'Error':<30} | {res['error']:<20}")
            continue

        # Case 3: Multiclass (Genre, Key, etc)
        if "prediction" in res:
            main_pred = res.get("label", str(res["prediction"]))
            detail = f"Conf: {res['confidence']}"
            print(f"{display_name:<30} | {main_pred:<30} | {detail:<20}")

        # Case 4: Multilabel (Tagging)
        elif "top_indices" in res:
            # Join top 3 indices for brevity
            indices_str = ", ".join(map(str, res['top_indices']))
            scores_str = ", ".join(res['top_scores'])
            print(f"{display_name:<30} | IDs: {indices_str:<25} | {scores_str:<20}")

        # Case 5: Regression (Emotion)
        elif "arousal" in res:
            val_str = f"A: {res['arousal']}, V: {res['valence']}"
            print(f"{display_name:<30} | {val_str:<30} | {'(Range -1 to 1)':<20}")

    print("="*85 + "\n")