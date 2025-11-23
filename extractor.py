# mert_extractor.py
import os
import glob
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import config

def extract_and_save_features(input_root, output_folder):
    print(f"--- Starting Feature Extraction for {input_root} ---")
    print(f"Settings: Limit={config.MAX_DURATION}s | Chunk={config.CHUNK_SEC}s | Dim={config.HIDDEN_DIM}")
    
    # Load MERT Model
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    model.eval()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recursive search for .wav files
    search_path = os.path.join(input_root, "**", "*.wav")
    files = glob.glob(search_path, recursive=True)
    print(f"Found {len(files)} files. Processing...")
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            save_name = filename.replace('.wav', '.pt')
            save_path = os.path.join(output_folder, save_name)
            
            if os.path.exists(save_path):
                continue

            # 1. Load & Resample
            waveform, sr = torchaudio.load(file_path)
            if sr != config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True) # Mono

            # 2. LIMIT DURATION (The "Pre-Process" Cut)
            # Use MAX_DURATION from config (e.g., 30 seconds)
            max_len = int(config.SAMPLE_RATE * config.MAX_DURATION)
            if waveform.shape[1] > max_len:
                waveform = waveform[:, :max_len]
            # If shorter, we don't pad yet; the chunking loop handles it.

            # 3. CHUNKING STRATEGY (To avoid OOM on GPU)
            # MERT loves 5-second inputs [cite: 185]
            chunk_size = int(config.SAMPLE_RATE * config.CHUNK_SEC) # 5s samples
            stride = chunk_size # Non-overlapping chunks for training data efficiency
            
            # Unfold creates windows: (Num_Chunks, Chunk_Size)
            if waveform.shape[1] < chunk_size:
                # Pad if the TOTAL file is shorter than 5s
                padding = chunk_size - waveform.shape[1]
                chunks = torch.nn.functional.pad(waveform, (0, padding))
            else:
                chunks = waveform.squeeze().unfold(0, chunk_size, stride)
                # Handle remaining frames if they don't fit a full chunk? 
                # Usually easiest to drop the partial tail for training consistency.

            # 4. Process Chunks in Batch
            processed_chunks = []
            
            # Process chunks (we do this in a loop to be safe on memory)
            # or pass all chunks as one batch if GPU RAM allows.
            # Let's do batch size of 4 to be safe.
            BATCH_PROC_SIZE = 4
            for i in range(0, len(chunks), BATCH_PROC_SIZE):
                batch_audio = chunks[i : i + BATCH_PROC_SIZE]
                
                # Ensure 2D shape for processor: (Batch, Samples)
                if batch_audio.dim() == 1: 
                    batch_audio = batch_audio.unsqueeze(0)
                
                inputs = processor(batch_audio.numpy(), 
                                 sampling_rate=config.SAMPLE_RATE, 
                                 return_tensors="pt", 
                                 padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    # Shape: (Batch, 375, Hidden_Dim)
                    processed_chunks.append(outputs.last_hidden_state)

            # 5. Concatenate Back Together
            # Shape: (Total_Time_Steps, Hidden_Dim)
            if len(processed_chunks) > 0:
                full_embedding = torch.cat(processed_chunks, dim=0)
                # Flatten batch dimension if it exists from concatenation
                full_embedding = full_embedding.view(-1, config.HIDDEN_DIM)
                
                # Save Tensor
                torch.save(full_embedding.unsqueeze(0), save_path) # Save as (1, Time, Dim)
                print(f"Saved {save_name}: {full_embedding.shape}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Point to your RAW WAV data (processed by preprocess.py)
    INPUT_ROOT = "./raw_data/genres" 
    OUTPUT_DIR = "./features/gtzan"
    
    extract_and_save_features(INPUT_ROOT, OUTPUT_DIR)