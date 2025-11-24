import os
import glob
import torch
import torchaudio
import argparse
import sys
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import config

def extract_and_save_features(input_root, output_folder):
    print(f"--- Starting Feature Extraction ---")
    print(f"Input:  {input_root}")
    print(f"Output: {output_folder}")
    print(f"Settings: Limit={config.EXTRACT_DURATION}s | Chunk={config.CHUNK_SEC}s | Dim={config.HIDDEN_DIM}")
    
    # Load MERT Model
    try:
        print(f"Loading model: {config.MODEL_ID}...")
        processor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        model = AutoModel.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Search for all common audio formats recursively
    extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
    files = []
    for ext in extensions:
        # Search for lowercase extensions
        files.extend(glob.glob(os.path.join(input_root, "**", ext), recursive=True))
        # Search for uppercase extensions (e.g., .WAV, .MP3)
        files.extend(glob.glob(os.path.join(input_root, "**", ext.upper()), recursive=True))

    # Remove duplicates and sort
    files = sorted(list(set(files)))

    if not files:
        print(f"No audio files found in {input_root}")
        return

    print(f"Found {len(files)} audio files. Processing...")
    
    for i, file_path in enumerate(files):
        try:
            # Generate Output Filename
            # Logic: Song.mp3 -> Song.pt
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            save_name = f"{name_without_ext}.pt"
            save_path = os.path.join(output_folder, save_name)
            
            if os.path.exists(save_path):
                continue

            # 1. Load & Resample (Torchaudio handles mp3/flac decoding)
            waveform, sr = torchaudio.load(file_path)
            
            if sr != config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True) # Mono

            # 2. LIMIT DURATION
            max_len = int(config.SAMPLE_RATE * config.EXTRACT_DURATION)
            if waveform.shape[1] > max_len:
                waveform = waveform[:, :max_len]

            # 3. CHUNKING STRATEGY
            chunk_size = int(config.SAMPLE_RATE * config.CHUNK_SEC)
            stride = chunk_size
            
            if waveform.shape[1] < chunk_size:
                # Pad if too short
                padding = chunk_size - waveform.shape[1]
                chunks = torch.nn.functional.pad(waveform, (0, padding))
                if chunks.dim() == 1: chunks = chunks.unsqueeze(0) # Ensure (1, Samples)
            else:
                # Unfold creates (Num_Chunks, Chunk_Size)
                chunks = waveform.squeeze().unfold(0, chunk_size, stride)

            # 4. Process Chunks
            processed_chunks = []
            
            # Process in small batches to save GPU memory
            BATCH_PROC_SIZE = 4
            for j in range(0, len(chunks), BATCH_PROC_SIZE):
                batch_audio = chunks[j : j + BATCH_PROC_SIZE]
                
                if batch_audio.dim() == 1: 
                    batch_audio = batch_audio.unsqueeze(0)
                
                inputs = processor(batch_audio.numpy(), 
                                 sampling_rate=config.SAMPLE_RATE, 
                                 return_tensors="pt", 
                                 padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    # Get last layer: (Batch, Time, Dim)
                    processed_chunks.append(outputs.last_hidden_state)

            # 5. Concatenate & Save
            if len(processed_chunks) > 0:
                full_embedding = torch.cat(processed_chunks, dim=0)
                # Reshape to (Total_Time, Dim)
                full_embedding = full_embedding.view(-1, config.HIDDEN_DIM)
                
                # Save Tensor
                torch.save(full_embedding, save_path)
                
                # Simple progress indicator
                print(f"[{i}/{len(files)}] Saved {save_name} ({full_embedding.shape})")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MERT features from a folder of audio files.")
    
    # Arguments for input and output folders
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the folder containing audio files (mp3, wav, flac, etc.)")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save the extracted feature tensors (.pt)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        sys.exit(1)
        
    extract_and_save_features(args.input, args.output)