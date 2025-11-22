# mert_extractor.py
import os
import glob
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import config  # Imports variables from config.py

def extract_and_save_features(audio_folder, output_folder):
    print(f"--- Starting Feature Extraction for {audio_folder} ---")
    
    # Load MERT Model
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    model.eval() # Freeze model
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = glob.glob(f"{audio_folder}/*.wav")
    
    for file_path in files:
        try:
            # 1. Load & Resample
            waveform, sr = torchaudio.load(file_path)
            if sr != config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # Mix to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 2. Chunking (Simplified: taking first 5s only)
            target_len = config.SAMPLE_RATE * config.CHUNK_SEC
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            else:
                padding = target_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # 3. Run MERT
            inputs = processor(waveform.squeeze(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                embeddings = outputs.last_hidden_state 
                
            # 4. Save Tensor
            file_name = os.path.basename(file_path).replace('.wav', '.pt')
            save_path = os.path.join(output_folder, file_name)
            torch.save(embeddings, save_path)
            print(f"Processed: {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Example usage:
    # Create folders 'raw_audio' and 'features' before running
    extract_and_save_features("./dataset", "./features")