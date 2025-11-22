import os
import glob
import torch
import torchaudio
import argparse
from tqdm import tqdm  # pip install tqdm

# --- CONFIGURATION ---
TARGET_SAMPLE_RATE = 24000  # Required by MERT
TARGET_CHANNELS = 1         # MERT typically processes Mono audio

def preprocess_audio_file(input_path, output_path):
    """
    Reads an audio file, converts it to Mono 24kHz WAV, and saves it.
    """
    try:
        # 1. Load Audio (Handles MP3, FLAC, WAV, etc.)
        # info object helps us check sample rate before loading if we wanted optimization,
        # but loading directly is safer for format conversion.
        waveform, sr = torchaudio.load(input_path)

        # 2. Convert to Mono (if Stereo)
        # Shape is (Channels, Time)
        if waveform.shape[0] > 1:
            # Average across channels to get mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Resample to 24kHz
        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        # 4. Save as Raw WAV (PCM Float or 16-bit Int)
        # encoding='PCM_S16' is standard 16-bit WAV, compatible with everything.
        torchaudio.save(output_path, waveform, TARGET_SAMPLE_RATE, encoding="PCM_S16")
        return True

    except Exception as e:
        print(f"\n[Error] Could not process {input_path}: {e}")
        return False

def batch_process(input_dir, output_dir):
    """
    Scans input_dir for audio files and saves processed versions to output_dir.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # List of supported extensions
    extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"Found {len(files)} audio files in '{input_dir}'.")
    
    # Process files with a progress bar
    success_count = 0
    for file_path in tqdm(files, desc="Converting"):
        # Create output filename (keep name, force .wav extension)
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(output_dir, name_without_ext + ".wav")

        if preprocess_audio_file(file_path, output_path):
            success_count += 1

    print(f"\n--- Processing Complete ---")
    print(f"Successfully converted: {success_count}/{len(files)}")
    print(f"Clean raw files are located in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-preprocess audio to MERT-ready WAV format.")
    parser.add_argument("--input", required=True, help="Folder containing input MP3/audio files")
    parser.add_argument("--output", required=True, help="Folder to save clean WAV files")
    
    args = parser.parse_args()
    
    batch_process(args.input, args.output)