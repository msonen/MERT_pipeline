import pandas as pd
import os
import csv
import argparse

def prepare_mtt(mtt_root, output_csv):
    """
    Parses MagnaTagATune annotations to generate a clean label file 
    for the Top 50 tags.
    """
    annot_path = os.path.join(mtt_root, "annotations_final.csv")
    if not os.path.exists(annot_path):
        print(f"Error: Could not find {annot_path}")
        return

    print("Loading annotations...")
    df = pd.read_csv(annot_path, sep="\t")
    
    # 1. Identify Top 50 Tags
    # Drop non-tag columns to count frequencies
    tag_cols = df.columns[1:-1] # Skip 'clip_id' and 'mp3_path'
    tag_counts = df[tag_cols].sum().sort_values(ascending=False)
    top_50_tags = tag_counts.head(50).index.tolist()
    
    print(f"--- Top 50 Tags Selected ---")
    print(top_50_tags[:10], "...")

    # 2. Filter DataFrame to keep only Top 50 columns + metadata
    keep_cols = ['mp3_path'] + top_50_tags
    df_filtered = df[keep_cols].copy()
    
    # 3. Create a compact mapping: filename -> list of active indices
    # We will save a CSV with: filename, tag_indices_str (e.g., "0,4,12")
    processed_data = []
    
    print("Processing rows...")
    for idx, row in df_filtered.iterrows():
        # Get indices where tag is 1
        active_indices = [i for i, tag in enumerate(top_50_tags) if row[tag] == 1]
        
        # Skip files with no top-50 tags
        if not active_indices:
            continue
            
        # Clean path: 'f/american_bach_soloists...' -> 'american_bach_soloists...'
        # This depends on how your extractor saved the files. 
        # Usually, MERT extractor uses os.path.basename.
        # Let's assume we match by basename.
        full_path = row['mp3_path']
        base_name = os.path.basename(full_path)
        # Ensure it maps to .wav (as expected by dataset.py)
        wav_name = os.path.splitext(base_name)[0] + ".wav"
        
        indices_str = ";".join(map(str, active_indices))
        processed_data.append([wav_name, indices_str])

    # 4. Save
    out_df = pd.DataFrame(processed_data, columns=['filename', 'tag_indices'])
    out_df.to_csv(output_csv, index=False)
    
    # Save the tag vocabulary for later reference
    vocab_path = output_csv.replace(".csv", "_vocab.txt")
    with open(vocab_path, "w") as f:
        for t in top_50_tags:
            f.write(f"{t}\n")
            
    print(f"Saved processed labels to {output_csv} ({len(out_df)} samples)")
    print(f"Saved vocabulary to {vocab_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to folder containing annotations_final.csv")
    parser.add_argument("--output", type=str, default="./meta/mtt_top50.csv")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    prepare_mtt(args.input, args.output)