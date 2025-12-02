import os
import glob
import pandas as pd
import argparse

# Comprehensive list of VocalSet keywords
# We map variations (fast_forte, forte) to standard IDs
TECHNIQUE_MAP = {
    'belt': 'belt',
    'breathy': 'breathy',
    'inhaled': 'inhaled',
    'lip_trill': 'lip_trill',
    'spoken': 'spoken',
    'straight': 'straight',
    'trill': 'trill',
    'vibrato': 'vibrato',
    'vocal_fry': 'vocal_fry',
    'fry': 'vocal_fry',
    # Dynamics often treated as techniques in VocalSet
    'fast_forte': 'fast_forte',
    'fast_piano': 'fast_piano',
    'slow_forte': 'slow_forte',
    'slow_piano': 'slow_piano',
    'forte': 'fast_forte', # Fallback
    'piano': 'fast_piano', # Fallback
    'glissando': 'glissando',
    'runs': 'runs',
    'arpeggio': 'arpeggio', 
    'scale': 'scale'
}

def generate_vocalset_csv(input_root, output_csv):
    print(f"--- Scanning VocalSet in {input_root} ---")
    
    files = glob.glob(os.path.join(input_root, "**", "*.wav"), recursive=True)
    if not files:
        print("Error: No WAV files found.")
        return

    data = []
    singers = set()
    found_techs = set()

    print(f"Found {len(files)} files. Parsing metadata...")

    for file_path in files:
        # Normalize path
        norm_path = os.path.normpath(file_path)
        parts = norm_path.split(os.sep)
        
        singer_id = None
        tech_id = "other" # Default if not found
        
        # 1. Detect Singer (Required)
        for part in parts:
            part_lower = part.lower()
            if (part_lower.startswith('female') or part_lower.startswith('male')) and any(c.isdigit() for c in part):
                singer_id = part_lower
                break
        
        if not singer_id:
            continue # Skip files that aren't inside a singer folder
            
        singers.add(singer_id)

        # 2. Detect Technique (Optional - greedy match)
        for part in parts:
            part_lower = part.lower()
            # Check against our map
            for key, val in TECHNIQUE_MAP.items():
                if key in part_lower:
                    tech_id = val
                    break
            if tech_id != "other":
                break
        
        found_techs.add(tech_id)
        
        data.append({
            'filename': os.path.basename(file_path),
            'singer': singer_id,
            'technique': tech_id
        })

    # Sort for consistent Class IDs
    sorted_singers = sorted(list(singers))
    # Ensure 'other' is at the end or specific spot if you prefer
    sorted_techs = sorted(list(found_techs))
    
    singer_map = {s: i for i, s in enumerate(sorted_singers)}
    tech_map = {t: i for i, t in enumerate(sorted_techs)}
    
    print(f"\nIdentified {len(sorted_singers)} Singers (Classes 0-{len(sorted_singers)-1})")
    print(f"Identified {len(sorted_techs)} Techniques (Classes 0-{len(sorted_techs)-1})")
    print(f"Techniques Found: {sorted_techs}")
    
    # Compile Final Data
    csv_rows = []
    for entry in data:
        csv_rows.append({
            'filename': entry['filename'],
            'singer_label': singer_map[entry['singer']],
            'tech_label': tech_map[entry['technique']],
            'singer_str': entry['singer'],
            'tech_str': entry['technique']
        })
        
    df = pd.DataFrame(csv_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Saved metadata to: {output_csv}")
    print(f"Total samples processed: {len(df)}") # Should now match ~3613

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="./train_data/meta/vocalset.csv")
    args = parser.parse_args()
    
    generate_vocalset_csv(args.input, args.output)