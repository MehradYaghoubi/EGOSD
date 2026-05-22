import json
from pathlib import Path
from tqdm import tqdm
from g2p_en import G2p
import re
from collections import Counter
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Configs.config import MainConfig

BLANK_TOKEN = '<blank>'
WORD_BOUNDARY_TOKEN = '<wb>'  # Using <wb> instead of '|' for better compatibility
SIL_TOKEN = 'sil'

def generate_phoneme_map():
    """
    Generates a phoneme map by reading all .align files directly.
    """
    print("--- Starting Phoneme Map Generation from .align files ---")
    
    config = MainConfig()
    aligns_root = config.dataset.RAW_ALIGNS_DIR
    output_phoneme_map_path = config.dataset.PROCESSED_DATASET_ROOT / "phoneme_to_id.json"
    output_id_to_phoneme_map_path = config.dataset.PROCESSED_DATASET_ROOT / "id_to_phoneme.json"
    
    try:
        import nltk
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('cmudict', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return

    g2p = G2p()
    all_phonemes = set()

    print("Step 1/3: Reading all .align files...")
    align_files = list(aligns_root.glob("*/*.align"))
    for align_file in tqdm(align_files, desc="Processing .align files"):
        with open(align_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    label = parts[2].lower()
                    if label == 'sil':
                        continue # 'sil' is a special token, added later
                    elif label not in ['sp', 'unk', '<unk>']:
                        word_phonemes = g2p(label)
                        word_phonemes = [re.sub(r'\d', '', p) for p in word_phonemes]
                        all_phonemes.update(word_phonemes)

    if not all_phonemes:
        print("Error: No phonemes found. Please check the .align files.")
        return

    print(f"\nStep 2/3: Building map from {len(all_phonemes)} unique phonemes...")

    sorted_phonemes = sorted(list(all_phonemes))
    
    phoneme_to_id = {}
    
    for phoneme in sorted_phonemes:
        if phoneme not in phoneme_to_id:
            phoneme_to_id[phoneme] = len(phoneme_to_id)
    
    # Add special tokens
    for special_token in [SIL_TOKEN, BLANK_TOKEN]:
        if special_token not in phoneme_to_id:
            phoneme_to_id[special_token] = len(phoneme_to_id)
            
    id_to_phoneme = {idx: ph for ph, idx in phoneme_to_id.items()}

    print(f"Map created with {len(phoneme_to_id)} tokens.")
    print("Sample phonemes:", list(phoneme_to_id.keys())[:15])

    print(f"\nStep 3/3: Saving maps to JSON files...")
    try:
        output_phoneme_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_phoneme_map_path, 'w', encoding='utf-8') as f:
            json.dump(phoneme_to_id, f, indent=4, ensure_ascii=False)
        print(f" Phoneme to ID map saved successfully to {output_phoneme_map_path}")
        
        with open(output_id_to_phoneme_map_path, 'w', encoding='utf-8') as f:
            json.dump(id_to_phoneme, f, indent=4, ensure_ascii=False)
        print(f" ID to Phoneme map saved successfully to {output_id_to_phoneme_map_path}")

    except Exception as e:
        print(f"Error saving JSON files: {e}")

    print("\n--- Phoneme Map Generation Completed! ---")

if __name__ == "__main__":
    generate_phoneme_map()
