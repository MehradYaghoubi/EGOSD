import sys
import os
from pathlib import Path
import json
import re
import numpy as np
from tqdm import tqdm
import cv2
from typing import List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Configs.config import MainConfig
from DataProcessor.generate_viseme_map import generate_viseme_mappings
from g2p_en import G2p

# --- Constants ---
WORD_BOUNDARY_TOKEN = '<wb>'
SIL_TOKEN = 'sil'

def get_num_frames(npy_path):
    # این دستور فایل را لود نمی‌کند، فقط آن را Map می‌کند
    data = np.load(npy_path, mmap_mode='r')
    return data.shape[0]

def parse_align_file(align_file_path: Path, g2p_instance: G2p) -> Tuple[List[str], List[List[str]]]:
    """
    Reads an alignment file, extracts words, and converts them to phonemes.
    Returns the word sequence and a list of phoneme lists (one list per word).
    """
    word_sequence = []
    phoneme_groups = []
    try:
        with open(align_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) == 3:
                    label = parts[2].lower()
                    if label == 'sp': # Skip short pauses, they are not words
                        continue
                    
                    word_sequence.append(label)
                    
                    if label == SIL_TOKEN:
                        phoneme_groups.append([SIL_TOKEN])
                    elif label not in ['unk', '<unk>']:
                        word_phonemes = g2p_instance(label)
                        word_phonemes = [re.sub(r'\d', '', p) for p in word_phonemes]
                        phoneme_groups.append(word_phonemes)
    except Exception as e:
        print(f"Error reading align file '{align_file_path.name}': {e}", flush=True)
        return [], []

    return word_sequence, phoneme_groups

def generate_data_lists(main_config: MainConfig):
    """
    Generates data lists (train.json, val.json, test.json), the LM corpus, and decoder labels.
    """
    print("--- Starting Data List Generation ---", flush=True)
    
    # This script assumes generate_phoneme_map has been run and the file exists.
    # We do not call it from here to maintain separation of concerns.
    
    main_config.dataset.PROCESSED_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        with open(main_config.dataset.PHONEME_MAP_FILE, 'r', encoding='utf-8') as f:
            phoneme_to_id = json.load(f)
        with open(main_config.dataset.PROCESSED_DATASET_ROOT / "character_to_id.json", 'r', encoding='utf-8') as f:
            char_to_id = json.load(f)
        with open(main_config.dataset.PROCESSED_DATASET_ROOT / "viseme_to_id.json", 'r', encoding='utf-8') as f:
            viseme_to_id = json.load(f)
        with open(main_config.dataset.RAW_DATASET_ROOT.parent / "DataProcessor" / "phoneme_to_viseme.json", 'r', encoding='utf-8') as f:
            phoneme_to_viseme = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Required map file not found. Run generate_phoneme_map.py first. Details: {e}")
        return

    try:
        with open(main_config.dataset.SPEAKER_META_FILE, 'r', encoding='utf-8') as f:
            speaker_metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Speaker metadata not found at {main_config.dataset.SPEAKER_META_FILE}. Run speaker_segmenter.py.", flush=True)
        return

    g2p = G2p()
    split_entries_map = {"train": [], "val": [], "test": []}
    all_lm_sentences = []
    
    speaker_to_split_map = {s_id: split for split, s_list in speaker_metadata.items() for s_id in s_list}
    all_known_speakers = sorted(list(speaker_to_split_map.keys()))

    print("Gathering clip information and transcripts...", flush=True)
    for speaker_id in tqdm(all_known_speakers, desc="Processing speakers"):
        split_name = speaker_to_split_map.get(speaker_id, "unknown")
        processed_speaker_video_dir = main_config.dataset.PROCESSED_VIDEO_DIR / split_name / speaker_id
        raw_speaker_align_dir = main_config.dataset.RAW_ALIGNS_DIR / speaker_id
        
        if not processed_speaker_video_dir.exists() or not raw_speaker_align_dir.exists():
            continue

        for npy_file_path in processed_speaker_video_dir.glob("*.npy"):
            base_filename = npy_file_path.stem
            align_file_path = raw_speaker_align_dir / f"{base_filename}.align"
            
            if not align_file_path.exists():
                continue
            
            word_sequence, phoneme_groups = parse_align_file(align_file_path, g2p)
            if not phoneme_groups:
                continue

            # 1. Create the string for the Language Model Corpus
            # This logic correctly places '|' between word-phoneme groups and handles 'sil' tokens.
            word_strings = []
            # This flag tracks if the last added element was a word, to correctly place separators.
            last_was_word = False
            for group in phoneme_groups:
                if group == [SIL_TOKEN]:
                    # If the last element was a word, add a space, otherwise it's a new sentence.
                    if last_was_word:
                        word_strings.append(SIL_TOKEN)
                    else:
                        word_strings.append(SIL_TOKEN)
                    last_was_word = False
                elif group: # It's a word
                    word_strings.append(" ".join(group))
                    last_was_word = True
            lm_corpus_line = " ".join(word_strings)
            all_lm_sentences.append(lm_corpus_line)

            # 2. Create the flat list of phonemes for CTC model training (no word boundaries)
            ctc_phoneme_list = []
            for i, group in enumerate(phoneme_groups):
                ctc_phoneme_list.extend(group)
                # add '|' between words if next group isn't sil or end
            
            # 3. Create a human-readable string for logging in the JSON file
            phoneme_string_for_json = lm_corpus_line.replace("<wb>", "|")

            # --- Video processing to get FPS (simplified for brevity) ---
            # raw_video_path = next((p for p in (main_config.dataset.RAW_VIDEOSET_ROOT / speaker_id).glob(f"{base_filename}.*") if p.suffix in ['.mpg', '.mp4', '.avi']), None)
            # if not raw_video_path: continue
            # cap = cv2.VideoCapture(str(raw_video_path))
            # fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
            # cap.release()
            # if fps <= 0: continue

            relative_npy_path = npy_file_path.relative_to(main_config.dataset.PROCESSED_DATASET_ROOT)
            fps = get_num_frames(npy_file_path)
            
            if split_name in split_entries_map:
                phoneme_ids = [phoneme_to_id[p] for p in ctc_phoneme_list if p in phoneme_to_id]
                # viseme_ids = [viseme_to_id.get(phoneme_to_viseme.get(p, 'viseme_S')) for p in ctc_phoneme_list]
                # viseme_ids = [v_id for v_id in viseme_ids if v_id is not None]
                mapped_words = []
                for w in word_sequence:
                    if w == SIL_TOKEN:
                        mapped_words.append("§")   # silence as a single-character symbol
                    else:
                        mapped_words.append(w)

                text_string = " ".join(mapped_words)
                character_ids = [char_to_id[c] for c in text_string if c in char_to_id]

                split_entries_map[split_name].append({
                    "video_path": str(relative_npy_path),
                    "text": " ".join(word_sequence),
                    "phonemes": phoneme_string_for_json,
                    "phoneme_ids": phoneme_ids,
                    "character_ids": character_ids,
                    "num_frames": fps,
                })

    # --- Write LM Corpus and Decoder Labels ---
    print("\nWriting Language Model and Decoder files...", flush=True)
    output_dir = main_config.dataset.PROCESSED_DATASET_ROOT
    
    lm_corpus_path = main_config.lm_builder.LM_CORPUS_PATH
    with open(lm_corpus_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_lm_sentences))
    print(f"LM corpus with {len(all_lm_sentences)} sentences saved to {lm_corpus_path}", flush=True)

    labels_path = main_config.lm_builder.LABELS_PATH
    try:
        sorted_phonemes = sorted(phoneme_to_id.items(), key=lambda item: item[1])
        # The labels for pyctcdecode must not contain the blank token.
        labels = [phoneme for phoneme, _ in sorted_phonemes if phoneme != '<blank>']
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False)
        print(f"Decoder labels file with {len(labels)} tokens saved to {labels_path}", flush=True)
    except Exception as e:
        print(f"Error writing decoder labels file: {e}", flush=True)

    # --- Write final JSON data lists ---
    print("\nWriting final data lists (JSON)...", flush=True)
    for split, entries in split_entries_map.items():
        with open(output_dir / f"{split}.json", 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=4)
        print(f"{split}.json created with {len(entries)} entries.", flush=True)

    print("\n--- Data List Generation Completed! ---", flush=True)

if __name__ == "__main__":
    main_config_instance = MainConfig()
    generate_data_lists(main_config_instance)