

import os, sys, json, random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- Local Imports ---
from Configs.config import MainConfig
from DataProcessor.ULBiTP_data_loader import ULBiTPDataset, ULBiTP_collate_fn
from Utils.metrics import calculate_per

# --- IMPORT YOUR MODELS HERE ---
# مطمئن شوید نام فایل‌ها و کلاس‌ها دقیق است
from Model.base import ULBiTP_R_v7  # معماری پایه
from Model.egosd import ULBiTP_R_v8_VLBP_FullFeedback # معماری پیشنهادی (Belief)

def load_checkpoint(model, path, device):
    print(f" Loading: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    ckpt = torch.load(path, map_location=device)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    
    # حذف پیشوند module. اگر وجود دارد
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

def run_comparison(base_ckpt_path, belief_ckpt_path):
    cfg = MainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Starting Comparison on {device}")

    # 1. Load Vocab
    with open(cfg.dataset.PHONEME_MAP_FILE, "r") as f:
        phoneme_to_id = json.load(f)
    sorted_phonemes = [k for k, v in sorted(phoneme_to_id.items(), key=lambda x: x[1])]
    blank_id = phoneme_to_id.get("<blank>", 0)

    # 2. Data Loader
    test_ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT,  "test", "phoneme")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=ULBiTP_collate_fn)
    print(f" Test Samples: {len(test_ds)}")

    # 3. Initialize Models
    print("\n Initializing Models...")
    # Base Model
    model_base = ULBiTP_R_v7(cfg.rlx_config).to(device)
    load_checkpoint(model_base, base_ckpt_path, device)
    
    # Belief Model
    model_belief = ULBiTP_R_v8_VLBP_FullFeedback(cfg.rlx_config).to(device)
    load_checkpoint(model_belief, belief_ckpt_path, device)

    # 4. Comparison Loop
    base_preds = []
    belief_preds = []
    ground_truths = []
    
    # لیست‌هایی برای تحلیل کیفی
    ugml_wins = []   # Base غلط، Belief درست
    base_wins = []   # Base درست، Belief غلط
    both_wrong = []  # هر دو غلط
    
    print("\n Running Inference Duel...")
    with torch.no_grad():
        for videos, ctc_targets, video_lengths, target_lengths, _ in tqdm(test_loader):
            videos = videos.to(device)
            video_lengths = video_lengths.to(device)
            
            # --- Run Base Model ---
            out_base = model_base(videos, input_lengths=video_lengths)
            logits_base = out_base[0] if isinstance(out_base, tuple) else out_base
            
            # --- Run Belief Model ---
            out_belief = model_belief(videos, input_lengths=video_lengths)
            logits_belief = out_belief[0] if isinstance(out_belief, tuple) else out_belief
            
            # --- Process Batch ---
            logits_base_np = logits_base.permute(1,0,2).cpu().numpy()
            logits_belief_np = logits_belief.permute(1,0,2).cpu().numpy()
            ctc_targets_cpu = ctc_targets.cpu()
            
            start = 0
            for i, L in enumerate(target_lengths.cpu()):
                L = int(L)
                valid_T = int(video_lengths[i].item())
                
                # 1. Get Ground Truth
                gt_toks = [sorted_phonemes[int(t)] for t in ctc_targets_cpu[start:start+L]]
                gt_text = " ".join(gt_toks)
                ground_truths.append(gt_text)
                start += L
                
                # 2. Decode Base
                base_idxs = np.argmax(logits_base_np[i][:valid_T], axis=-1)
                base_text = decode_greedy(base_idxs, sorted_phonemes, blank_id)
                base_preds.append(base_text)
                
                # 3. Decode Belief
                belief_idxs = np.argmax(logits_belief_np[i][:valid_T], axis=-1)
                belief_text = decode_greedy(belief_idxs, sorted_phonemes, blank_id)
                belief_preds.append(belief_text)
                
                # 4. Compare
                base_correct = (base_text == gt_text)
                belief_correct = (belief_text == gt_text)
                
                sample_info = {
                    "GT": gt_text,
                    "Base": base_text,
                    "Belief": belief_text
                }
                
                if not base_correct and belief_correct:
                    ugml_wins.append(sample_info)
                elif base_correct and not belief_correct:
                    base_wins.append(sample_info)
                elif not base_correct and not belief_correct:
                    both_wrong.append(sample_info)

    # 5. Metrics & Reporting
    per_base = calculate_per(base_preds, ground_truths, standard=True) * 100
    per_belief = calculate_per(belief_preds, ground_truths, standard=True) * 100
    
    print("\n" + "="*50)
    print(f"{'Model':<20} | {'PER (%)':<10}")
    print("-" * 50)
    print(f"{'Baseline (v7)':<20} | {per_base:.2f}%")
    print(f"{'Proposed (UGML)':<20} | {per_belief:.2f}%")
    print("-" * 50)
    print(f"improvement: {per_base - per_belief:.2f}% (Absolute)")
    print("="*50)
    
    # 6. Write Detailed Analysis File
    log_file = "model_comparison_analysis.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"COMPARISON REPORT\n")
        f.write(f"Baseline PER: {per_base:.2f}%\n")
        f.write(f"Proposed PER: {per_belief:.2f}%\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"🏆 SECTION 1: PROPOSED MODEL WINS ({len(ugml_wins)} samples)\n")
        f.write(f"(Cases where Baseline failed but Proposed succeeded)\n")
        f.write("-" * 60 + "\n")
        for s in ugml_wins[:30]: # 30 نمونه اول
            f.write(f"GT      : {s['GT']}\n")
            f.write(f"Base ❌ : {s['Base']}\n")
            f.write(f"Belief ✅: {s['Belief']}\n")
            f.write("\n")
            
        f.write(f"\n📉 SECTION 2: BASELINE WINS ({len(base_wins)} samples)\n")
        f.write("-" * 60 + "\n")
        for s in base_wins[:15]:
            f.write(f"GT      : {s['GT']}\n")
            f.write(f"Base ✅ : {s['Base']}\n")
            f.write(f"Belief ❌: {s['Belief']}\n")
            f.write("\n")
            
        f.write(f"\n💀 SECTION 3: BOTH FAILED ({len(both_wrong)} samples)\n")
        f.write("-" * 60 + "\n")
        for s in both_wrong[:15]:
            f.write(f"GT      : {s['GT']}\n")
            f.write(f"Base ❌ : {s['Base']}\n")
            f.write(f"Belief ❌: {s['Belief']}\n")
            f.write("\n")

    print(f"\n Detailed comparison log saved to: {log_file}")

def decode_greedy(idxs, vocab, blank_id):
    toks = []
    last = -1
    for idx in idxs:
        if idx != blank_id and idx != last:
            toks.append(vocab[idx])
        last = idx
    return " ".join(toks)

if __name__ == "__main__":
    # --- PATHS ---
    # مسیر چک پوینت مدل پایه (که الان دارید آموزش می‌دهید)
    BASE_CKPT = "./ch/ULBiTPR_RESUMED_0217_223430_best_per_0.086.pth" 
    
    # مسیر چک پوینت مدل جدید (که ۶.۷٪ شد)
    BELIEF_CKPT = "./ch/ULBiTP_EGOSD_CTC_0218_0007_best.pth"
    
    if os.path.exists(BASE_CKPT) and os.path.exists(BELIEF_CKPT):
        run_comparison(BASE_CKPT, BELIEF_CKPT)
    else:
        print(" Please set the correct checkpoint paths in the script.")
