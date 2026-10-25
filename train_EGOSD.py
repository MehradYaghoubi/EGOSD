

import argparse
import os, sys, json, math, random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
import logging
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# --- Local Imports ---
from Configs.config import MainConfig
from DataProcessor.ULBiTP_data_loader import  ULBiTPDataset, ULBiTP_collate_fn
# Import updated model and loss class
from Model.egosd import (
    ULBiTP_R_v8_VLBP_FullFeedback as ULBiTP_R_v8_EGOSD_FullFeedback,
    UncertaintyAwareLoss as EGOSD_Loss,
    count_parameters,
)
from DataProcessor.kornia_augs import KorniaAugmentation
from Utils.custom_loss import LabelSmoothedCTCLoss
from Utils.metrics import calculate_per

# ---------------- Logging ----------------
def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        ch.setFormatter(fmt); fh.setFormatter(fmt)
        logger.addHandler(ch); logger.addHandler(fh)
    return logger

# ---------------- Decoding & Utils ----------------
def greedy_decode_batch(log_probs, id_to_phoneme, blank_id):
    seq = log_probs.permute(1, 0, 2) # (B, T, C)
    preds = torch.argmax(seq, dim=-1)
    decoded = []
    for row in preds:
        toks, last = [], -1
        for x in row:
            idx = int(x)
            if idx != blank_id and idx != last:
                toks.append(id_to_phoneme.get(idx, "?"))
            last = idx
        decoded.append(" ".join(toks))
    return decoded

def get_lr(optimizer): return optimizer.param_groups[0]["lr"]

def train_ulbitp_egosd(resume_checkpoint=None):
    # 1. Setup Config & Environment
    cfg = MainConfig()
    if resume_checkpoint:
        run_name = f"ULBiTP_EGOSD_RESUMED_{datetime.now().strftime('%m%d_%H%M')}"
    else:
        run_name = f"ULBiTP_EGOSD_CTC_{datetime.now().strftime('%m%d_%H%M')}"
        
    logger = setup_logging(cfg.training.LOG_DIR, run_name)
    device = torch.device(cfg.training.DEVICE)
    logger.info(f"🚀 Starting Training: {run_name} on {device}")
    
    if resume_checkpoint:
        logger.info(f"Resuming from: {resume_checkpoint}")
    
    seed = getattr(cfg.training, "SEED", 42)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # 2. Load Phoneme Map
    with open(cfg.dataset.PHONEME_MAP_FILE, "r", encoding="utf-8") as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {int(v): k for k, v in phoneme_to_id.items()}
    blank_id = phoneme_to_id.get("<blank>", 0)

    # 3. Data Loaders
    train_ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT, "train", "phoneme")
    val_ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT, "val", "phoneme")
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.training.NUM_WORKERS, collate_fn=ULBiTP_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.INFERENCE_BATCH_SIZE, shuffle=False,
                            num_workers=getattr(cfg.training, "VAL_NUM_WORKERS", 2), collate_fn=ULBiTP_collate_fn)

    # 4. Model Init
    model = ULBiTP_R_v8_EGOSD_FullFeedback(cfg.rlx_config).to(device)
    logger.info(f"🧠 Model Parameters: {count_parameters(model):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    
    augmentation = KorniaAugmentation(cfg.train_aug, cfg.dataset).to(device)

    # -----------------------------------------------------
    # LOSS SETUP (Hybrid Approach)
    # -----------------------------------------------------
    # 1. Main Loss (with Label Smoothing and Alpha Decay) - KEPT!
    ctc_loss_fn = torch.nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True).to(device)
    # ctc_loss_fn = LabelSmoothedCTCLoss(cfg.dataset.NUM_PHONEME_CLASSES, blank_id,
    #                                    alpha=cfg.training.label_smoothing.DEFAULT_ALPHA).to(device)
    
    # 2. Consistency Loss Calculator (The Logic Mover) - NEW!
    # This class now handles KL, Masking, and Belief CTC internally
    consistency_calc = EGOSD_Loss().to(device)

    # Hyperparameters
    KL_START_EPOCH = 12       
    CONSISTENCY_WEIGHT = 2.0 
    BELIEF_CTC_WEIGHT = 0.5   

    best_val_per = float("inf")
    train_losses, val_pers = [], []
    per_no_improve_count = 0

    # <<< MODIFIED: RESUME LOGIC >>>
    if resume_checkpoint:
        if os.path.isfile(resume_checkpoint):
            logger.info(f"Loading checkpoint '{resume_checkpoint}' ...")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            
            model.load_state_dict(checkpoint['model_state'])
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            start_epoch = checkpoint.get('epoch', 0)
            best_val_per = checkpoint.get('best_val_per', float('inf'))
            
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
        else:
            logger.error(f"Checkpoint file not found: {resume_checkpoint}")
            return

    # <<< MODIFIED: FORCE LR LOGIC >>>

    # 5. Training Loop
    for epoch in range(cfg.training.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_cons_loss = 0.0
        total_belief_loss = 0.0
        avg_uncertainty = 0.0
        
        is_warmup = epoch < KL_START_EPOCH
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", ncols=140)
        
        for step, (videos, ctc_targets, video_lengths, target_lengths, _) in enumerate(pbar):
            videos = videos.to(device, non_blocking=True)
            ctc_targets = ctc_targets.to(device, non_blocking=True)
            video_lengths = video_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            if getattr(cfg.train_aug, "USE_AUGMENTATION", True):
                videos = augmentation(videos)

            optimizer.zero_grad()

            # --- Forward Pass ---
            out = model(videos, input_lengths=video_lengths)
            
            log_probs = out[0]          # Main Logits
            out_lens = out[1]
            aux_logp = out[2]
            belief_logits = out[3]      # Belief Logits
            uncertainty = out[4]        # Uncertainty

            # --- 1. Main Task Loss (Original Logic) ---
            loss_main = ctc_loss_fn(log_probs, ctc_targets, out_lens, target_lengths)

            # --- 2. Aux Loss ---
            loss_aux = torch.tensor(0.0, device=device)
            if cfg.rlx_config.use_aux_head and aux_logp is not None:
                loss_aux = ctc_loss_fn(aux_logp, ctc_targets, out_lens, target_lengths)

            # --- 3. Belief & Consistency (Calculated via Class) ---
            loss_belief = torch.tensor(0.0, device=device)
            loss_consistency = torch.tensor(0.0, device=device)

            if belief_logits is not None:
                # Use the class to calculate robust losses
                loss_dict = consistency_calc(
                    main_log_probs=log_probs,
                    belief_logits=belief_logits,
                    uncertainty=uncertainty,
                    targets=ctc_targets,
                    input_lengths=video_lengths,
                    target_lengths=target_lengths
                )
                
                # Apply weights
                loss_belief = loss_dict["belief"] * BELIEF_CTC_WEIGHT
                
                if not is_warmup:
                    loss_consistency = loss_dict["consistency"] * CONSISTENCY_WEIGHT

            # --- Total Loss ---
            final_loss = loss_main + \
                         (cfg.rlx_config.aux_loss_weight * loss_aux) + \
                         loss_belief + \
                         loss_consistency

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.GRAD_NORM_CLIP)
            optimizer.step()

            # Logging
            total_loss += final_loss.item()
            total_cons_loss += loss_consistency.item()
            total_belief_loss += loss_belief.item()
            if uncertainty is not None:
                avg_uncertainty += uncertainty.mean().item()

            pbar.set_postfix(
                loss=final_loss.item(), 
                cons=loss_consistency.item(),
                unc=uncertainty.mean().item() if uncertainty is not None else 0.0
            )

        # --- End of Epoch Stats ---
        steps = len(train_loader)
        avg_loss = total_loss / steps
        avg_cons = total_cons_loss / steps
        avg_unc = avg_uncertainty / steps
        
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Consist: {avg_cons:.4f} | Uncert: {avg_unc:.4f} |  lr = {get_lr(optimizer)}")
        train_losses.append(avg_loss)

        # 6. Validation Loop
        model.eval()
        val_loss = 0.0; all_preds = []; all_gts = []
        
        with torch.no_grad():
            for videos, ctc_targets, video_lengths, target_lengths, _ in tqdm(val_loader, desc="[Val]"):
                videos = videos.to(device, non_blocking=True)
                ctc_targets = ctc_targets.to(device, non_blocking=True)
                video_lengths = video_lengths.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)

                out_val = model(videos, input_lengths=video_lengths)
                
                if isinstance(out_val, tuple):
                    log_probs_val = out_val[0]
                    out_lens_val = out_val[1]
                else:
                    log_probs_val, out_lens_val = out_val
                
                val_loss += ctc_loss_fn(log_probs_val, ctc_targets, out_lens_val, target_lengths).item()

                preds = greedy_decode_batch(log_probs_val, id_to_phoneme, blank_id)
                all_preds.extend(preds)

                start = 0
                for L in target_lengths.cpu(): 
                    L = int(L)
                    toks = [id_to_phoneme.get(int(t), "?") for t in ctc_targets[start:start+L].tolist()]
                    all_gts.append(" ".join(toks))
                    start += L

        val_per = calculate_per(all_preds, all_gts, standard=True)
        avg_val_loss = val_loss / len(val_loader)
        val_pers.append(val_per)
        
        logger.info(f"Epoch {epoch+1} VAL | Loss: {avg_val_loss:.4f} | PER: {(val_per*100):.2f}%")

        # <<< MODIFIED: FIXED ORDER - Scheduler step BEFORE saving >>>
        scheduler.step(val_per)

        # 7. Checkpointing
        # Save Last Model (for resuming)
        ckpt = {
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), # Now contains updated LR
            "best_val_per": best_val_per,
        }
        torch.save(ckpt, cfg.training.CHECKPOINT_DIR / f"{run_name}_last_epoch_{epoch+1}.pth")

        # 7. Checkpointing
        if val_per < best_val_per:
            best_val_per = val_per
            torch.save(model.state_dict(), cfg.training.CHECKPOINT_DIR / f"{run_name}_best.pth")
            logger.info(f" BEST PER: {(val_per*100):.2f}%")
            per_no_improve_count = 0
        else:
            per_no_improve_count += 1
            # --- ALPHA DECAY ---
            # if per_no_improve_count >= 2:
            #     old_alpha = ctc_loss_fn.alpha
            #     new_alpha = max(old_alpha * 0.6, 0.0) 
            #     ctc_loss_fn.alpha = new_alpha
            #     logger.info(f"📉 Plateau detected! Decaying Label Smoothing Alpha: {old_alpha:.4f} -> {new_alpha:.4f}")
            #     per_no_improve_count = 0 
            if per_no_improve_count >= 6:
                logger.info("No improvement for 6 epochs, stopping training.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ULBiTP Model (egosd)")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()
    
    train_ulbitp_egosd(resume_checkpoint=args.resume)
