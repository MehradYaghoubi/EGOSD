
import os, sys, json, random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

# --- Local Imports ---
from Configs.config import MainConfig
from DataProcessor.ULBiTP_data_loader import ULBiTPDataset, ULBiTP_collate_fn
from Model.egosd import ULBiTP_R_v8_VLBP_FullFeedback, count_parameters
from DataProcessor.kornia_augs import KorniaAugmentation
from Utils.metrics import calculate_per

# ---------------- Logging ----------------
import logging
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

# ---------------- No-Gate Consistency Loss ----------------
class UncertaintyAwareLossNoGate(nn.Module):
    """Same as EGOSD loss, but all frames treated equally (no gating)"""
    def __init__(self):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.kl_div = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, main_log_probs, belief_logits, uncertainty, targets, input_lengths, target_lengths):
        # Belief CTC
        belief_log_probs = torch.log_softmax(belief_logits, dim=-1).permute(1,0,2)
        loss_belief = self.ctc_loss(belief_log_probs, targets, input_lengths, target_lengths)

        # Consistency Loss (No Gate)
        p_main = main_log_probs.permute(1,0,2)
        p_belief_target = torch.log_softmax(belief_logits.detach(), dim=-1)

        kl = self.kl_div(p_main, p_belief_target).sum(dim=-1)

        # --- Masking for padding ---
        if input_lengths is not None:
            max_len = kl.size(1)
            mask = torch.arange(max_len, device=kl.device).unsqueeze(0) < input_lengths.unsqueeze(1)
            kl = kl * mask
            weighted_consistency = kl.sum() / (mask.sum() + 1e-8)
        else:
            weighted_consistency = kl.mean()

        return {"belief": loss_belief, "consistency": weighted_consistency}

# ---------------- Training Loop ----------------
def train_ulbitp_vlbp_grid_no_gate(resume_checkpoint=None):
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

    # --- Seed ---
    seed = getattr(cfg.training, "SEED", 42)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # --- Phoneme Map ---
    with open(cfg.dataset.PHONEME_MAP_FILE, "r", encoding="utf-8") as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {int(v): k for k, v in phoneme_to_id.items()}
    blank_id = phoneme_to_id.get("<blank>", 0)

    # --- Data ---
    train_ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT, "train", "phoneme")
    val_ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT, "val", "phoneme")

    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.training.NUM_WORKERS, collate_fn=ULBiTP_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.INFERENCE_BATCH_SIZE, shuffle=False,
                            num_workers=getattr(cfg.training, "VAL_NUM_WORKERS", 2), collate_fn=ULBiTP_collate_fn)

    # --- Model ---
    model = ULBiTP_R_v8_VLBP_FullFeedback(cfg.rlx_config).to(device)
    logger.info(f"🧠 Model Parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    augmentation = KorniaAugmentation(cfg.train_aug, cfg.dataset).to(device)

    # --- Loss ---
    ctc_loss_fn = torch.nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True).to(device)
    # ctc_loss_fn = LabelSmoothedCTCLoss(cfg.dataset.NUM_PHONEME_CLASSES, blank_id,
    #                                    alpha=cfg.training.label_smoothing.DEFAULT_ALPHA).to(device)
    consistency_calc = UncertaintyAwareLossNoGate().to(device)

    best_val_per = float("inf")
    train_losses, val_pers = [], []

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

    # --- Training Loop ---
    for epoch in range(cfg.training.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for videos, ctc_targets, video_lengths, target_lengths, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            videos = videos.to(device)
            ctc_targets = ctc_targets.to(device)
            video_lengths = video_lengths.to(device)
            target_lengths = target_lengths.to(device)

            if getattr(cfg.train_aug, "USE_AUGMENTATION", True):
                videos = augmentation(videos)

            optimizer.zero_grad()
            out = model(videos, input_lengths=video_lengths)
            log_probs = out[0]; out_lens = out[1]; belief_logits = out[3]; uncertainty = out[4]

            # --- Main Loss ---
            loss_main = ctc_loss_fn(log_probs, ctc_targets, out_lens, target_lengths)

            # --- Belief + Consistency (No Gate) ---
            if belief_logits is not None:
                loss_dict = consistency_calc(
                    main_log_probs=log_probs,
                    belief_logits=belief_logits,
                    uncertainty=uncertainty,
                    targets=ctc_targets,
                    input_lengths=video_lengths,
                    target_lengths=target_lengths
                )
                loss_belief = loss_dict["belief"]
                loss_consistency = loss_dict["consistency"]
            else:
                loss_belief = torch.tensor(0.0, device=device)
                loss_consistency = torch.tensor(0.0, device=device)

            # --- Total Loss ---
            final_loss = loss_main + loss_belief + loss_consistency
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.GRAD_NORM_CLIP)
            optimizer.step()
            total_loss += final_loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} | Avg Train Loss: {avg_loss:.4f}")
        train_losses.append(avg_loss)

        # --- Validation ---
        model.eval()
        all_preds, all_gts = [], []
        val_loss = 0.0
        with torch.no_grad():
            for videos, ctc_targets, video_lengths, target_lengths, _ in val_loader:
                videos = videos.to(device)
                ctc_targets = ctc_targets.to(device)
                video_lengths = video_lengths.to(device)
                target_lengths = target_lengths.to(device)

                out_val = model(videos, input_lengths=video_lengths)
                log_probs_val = out_val[0]; out_lens_val = out_val[1]; belief_logits_val = out_val[3]

                val_loss += ctc_loss_fn(log_probs_val, ctc_targets, out_lens_val, target_lengths).item()

                # Decode
                preds = []
                seq = log_probs_val.permute(1,0,2)
                for row in seq.argmax(dim=-1):
                    toks, last = [], -1
                    for x in row:
                        x = int(x)
                        if x != blank_id and x != last:
                            toks.append(id_to_phoneme.get(x, "?"))
                        last = x
                    preds.append(" ".join(toks))
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
            torch.save(model.state_dict(), cfg.training.CHECKPOINT_DIR / f"{run_name}_best_nogate.pth")
            logger.info(f" BEST PER: {(val_per*100):.2f}%")
            per_no_improve_count = 0
        else:
            per_no_improve_count += 1
             # # --- ALPHA DECAY ---
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
    train_ulbitp_vlbp_grid_no_gate()
