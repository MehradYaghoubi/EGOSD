

import argparse
import os, sys, json, math, random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
import logging
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA

from Configs.config import MainConfig
from DataProcessor.ULBiTP_data_loader import ULBiTPDataset, ULBiTP_collate_fn
from Model.base import ULBiTP_R_v7, count_parameters
from DataProcessor.kornia_augs import KorniaAugmentation
from Utils.custom_loss import LabelSmoothedCTCLoss
from Utils.metrics import calculate_per

# ---------------- Logging ----------------
def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    
    # جلوگیری از تکرار هندلرها در صورت اجرای مجدد در نوت‌بوک
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        ch.setFormatter(fmt); fh.setFormatter(fmt)
        logger.addHandler(ch); logger.addHandler(fh)
    return logger

# ---------------- Decoding ----------------
def greedy_decode_batch(log_probs, id_to_phoneme, blank_id):
    seq = log_probs.permute(1, 0, 2)
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

def compute_grad_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.data.float().pow(2).sum().item())
    return math.sqrt(total_sq) if total_sq > 0.0 else 0.0

# ---------------- utilities & plots ----------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def align_sequences(gt, pred):
    # simple DP Levenshtein backtrack (returns aligned lists)
    n, m = len(gt), len(pred)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i
    for j in range(1, m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if gt[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    i, j = n, m
    ag, ap = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i-1] == pred[j-1]:
            ag.append(gt[i-1]); ap.append(pred[j-1]); i -= 1; j -= 1
        else:
            candidates = []
            if i > 0 and j > 0:
                candidates.append((dp[i-1][j-1], 'rep'))
            if i > 0:
                candidates.append((dp[i-1][j], 'del'))
            if j > 0:
                candidates.append((dp[i][j-1], 'ins'))
            best = min(candidates, key=lambda x: x[0])[1]
            if best == 'rep':
                ag.append(gt[i-1]); ap.append(pred[j-1]); i -= 1; j -= 1
            elif best == 'del':
                ag.append(gt[i-1]); ap.append('<del>'); i -= 1
            else:
                ag.append('<ins>'); ap.append(pred[j-1]); j -= 1
    return ag[::-1], ap[::-1]

def plot_confusion_matrix(cm, labels, epoch, save_path):
    cm = cm.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax, cbar=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ---------------- TRAIN ----------------
def train_ulbitp_aug(resume_checkpoint=None):
    cfg = MainConfig()
    if resume_checkpoint:
        run_name = f"ULBiTPR_RESUMED_{datetime.now().strftime('%m%d_%H%M%S')}"
    else:
        run_name = f"ULBiTPR_Base_CTC_{datetime.now().strftime('%m%d_%H%M%S')}"
        
    logger = setup_logging(cfg.training.LOG_DIR, run_name)
    device = torch.device(cfg.training.DEVICE)
    logger.info(f"Device: {device}")
    
    if resume_checkpoint:
        logger.info(f"Resuming training from: {resume_checkpoint}")


    seed = getattr(cfg.training,"SEED",42)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    with open(cfg.dataset.PHONEME_MAP_FILE,"r",encoding="utf-8") as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {int(v): k for k, v in phoneme_to_id.items()}
    blank_id = phoneme_to_id.get("<blank>", 0)
    K = len(phoneme_to_id)
    logger.info(f"Loaded phoneme map (%d tokens)", K)

    train_ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT, "train", "phoneme")
    val_ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT, "val", "phoneme")

    val_num_workers = getattr(cfg.training, "VAL_NUM_WORKERS", 0)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.training.NUM_WORKERS, collate_fn=ULBiTP_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.INFERENCE_BATCH_SIZE, shuffle=False,
                            num_workers=val_num_workers, collate_fn=ULBiTP_collate_fn)

    logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))

    model = ULBiTP_R_v7(cfg.rlx_config).to(device)
    logger.info("Model parameters: %s", f"{count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.training.LEARNING_RATE,
                            betas=getattr(cfg.training,"BETAS",(0.9,0.98)),
                            eps=getattr(cfg.training,"EPS",1e-8),
                            weight_decay=getattr(cfg.training,"WEIGHT_DECAY",1e-2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    ctc_loss_fn = torch.nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True).to(device)
    # ctc_loss_fn = LabelSmoothedCTCLoss(cfg.dataset.NUM_PHONEME_CLASSES, blank_id,
    #                                    alpha=cfg.training.label_smoothing.DEFAULT_ALPHA).to(device)

    augmentation = KorniaAugmentation(cfg.train_aug, cfg.dataset).to(device)

    start_epoch = 0
    best_val_per = float("inf")
    
    if resume_checkpoint:
        if os.path.isfile(resume_checkpoint):
            logger.info(f"Loading checkpoint '{resume_checkpoint}' ...")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            
            # Load Model
            model.load_state_dict(checkpoint['model_state'])
            
            # Load Optimizer
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Load Scheduler
            if 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                
            # Load Epoch and Metrics
            start_epoch = checkpoint.get('epoch', 0)
            best_val_per = checkpoint.get('best_val_per', float('inf'))
            
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
        else:
            logger.error(f"No checkpoint found at '{resume_checkpoint}'")
            return
    # <<< MODIFIED: RESUME LOGIC ENDS HERE >>>

    accum_steps = max(1, int(getattr(cfg.training,"GRAD_ACCUM_BATCHES", max(1, 32 // cfg.training.BATCH_SIZE))))
    logger.info("Using accumulation steps = %d", accum_steps)
    ckpt_dir = cfg.training.CHECKPOINT_DIR; ckpt_dir.mkdir(parents=True, exist_ok=True)


    per_decay_patience = getattr(cfg.training,"PER_DECAY_PATIENCE",2)
    per_decay_factor = getattr(cfg.training,"PER_DECAY_FACTOR",0.6)
    per_no_improve_count = 0

    PLOT_EVERY = getattr(cfg.training, "PLOT_EVERY", 5)

    plot_dir = getattr(cfg.training, "PLOTS_DIR", cfg.training.LOG_DIR / "plots") / run_name
    ensure_dir(plot_dir)

    train_losses, val_losses, val_pers = [], [], []

    for epoch in range(start_epoch , cfg.training.NUM_EPOCHS):
        model.train()
        total_loss = 0.0; grad_norms=[]
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.NUM_EPOCHS} [Train]", ncols=140)

        for step, (videos, ctc_targets, video_lengths, target_lengths, _) in enumerate(pbar):
            videos = videos.to(device, non_blocking=True)
            ctc_targets = ctc_targets.to(device, non_blocking=True)
            video_lengths = video_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            if getattr(cfg.train_aug, "USE_AUGMENTATION", True):
                videos = augmentation(videos)

            log_probs, out_lens, aux_logp = model(videos, video_lengths)

            primary_ctc = ctc_loss_fn(log_probs, ctc_targets, out_lens, target_lengths)
            aux_ctc = None
            if cfg.rlx_config.use_aux_head and aux_logp is not None:
                aux_ctc = ctc_loss_fn(aux_logp, ctc_targets, out_lens, target_lengths)

            # combine
            loss_batch = primary_ctc
            if aux_ctc is not None: loss_batch = loss_batch + cfg.rlx_config.aux_loss_weight * aux_ctc

            loss = loss_batch / accum_steps
            loss.backward()
            grad_norms.append(compute_grad_norm(model))

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.GRAD_NORM_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=(total_loss / (step + 1)), lr=get_lr(optimizer))

        avg_grad_norm = sum(grad_norms)/len(grad_norms) if grad_norms else 0.0
        avg_train_loss = total_loss / len(train_loader)
        logger.info("Epoch %d TRAIN loss=%.4f ,GradNorm=%.4f", epoch+1, avg_train_loss, avg_grad_norm )
        train_losses.append(avg_train_loss)

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0.0; all_preds = []; all_gts = []
        with torch.no_grad():
            for videos, ctc_targets, video_lengths, target_lengths, _ in tqdm(val_loader, desc="[Val]", ncols=120):
                videos = videos.to(device, non_blocking=True)
                ctc_targets = ctc_targets.to(device, non_blocking=True)
                video_lengths = video_lengths.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)

                log_probs, out_lens, aux_logp = model(videos, video_lengths)
                val_loss += ctc_loss_fn(log_probs, ctc_targets, out_lens, target_lengths).item()

                preds = greedy_decode_batch(log_probs, id_to_phoneme, blank_id)
                all_preds.extend(preds)

                start = 0
                for L in target_lengths:
                    L = int(L)
                    toks = [id_to_phoneme.get(int(t), "?") for t in ctc_targets[start:start+L].tolist()]
                    all_gts.append(" ".join(toks))
                    start += L

        avg_val_loss = val_loss / len(val_loader)
        val_per = calculate_per(all_preds, all_gts, standard=True)
        logger.info("Epoch %d VAL loss=%.4f, PER=%.4f", epoch+1, avg_val_loss, val_per)
        val_losses.append(avg_val_loss)
        val_pers.append(val_per)

        logger.info("--- Sample Predictions ---")
        for i in range(min(5, len(all_preds))):
            logger.info("GT  : %s", all_gts[i])
            logger.info("Pred: %s", all_preds[i])
            logger.info("-"*20)

        # ---------------- Plots (conditional) ----------------
        should_plot = ((epoch+1) % PLOT_EVERY == 0) or (val_per < best_val_per)
        if should_plot:
            plt.figure(figsize=(8,4))
            plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
            plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
            plt.legend(); plt.grid(True)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(plot_dir / f"epoch_{epoch+1}_loss.png", dpi=200)
            plt.close()

            plt.figure(figsize=(8,4))
            plt.plot(range(1, len(val_pers)+1), val_pers, label="Val PER")
            plt.xlabel("Epoch"); plt.ylabel("PER"); plt.title("Validation PER")
            plt.legend(); plt.grid(True)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(plot_dir / f"epoch_{epoch+1}_per.png", dpi=200)
            plt.close()

        # Confusion matrix: exclude special tokens from axes
        canonical_labels = [id_to_phoneme[i] for i in range(K) if id_to_phoneme[i] != "<blank>"]
        all_aligned_gts = []; all_aligned_preds = []
        for gt_seq, pred_seq in zip(all_gts, all_preds):
            gt_tokens = gt_seq.split()
            pred_tokens = pred_seq.split()
            aligned_gt, aligned_pred = align_sequences(gt_tokens, pred_tokens)
            all_aligned_gts.extend(aligned_gt)
            all_aligned_preds.extend(aligned_pred)

        if len(all_aligned_gts) > 0 and len(canonical_labels) > 0:
            try:
                cm = confusion_matrix(all_aligned_gts, all_aligned_preds, labels=canonical_labels)
                if should_plot:
                    plot_confusion_matrix(cm, canonical_labels, epoch+1, plot_dir / f"epoch_{epoch+1}_confusion.png")
            except Exception as e:
                logger.warning("Failed building confusion matrix: %s", e)
        else:
            logger.warning("No aligned tokens to build confusion matrix for epoch %d", epoch+1)

        # ---------------- Checkpointing ----------------
        scheduler.step(val_per)

        ckpt = {
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "best_val_per": best_val_per,
        }
        torch.save(ckpt, ckpt_dir / f"{run_name}_last_epoch_{epoch+1}.pth")

        if val_per < best_val_per:
            best_val_per = val_per
            torch.save(model.state_dict(), ckpt_dir / f"{run_name}_best_per_{val_per:.3f}.pth")
            logger.info(" New best model saved (PER=%.4f)", val_per)
            per_no_improve_count = 0
        else:
            per_no_improve_count += 1
            # if per_no_improve_count >= per_decay_patience:
            #     old_alpha = ctc_loss_fn.alpha
            #     ctc_loss_fn.alpha = max(old_alpha * per_decay_factor, 0.0)
            #     logger.info("Label smoothing α decayed %.4f → %.4f", old_alpha, ctc_loss_fn.alpha)
            #     per_no_improve_count = 0
            if per_no_improve_count >= 6:
                logger.info("No improvement for 6 epochs, stopping training.")
                break

        
    logger.info("Training complete ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ULBiTP Model")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint file to resume from (e.g., checkpoints/run_name_last_epoch_5.pth)")
    args = parser.parse_args()
    
    train_ulbitp_aug(resume_checkpoint=args.resume)
