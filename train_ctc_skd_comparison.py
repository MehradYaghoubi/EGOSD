#!/usr/bin/env python3
"""
Training selects STUDENT validation PER; test requires a separate invocation.
Run from the project root:
  python train_ctc_skd_comparison.py --seed 42
The script does not implement resume; last_training.pth is a recovery snapshot.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "Configs").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Configs.config import MainConfig
from DataProcessor.ULBiTP_data_loader import ULBiTPDataset, ULBiTP_collate_fn
from DataProcessor.kornia_augs import KorniaAugmentation
from Model.base import ULBiTP_R_v7, count_parameters
from Utils.metrics import calculate_per

AUXILIARY_CTC_BLANK = 0


def settings(config):
    # Unannotated model class attributes are absent from dataclasses.asdict().
    return {
        k: deepcopy(getattr(config, k))
        for k in dir(config)
        if not k.startswith("_")
        and isinstance(getattr(config, k), (bool, int, float, str, list, tuple))
    }


class SharedPrefixSKD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.student = ULBiTP_R_v7(config)
        self.teacher_layer = deepcopy(self.student.encoder.transformer.layers[-1])
        self.teacher_classifier = deepcopy(self.student.classifier)

    def forward(self, videos, lengths):
        # Same student operations as basexP/proxxV; no change to existing modules.
        features = self.student.frontend(videos)
        pooled = self.student.pool(features, input_lengths=lengths)
        encoded, mid_out = self.student.encoder(self.student.proj(pooled))
        student_features = F.layer_norm(encoded, encoded.shape[-1:])
        student_logp = F.log_softmax(self.student.classifier(student_features), -1)
        teacher_features = self.teacher_layer(encoded)
        teacher_features = F.layer_norm(teacher_features, teacher_features.shape[-1:])
        teacher_logp = F.log_softmax(self.teacher_classifier(teacher_features), -1)
        aux_logp = None
        if self.student.use_aux:
            mid_features = F.layer_norm(mid_out, mid_out.shape[-1:])
            aux_logp = F.log_softmax(self.student.aux_head(mid_features), -1)
            aux_logp = aux_logp.transpose(0, 1).contiguous()
        return (
            student_logp.transpose(0, 1).contiguous(),
            teacher_logp.transpose(0, 1).contiguous(),
            aux_logp,
        )


def frame_skd(student_logp, teacher_logp, lengths):
    if student_logp.shape != teacher_logp.shape or student_logp.ndim != 3:
        raise ValueError("SKD requires matching (T, B, C) outputs.")
    if lengths.numel() != student_logp.shape[1]:
        raise ValueError("One valid length is required per video.")
    if bool(((lengths <= 0) | (lengths > student_logp.shape[0])).any()):
        raise ValueError("Invalid video lengths for SKD.")
    valid = torch.arange(student_logp.shape[0], device=student_logp.device)[:, None]
    valid = valid < lengths[None, :]
    # Eq. (7) soft cross-entropy, not KL. Keep blank; mask padding only.
    per_frame = -(teacher_logp.detach().exp() * student_logp).sum(-1)
    return per_frame.masked_select(valid).mean()


def scheduled_alpha(epoch, total_epochs):
    if total_epochs < 2 or not 0 <= epoch < total_epochs:
        raise ValueError("Alpha scheduling requires at least two epochs.")
    return min(max(epoch / (total_epochs - 1), 0.3), 0.7)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_hashes(config):
    root = Path(config.dataset.PROCESSED_DATASET_ROOT)
    return {
        s: sha256(root / f"{s}.json")
        for s in ("train", "val", "test")
        if (root / f"{s}.json").is_file()
    }


def load_map(config):
    mapping = json.loads(
        Path(config.dataset.PHONEME_MAP_FILE).read_text(encoding="utf-8")
    )
    ids = list(mapping.values())
    if "<blank>" not in mapping or any(type(i) is not int for i in ids):
        raise ValueError("Phoneme map requires integer IDs and explicit <blank>.")
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("Phoneme IDs must be unique and contiguous from zero.")
    return mapping


def make_loader(config, split, mapping, training=False):
    ds = ULBiTPDataset(config.dataset.PROCESSED_DATASET_ROOT, split, "phoneme")
    if not len(ds):
        raise ValueError(f"Empty {split} dataset.")
    # Stop before existing loader fallbacks can silently substitute missing data.
    for sample in ds.data_samples:
        path = ds.data_root / sample["video_path"].replace(chr(92), "/")
        ids = sample.get("phoneme_ids", [])
        frames = sample.get("num_frames", 0)
        if not path.is_file():
            raise FileNotFoundError(path)
        if not ids or any(type(i) is not int or not 0 <= i < len(mapping) for i in ids):
            raise ValueError(f"Invalid phoneme targets: {path}")
        if mapping["<blank>"] in ids:
            raise ValueError(f"CTC targets contain blank: {path}")
        repeats = sum(a == b for a, b in zip(ids, ids[1:]))
        if type(frames) is not int or frames < len(ids) + repeats:
            raise ValueError(f"Insufficient frames for a valid CTC alignment: {path}")
    return DataLoader(
        ds,
        batch_size=(
            config.training.BATCH_SIZE
            if training
            else config.training.INFERENCE_BATCH_SIZE
        ),
        shuffle=training,
        num_workers=(
            config.training.NUM_WORKERS
            if training
            else getattr(config.training, "VAL_NUM_WORKERS", 2)
        ),
        collate_fn=ULBiTP_collate_fn,
        pin_memory=training,
    )


def decode(logp, lengths, inverse, blank):
    decoded = []
    for row in logp.argmax(-1).transpose(0, 1).tolist():
        tokens, last = [], -1
        # Match train_graphXV.greedy_decode_batch: decode the complete output row.
        for token in row:
            if token != blank and token != last:
                tokens.append(inverse[token])
            last = token
        decoded.append(" ".join(tokens))
    return decoded


@torch.no_grad()
def evaluate_student(student, loader, ctc, mapping, device):
    student.eval()
    inverse = {v: k for k, v in mapping.items()}
    predictions, references, losses = [], [], []
    for videos, targets, lengths, target_lengths, _ in tqdm(
        loader, desc=loader.dataset.split
    ):
        logp, out_lengths, _ = student(videos.to(device), lengths.to(device))
        losses.append(
            ctc(logp, targets.to(device), out_lengths, target_lengths.to(device)).item()
        )
        predictions.extend(decode(logp, out_lengths, inverse, mapping["<blank>"]))
        offset = 0
        for length in target_lengths.tolist():
            references.append(
                " ".join(inverse[i] for i in targets[offset : offset + length].tolist())
            )
            offset += length
    return {
        "loss": sum(losses) / len(losses),
        "per": calculate_per(predictions, references, standard=True),
        "samples": len(predictions),
    }


def train(config, args, mapping):
    if config.training.NUM_EPOCHS < 2:
        raise ValueError("NUM_EPOCHS must be at least two.")
    seed = args.seed if args.seed is not None else getattr(config.training, "SEED", 43)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config.training.DEVICE)
    name = f"CTC_SKD_Adapted_seed{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run_dir = Path(config.training.CHECKPOINT_DIR) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir = Path(config.training.LOG_DIR) / name
    log_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8"),
    ):
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.info("Device: %s", device)
    # rlx matches EGOSD; rlxx currently selects the enlarged ten-TCN-block base.
    model_config = settings(config.rlx_config)
    if model_config["NUM_PHONEME_CLASSES"] != len(mapping):
        raise ValueError("Config class count differs from the phoneme map.")
    model = SharedPrefixSKD(config.rlx_config).to(device)
    if model.student.cfg is not config.rlx_config:
        raise RuntimeError("Student must use MainConfig.rlx_config directly.")
    if len(model.student.encoder.blocks) != config.rlx_config.n_blocks:
        raise RuntimeError("TCN depth differs from rlx_config.")
    if len(model.student.encoder.transformer.layers) != max(
        1, config.rlx_config.n_transformer_layers
    ):
        raise RuntimeError("Transformer depth differs from rlx_config.")
    train_loader = make_loader(config, "train", mapping, training=True)
    val_loader = make_loader(config, "val", mapping)
    ctc = nn.CTCLoss(blank=mapping["<blank>"], reduction="mean", zero_infinity=True).to(
        device
    )
    teacher_ctc_fn = nn.CTCLoss(
        blank=AUXILIARY_CTC_BLANK, reduction="mean", zero_infinity=True
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.LEARNING_RATE, weight_decay=0.5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    augmentation = KorniaAugmentation(config.train_aug, config.dataset).to(device)
    protocol = {
        "method": "Kim2024 frame-level CTC SKD, adapted to VSR",
        "reference": "https://doi.org/10.21437/Interspeech.2024-363",
        "seed": seed,
        "model_config": model_config,
        "student_parameters": count_parameters(model.student),
        "training_parameters": count_parameters(model),
        "teacher_extra_transformer_layers": 1,
        "data_root": str(Path(config.dataset.PROCESSED_DATASET_ROOT).resolve()),
        "split_sha256": split_hashes(config),
        "phoneme_map": mapping,
        "script_sha256": sha256(__file__),
        "epochs": config.training.NUM_EPOCHS,
        "batch_size": config.training.BATCH_SIZE,
        "inference_batch_size": config.training.INFERENCE_BATCH_SIZE,
        "optimizer": "AdamW",
        "lr": config.training.LEARNING_RATE,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "scheduler": "ReduceLROnPlateau(student_val_per), factor=0.5, patience=2, min_lr=1e-6",
        "gradient_clip": config.training.GRAD_NORM_CLIP,
        "gradient_accumulation": 1,
        "early_stopping_patience": 6,
        "augmentation": settings(config.train_aug),
        "alpha": "clip(epoch_index/(NUM_EPOCHS-1), 0.3, 0.7)",
        "skd_reduction": "valid-frame mean, blank included, teacher detached",
        "main_student_ctc_blank": mapping["<blank>"],
        "auxiliary_teacher_ctc_blank": AUXILIARY_CTC_BLANK,
        "decode_blank": mapping["<blank>"],
        "selection": "student validation PER; no test use during training",
        "adaptations": [
            "visual frontend",
            "training from scratch",
            "one extra teacher Transformer layer",
            "frame-mean SKD",
        ],
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Protocol: %s", json.dumps(protocol, ensure_ascii=False))
    logger.info(
        "EGOSD blank convention: main/student=%d, auxiliary/teacher=%d, decoding=%d.",
        ctc.blank,
        teacher_ctc_fn.blank,
        mapping["<blank>"],
    )
    if teacher_ctc_fn.blank != ctc.blank:
        logger.warning(
            "Historical EGOSD auxiliary CTC blank mismatch retained for the comparison by request."
        )
    zero_target_clips = sum(
        AUXILIARY_CTC_BLANK in s["phoneme_ids"]
        for s in train_loader.dataset.data_samples
    )
    if zero_target_clips:
        logger.warning(
            "%d training clips contain target ID %d, also used as auxiliary CTC blank. This reproduces the historical auxiliary convention, not standard blank-consistent CTC.",
            zero_target_clips,
            AUXILIARY_CTC_BLANK,
        )
    logger.info(
        "Train=%d Val=%d; student=%s training=%s parameters",
        len(train_loader.dataset),
        len(val_loader.dataset),
        f"{count_parameters(model.student):,}",
        f"{count_parameters(model):,}",
    )
    logger.info("Output: %s", run_dir.resolve())
    best_per, stale, history = float("inf"), 0, []
    for epoch in range(config.training.NUM_EPOCHS):
        model.train()
        alpha = scheduled_alpha(epoch, config.training.NUM_EPOCHS)
        sums = np.zeros(4, dtype=float)
        for videos, targets, lengths, target_lengths, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1} [Train]"
        ):
            videos, targets = videos.to(device), targets.to(device)
            lengths, target_lengths = lengths.to(device), target_lengths.to(device)
            if getattr(config.train_aug, "USE_AUGMENTATION", True):
                videos = augmentation(videos)
            optimizer.zero_grad(set_to_none=True)
            student_logp, teacher_logp, aux_logp = model(videos, lengths)
            student_ctc = ctc(student_logp, targets, lengths, target_lengths)
            teacher_ctc = teacher_ctc_fn(teacher_logp, targets, lengths, target_lengths)
            skd = frame_skd(student_logp, teacher_logp, lengths)
            loss = (1 - alpha) * teacher_ctc + alpha * (student_ctc + skd)
            if config.rlx_config.use_aux_head and aux_logp is not None:
                loss = loss + config.rlx_config.aux_loss_weight * ctc(
                    aux_logp, targets, lengths, target_lengths
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}.")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.GRAD_NORM_CLIP,
                error_if_nonfinite=True,
            )
            optimizer.step()
            sums += [loss.item(), teacher_ctc.item(), student_ctc.item(), skd.item()]
        result = evaluate_student(model.student, val_loader, ctc, mapping, device)
        scheduler.step(result["per"])
        improved = result["per"] < best_per
        if improved:
            best_per, stale = result["per"], 0
        else:
            stale += 1
        row = dict(
            zip(
                ("train_loss", "teacher_ctc", "student_ctc", "skd"),
                (sums / len(train_loader)).tolist(),
            )
        )
        row.update(
            epoch=epoch + 1,
            alpha=alpha,
            val_loss=result["loss"],
            val_per=result["per"],
            lr=optimizer.param_groups[0]["lr"],
        )
        history.append(row)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        logger.info(
            "Epoch %d alpha=%.4f loss=%.4f VAL student PER=%.3f%% best=%.3f%% LR=%g",
            epoch + 1,
            alpha,
            row["train_loss"],
            result["per"] * 100,
            best_per * 100,
            row["lr"],
        )
        if improved:
            # Only the deployed student, with original basexP-compatible keys.
            torch.save(
                {
                    "model_state": model.student.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_per": best_per,
                    "protocol": protocol,
                },
                run_dir / "best_student.pth",
            )
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_val_per": best_per,
                "protocol": protocol,
            },
            run_dir / "last_training.pth",
        )
        if stale >= 6:
            logger.info("No improvement for 6 epochs; stopping as in train_graphXV.py.")
            break
    logger.info(
        "Finished. Test best_student.pth separately; best VAL PER is not TEST PER."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--evaluate",
        type=Path,
        help="Evaluate this script's best_student.pth; never starts training.",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    config = MainConfig()
    mapping = load_map(config)
    if args.evaluate is None:
        train(config, args, mapping)
        return
    device = torch.device(config.training.DEVICE)
    checkpoint = torch.load(args.evaluate, map_location=device, weights_only=False)
    protocol = checkpoint["protocol"]
    if protocol["phoneme_map"] != mapping:
        raise ValueError("Evaluation phoneme map differs from the training map.")
    expected_hash = protocol["split_sha256"].get(args.split)
    if expected_hash is None or expected_hash != split_hashes(config).get(args.split):
        raise ValueError(
            "Evaluation split is missing or differs from the recorded protocol."
        )
    student = ULBiTP_R_v7(SimpleNamespace(**protocol["model_config"])).to(device)
    student.load_state_dict(checkpoint["model_state"], strict=True)
    loader = make_loader(config, args.split, mapping)
    if protocol["main_student_ctc_blank"] != mapping["<blank>"]:
        raise ValueError("Checkpoint main/student blank differs from the phoneme map.")
    ctc = nn.CTCLoss(
        blank=protocol["main_student_ctc_blank"], reduction="mean", zero_infinity=True
    ).to(device)
    result = evaluate_student(student, loader, ctc, mapping, device)
    result.update(
        split=args.split,
        per_percent=result["per"] * 100,
        checkpoint=str(args.evaluate.resolve()),
        checkpoint_sha256=sha256(args.evaluate),
    )
    output = args.evaluate.parent / f"{args.evaluate.stem}_{args.split}_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{args.split.upper()} STUDENT PER: {result['per_percent']:.3f}% ({result['samples']} samples)"
    )
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
