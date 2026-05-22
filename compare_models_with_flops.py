

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from Configs.config import MainConfig
from DataProcessor.ULBiTP_data_loader import ULBiTPDataset, ULBiTP_collate_fn
from Utils.metrics import calculate_per

from Model.base import ULBiTP_R_v7
from Model.egosd import ULBiTP_R_v8_VLBP_FullFeedback

from ptflops import get_model_complexity_info


def load_checkpoint(model, path, device):
    print(f" Loading checkpoint: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model


def decode_greedy(idxs, vocab, blank_id):
    toks = []
    last = -1
    for idx in idxs:
        if idx != blank_id and idx != last:
            toks.append(vocab[idx])
        last = idx
    return " ".join(toks)


def evaluate_model(model, test_loader, sorted_phonemes, blank_id, device, model_name="Model"):
    preds = []
    gts = []

    print(f"\n Evaluating {model_name} ...")
    with torch.no_grad():
        for videos, ctc_targets, video_lengths, target_lengths, _ in tqdm(test_loader, desc=model_name):
            videos = videos.to(device)
            video_lengths = video_lengths.to(device)

            outputs = model(videos, input_lengths=video_lengths)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            logits_np = logits.permute(1, 0, 2).cpu().numpy()
            ctc_targets_cpu = ctc_targets.cpu()

            start = 0
            for i, L in enumerate(target_lengths.cpu()):
                L = int(L)
                valid_T = int(video_lengths[i].item())

                gt_toks = [sorted_phonemes[int(t)] for t in ctc_targets_cpu[start:start + L]]
                gt_text = " ".join(gt_toks)
                gts.append(gt_text)
                start += L

                pred_idxs = np.argmax(logits_np[i][:valid_T], axis=-1)
                pred_text = decode_greedy(pred_idxs, sorted_phonemes, blank_id)
                preds.append(pred_text)

    per = calculate_per(preds, gts, standard=True) * 100
    return per, preds, gts


def profile_model_flops(model, sample_videos, sample_lengths, device, model_name="Model"):
    """
    FLOPs/Params را برای هر معماری به صورت مستقل محاسبه می‌کند.
    از یک sample واقعی از دیتاست برای تعیین shape استفاده می‌شود.
    """

    model = model.to(device)
    model.eval()

    # ورودی نمونه واقعی
    sample_videos = sample_videos[:1].to(device)     # batch=1 برای گزارش استاندارد
    sample_lengths = sample_lengths[:1].to(device)

    input_shape = tuple(sample_videos.shape[1:])  # بدون batch
    print(f"\n Profiling {model_name}")
    print(f"   Input shape used for FLOPs: (1, {', '.join(map(str, input_shape))})")
    print(f"   Input lengths used: {sample_lengths.tolist()}")

    class WrappedModel(torch.nn.Module):
        def __init__(self, model, input_lengths):
            super().__init__()
            self.model = model
            self.input_lengths = input_lengths

        def forward(self, x):
            out = self.model(x, input_lengths=self.input_lengths)
            return out[0] if isinstance(out, tuple) else out

    wrapped_model = WrappedModel(model, sample_lengths)

    try:
        macs, params = get_model_complexity_info(
            wrapped_model,
            input_res=input_shape,
            as_strings=False,
            print_per_layer_stat=True,
            verbose=False
        )

        # در بسیاری از ابزارها MACs گزارش می‌شود؛ FLOPs ≈ 2 * MACs
        flops = 2 * macs

        print(f"✅ {model_name} complexity:")
        print(f"   Params : {params:,}")
        print(f"   MACs   : {macs:,}")
        print(f"   FLOPs  : {flops:,}")

        return {
            "params": params,
            "macs": macs,
            "flops": flops,
            "input_shape": (1, *input_shape),
            "input_lengths": sample_lengths.tolist()
        }

    except Exception as e:
        print(f"❌ FLOPs profiling failed for {model_name}: {e}")
        return None


def compare_two_models(base_ckpt_path, belief_ckpt_path):
    cfg = MainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # ---------- Load vocab ----------
    with open(cfg.dataset.PHONEME_MAP_FILE, "r") as f:
        phoneme_to_id = json.load(f)

    sorted_phonemes = [k for k, v in sorted(phoneme_to_id.items(), key=lambda x: x[1])]
    blank_id = phoneme_to_id.get("<blank>", 0)

    # ---------- Data ----------
    test_ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT, "test", "phoneme")
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=ULBiTP_collate_fn
    )
    print(f"📂 Test samples: {len(test_ds)}")

    # یک batch واقعی برای تعیین shape پروفایل
    sample_batch = next(iter(test_loader))
    sample_videos, _, sample_video_lengths, _, _ = sample_batch

    # ==========================================================
    # 1) BASE MODEL -- مستقل
    # ==========================================================
    print("\n" + "=" * 70)
    print("🏗️ BASE MODEL (Base)")
    print("=" * 70)

    model_base = ULBiTP_R_v7(cfg.rlx_config)
    model_base = load_checkpoint(model_base, base_ckpt_path, device)

    base_profile = profile_model_flops(
        model=model_base,
        sample_videos=sample_videos,
        sample_lengths=sample_video_lengths,
        device=device,
        model_name="Baseline "
    )

    per_base, base_preds, ground_truths = evaluate_model(
        model=model_base,
        test_loader=test_loader,
        sorted_phonemes=sorted_phonemes,
        blank_id=blank_id,
        device=device,
        model_name="Baseline "
    )

    # آزادسازی حافظه
    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================================
    # 2) PROPOSED MODEL -- مستقل
    # ==========================================================
    print("\n" + "=" * 70)
    print(" PROPOSED MODEL (EGOSD)")
    print("=" * 70)

    model_belief = ULBiTP_R_v8_VLBP_FullFeedback(cfg.rlx_config)
    model_belief = load_checkpoint(model_belief, belief_ckpt_path, device)

    belief_profile = profile_model_flops(
        model=model_belief,
        sample_videos=sample_videos,
        sample_lengths=sample_video_lengths,
        device=device,
        model_name="Proposed (EGOSD)"
    )

    per_belief, belief_preds, _ = evaluate_model(
        model=model_belief,
        test_loader=test_loader,
        sorted_phonemes=sorted_phonemes,
        blank_id=blank_id,
        device=device,
        model_name="Proposed (EGOSD)"
    )

    del model_belief
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- Qualitative comparison ----------
    EGOSD_wins, base_wins, both_wrong = [], [], []

    for gt, bp, up in zip(ground_truths, base_preds, belief_preds):
        base_correct = (bp == gt)
        belief_correct = (up == gt)

        sample_info = {"GT": gt, "Base": bp, "Belief": up}

        if not base_correct and belief_correct:
            EGOSD_wins.append(sample_info)
        elif base_correct and not belief_correct:
            base_wins.append(sample_info)
        elif not base_correct and not belief_correct:
            both_wrong.append(sample_info)

    # ---------- Final report ----------
    print("\n" + "=" * 70)
    print(f"{'Model':<25} | {'PER (%)':<10} | {'FLOPs':<20} | {'Params':<15}")
    print("-" * 70)

    base_flops_str = f"{base_profile['flops']:,}" if base_profile else "N/A"
    base_params_str = f"{base_profile['params']:,}" if base_profile else "N/A"

    belief_flops_str = f"{belief_profile['flops']:,}" if belief_profile else "N/A"
    belief_params_str = f"{belief_profile['params']:,}" if belief_profile else "N/A"

    print(f"{'Baseline ':<25} | {per_base:<10.2f} | {base_flops_str:<20} | {base_params_str:<15}")
    print(f"{'Proposed (EGOSD)':<25} | {per_belief:<10.2f} | {belief_flops_str:<20} | {belief_params_str:<15}")
    print("-" * 70)
    print(f"Absolute PER improvement: {per_base - per_belief:.2f}%")
    print("=" * 70)

    # ---------- Save report ----------
    log_file = "model_comparison_analysis.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("[1] PERFORMANCE\n")
        f.write(f"Baseline PER : {per_base:.2f}%\n")
        f.write(f"Proposed PER : {per_belief:.2f}%\n")
        f.write(f"Absolute Improvement : {per_base - per_belief:.2f}%\n\n")

        f.write("[2] COMPUTATIONAL COST\n")
        if base_profile:
            f.write("Baseline \n")
            f.write(f"  Input shape : {base_profile['input_shape']}\n")
            f.write(f"  Input lengths : {base_profile['input_lengths']}\n")
            f.write(f"  Params : {base_profile['params']:,}\n")
            f.write(f"  MACs   : {base_profile['macs']:,}\n")
            f.write(f"  FLOPs  : {base_profile['flops']:,}\n\n")

        if belief_profile:
            f.write("Proposed (EGOSD)\n")
            f.write(f"  Input shape : {belief_profile['input_shape']}\n")
            f.write(f"  Input lengths : {belief_profile['input_lengths']}\n")
            f.write(f"  Params : {belief_profile['params']:,}\n")
            f.write(f"  MACs   : {belief_profile['macs']:,}\n")
            f.write(f"  FLOPs  : {belief_profile['flops']:,}\n\n")

        f.write("[3] QUALITATIVE ANALYSIS\n\n")

        f.write(f"SECTION A - Proposed wins ({len(EGOSD_wins)} samples)\n")
        f.write("-" * 60 + "\n")
        for s in EGOSD_wins[:30]:
            f.write(f"GT      : {s['GT']}\n")
            f.write(f"Base    : {s['Base']}\n")
            f.write(f"Belief  : {s['Belief']}\n\n")

        f.write(f"\nSECTION B - Baseline wins ({len(base_wins)} samples)\n")
        f.write("-" * 60 + "\n")
        for s in base_wins[:15]:
            f.write(f"GT      : {s['GT']}\n")
            f.write(f"Base    : {s['Base']}\n")
            f.write(f"Belief  : {s['Belief']}\n\n")

        f.write(f"\nSECTION C - Both failed ({len(both_wrong)} samples)\n")
        f.write("-" * 60 + "\n")
        for s in both_wrong[:15]:
            f.write(f"GT      : {s['GT']}\n")
            f.write(f"Base    : {s['Base']}\n")
            f.write(f"Belief  : {s['Belief']}\n\n")

    print(f"\n Detailed report saved to: {log_file}")


if __name__ == "__main__":
    BASE_CKPT = "./ch/ULBiTPR_RESUMED_0217_223430_best_per_0.086.pth"
    BELIEF_CKPT = "./ch/ULBiTP_EGOSD_CTC_0218_0007_best.pth"

    if os.path.exists(BASE_CKPT) and os.path.exists(BELIEF_CKPT):
        compare_two_models(BASE_CKPT, BELIEF_CKPT)
    else:
        print("Please set correct checkpoint paths.")
