import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

# تنظیمات گرافیکی برای ژورنال
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'figure.dpi': 300
})

try:
    from Configs.config import MainConfig
    from DataProcessor.ULBiTP_data_loader import ULBiTPDataset, ULBiTP_collate_fn
    # اصلاح اول: فراخوانی فایل و کلاس جدید معماری EGOSD
    from Model.egosd import ULBiTP_R_v8_VLBP_FullFeedback
except ImportError:
    print(" Error: Run from project root.")
    sys.exit()

def analyze_limited_scatter(num_batches=5):
    # --- CONFIG ---
    CKPT_PATH = "./ch/ULBiTP_EGOSD_CTC_0218_0007_best.pth"
    cfg = MainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # اصلاح دوم: نمونه‌سازی از کلاس جدید
    model = ULBiTP_R_v8_VLBP_FullFeedback(cfg.rlx_config).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.eval()

    # Load Data (تغییرات دیتالودر شما که max_frames ندارد در این نمونه اعمال شده است)
    ds = ULBiTPDataset(cfg.dataset.PROCESSED_DATASET_ROOT, "test", "phoneme")
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, collate_fn=ULBiTP_collate_fn)

    all_h = []
    all_kl = []

    print(f" Collecting data from {num_batches} batches...")
    with torch.no_grad():
        for i, (vid, _, vlen, _, _) in enumerate(tqdm(loader, total=num_batches)):
            if i >= num_batches: break
            
            vid, vlen = vid.to(device), vlen.to(device)
            outs = model(vid, input_lengths=vlen)
            
            s_logp = outs[0].permute(1, 0, 2)       # Student (B, T, C)
            t_logp = F.log_softmax(outs[3], dim=-1) # Teacher (B, T, C) -> خروجی بخش EGOSD
            entropy = outs[4]                       # Uncertainty (B, T)

            # محاسبه واگرایی KL فریم‌به‌فریم بر اساس فرمول Loss
            kl = F.kl_div(s_logp, t_logp, reduction='none', log_target=True).sum(-1)

            # استخراج فریم‌های معتبر بر اساس vlen (حذف Paddingها)
            for b in range(vid.shape[0]):
                length = int(vlen[b].item())
                all_h.extend(entropy[b, :length].cpu().numpy())
                all_kl.extend(kl[b, :length].cpu().numpy())

    all_h = np.array(all_h)
    all_kl = np.array(all_kl)

    # --- Trend Statistics ---
    num_bins = 25
    bin_means_kl, bin_edges, _ = binned_statistic(all_h, all_kl, statistic='mean', bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # 1. SCATTER PLOT
    ax1.scatter(all_h, all_kl, s=12, color='#34495e', alpha=0.4, 
                edgecolors='none', label='Phoneme Frames (Test Subset)')

    # 2. KL TREND LINE
    color_kl = '#c0392b'
    ax1.plot(bin_centers, bin_means_kl, color=color_kl, linewidth=3, 
             marker='o', markersize=4, label='Mean Teacher-Student Conflict')
    
    ax1.set_xlabel('Predictive Entropy $H(t)$ (Uncertainty)', fontweight='bold')
    ax1.set_ylabel('Divergence ($D_{KL}$)', color='#2c3e50', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 3. GATING WEIGHT CURVE
    ax2 = ax1.twinx()
    color_w = '#27ae60'
    h_range = np.linspace(all_h.min(), all_h.max(), 100)
    w_curve = np.exp(-h_range)
    ax2.plot(h_range, w_curve, color=color_w, linewidth=4, 
             label='Gating Weight $w_t = \exp(-H)$')
    
    ax2.set_ylabel('Gating Weight ($w_t$)', color=color_w, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_w)
    ax2.set_ylim(-0.05, 1.05)

    # LEGEND
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
               frameon=True, facecolor='white', framealpha=0.9)

    plt.title('Mechanism Analysis: Gating Reliability vs. Data Uncertainty', fontsize=13, pad=20)
    
    # یادداشت‌های روی نمودار
    ax1.annotate('Teacher Confident:\nGate Opens for KD', 
                     xy=(all_h.min() + 0.3, bin_means_kl[1]), 
                     xytext=(all_h.min() + 0.02, 1.05*0.45),
                     arrowprops=dict(facecolor='#27ae60', arrowstyle='->', connectionstyle="arc3,rad=.1"),
                     fontsize=9, fontweight='bold', color='#1e824c')

        # یادداشت برای مهار هوشمند تضاد در مدل نهایی
    idx_target = -4 if len(bin_means_kl) > 4 else -1
    ax1.annotate('Teacher Uncertain \n$D_{KL}$ Gate Restricted', 
                     xy=(bin_centers[idx_target], bin_means_kl[idx_target]), 
                     xytext=(bin_centers[idx_target] - 1.2, bin_means_kl[idx_target] + 1.05*0.12),
                     arrowprops=dict(facecolor='#c0392b', arrowstyle='->', connectionstyle="arc3,rad=-0.1"),
                     fontsize=9, fontweight='bold', color='#962d22')

    plt.tight_layout()
    plt.savefig("Limited_Scatter_Analysis.pdf")
    plt.savefig("Limited_Scatter_Analysis.png")
    print(f" Created plot using {num_batches} batches.")

if __name__ == "__main__":
    analyze_limited_scatter(num_batches=60)