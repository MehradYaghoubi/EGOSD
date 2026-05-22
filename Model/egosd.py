# Model/ULBiTP_R_v7_VLBP_stable.py

import math
from pathlib import Path
import sys
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Configs.config import rlx

try:
    from dropblock import DropBlock3D
except Exception:
    DropBlock3D = None

# --- 1. Frontend ---
class Efficient3DFrontendLite(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        def GN(c):
            g = max(2, min(16, c // 8))
            return nn.GroupNorm(g, c)
        
        self.layers = nn.Sequential(
            nn.Conv3d(3, 32, (3,5,5), (1,2,2), (1,2,2), bias=False),
            GN(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, (3,3,3), (1,2,2), (1,1,1), bias=False),
            GN(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, out_ch, (3,3,3), (1,1,1), (1,1,1), bias=False),
            GN(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.layers(x)

# --- 2. Pooling ---
class SpatialAttnPoolSimple(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch,1,1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x, input_lengths: Optional[torch.Tensor]=None):
        B,C,T,H,W = x.shape
        att = self.conv(x).view(B,1,T,H*W)
        att = att / math.sqrt(float(H * W) + 1e-12)
        if input_lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= input_lengths.unsqueeze(1)
            mask = mask.view(B,1,T,1).expand(-1,-1,-1,H*W)
            att = att.masked_fill(mask, float('-1e9'))
        att = self.softmax(att)
        feat = (x.view(B,C,T,H*W) * att).sum(-1)
        return feat 

# --- 3. TCN Blocks ---
class DepthwiseTCNBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1, dropout=0.12):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.dw = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
        self.norm = nn.GroupNorm(1, ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        res = x
        y = self.dw(x)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        return res + self.dropout(y)

# --- 4. Encoder ---
class TemporalEncoder(nn.Module):
    def __init__(self, cfg: rlx):
        super().__init__()
        self.blocks = nn.ModuleList([
            DepthwiseTCNBlock(cfg.d_model,
                              kernel=cfg.kernel_list[i % len(cfg.kernel_list)],
                              dilation=min(2**i, getattr(cfg,"max_dilation",16)),
                              dropout=0.12 if getattr(cfg, "use_temporal_dropout", True) else 0.0)
            for i in range(cfg.n_blocks)
        ])
        self.mid_index = max(1, cfg.n_blocks // 2)
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ),
            num_layers=max(1, cfg.n_transformer_layers)
        )

    def forward(self, x):
        out = x
        mid_out = None
        for i, blk in enumerate(self.blocks):
            out = blk(out)
            if i == self.mid_index - 1:
                mid_out = out
        if mid_out is None:
            mid_out = out
        out = out.permute(0,2,1)
        mid_out = mid_out.permute(0,2,1)
        out = self.pre_norm(out)
        out = self.transformer(out)
        return out, mid_out

# --- 5. Belief Module ---
class VLBPModule_v3(nn.Module):
    def __init__(self, d_model: int, num_phonemes: int, belief_hidden: int = None,
                 detach_hidden_every: int = 2, proj_p_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.detach_hidden_every = max(1, int(detach_hidden_every))
        if belief_hidden is None:
            belief_hidden = d_model
        
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, proj_p_dim),
            nn.LayerNorm(proj_p_dim),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRUCell(proj_p_dim, belief_hidden)
        self.h_norm = nn.LayerNorm(belief_hidden)
        self.belief_classifier = nn.Linear(belief_hidden, num_phonemes)

    def forward(self, enc_out: torch.Tensor, input_lengths: Optional[torch.Tensor] = None):
        B, T, C = enc_out.shape
        device = enc_out.device
        h = torch.zeros(B, self.gru.hidden_size, device=device)
        belief_logits_list = []
        uncertainty_list = [] 

        for t in range(T):
            e_t = enc_out[:, t, :]
            gru_in = self.input_proj(e_t)
            
            if (t % self.detach_hidden_every) == 0 and t != 0:
                h = h.detach()
            
            h = self.gru(gru_in, h)
            h = self.h_norm(h)
            
            b_logits = self.belief_classifier(h)
            belief_logits_list.append(b_logits)
            
            # --- Uncertainty (Entropy) ---
            probs = F.softmax(b_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) 
            uncertainty_list.append(entropy)

            if input_lengths is not None:
                finished = (input_lengths <= (t + 1)).to(device)
                if finished.any():
                    h = h.masked_fill(finished.unsqueeze(-1), 0.0)

        belief_logits_seq = torch.stack(belief_logits_list, dim=1) 
        uncertainty_seq = torch.stack(uncertainty_list, dim=1) 
        return belief_logits_seq, uncertainty_seq

# --- 6. Main Model Wrapper ---
class ULBiTP_R_v8_VLBP_FullFeedback(nn.Module):
    def __init__(self, cfg: rlx):
        super().__init__()
        self.cfg = cfg
        self.frontend = Efficient3DFrontendLite(cfg.frontend_out_ch)
        self.pool = SpatialAttnPoolSimple(cfg.frontend_out_ch)
        self.proj = nn.Conv1d(cfg.frontend_out_ch, cfg.d_model, 1)
        self.encoder = TemporalEncoder(cfg)
        
        self.use_vlbp = getattr(cfg, "use_vlbp", True)
        if self.use_vlbp:
            self.vlbp = VLBPModule_v3(
                d_model=cfg.d_model,
                num_phonemes=cfg.NUM_PHONEME_CLASSES,
                belief_hidden=getattr(cfg, "belief_hidden_dim", cfg.d_model)
            )
            
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(0.2),
            nn.Linear(cfg.d_model, cfg.NUM_PHONEME_CLASSES)
        )

    def forward(self, x: torch.Tensor, input_lengths: Optional[torch.Tensor]=None, **kwargs):
        feat3d = self.frontend(x)
        pooled = self.pool(feat3d, input_lengths=input_lengths)
        proj = self.proj(pooled)
        enc_out, mid_out = self.encoder(proj) 

        # 1. Main Path
        enc_final = F.layer_norm(enc_out, enc_out.shape[-1:])
        logits = self.classifier(enc_final)
        logp = F.log_softmax(logits, dim=-1).permute(1,0,2).contiguous()

        # 2. Belief Path
        belief_logits_seq = None
        uncertainty_seq = None
        if self.use_vlbp:
            # Note: We pass raw encoder features. Detaching logic happens in Loss.
            belief_input = enc_final 
            belief_logits_seq, uncertainty_seq = self.vlbp(belief_input, input_lengths=input_lengths)

        return logp, input_lengths, None, belief_logits_seq, uncertainty_seq

# --- 7. The LOSS CLASS (EGOSD) ---
class UncertaintyAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard losses used internally for Belief Branch
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.kl_div = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, main_log_probs, belief_logits, uncertainty, targets, input_lengths, target_lengths):
        """
        Calculates:
        1. Belief Module's CTC Loss
        2. Uncertainty-Gated Consistency Loss (Weighted KL)
        
        Note: Main Task CTC is calculated outside to preserve Alpha Decay logic.
        """
        # A) Belief CTC Loss
        # belief_logits is (B, T, C) -> Permute to (T, B, C)
        belief_log_probs = F.log_softmax(belief_logits, dim=-1).permute(1,0,2)
        loss_belief = self.ctc_loss(belief_log_probs, targets, input_lengths, target_lengths)

        # B) Uncertainty-Gated Consistency
        # Main (Student) -> (B, T, C) - Already log_softmax from model output, just permute back
        p_main = main_log_probs.permute(1,0,2)
        
        # Belief (Teacher) -> (B, T, C) - Detach to stop gradient flow to Teacher
        p_belief_target = F.log_softmax(belief_logits.detach(), dim=-1)

        # KL Divergence: (B, T)
        kl = self.kl_div(p_main, p_belief_target).sum(dim=-1)

        # Gating Weight: exp(-H)
        consistency_weights = torch.exp(-uncertainty)

        # --- MASKING (The Critical Part from train.py) ---
        if input_lengths is not None:
            max_len = kl.size(1)
            # Create mask: True for valid frames, False for padding
            mask = torch.arange(max_len, device=kl.device).unsqueeze(0) < input_lengths.unsqueeze(1)
            
            kl = kl * mask
            consistency_weights = consistency_weights * mask
            
            # Weighted Average over Valid Frames Only
            weighted_consistency = (kl * consistency_weights).sum() / (mask.sum() + 1e-8)
        else:
            weighted_consistency = (kl * consistency_weights).mean()

        return {
            "belief": loss_belief, 
            "consistency": weighted_consistency
        }

def count_parameters(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
