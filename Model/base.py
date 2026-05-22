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

# --- 2. Pooling  ---
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

# --- 3. Encoder Blocks ---
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

class TemporalEncoder(nn.Module):
    def __init__(self, cfg: rlx):
        super().__init__()
        
        # TCN Part
        self.blocks = nn.ModuleList([
            DepthwiseTCNBlock(cfg.d_model,
                              kernel=cfg.kernel_list[i % len(cfg.kernel_list)],
                              dilation=min(2**i, getattr(cfg,"max_dilation",16)),
                              dropout=0.12 if getattr(cfg, "use_temporal_dropout", True) else 0.0)
            for i in range(cfg.n_blocks)
        ])
        self.mid_index = max(1, cfg.n_blocks // 2)
        
        # Transformer Part
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        
        actual_transformer_layers = max(1, getattr(cfg, "n_transformer_layers", 1))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ),
            num_layers=actual_transformer_layers
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
            
        out = out.permute(0,2,1) # (B, C, T) -> (B, T, C)
        mid_out = mid_out.permute(0,2,1)
        
        out = self.pre_norm(out)
        out = self.transformer(out)
        
        return out, mid_out

# --- 4. Main Base Model ---
class ULBiTP_R_v7(nn.Module):
    def __init__(self, cfg: rlx):
        super().__init__()
        self.cfg = cfg
        
        # Components
        self.frontend = Efficient3DFrontendLite(cfg.frontend_out_ch)
        self.pool = SpatialAttnPoolSimple(cfg.frontend_out_ch)
        self.proj = nn.Conv1d(cfg.frontend_out_ch, cfg.d_model, 1)
        self.encoder = TemporalEncoder(cfg)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(0.2),
            nn.Linear(cfg.d_model, cfg.NUM_PHONEME_CLASSES)
        )
        
        # Aux Head
        self.use_aux = getattr(cfg, "use_aux_head", False)
        if self.use_aux:
            self.aux_head = nn.Linear(cfg.d_model, cfg.NUM_PHONEME_CLASSES)

    def forward(self, x: torch.Tensor, input_lengths: Optional[torch.Tensor] = None):
        # Input Check
        if x.ndim != 5:
            raise ValueError("Input must be 5D tensor (B,C,T,H,W) or (B,T,C,H,W)")
        if x.shape[1] not in (1,3):
            x = x.permute(0,2,1,3,4).contiguous()
            
        # 1. Visual Frontend
        feat3d = self.frontend(x)
        pooled = self.pool(feat3d, input_lengths=input_lengths)
        proj = self.proj(pooled)
        
        # 2. Temporal Encoder
        enc_out, mid_out = self.encoder(proj)
        
        # 3. Classification
        enc_final = F.layer_norm(enc_out, enc_out.shape[-1:])
        logits = self.classifier(enc_final)
        logp = F.log_softmax(logits, dim=-1).permute(1,0,2).contiguous() # (T, B, C)
        
        # 4. Aux Output
        aux_logp = None
        if self.use_aux:
            mid_out_ln = F.layer_norm(mid_out, mid_out.shape[-1:])
            aux_logits = self.aux_head(mid_out_ln)
            aux_logp = F.log_softmax(aux_logits, dim=-1).permute(1,0,2).contiguous() # (T, B, C)
            
        out_lens = torch.full((logp.size(1),), logp.size(0), dtype=torch.long, device=x.device) if input_lengths is None else input_lengths
        
        # Return matches what train_base.py expects: log_probs, out_lens, aux_logp
        return logp, out_lens, aux_logp

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)