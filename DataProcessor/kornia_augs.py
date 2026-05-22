import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from typing import Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Configs.config import AugConfigTrain


class KorniaAugmentation(nn.Module):
    """
    GPU-friendly Kornia augmentation pipeline for lip-reading videos.
    Input: (N, C, T, H, W)
    Includes: spatial (resize, flip, rotation, affine, jitter, noise, blur)
              temporal jitter, masking, blur.
    """

    def __init__(self, aug_cfg: AugConfigTrain, dataset_cfg):
        super().__init__()
        self.cfg = aug_cfg
        self.img_h = dataset_cfg.IMG_H
        self.img_w = dataset_cfg.IMG_W
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Build spatial augmentations ===
        aug_list = []

        if self.cfg.RESIZED_CROP:
            aug_list.append(K.RandomResizedCrop(
                size=(self.img_h, self.img_w),
                scale=self.cfg.RESIZED_CROP_SCALE,
                ratio=self.cfg.RESIZED_CROP_RATIO,
                p=self.cfg.RESIZED_CROP_PROB
            ))
        if self.cfg.HORIZONTAL_FLIP:
            aug_list.append(K.RandomHorizontalFlip(p=self.cfg.FLIP_PROB))
        if self.cfg.ROTATION:
            aug_list.append(K.RandomRotation(
                degrees=self.cfg.ROTATION_DEGREES,
                p=self.cfg.ROTATION_PROB
            ))
        if self.cfg.AFFINE:
            aug_list.append(K.RandomAffine(
                degrees=0,
                translate=self.cfg.AFFINE_TRANSLATE,
                p=self.cfg.AFFINE_PROB
            ))
        if self.cfg.COLOR_JITTER:
            aug_list.append(K.ColorJitter(
                brightness=self.cfg.BRIGHTNESS,
                contrast=self.cfg.CONTRAST,
                saturation=self.cfg.SATURATION,
                hue=self.cfg.HUE,
                p=self.cfg.JITTER_PROB
            ))
        if self.cfg.GAUSSIAN_NOISE:
            aug_list.append(K.RandomGaussianNoise(
                mean=self.cfg.NOISE_MEAN,
                std=self.cfg.NOISE_STD,
                p=self.cfg.NOISE_PROB
            ))
        if self.cfg.GAUSSIAN_BLUR:
            aug_list.append(K.RandomGaussianBlur(
                kernel_size=(self.cfg.BLUR_KERNEL_SIZE, self.cfg.BLUR_KERNEL_SIZE),
                sigma=self.cfg.BLUR_SIGMA,
                p=self.cfg.BLUR_PROB
            ))

        self.spatial_transforms = K.AugmentationSequential(
            *aug_list,
            data_keys=["input"],
            same_on_batch=True,
            keepdim=False
        ) if aug_list else None

    @torch.no_grad()
    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_batch: (N, C, T, H, W), float32, [0,1] or [-1,1]
        Returns:
            video_batch: augmented tensor of same shape
        """
        N, C, T, H, W = video_batch.shape
        video_batch = video_batch.to(self.device, non_blocking=True)

        # --- Temporal jitter ---
        if self.cfg.TEMPORAL_JITTER and self.training:
            for i in range(N):
                if torch.rand(1).item() < self.cfg.TEMPORAL_JITTER_PROB:
                    shift = torch.randint(-self.cfg.MAX_TEMPORAL_SHIFT, self.cfg.MAX_TEMPORAL_SHIFT + 1, (1,)).item()
                    if shift != 0:
                        video_batch[i] = torch.roll(video_batch[i], shifts=shift, dims=1)

        # --- Time masking ---
        if self.cfg.TIME_MASKING and self.training:
            for i in range(N):
                if torch.rand(1).item() < self.cfg.TIME_MASK_PROB:
                    mask_len = torch.randint(1, min(self.cfg.MAX_MASK_FRAMES, T) + 1, (1,)).item()
                    start = torch.randint(0, T - mask_len + 1, (1,)).item()
                    video_batch[i, :, start:start + mask_len, :, :] = 0.0

        # --- Spatial augmentation ---
        if self.spatial_transforms is not None and self.training:
            # Flatten (N, C, T, H, W) → (N*T, C, H, W)
            flat = video_batch.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
            flat_aug = self.spatial_transforms(flat)
            video_batch = flat_aug.view(N, T, C, H, W).permute(0, 2, 1, 3, 4)

        # --- Temporal blur ---
        if self.cfg.TEMPORAL_BLUR and self.training:
            blur_kernel = torch.tensor([0.25, 0.5, 0.25], device=self.device, dtype=video_batch.dtype).view(1, 1, 3, 1, 1)
            for i in range(N):
                if torch.rand(1).item() < getattr(self.cfg, 'TEMPORAL_BLUR_PROB', 0.2):
                    tb = F.conv3d(
                        video_batch[i].unsqueeze(0),
                        blur_kernel.repeat(C, 1, 1, 1, 1),
                        padding=(1, 0, 0),
                        groups=C
                    )
                    video_batch[i] = tb.squeeze(0)

        return video_batch
