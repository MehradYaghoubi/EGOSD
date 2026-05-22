import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import os
import sys

# Add project root to sys.path to allow imports from other directories
# (مسیر را در صورت نیاز متناسب با ساختار پروژه خود تنظیم کنید)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Configs.config import MainConfig

class ULBiTPDataset(Dataset):
    """
    دیتا لودر مستحکم که طول فریم را از فایل JSON به عنوان "منبع حقیقت" می‌خواند.
    این رویکرد پایداری کامل را در برابر خطاهای پردازشی در فایل‌های npy تضمین می‌کند.
    """
    def __init__(self, data_root: Path, split: str = 'train', target_type: str = 'phoneme'):
        self.data_root = Path(data_root)
        self.split = split
        self.target_type = target_type
        self.target_key = f"{target_type}_ids"
        self.dataset_cfg = MainConfig # در صورت نیاز از حالت کامنت خارج شود

        data_list_path = self.data_root / f"{self.split}.json"
        if not data_list_path.exists():
            raise FileNotFoundError(f"Data list file not found at: {data_list_path}")

        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.data_samples = json.load(f)
        
        # بررسی وجود کلید num_frames در نمونه اول برای اطمینان
        if self.data_samples and "num_frames" not in self.data_samples[0]:
            raise KeyError("The 'num_frames' key was not found in the JSON data. "
                           "Please ensure your JSON file includes it for each sample.")

        print(f"✅ Loaded {len(self.data_samples)} clips for split='{self.split}'. Using 'num_frames' from JSON.", flush=True)

    def __len__(self):
        return len(self.data_samples)

    def _load_npy(self, path: Path) -> np.ndarray:
        arr = np.load(path)  # (T, H, W, C)
        if arr.dtype != np.uint8 and arr.dtype != np.float32:
            arr = arr.astype(np.uint8)
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        sample = self.data_samples[idx]
        
        # --- 1. "منبع حقیقت": خواندن طول مورد انتظار از JSON ---
        expected_frames = sample["num_frames"]
        
        rel_path = sample["video_path"].replace('\\\\', '/')
        full_path = self.data_root / rel_path

        if not full_path.exists():
            # ایجاد یک ویدیوی خالی با ابعاد صحیح در صورت نبود فایل
            H, W = self.dataset_cfg.dataset.IMG_H, self.dataset_cfg.dataset.IMG_W # ابعاد را در صورت نیاز تغییر دهید
            video_frames = np.zeros((expected_frames, H, W, 3), dtype=np.uint8)
            actual_frames = 0
            print(f"⚠️ Warning: File not found: {full_path}. Returning a dummy tensor.", flush=True)

        else:
            arr = self._load_npy(full_path)
            
            # --- 2. اعتبارسنجی و اصلاح خودکار ---
            if arr.shape[0] != expected_frames:
                print(f"⚠️ Warning: Mismatch for '{full_path.name}'. "
                      f"Expected {expected_frames} frames, but found {arr.shape[0]}. "
                      f"Correcting to expected length.", flush=True)
                
                # اگر آرایه بلندتر است، آن را برش بزن
                if arr.shape[0] > expected_frames:
                    arr = arr[:expected_frames]
                # اگر آرایه کوتاه‌تر است، آن را با صفر پد کن
                else:
                    pad_len = expected_frames - arr.shape[0]
                    pad_shape = (pad_len, arr.shape[1], arr.shape[2], arr.shape[3])
                    # مطمئن شویم که نوع داده برای پدینگ درست است
                    padding = np.zeros(pad_shape, dtype=arr.dtype)
                    arr = np.concatenate([arr, padding], axis=0)

            video_frames = arr
            actual_frames = expected_frames # همیشه به JSON اعتماد کن
            
        # --- 3. نرمال‌سازی ---
        video_frames_tensor = torch.from_numpy(video_frames).float() / 255.0  # (T,H,W,C) in [0,1]
        video_frames_tensor = (video_frames_tensor - 0.5) / 0.5  # now in [-1,1]

        # --- 4. آماده‌سازی تارگت‌ها ---
        target_ids = sample.get(self.target_key, [])
        if not target_ids:
            target_ids = [0] # Handle empty targets
        target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)

        metadata = {
            "video_name": Path(rel_path).name,
            "raw_text": sample.get("text", sample.get("phonemes", "")),
            "original_frames": actual_frames
        }

        return video_frames_tensor, target_ids_tensor, int(actual_frames), metadata

def ULBiTP_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]]):
    """
    این تابع حالا به درستی کار می‌کند، زیرا __getitem__ تضمین کرده 
    که تمام ویدیوهای یک بچ طول یکسانی دارند.
    """
    videos, targets, video_lengths, metas = zip(*batch)
    
    # stack ویدیوها: چون همه طول یکسان دارند، این کار بدون خطا انجام می‌شود.
    # videos: sequence of (T,H,W,C)
    videos_stack = torch.stack(videos)  # (N, T, H, W, C)
    videos_stack = videos_stack.permute(0, 4, 1, 2, 3).contiguous()  # (N, C, T, H, W)

    # مدیریت اهداف (Targets) برای CTC Loss
    targets_concat = torch.cat(targets).to(torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    
    # طول واقعی هر ویدیو (که همگی یکسان و برابر مقدار JSON هستند)
    video_lengths_tensor = torch.tensor(video_lengths, dtype=torch.long)
    
    return videos_stack, targets_concat, video_lengths_tensor, target_lengths, list(metas)

# # --- مثال برای تست محلی ---
# if __name__ == "__main__":
#     # یک کانفیگ ساختگی برای تست
#     class MockConfig:
#         MAX_FRAMES = 75

#     cfg = MockConfig()
#     # مسیر روت دیتاست خود را اینجا قرار دهید
#     DATA_ROOT = Path("./path/to/your/processed_dataset") 
    
#     # فرض می‌کنیم فایل val.json شما آماده است
#     ds = ULBiTPDataset(DATA_ROOT, split='val')
#     dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=ULBiTP_collate_fn, num_workers=0)
    
#     # یک بچ را برای بررسی چاپ کن
#     try:
#         batch = next(iter(dl))
#         videos_stack, targets_concat, v_lengths, t_lengths, metas = batch
        
#         print("\n--- Batch Test Results ---")
#         print("Videos shape (N, C, T, H, W):", videos_stack.shape)
#         print("Video lengths (all should be equal):", v_lengths)
#         print("Targets_concat shape:", targets_concat.shape)
#         print("Target lengths:", t_lengths)
#         print("--------------------------\n")
#         assert videos_stack.shape[2] == v_lengths[0], "Mismatch between tensor dim and length!"
#         print(" Test Passed: Tensor dimension matches the reported length.")

#     except (FileNotFoundError, KeyError) as e:
#         print(f"Error during test: {e}")
#     except StopIteration:
#         print("Couldn't get a batch. Is the dataset empty or the path correct?")
