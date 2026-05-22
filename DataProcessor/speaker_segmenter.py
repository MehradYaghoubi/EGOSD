# DataProcessor/speaker_segmenter.py

import sys
import os
from pathlib import Path
import json
import random # برای قابلیت تصادفی‌سازی (اختیاری، اما در اینجا برای اطمینان از بازتولیدپذیری از مرتب‌سازی استفاده می‌شود)
from tqdm import tqdm # برای نمایش نوار پیشرفت

# اطمینان حاصل می کند که پایتون می تواند ماژول 'Configs' را پیدا کند.
# این خط برای اجرا از دایرکتوری اصلی پروژه ضروری است.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# وارد کردن MainConfig و DatasetConfig از فایل پیکربندی مرکزی
from Configs.config import MainConfig, DatasetConfig 

# --- فیلتر کردن هشدارهای GLOG برای خروجی تمیزتر ---
os.environ['GLOG_minloglevel'] = '2' # فقط پیام‌های با سطح 'WARNING' و بالاتر را نمایش می‌دهد.
# --- پایان فیلتر کردن ---

def perform_speaker_segmentation(dataset_config: DatasetConfig):
    """
    گوینده‌ها را به مجموعه‌های آموزش (train)، اعتبارسنجی (validation) و تست (test) بر اساس نسبت‌های تعریف شده
    در `DatasetConfig` تقسیم می‌کند و نتایج را در `speaker_meta.json` ذخیره می‌کند.
    """
    print("--- شروع تقسیم‌بندی گوینده‌ها ---", flush=True)

    # اطمینان حاصل شود که دایرکتوری داده‌های پردازش شده وجود دارد.
    dataset_config.PROCESSED_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    # یافتن همه دایرکتوری‌های گوینده (با فرمت `sXX`) در مسیر ویدیوهای خام.
    raw_videos_root = dataset_config.RAW_DATASET_ROOT / "videos"
    
    if not raw_videos_root.exists():
        print(f"خطا: دایرکتوری ویدیوهای خام یافت نشد: {raw_videos_root}. "
              "لطفاً مطمئن شوید که داده‌ها به درستی سازماندهی شده‌اند.", flush=True)
        return

    all_speaker_ids_numeric = []
    print(f"در حال جمع آوری ID گوینده‌ها از {raw_videos_root}...", flush=True)
    # پیمایش دایرکتوری ویدیوهای خام برای یافتن دایرکتوری‌های گوینده.
    for item in raw_videos_root.iterdir():
        if item.is_dir() and item.name.startswith('s'):
            try:
                # استخراج قسمت عددی ID گوینده (مثلاً از 's01' -> 1).
                speaker_num = int(item.name[1:])
                all_speaker_ids_numeric.append(speaker_num)
            except ValueError:
                print(f"هشدار: دایرکتوری غیرمعمولی یافت شد: {item.name}. نادیده گرفته شد.", flush=True)

    if not all_speaker_ids_numeric:
        print(f"هشدار: هیچ گوینده‌ای در {raw_videos_root} یافت نشد. "
              "لطفاً مطمئن شوید که زیرپوشه‌های گوینده (مانند s01, s02) وجود دارند.", flush=True)
        return

    # مرتب‌سازی ID گوینده‌ها برای اطمینان از تقسیم‌بندی بازتولیدپذیر (reproducible).
    all_speaker_ids_numeric.sort()
    num_speakers = len(all_speaker_ids_numeric)
    print(f"تعداد کل گوینده‌های شناسایی شده: {num_speakers}", flush=True)

    # محاسبه تعداد گوینده‌ها برای هر تقسیم‌بندی (train, val, test).
    num_train = int(num_speakers * dataset_config.TRAIN_SPLIT_RATIO)
    num_val = int(num_speakers * dataset_config.VAL_SPLIT_RATIO)
    # تعداد تست به عنوان باقیمانده برای اطمینان از اینکه همه گوینده‌ها پوشش داده می‌شوند.
    num_test = num_speakers - num_train - num_val

    # اعمال تقسیم‌بندی بر اساس لیست مرتب شده IDها.
    train_speakers_numeric = all_speaker_ids_numeric[:num_train]
    val_speakers_numeric = all_speaker_ids_numeric[num_train : num_train + num_val]
    test_speakers_numeric = all_speaker_ids_numeric[num_train + num_val : num_train + num_val + num_test]

    # تبدیل IDهای عددی به فرمت رشته‌ای 'sXX' (مثلاً 1 -> 's01').
    train_speakers = [f"s{s:02d}" for s in train_speakers_numeric]
    val_speakers = [f"s{s:02d}" for s in val_speakers_numeric]
    test_speakers = [f"s{s:02d}" for s in test_speakers_numeric]

    # ساختاردهی متادیتا برای ذخیره در فایل JSON.
    speaker_meta = {
        "train": train_speakers,
        "val": val_speakers,
        "test": test_speakers
    }

    # ذخیره متادیتا در فایل JSON.
    output_file = dataset_config.SPEAKER_META_FILE
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(speaker_meta, f, indent=4) # 'indent=4' برای خوانایی بهتر فایل JSON.

    print(f"\nتقسیم‌بندی گوینده‌ها با موفقیت انجام شد و در {output_file} ذخیره شد:", flush=True)
    print(f"   آموزش (Train): {len(train_speakers)} گوینده", flush=True)
    print(f"   اعتبارسنجی (Validation): {len(val_speakers)} گوینده", flush=True)
    print(f"   تست (Test): {len(test_speakers)} گوینده", flush=True)
    print(f"   نمونه‌هایی از گوینده‌های آموزش: {train_speakers[:5]}...", flush=True)
    print(f"   نمونه‌هایی از گوینده‌های اعتبارسنجی: {val_speakers[:5]}...", flush=True)
    print(f"   نمونه‌هایی از گوینده‌های تست: {test_speakers[:5]}...", flush=True)

    # --- بررسی همپوشانی بین مجموعه‌ها (باید صفر باشد) ---
    train_set = set(train_speakers)
    val_set = set(val_speakers)
    test_set = set(test_speakers)

    assert len(train_set.intersection(val_set)) == 0, "همپوشانی بین گوینده‌های آموزش و اعتبارسنجی یافت شد!"
    assert len(train_set.intersection(test_set)) == 0, "همپوشانی بین گوینده‌های آموزش و تست یافت شد!"
    assert len(val_set.intersection(test_set)) == 0, "همپوشانی بین گوینده‌های اعتبارسنجی و تست یافت شد!"
    print("\nاعتبارسنجی: هیچ همپوشانی بین مجموعه‌های گوینده یافت نشد.", flush=True)

    print("\n--- تقسیم‌بندی گوینده‌ها تکمیل شد! ---", flush=True)

if __name__ == "__main__":
    # ایجاد یک نمونه از `MainConfig` برای دسترسی به تنظیمات `Dataset`.
    main_config = MainConfig()
    # اجرای تابع تقسیم‌بندی گوینده با ارسال بخش `dataset` از `MainConfig`.
    perform_speaker_segmentation(main_config.dataset)
