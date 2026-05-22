# DataProcessor\generate_character_map.py

import sys
import os
from pathlib import Path
import json
from tqdm import tqdm
from typing import List

# اطمینان حاصل می‌شود که پایتون می‌تواند ماژول 'Configs' را پیدا کند.
# این خط برای اجرای اسکریپت از دایرکتوری اصلی پروژه ضروری است.
sys.path.append(str(Path(__file__).resolve().parent.parent))
# وارد کردن کلاس‌های پیکربندی 'MainConfig' و 'DatasetConfig' از فایل پیکربندی مرکزی
from Configs.config import MainConfig, DatasetConfig 

# --- فیلتر کردن هشدارهای GLOG برای خروجی تمیزتر ---
# 'GLOG_minloglevel = 2' باعث می‌شود که فقط پیام‌های با سطح 'WARNING' و بالاتر نمایش داده شوند.
os.environ['GLOG_minloglevel'] = '2' 
# --- پایان فیلتر کردن ---

def preprocess_align(align_path: Path) -> List[str]:
    """
    فایل هم‌ترازی (alignment file) را می‌خواند، توکن‌های کلمه‌ای مرتبط را استخراج می‌کند،
    آن‌ها را به یک رشته واحد متصل می‌کند و یک لیست از کاراکترهای مجزا
    (شامل حروف الفبایی کوچک و کاراکتر فاصله) را از آن رشته بازمی‌گرداند.
    این تابع برای فرمت‌های فایل 'align' که ممکن است شامل زمان‌بندی و توکن‌ها باشند،
    طراحی شده است.

    Args:
        align_path (Path): مسیر کامل فایل هم‌ترازی (`.align`) برای پردازش.

    Returns:
        List[str]: لیستی از کاراکترهای مجزای استخراج شده (حروف کوچک و فاصله).
                   در صورت بروز خطا یا یافت نشدن فایل، یک لیست خالی بازمی‌گرداند.
    """
    all_chars_in_file = []
    try:
        with open(align_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        words = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                token = parts[2].lower() # توکن (کلمه یا فونم) را به حروف کوچک تبدیل می‌کند.
                # توکن‌های خاص مربوط به سکوت یا ناشناخته را فیلتر می‌کند.
                if token not in ['', '<unk>']:
                    words.append(token)
        
        # کلمات را با فاصله به هم می‌چسباند تا یک جمله منسجم تشکیل شود.
        # این کاراکترهای فاصله بین کلمات را حفظ می‌کند.
        full_transcription = " ".join(words)

        # کاراکترهای تکی را از متن کامل استخراج می‌کند.
        # فقط حروف کوچک انگلیسی ('a' تا 'z') و کاراکتر فاصله (' ') را شامل می‌شود.
        for char in full_transcription:
            if 'a' <= char <= 'z' or char == ' ':
                all_chars_in_file.append(char)

    except FileNotFoundError:
        print(f"خطا: فایل هم‌ترازی در {align_path} یافت نشد.", flush=True)
    except Exception as e:
        print(f"خطا در پردازش فایل هم‌ترازی {align_path}: {e}", flush=True)
    return all_chars_in_file


def generate_character_map(dataset_config: DatasetConfig):
    """
    این تابع فرآیند جمع‌آوری تمامی کاراکترهای منحصر به فرد از فایل‌های هم‌ترازی خام
    و سپس ایجاد نگاشت (mapping) بین کاراکترها و IDهای عددی را اجرا می‌کند.
    این نگاشت در یک فایل JSON ذخیره می‌شود.

    Args:
        dataset_config (DatasetConfig): شیء پیکربندی مجموعه داده که حاوی مسیرهای لازم است.
    """
    print("--- شروع جمع‌آوری کاراکترها برای تولید نگاشت (character_to_id.json) ---", flush=True)

    # --- ایجاد دایرکتوری خروجی ---
    # اطمینان حاصل می‌شود که دایرکتوری اصلی برای داده‌های پیش‌پردازش شده وجود دارد.
    dataset_config.PROCESSED_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    # ----------------------------------------------------

    # یک مجموعه (set) خالی برای جمع‌آوری تمامی کاراکترهای منحصر به فرد مقداردهی اولیه می‌شود.
    all_characters_collected = set() 

    print("در حال جمع‌آوری کاراکترها از فایل‌های هم‌ترازی خام (تبدیل کلمات به کاراکترها)...", flush=True)
    
    # مسیر دایرکتوری حاوی فایل‌های هم‌ترازی خام (alignments)
    raw_align_root = dataset_config.RAW_ALIGNS_DIR 
    
    # پیدا کردن تمامی دایرکتوری‌های گوینده (با نام‌های شروع شده با 's') در مسیر هم‌ترازی خام.
    raw_align_speaker_dirs = sorted([d for d in raw_align_root.iterdir() if d.is_dir() and d.name.startswith('s')])

    if not raw_align_speaker_dirs:
        print(f"هشدار: هیچ دایرکتوری گوینده‌ای در {raw_align_root} یافت نشد. "
              "لطفاً مطمئن شوید که داده‌های خام به درستی سازماندهی شده‌اند.", flush=True)
    
    # --- حلقه برای پیمایش تمامی فایل‌های '.align' و استخراج کاراکترها ---
    for raw_speaker_align_dir in tqdm(raw_align_speaker_dirs, desc="در حال پردازش فایل‌های هم‌ترازی"):
        
        if not raw_speaker_align_dir.exists():
            print(f"هشدار: دایرکتوری هم‌ترازی برای {raw_speaker_align_dir.name} در {raw_speaker_align_dir} یافت نشد. جمع‌آوری کاراکتر برای این دایرکتوری نادیده گرفته شد.", flush=True)
            continue
        
        # پیدا کردن تمامی فایل‌های '.align' در دایرکتوری فعلی گوینده.
        for align_file_path in raw_speaker_align_dir.glob("*.align"):
            try:
                # تابع 'preprocess_align' اکنون کاراکترها را شامل فاصله نیز بازمی‌گرداند.
                chars_in_file = preprocess_align(align_file_path)
                all_characters_collected.update(chars_in_file) # اضافه کردن کاراکترهای استخراج شده به مجموعه
            except Exception as e:
                print(f"خطا در خواندن فایل هم‌ترازی {align_file_path} برای جمع‌آوری کاراکتر: {e}", flush=True)
                
    # --- افزودن کاراکترهای خاص و مرتب‌سازی ---
    # اطمینان حاصل می‌شود که کاراکتر فاصله (' ') نیز در مجموعه کاراکترها وجود دارد.
    if ' ' not in all_characters_collected:
        all_characters_collected.add(' ')
    

    # اضافه کردن 'sil' به صورت صریح به مجموعه کاراکترها
    all_characters_collected.add('§') # <--- تغییر مهم: اضافه کردن 'sil' به صورت دستی

    # کاراکترهای منحصر به فرد جمع‌آوری شده را به ترتیب الفبا مرتب می‌کند.
    sorted_unique_chars = sorted(list(all_characters_collected))
    
    # --- ایجاد نگاشت کاراکتر به ID ---
    # '<blank>' (توکن خالی برای CTC) ID 0 را دریافت می‌کند.
    # سایر کاراکترها به ترتیب از 1 به بعد ID می‌گیرند.
    char_to_id = {'<blank>': 0} 
    for i, char in enumerate(sorted_unique_chars):
        char_to_id[char] = i + 1 

    # --- ایجاد نگاشت ID به کاراکتر (برای فرآیند دیکدینگ) ---
    id_to_char = {v: k for k, v in char_to_id.items()}

    # --- ذخیره نگاشت‌ها در فایل‌های JSON ---
    # مسیر فایل برای نگاشت کاراکتر به ID (برگرفته از تنظیمات پیکربندی)
    char_to_id_path = dataset_config.CHARACTER_MAP_FILE 
    # مسیر فایل جداگانه برای نگاشت ID به کاراکتر
    id_to_char_path = dataset_config.PROCESSED_DATASET_ROOT / "id_to_character.json" 

    with open(char_to_id_path, 'w', encoding='utf-8') as f:
        json.dump(char_to_id, f, indent=4, ensure_ascii=False)
    print(f"نگاشت کاراکتر به ID در {char_to_id_path.name} با {len(char_to_id) - 1} ورودی (بدون احتساب blank) ایجاد شد.", flush=True)
    print(f"تعداد کل کلاس‌ها (با احتساب blank): {len(char_to_id)}", flush=True)

    with open(id_to_char_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_char, f, indent=4, ensure_ascii=False)
    print(f"نگاشت ID به کاراکتر در {id_to_char_path.name} ایجاد شد.", flush=True)

    print("توجه: تقسیم‌بندی گوینده (speaker segmentation) توسط این اسکریپت انجام نمی‌شود. فایل 'speaker_meta.json' توسط این فرآیند ایجاد/به‌روز نمی‌شود.", flush=True)
    print("\n--- جمع‌آوری کاراکترها برای نگاشت تکمیل شد! ---", flush=True)

if __name__ == "__main__":
    # یک نمونه از 'MainConfig' برای دسترسی به تنظیمات 'Dataset' ایجاد می‌شود.
    main_config = MainConfig()
    # تابع 'generate_character_map' با ارسال بخش 'dataset' از 'MainConfig' اجرا می‌شود.
    generate_character_map(main_config.dataset)

