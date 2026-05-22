# DataProcessor\grid_dataset.py

import os
import shutil
import requests    # برای دانلود فایل‌ها از وب
import zipfile     # برای کار با فایل‌های .zip
import tarfile     # برای کار با فایل‌های .tar
import sys         # برای مدیریت خروجی خطا (stderr)
from tqdm import tqdm # برای نمایش نوار پیشرفت دانلود


def download_and_organize_grid():
    """
    این تابع دیتاست GRID را دانلود، استخراج و به ساختار
    Dataset/videos/sXX و Dataset/aligns/sXX سازماندهی می‌کند.
    فایل‌های موقت نیز پس از استفاده پاک می‌شوند.
    """
    # تعریف مسیر اصلی Dataset
    dataset_base_dir = "Dataset"

    # ایجاد دایرکتوری‌های اصلی 'Dataset' برای ذخیره ویدیوها و فایل‌های هم‌ترازی
    os.makedirs(os.path.join(dataset_base_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(dataset_base_dir, "aligns"), exist_ok=True)

    # *** تغییر مهم: مسیر فولدر موقت 'grid' حالا داخل 'Dataset' قرار می‌گیرد ***
    grid_temp_dir = os.path.join(dataset_base_dir, "grid_temp")
    os.makedirs(grid_temp_dir, exist_ok=True)


    # تعریف لیست گوینده‌ها از s1 تا s34 به جز s21
    speakers_raw = [f"s{i}" for i in range(1, 35) if i != 21]

    for spk_raw in speakers_raw:
        # فرمت‌دهی نام پوشه به صورت s01, s02, ..., s34
        spk_num = int(spk_raw[1:])
        spk_formatted = f"s{spk_num:02d}"

        print(f" در حال دانلود و سازماندهی داده‌ها برای {spk_raw} (به عنوان {spk_formatted})...", flush=True)

        # ایجاد دایرکتوری‌های موقت در ساختار 'grid_temp' (که داخل Dataset است)
        os.makedirs(os.path.join(grid_temp_dir, "video", spk_raw), exist_ok=True)
        os.makedirs(os.path.join(grid_temp_dir, "align", spk_raw), exist_ok=True)

        # تعریف لینک‌های دانلود برای ویدیو و فایل‌های هم‌ترازی هر گوینده
        video_url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{spk_raw}/video/{spk_raw}.mpg_vcd.zip"
        align_url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{spk_raw}/align/{spk_raw}.tar"

        # --- دانلود و استخراج فایل‌های ویدیویی (.zip) ---
        print(f"  در حال دانلود ویدیو برای {spk_raw} از {video_url}...", flush=True)
        temp_video_zip_path = os.path.join(grid_temp_dir, "video", spk_raw, f"{spk_raw}.mpg_vcd.zip")
        
        try:
            with requests.get(video_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total_size_in_bytes = int(r.headers.get('content-length', 0))
                block_size = 8192
                
                with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"  {spk_formatted} ویدیو") as pbar:
                    with open(temp_video_zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size): 
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            print(f"  دانلود فایل ویدیو {os.path.basename(temp_video_zip_path)} به پایان رسید.", flush=True)

            print(f"  در حال استخراج ویدیو برای {spk_raw}...", flush=True)
            # مقصد استخراج: پوشه موقت ویدیو داخل grid_temp
            # این باعث ایجاد ساختار Dataset/grid_temp/video/sXX/sXX/video_files.mpg می‌شود
            with zipfile.ZipFile(temp_video_zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(grid_temp_dir, "video", spk_raw)) 
            print(f"  استخراج ویدیو برای {spk_raw} به پایان رسید.", flush=True)

        except requests.exceptions.Timeout:
            print(f"خطا: زمان دانلود ویدیو برای {spk_raw} از {video_url} به پایان رسید.", file=sys.stderr, flush=True)
            continue
        except requests.exceptions.RequestException as e:
            print(f"خطا در دانلود ویدیو برای {spk_raw} از {video_url}: {e}", file=sys.stderr, flush=True)
            continue
        except zipfile.BadZipFile:
            print(f"خطا: فایل {os.path.basename(temp_video_zip_path)} یک فایل زیپ نامعتبر است. رد شدن از استخراج.", file=sys.stderr, flush=True)
            continue
        except Exception as e:
            print(f"خطای ناشناخته در دانلود/استخراج ویدیو برای {spk_raw}: {e}", file=sys.stderr, flush=True)
            continue

        # تعریف مسیر نهایی برای ذخیره فایل‌های ویدیویی
        final_video_spk_dir = os.path.join(dataset_base_dir, "videos", spk_formatted)
        os.makedirs(final_video_spk_dir, exist_ok=True) # ایجاد پوشه نهایی برای گوینده
        
        # مسیر دایرکتوری استخراج شده موقت حاوی ویدیوها (یعنی Dataset/grid_temp/video/sXX/sXX/)
        extracted_video_sub_dir = os.path.join(grid_temp_dir, "video", spk_raw, spk_raw) # <--- تغییر اینجا! اضافه شدن spk_raw دوم

        if os.path.exists(extracted_video_sub_dir):
            # *** تغییر مهم: انتقال فایل‌های ویدیویی به جای فولدر والد ***
            # لیست کردن تمام فایل‌های ویدیویی (mpg.) در فولدر داخلی sXX
            for item_name in os.listdir(extracted_video_sub_dir):
                source_path = os.path.join(extracted_video_sub_dir, item_name)
                destination_path = os.path.join(final_video_spk_dir, item_name)
                # اطمینان حاصل کنید که فقط فایل‌ها را منتقل می‌کنیم، نه زیرپوشه‌های احتمالی
                if os.path.isfile(source_path):
                    shutil.move(source_path, destination_path)
            print(f"  فایل‌های ویدیویی از {extracted_video_sub_dir} به {final_video_spk_dir} منتقل شدند.", flush=True)
            # پس از انتقال فایل‌ها، فولدر داخلی sXX را پاک می‌کنیم
            shutil.rmtree(extracted_video_sub_dir)
            print(f"  فولدر موقت داخلی {extracted_video_sub_dir} پاک شد.", flush=True)
        else:
            print(f"هشدار: دایرکتوری ویدیو {extracted_video_sub_dir} برای {spk_raw} یافت نشد. انتقال نادیده گرفته می‌شود.", flush=True)


        # --- پاکسازی فایل زیپ موقت ویدیو ---
        if os.path.exists(temp_video_zip_path):
            os.remove(temp_video_zip_path)
            print(f"  فایل زیپ موقت ویدیو پاک شد: {temp_video_zip_path}", flush=True)

        # --- دانلود و استخراج فایل‌های هم‌ترازی (.tar) ---
        print(f"  در حال دانلود فایل‌های هم‌ترازی برای {spk_raw} از {align_url}...", flush=True)
        temp_align_tar_path = os.path.join(grid_temp_dir, "align", spk_raw, f"{spk_raw}.tar")
        
        try:
            with requests.get(align_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total_size_in_bytes = int(r.headers.get('content-length', 0))
                block_size = 8192
                
                with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"  {spk_formatted} هم‌ترازی") as pbar:
                    with open(temp_align_tar_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            print(f"  دانلود فایل هم‌ترازی {os.path.basename(temp_align_tar_path)} به پایان رسید.", flush=True)

            print(f"  در حال استخراج هم‌ترازی برای {spk_raw}...", flush=True)
            with tarfile.open(temp_align_tar_path, 'r') as tar_ref:
                tar_ref.extractall(os.path.join(grid_temp_dir, "align", spk_raw))
            print(f"  استخراج فایل‌های هم‌ترازی برای {spk_raw} به پایان رسید.", flush=True)

        except requests.exceptions.Timeout:
            print(f"خطا: زمان دانلود فایل هم‌ترازی برای {spk_raw} از {align_url} به پایان رسید.", file=sys.stderr, flush=True)
            continue
        except requests.exceptions.RequestException as e:
            print(f"خطا در دانلود فایل هم‌ترازی برای {spk_raw} از {align_url}: {e}", file=sys.stderr, flush=True)
            continue
        except tarfile.ReadError:
            print(f"خطا: فایل {os.path.basename(temp_align_tar_path)} یک فایل tar نامعتبر است. رد شدن از استخراج.", file=sys.stderr, flush=True)
            continue
        except Exception as e:
            print(f"خطای ناشناخته در دانلود/استخراج فایل هم‌ترازی برای {spk_raw}: {e}", file=sys.stderr, flush=True)
            continue

        # تعریف مسیر نهایی برای ذخیره فایل‌های هم‌ترازی
        final_align_path = os.path.join(dataset_base_dir, "aligns", spk_formatted)
        # مسیر دایرکتوری موقت حاوی فایل‌های هم‌ترازی (پوشه 'align')
        extracted_align_contents_dir = os.path.join(grid_temp_dir, "align", spk_raw, "align")

        if os.path.exists(extracted_align_contents_dir):
            os.makedirs(final_align_path, exist_ok=True)
            for item in os.listdir(extracted_align_contents_dir):
                shutil.move(os.path.join(extracted_align_contents_dir, item), final_align_path)
            print(f"  فایل‌های هم‌ترازی به {final_align_path} منتقل شدند.", flush=True)
        else:
            print(f"هشدار: دایرکتوری هم‌ترازی {extracted_align_contents_dir} برای {spk_raw} یافت نشد. انتقال نادیده گرفته می‌شود.", flush=True)

        # --- پاکسازی فایل 'tar' موقت و دایرکتوری‌های موقت 'align' ---
        if os.path.exists(temp_align_tar_path):
            os.remove(temp_align_tar_path)
            print(f"  فایل 'tar' موقت هم‌ترازی پاک شد: {temp_align_tar_path}", flush=True)

        # پاکسازی دایرکتوری‌های موقت هر گوینده در ساختار 'grid_temp'
        # برای ویدیو: حالا که فایل‌ها رو منتقل کردیم و فولدر داخلی رو پاک کردیم، فقط فولدر sXX بیرونی رو پاک می‌کنیم
        if os.path.exists(os.path.join(grid_temp_dir, "video", spk_raw)): # نیازی به چک کردن خالی بودن نیست چون sXX داخلی رو پاک کردیم
            try:
                shutil.rmtree(os.path.join(grid_temp_dir, "video", spk_raw)) # <--- تغییر اینجا!
            except OSError as e:
                print(f"هشدار: حذف دایرکتوری {os.path.join(grid_temp_dir, 'video', spk_raw)} امکان‌پذیر نیست: {e}", file=sys.stderr, flush=True)
        
        if os.path.exists(os.path.join(grid_temp_dir, "align", spk_raw)):
            try:
                shutil.rmtree(os.path.join(grid_temp_dir, "align", spk_raw))
            except OSError as e:
                print(f"هشدار: حذف دایرکتوری {os.path.join(grid_temp_dir, 'align', spk_raw)} امکان‌پذیر نیست: {e}", file=sys.stderr, flush=True)

    # --- پاکسازی نهایی دایرکتوری موقت 'grid_temp' ---
    print("\nدر حال پاکسازی دایرکتوری موقت 'grid_temp'...", flush=True)
    if os.path.exists(grid_temp_dir):
        try:
            shutil.rmtree(grid_temp_dir)
        except OSError as e:
            print(f"هشدار: حذف دایرکتوری نهایی 'grid_temp' امکان‌پذیر نیست: {e}", file=sys.stderr, flush=True)
    print("دانلود و سازماندهی دیتاست به پایان رسید!", flush=True)


# --- بخش اجرای اصلی اسکریپت (Main Execution Block) ---
if __name__ == "__main__":
    # مرحله آماده‌سازی: حذف هر گونه دایرکتوری 'Dataset' (شامل grid_temp داخلی) از اجرای قبلی
    print("--- آماده‌سازی برای اجرای جدید ---", flush=True)
    if os.path.exists("Dataset"):
        print("در حال حذف دایرکتوری موجود 'Dataset' برای اجرای تست تمیز...", flush=True)
        shutil.rmtree("Dataset")
    print("--- آماده‌سازی به پایان رسید ---", flush=True)

    # مرحله اول: اجرای تابع دانلود و سازماندهی GRID
    print("\n--- شروع دانلود و سازماندهی دیتاست ---", flush=True)
    download_and_organize_grid()
    
    print("\n--- دانلود و سازماندهی دیتاست به طور کامل به پایان رسید. ---", flush=True)
    print("\n عملیات دانلود و سازماندهی دیتاست با موفقیت به پایان رسید! ", flush=True)