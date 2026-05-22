import re
from pathlib import Path
from tqdm import tqdm

def convert_tcd_timit_txt_to_mfa(input_root: str, output_root: str):
    """
    Converts TCD-TIMIT .txt files to MFA-ready plain text, keeping speaker folder structure.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # پیدا کردن همه فایل‌های txt در تمام زیرپوشه‌ها
    txt_files = list(input_root.rglob("*.txt"))
    print(f"Found {len(txt_files)} text files.")

    for txt_path in tqdm(txt_files, desc="Converting TXT files"):
        # خواندن محتوای فایل
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        words = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # حذف تایم‌استمپ (مثلاً "2400 8160 word")
            parts = re.split(r"\s+", line, maxsplit=2)
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                word = parts[2]
            else:
                word = parts[-1]

            # تمیز کردن کلمات (حروف انگلیسی و ')
            word = re.sub(r"[^a-zA-Z']", "", word).lower()
            if word:
                words.append(word)

        clean_text = " ".join(words)

        # ساخت مسیر خروجی بر اساس فولدر گوینده و اسم فایل
        # rel_path = مسیر نسبی از input_root
        rel_path = txt_path.relative_to(input_root)
        out_txt_path = output_root / rel_path
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(clean_text + "\n")

    print(" Conversion completed. MFA-ready text files generated.")

# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    input_dir = "./Dataset/aligns"  # مسیر فولدر ورودی شما
    output_dir = "./Dataset/MFA-ready"               # مسیر خروجی آماده MFA
    convert_tcd_timit_txt_to_mfa(input_dir, output_dir)
