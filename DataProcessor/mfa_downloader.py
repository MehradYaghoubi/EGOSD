import subprocess
from pathlib import Path
from montreal_forced_aligner.models import download_acoustic_model, download_dictionary


MFA_MODELS_DIR = Path("./MFA_models")
MFA_MODELS_DIR.mkdir(parents=True, exist_ok=True)


acoustic_model_path = download_acoustic_model("english_mfa", output_dir=MFA_MODELS_DIR)
print(" Acoustic model downloaded at:", acoustic_model_path)

# دانلود دیکشنری انگلیسی
dictionary_path = download_dictionary("english_mfa", output_dir=MFA_MODELS_DIR)
print(" Dictionary downloaded at:", dictionary_path)


dataset_dir = Path("./Dataset/voice")  # مسیر فایل‌های wav و txt
aligned_dir = Path("./Dataset/Aligned")  # مسیر خروجی TextGrid
aligned_dir.mkdir(parents=True, exist_ok=True)


command = [
    "mfa",
    "align",
    str(dataset_dir),
    str(acoustic_model_path),
    str(dictionary_path),
    str(aligned_dir)
]

print(" Running MFA...")
subprocess.run(command)
print(" MFA alignment completed. Check folder:", aligned_dir)
