# Configs\config.py

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# --- تنظیمات مجموعه داده (Dataset Configuration) ---
@dataclass
class DatasetConfig:
    RAW_DATASET_ROOT: Path = Path('./Dataset')
    PROCESSED_DATASET_ROOT: Path = Path('./dataset_processed')
    CHARACTER_MAP_FILE: Path = field(init=False)
    PHONEME_MAP_FILE: Path = field(init=False)
    VISEME_MAP_FILE: Path = field(init=False) # For viseme pre-training
    SPEAKER_META_FILE: Path = field(init=False)
    IMG_H: int = 70
    IMG_W: int = 105
    MAX_FRAMES: int = 75
    FRAME_RATE: int = 25
    LIP_PROPORTIONAL_MARGIN_X: float = 0.0
    LIP_PROPORTIONAL_MARGIN_Y: float = 0.02
    LIP_PADDING_TOP_PX: int = 0
    LIP_PADDING_BOTTOM_PX: int = 0
    LIP_PADDING_LEFT_PX: int = 0
    LIP_PADDING_RIGHT_PX: int = 0
    TRAIN_SPLIT_RATIO: float = 0.8
    VAL_SPLIT_RATIO: float = 0.15
    TEST_SPLIT_RATIO: float = 0.05
    NUM_VISEME_CLASSES: int = field(init=False)
    NUM_PHONEME_CLASSES: int = field(init=False)
    NUM_CHAR_CLASSES: int = field(init=False)
    NUM_SPEAKERS: int = field(init=False)
    RAW_VIDEOSET_ROOT: Path = field(init=False)
    RAW_ALIGNS_DIR: Path = field(init=False)
    PROCESSED_VIDEO_DIR: Path = field(init=False)
    PROCESSED_ALIGN_DIR: Path = field(init=False)
    NUM_CPU_CORES: int = field(default_factory=lambda: os.cpu_count() - 1 if os.cpu_count() is not None and os.cpu_count() > 1 else 1)
    SPEAKERS_TO_PROCESS: Optional[List[str]] = None
    APPLY_EMA_SMOOTHING: bool = True
    EMA_ALPHA: float = 0.3
    BLACK_OUT_NON_LIP_AREAS: bool = False
    CALCULATE_RTF: bool = False
    MAX_PROBLEMATIC_FRAMES_PERCENTAGE: int = 10
    APPLY_HISTOGRAM_MATCHING: bool = False
    HW_ACCELERATION_DEVICE: str = "cuda"
    SAVE_DEBUG_FRAMES: bool = False
    MAX_DEBUG_FRAMES: int = 5
    DEBUG_FRAMES_DIR: Path = Path("debug_output/preprocessing_debug_frames")
    CONVERT_TO_MP4_IF_NEEDED: bool = True
    MP4_TEMP_DIR: Path = field(init=False)

    def __post_init__(self):
        self.CHARACTER_MAP_FILE = self.PROCESSED_DATASET_ROOT / "character_to_id.json"
        self.PHONEME_MAP_FILE = self.PROCESSED_DATASET_ROOT / "phoneme_to_id.json"
        self.VISEME_MAP_FILE = self.PROCESSED_DATASET_ROOT / "viseme_to_id.json"
        self.SPEAKER_META_FILE = self.PROCESSED_DATASET_ROOT / "speaker_meta.json"
        self.RAW_VIDEOSET_ROOT = self.RAW_DATASET_ROOT / "videos"
        self.RAW_ALIGNS_DIR = self.RAW_DATASET_ROOT / "aligns"
        self.PROCESSED_VIDEO_DIR = self.PROCESSED_DATASET_ROOT / "video" 
        self.PROCESSED_ALIGN_DIR = self.PROCESSED_DATASET_ROOT / "align" 
        self.MP4_TEMP_DIR = self.RAW_DATASET_ROOT / "Temp_MP4_Conversions"
        if self.VISEME_MAP_FILE.exists():
            with open(self.VISEME_MAP_FILE, 'r', encoding='utf-8') as f:
                self.NUM_VISEME_CLASSES = len(json.load(f))
        else:
            self.NUM_VISEME_CLASSES = 0

        if self.PHONEME_MAP_FILE.exists():
            with open(self.PHONEME_MAP_FILE, 'r', encoding='utf-8') as f:
                self.NUM_PHONEME_CLASSES = len(json.load(f))
        else:
            self.NUM_PHONEME_CLASSES = 0
            
        if self.CHARACTER_MAP_FILE.exists():
            with open(self.CHARACTER_MAP_FILE, 'r', encoding='utf-8') as f:
                self.NUM_CHAR_CLASSES = len(json.load(f))
        else:
            self.NUM_CHAR_CLASSES = 0

        if self.SPEAKER_META_FILE.exists():
            with open(self.SPEAKER_META_FILE, 'r', encoding='utf-8') as f:
                speaker_meta = json.load(f)
            all_unique_speakers = set(sp for split in speaker_meta.values() for sp in split)
            self.NUM_SPEAKERS = len(all_unique_speakers)
        else:
            self.NUM_SPEAKERS = 0

# --- Augmentation Profiles ---

@dataclass
class AugConfigTrain:
    """Strong augmentation profile for main training (e.g., phoneme stage)."""
    USE_AUGMENTATION: bool = False
    # Spatial (Stronger)
    RESIZED_CROP: bool = False
    RESIZED_CROP_SCALE: Tuple[float, float] = (0.95, 1.0)
    RESIZED_CROP_RATIO: Tuple[float, float] = (0.9, 1.1)
    RESIZED_CROP_PROB: float = 0.3

    HORIZONTAL_FLIP: bool = False
    FLIP_PROB: float = 0.05

    ROTATION: bool = False
    ROTATION_DEGREES: float = 3.0
    ROTATION_PROB: float = 0.05

    AFFINE: bool = False
    AFFINE_TRANSLATE: Tuple[float, float] = (0.05, 0.05)
    AFFINE_PROB: float = 0.15

    # Color/Noise (Enabled for robustness)
    COLOR_JITTER: bool = False
    BRIGHTNESS: float = 0.2
    CONTRAST: float = 0.2
    SATURATION: float = 0.1
    HUE: float = 0.05
    JITTER_PROB: float = 0.05

    GAUSSIAN_NOISE: bool = False
    NOISE_MEAN: float = 0.0
    NOISE_STD: float = 0.01
    NOISE_PROB: float = 0.3

    GAUSSIAN_BLUR: bool = False
    BLUR_KERNEL_SIZE: int = 3
    BLUR_SIGMA: tuple = (0.05, 0.15)
    BLUR_PROB: float = 0.05

    # Time Masking
    TIME_MASKING: bool = False
    TIME_MASK_PROB: float = 0.05
    MAX_MASK_FRAMES: int = 4
    
    # --- Temporal ---
    TEMPORAL_JITTER: bool = False
    TEMPORAL_JITTER_PROB: float = 0.05
    MAX_TEMPORAL_SHIFT: int = 2
    TEMPORAL_BLUR: bool = False
    TEMPORAL_BLUR_PROB: float = 0.2


@dataclass
class rlx:
    NUM_PHONEME_CLASSES: int = field(init=False)
    NUM_CHAR_CLASSES: int = field(init=False)
    #fronbone
    use_aux_head = False            # auxiliary CTC head
    #  use_cross_scale = True         # LowRankCrossScaleFusion
    use_attn_dropblock = True      # spatial guided DropBlock
    use_temporal_dropout = True    # temporal drop

    d_model = 192                  # embedding dimension
    n_blocks = 5                   # تعداد TCN blocks
    kernel_list = [3,5,7]         # کرنل‌های TCN
    n_heads = 2                    # تعداد heads در Transformer
    n_transformer_layers = 2
    aux_loss_weight = 0.2
    frontend_out_ch = 128
    use_belief_branch = True
    tau = 0.5
    egosd_loss_weight= 1.0


# --- تنظیمات آموزش (Training Configuration) ---
@dataclass
class TrainingConfig:
    """
    این کلاس شامل تمامی تنظیمات و هایپرپارامترهای مربوط به فرایند آموزش مدل است.
    """
    MODEL_TO_TRAIN: str = "ULBiTP-Rxx" # Options: "ULBiTP-V1", "ULBiTP-V2", "ULBiTP-R", "ULBiTP-Rx", "ULBiTP-Rxx"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu" # دستگاه مورد استفاده برای آموزش (GPU یا CPU)


    NUM_EPOCHS: int = 90 # Increased for longer training possibility with early stopping
    BATCH_SIZE: int =  32
    
    # --- AdamW Optimizer Parameters ---
    LEARNING_RATE: float = 2e-4 # Increased from 1e-4 for training from scratch
    ADAM_BETAS: Tuple[float, float] = (0.9, 0.98) # مقادیر بتا برای AdamW
    ADAM_EPS: float = 1e-10 # مقدار اپسیلون برای AdamW
    WEIGHT_DECAY: float = 0.5 # مقدار تجزیه وزن (Weight Decay) - افزایش برای جلوگیری از overfit
    
    GRAD_NORM_CLIP: float = 10.0 
    WARMUP_STEPS : int = 5000 # تعداد مراحل گرم‌کردن (warm-up steps) برای برنامه زمان‌بندی نرخ یادگیری
    ACCUMULATION_STEPS = 1

    NUM_WORKERS: int = field(default_factory=lambda: os.cpu_count() - 1 if os.cpu_count() is not None and os.cpu_count() > 1 else 1)

    INFERENCE_BATCH_SIZE: int = 1 # اندازه بچ برای ارزیابی مدل در طول حلقه اعتبارسنجی (validation loop)

    CHECKPOINT_DIR: Path = Path('./checkpoints') # مسیر دایرکتوری برای ذخیره چک‌پوینت‌های مدل
    LOG_DIR: Path = Path('./logs')               # مسیر دایرکتوری برای ذخیره فایل‌های لاگ آموزش (مثلاً TensorBoard)
    PLOTS_DIR: Path = Path('./plots')

    RESUME_TRAINING: bool = False # آیا آموزش از آخرین چک‌پوینت یا یک چک‌پوینت مشخص ادامه یابد؟
    RESUME_FROM_CHECKPOINT: Optional[str] = "resume_checkpoint_epoch_040_per_0.3884.pth" # Let the script find the latest
    PRETRAINED_MODEL_PATH: Optional[str] = None # مسیر مدل پیش‌آموزش‌دیده برای بارگذاری (برای fine-tuning)
    SAVE_EVERY_N_EPOCHS: int = 1 # هر چند اپوک یک بار یک چک‌پوینت از مدل ذخیره شود

    # --- Sample Logging Parameters ---
    SAMPLE_LOG_INTERVAL: int = 1
    NUM_SAMPLES_TO_LOG: int = 5



    # --- Label Smoothing Configuration ---
    @dataclass
    class LabelSmoothingConfig:
        """Contains all settings for label smoothing, including annealing schedule."""
        USE_LABEL_SMOOTHING: bool = True
        # USE_ANNEALING is deprecated in favor of ConditionalAnnealingConfig
        USE_ANNEALING: bool = False
        # SCHEDULE is deprecated in favor of ConditionalAnnealingConfig
        SCHEDULE: List[Tuple[int, float]] = field(default_factory=lambda: [
            (50, 0.1),      # تا اپک 80 با alpha=0.1 بمان
            (1000, 0.05)    # بعد از آن alpha=0.05
        ])
        DEFAULT_ALPHA: float = 0.05

    label_smoothing: LabelSmoothingConfig = field(default_factory=LabelSmoothingConfig)



# --- کلاس پیکربندی اصلی (Main Configuration Class) ---
@dataclass
class MainConfig:
    """
    این کلاس پیکربندی اصلی است که تمامی پیکربندی‌های فرعی را در خود جای می‌دهد.
    با نمونه‌سازی از این کلاس، به تمامی تنظیمات پروژه دسترسی خواهید داشت.
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train_aug: AugConfigTrain = field(default_factory=AugConfigTrain)
    rlx_config: rlx = field(default_factory=rlx)
    training: TrainingConfig = field(default_factory=TrainingConfig)


    def __post_init__(self):
        """
        This method is executed after the constructor is called and is responsible for:
        1. Ensuring the necessary directories for checkpoints and logs exist.
        2. Initializing 'DatasetConfig' to load class counts and speaker data.
        3. Assigning class counts from 'DatasetConfig' to all model configs.
        """
        # --- Create necessary directories if they don't exist ---
        self.training.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.training.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # --- Call '__post_init__' for 'dataset' to load counts ---
        self.dataset.__post_init__() 
        
        
        # --- Propagate to new models ---
        self.rlx_config.NUM_PHONEME_CLASSES = self.dataset.NUM_PHONEME_CLASSES
        self.rlx_config.NUM_CHAR_CLASSES = self.dataset.NUM_CHAR_CLASSES
