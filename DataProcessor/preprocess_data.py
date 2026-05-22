# DataProcessor\preprocess_data.py

import sys
import os # Import os for environment variables
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import shutil
import logging
from typing import Tuple, Optional # Add Tuple and Optional for type hinting

# --- Set Environment Variables at the very top ---
# These should be set before any TensorFlow/MediaPipe related imports or operations
# to ensure they take effect.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppresses oneDNN warning from TensorFlow
os.environ['GLOG_minloglevel'] = '2'     # Suppresses verbose MediaPipe logs (INFO/WARNING)

# Configure logging for this script. This should generally align with your overall application's logging strategy.
# For debugging preprocessing, DEBUG or INFO levels are useful.
# The GLOG_minloglevel setting above will handle MediaPipe's internal verbosity.
logging.basicConfig(level=logging.ERROR , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger instance for this module

# Ensure the project root is in sys.path to allow importing 'Configs' module
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import LipExtractor and its config at the top level
from pylipextractor.lip_extractor import LipExtractor

# Import configuration classes 'MainConfig' and 'DatasetConfig'
from Configs.config import MainConfig, DatasetConfig


# Global counter for overall progress tracking across processes
global_clip_counter = multiprocessing.Value('i', 0)
clip_counter_lock = multiprocessing.Lock()

def get_next_global_clip_index():
    """
    Atomically increments the global clip counter and returns the next value.
    Used for tracking overall progress.
    """
    with clip_counter_lock:
        index = global_clip_counter.value
        global_clip_counter.value += 1
        return index

# --- Worker function for multiprocessing pool ---
def process_single_clip(task_args) -> Tuple[bool, Optional[float]]:
    """
    Processes a single video clip using LipExtractor. This function is designed to be
    executed by `multiprocessing` workers. The LipExtractor's configuration
    must be applied *within* this function for each child process.

    Args:
        task_args (tuple): A tuple containing the necessary arguments for processing:
            `(video_path, base_filename, processed_speaker_video_dir, dataset_config, speaker_id)`.

    Returns:
        Tuple[bool, Optional[float]]: `(True, rtf_value)` if processing was successful,
                                     `(False, None)` otherwise (e.g., skipped or failed).
    """
    video_path, base_filename, processed_speaker_video_dir, dataset_config, speaker_id = task_args # Unpack dataset_config

    # --- Apply LipExtractor configuration within each child process ---
    # This is CRUCIAL for multiprocessing, as child processes get a fresh namespace.
    LipExtractor.config.IMG_H = dataset_config.IMG_H
    LipExtractor.config.IMG_W = dataset_config.IMG_W
    LipExtractor.config.MAX_FRAMES = dataset_config.MAX_FRAMES
    LipExtractor.config.LIP_PROPORTIONAL_MARGIN_X = dataset_config.LIP_PROPORTIONAL_MARGIN_X
    LipExtractor.config.LIP_PROPORTIONAL_MARGIN_Y = dataset_config.LIP_PROPORTIONAL_MARGIN_Y
    LipExtractor.config.LIP_PADDING_TOP_PX = dataset_config.LIP_PADDING_TOP_PX
    LipExtractor.config.LIP_PADDING_BOTTOM_PX = dataset_config.LIP_PADDING_BOTTOM_PX
    LipExtractor.config.LIP_PADDING_LEFT_PX = dataset_config.LIP_PADDING_LEFT_PX
    LipExtractor.config.LIP_PADDING_RIGHT_PX = dataset_config.LIP_PADDING_RIGHT_PX

    LipExtractor.config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE = dataset_config.MAX_PROBLEMATIC_FRAMES_PERCENTAGE
    LipExtractor.config.CONVERT_TO_MP4_IF_NEEDED = dataset_config.CONVERT_TO_MP4_IF_NEEDED
    LipExtractor.config.MP4_TEMP_DIR = dataset_config.MP4_TEMP_DIR

    # NEW v2 Configs for EMA Smoothing, Blacking Out, and Histogram Matching
    LipExtractor.config.APPLY_EMA_SMOOTHING = dataset_config.APPLY_EMA_SMOOTHING
    LipExtractor.config.EMA_ALPHA = dataset_config.EMA_ALPHA
    LipExtractor.config.BLACK_OUT_NON_LIP_AREAS = dataset_config.BLACK_OUT_NON_LIP_AREAS
    LipExtractor.config.APPLY_HISTOGRAM_MATCHING = dataset_config.APPLY_HISTOGRAM_MATCHING

    # Debugging settings for pylipextractor itself (from your config)
    LipExtractor.config.SAVE_DEBUG_FRAMES = dataset_config.SAVE_DEBUG_FRAMES
    LipExtractor.config.MAX_DEBUG_FRAMES = dataset_config.MAX_DEBUG_FRAMES
    LipExtractor.config.DEBUG_OUTPUT_DIR = dataset_config.DEBUG_FRAMES_DIR

    LipExtractor.config.INCLUDE_LANDMARKS_ON_FINAL_OUTPUT = False # Typically set to False for final dataset generation
    LipExtractor.config.REFINE_LANDMARKS = False # This line was added in the previous update
    LipExtractor.config.PROFILE_CODE = False # Set to True if you want to profile the code execution time
    LipExtractor.config.HW_ACCELERATION_DEVICE = dataset_config.HW_ACCELERATION_DEVICE 
    LipExtractor.config.CALCULATE_RTF = dataset_config.CALCULATE_RTF # This line was updated in the previous fix

    # --- End of LipExtractor configuration for this process ---


    output_npy_path = processed_speaker_video_dir / f"{base_filename}.npy"

    # If the processed .npy file already exists, skip reprocessing.
    if output_npy_path.exists() and output_npy_path.stat().st_size > 0:
        logger.debug(f"Skipping '{video_path.name}': Processed .npy file already exists.")
        return True, None # Return None for RTF if skipped

    try:
        # Each process gets its own LipExtractor instance.
        # It will use the LipExtractor.config settings just applied above.
        extractor = LipExtractor()

        # Now unpacks both extracted_frames and rtf_value
        extracted_frames, rtf_value = extractor.extract_lip_frames(
            video_path=video_path,
            output_npy_path=output_npy_path, # LipExtractor handles saving directly
        )

        if extracted_frames is None or extracted_frames.shape[0] == 0:
            if output_npy_path.exists() and output_npy_path.stat().st_size == 0:
                os.remove(output_npy_path) # Clean up empty file if created
            logger.warning(f"No valid frames extracted for '{video_path.name}' (Speaker: {speaker_id}). Skipping.")
            return False, None # Return None for RTF if no frames extracted

        # Check rtf_value before formatting
        rtf_log_message = f"RTF: {rtf_value:.4f}" if rtf_value is not None else "RTF: N/A (محاسبه نشد)" # This line was updated in the previous fix
        logger.debug(f"Successfully processed '{video_path.name}' (Speaker: {speaker_id}) with {extracted_frames.shape[0]} frames. {rtf_log_message}")
        return True, rtf_value # Return RTF value on success

    except Exception as e:
        logger.error(f"Error processing '{video_path.name}' (Speaker: {speaker_id}). Details: {e}")
        if output_npy_path.exists():
            try:
                os.remove(output_npy_path)
            except OSError as cleanup_e:
                logger.warning(f"Failed to clean up '{output_npy_path}' after error: {cleanup_e}")
        return False, None # Return None for RTF on error


# --- Main Processing Orchestration Function ---
def run_processing(main_config: MainConfig, speakers_to_process_list: list[str]):
    """
    Orchestrates the preprocessing pipeline for specified speakers.
    Manages directory creation, task collection, and parallel execution.
    LipExtractor configuration specific to each process is now handled in `process_single_clip`.

    Args:
        main_config (MainConfig): The main configuration object.
        speakers_to_process_list (List[str]): List of speaker IDs (e.g., `['s01', 's02']`) to process.
    """
    logger.info("--- Starting Data Preprocessing ---")

    # Configure logging for pylipextractor's internal log messages as well
    # This ensures that pylipextractor's loggers (e.g., 'pylipextractor.lip_extractor')
    # also inherit the logging level and format.
    pylip_logger = logging.getLogger('pylipextractor')
    pylip_logger.setLevel(logging.INFO) # Or logging.DEBUG if you want more verbose pylipextractor logs

    logger.info("pylipextractor configuration will be applied within each child process based on MainConfig.dataset.")
    logger.info(f"pylipextractor Debugging Enabled: {main_config.dataset.SAVE_DEBUG_FRAMES} (Output to: {main_config.dataset.DEBUG_FRAMES_DIR})")
    logger.info(f"pylipextractor MP4 Conversion Enabled: {main_config.dataset.CONVERT_TO_MP4_IF_NEEDED} (Temp dir: {main_config.dataset.MP4_TEMP_DIR})")
    logger.info(f"pylipextractor EMA Smoothing Enabled: {main_config.dataset.APPLY_EMA_SMOOTHING} (Alpha: {main_config.dataset.EMA_ALPHA})")
    logger.info(f"pylipextractor Black Out Non-Lip Areas: {main_config.dataset.BLACK_OUT_NON_LIP_AREAS}")
    logger.info(f"pylipextractor Apply Histogram Matching: {main_config.dataset.APPLY_HISTOGRAM_MATCHING}")
    logger.info(f"pylipextractor Calculate RTF: {main_config.dataset.CALCULATE_RTF}") # This line was added in the previous fix


    # --- 2. Create necessary directories and clear old debug/temp files ---
    main_config.dataset.RAW_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    main_config.dataset.RAW_VIDEOSET_ROOT.mkdir(parents=True, exist_ok=True)
    main_config.dataset.PROCESSED_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    main_config.dataset.PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Clear pylipextractor's debug directory if its debug frames are enabled.
    # Note: This cleanup is done by the parent process. Child processes will also try to create/write.
    # If DEBUG_OUTPUT_DIR is shared, ensure pylipextractor handles concurrent writes (it usually does by video name subdirs).
    if main_config.dataset.SAVE_DEBUG_FRAMES and main_config.dataset.DEBUG_FRAMES_DIR.exists():
        try:
            for root, dirs, files in os.walk(main_config.dataset.DEBUG_FRAMES_DIR):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
            logger.info(f"pylipextractor debug frames directory '{main_config.dataset.DEBUG_FRAMES_DIR}' cleared for a fresh run.")
        except OSError as e:
            logger.warning(f"Could not clear pylipextractor debug directory '{main_config.dataset.DEBUG_FRAMES_DIR}': {e}")

    # Clear temporary MP4 directory if conversion is enabled.
    if main_config.dataset.CONVERT_TO_MP4_IF_NEEDED and main_config.dataset.MP4_TEMP_DIR.exists():
        try:
            for item in main_config.dataset.MP4_TEMP_DIR.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file():
                    os.remove(item)
            logger.info(f"Temporary MP4 conversion directory '{main_config.dataset.MP4_TEMP_DIR}' cleared for a fresh run.")
        except OSError as e:
            logger.warning(f"Could not clear temporary MP4 directory '{main_config.dataset.MP4_TEMP_DIR}': {e}")


    # --- 3. Load speaker metadata ---
    try:
        with open(main_config.dataset.SPEAKER_META_FILE, 'r', encoding='utf-8') as f:
            speaker_metadata = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Speaker metadata file ({main_config.dataset.SPEAKER_META_FILE}) not found. "
                     "Please run 'speaker_segmenter.py' first.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Speaker metadata file ({main_config.dataset.SPEAKER_META_FILE}) is not valid. "
                     "Please run 'speaker_segmenter.py' first.")
        return

    # Create a mapping from 'speaker_id' to split name (e.g., "train", "val", "test").
    speaker_to_split_map = {}
    for split_name, s_list in speaker_metadata.items():
        for s_id in s_list:
            speaker_to_split_map[s_id] = split_name

    all_tasks = [] # List to hold all clip processing tasks.
    speakers_for_loop = speakers_to_process_list
    logger.info(f"Collecting tasks for processing {len(speakers_for_loop)} specified speakers...")


    # --- 4. Collect processing tasks for each clip ---
    for speaker_id in tqdm(speakers_for_loop, desc="Collecting clips for processing"):
        split_name = speaker_to_split_map.get(speaker_id, "unknown") # Determine split.

        # Create output directories for processed frames.
        split_video_dir = main_config.dataset.PROCESSED_VIDEO_DIR / split_name
        split_video_dir.mkdir(parents=True, exist_ok=True) # Ensure split directory exists.

        raw_speaker_video_dir = main_config.dataset.RAW_VIDEOSET_ROOT / f"{speaker_id}"
        processed_speaker_video_dir = split_video_dir / f"{speaker_id}"
        processed_speaker_video_dir.mkdir(exist_ok=True) # Ensure speaker directory exists within split directory.

        # Find all raw video files for the current speaker.
        # Prioritize .mp4 if both exist, otherwise take first available type.
        video_files = sorted(list(raw_speaker_video_dir.glob("*.mp4")) + \
                             list(raw_speaker_video_dir.glob("*.mpg")) + \
                             list(raw_speaker_video_dir.glob("*.avi")))

        if not video_files:
            logger.warning(f"No video files found for speaker '{speaker_id}' in {raw_speaker_video_dir}. Skipping this speaker.")
            continue

        for video_file_path in video_files:
            base_filename = video_file_path.stem

            # Add task to the overall list of tasks.
            # CRITICAL: Pass main_config.dataset to the worker function.
            all_tasks.append((video_file_path, base_filename,
                              processed_speaker_video_dir,
                              main_config.dataset, speaker_id)) # Now passing dataset_config explicitly

    if not all_tasks:
        logger.info("No tasks to process. Please check valid speakers and ensure raw files exist.")
        return

    logger.info(f"\nStarting parallel processing of {len(all_tasks)} video clips using {main_config.dataset.NUM_CPU_CORES} CPU cores...")

    # Reset global clip counter.
    with clip_counter_lock:
        global_clip_counter.value = 0

    # List to collect RTF values
    collected_rtf_values = []

    # --- 5. Execute parallel processing with multiprocessing.Pool ---
    with multiprocessing.Pool(processes=main_config.dataset.NUM_CPU_CORES, maxtasksperchild=20) as pool:
        # imap_unordered is good for showing progress while tasks complete out of order
        results_iterator = pool.imap_unordered(process_single_clip, all_tasks, chunksize=1)

        # Use tqdm to show a progress bar for all tasks and collect RTF values
        for success, rtf_value in tqdm(results_iterator, total=len(all_tasks), desc="Overall Processing Progress"):
            if success and rtf_value is not None:
                collected_rtf_values.append(rtf_value)

    logger.info("\n--- Data Preprocessing Completed! ---")

    # --- Calculate and display average RTF ---
    if collected_rtf_values:
        average_rtf = np.mean(collected_rtf_values)
        logger.info(f"میانگین Real-Time Factor (RTF) برای کل ویدیوهای پردازش شده: {average_rtf:.4f}")
        logger.info(f"تعداد ویدیوهایی که RTF برای آنها محاسبه شد: {len(collected_rtf_values)}")
    else:
        logger.info("هیچ مقدار RTF برای محاسبه میانگین جمع‌آوری نشد. (ممکن است هیچ ویدیویی با موفقیت پردازش نشده باشد یا RTF فعال نباشد).")

if __name__ == "__main__":
    # Ensure this part of the code is run only when the script is executed directly
    # and not when imported by multiprocessing child processes.
    # This prevents redundant loading of metadata in child processes.
    if sys.platform.startswith('win'):
        # On Windows, 'spawn' start method is often default or required.
        # This block ensures that the main execution logic is wrapped in
        # if __name__ == '__main__': for multiprocessing safety.
        multiprocessing.freeze_support() # For Windows executable support if needed

    main_config_instance = MainConfig()

    try:
        # Load speaker metadata to determine which speakers are available
        with open(main_config_instance.dataset.SPEAKER_META_FILE, 'r', encoding='utf-8') as f:
            speaker_meta_data = json.load(f)

        all_speakers_from_meta = []
        for split_list in speaker_meta_data.values():
            all_speakers_from_meta.extend(split_list)
        # Get unique and sorted list of all available speakers.
        available_speakers = sorted(list(set(all_speakers_from_meta)))

        # --- Determine speakers to process based on config ---
        if main_config_instance.dataset.SPEAKERS_TO_PROCESS:
            # If specific speakers are defined in config, use them
            speakers_to_process = [
                s for s in main_config_instance.dataset.SPEAKERS_TO_PROCESS
                if s in available_speakers
            ]
            if not speakers_to_process:
                logger.error("Error: No valid speakers found in SPEAKERS_TO_PROCESS from config. "
                             "Please ensure the specified speaker IDs exist in speaker_meta.json.")
                sys.exit(1)
            elif len(speakers_to_process) < len(main_config_instance.dataset.SPEAKERS_TO_PROCESS):
                missing_speakers = set(main_config_instance.dataset.SPEAKERS_TO_PROCESS) - set(available_speakers)
                logger.warning(f"Some specified speakers were not found in speaker_meta.json and will be skipped: {list(missing_speakers)}")

            logger.info(f"Selected speakers for processing from config: {speakers_to_process}")
        else:
            # If SPEAKERS_TO_PROCESS is None or empty, process all available speakers
            speakers_to_process = available_speakers
            logger.info(f"No specific speakers defined in config. Processing ALL available speakers: {speakers_to_process}")


    except FileNotFoundError:
        logger.error(f"Error: Speaker metadata file ({main_config_instance.dataset.SPEAKER_META_FILE}) not found. "
                     "Cannot determine speakers to process. Please run 'speaker_segmenter.py' first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading speaker metadata: {e}")
        sys.exit(1)

    # Run the main preprocessing pipeline.
    run_processing(main_config_instance, speakers_to_process)
