import logging
from dataclasses import is_dataclass, asdict

def log_run_details(logger, training_config, aug_config, model_config, dataset_config, decoder_config=None):
    """
    Logs the configuration details of a training run in a structured format.
    """
    logger.info("--- Configuration Details ---")

    # Log Training Configuration
    logger.info("[Training Configuration]")
    for key, value in asdict(training_config).items():
        if is_dataclass(value):
            # Skip nested dataclasses here, as they are handled separately or not needed
            continue
        logger.info(f"  {key}: {value}")

    # Log Augmentation Configuration
    logger.info("\n[Augmentation Configuration]")
    for key, value in asdict(aug_config).items():
        logger.info(f"  {key}: {value}")

    # Log Model Configuration
    logger.info("\n[Model Configuration]")
    for key, value in asdict(model_config).items():
        logger.info(f"  {key}: {value}")

    # Log Dataset Configuration
    logger.info("\n[Dataset Configuration]")
    for key, value in asdict(dataset_config).items():
        if 'CLASSES' in key or 'SPEAKERS' in key: # Log class/speaker counts
             logger.info(f"  {key}: {value}")
        elif key in ['IMG_H', 'IMG_W', 'MAX_FRAMES', 'FRAME_RATE']: # Log key dataset params
             logger.info(f"  {key}: {value}")

    # Log Decoder Configuration if provided
    if decoder_config:
        logger.info("\n[Decoder Configuration]")
        for key, value in asdict(decoder_config).items():
            logger.info(f"  {key}: {value}")
            
    logger.info("-----------------------------")