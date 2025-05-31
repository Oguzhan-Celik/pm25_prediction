import os
import sys
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("improved_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the training function
from src.trainers.train import train_model

if __name__ == "__main__":
    try:
        # Specify the improved config file
        config_path = "configs/improved_model_config.yaml"

        # Create output directory for improved model
        os.makedirs("models/improved", exist_ok=True)

        # Log start of training
        logger.info(f"Starting training with improved configuration: {config_path}")

        # Train the model
        train_model(config_path)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
