import os
import sys
import torch
import logging
from pathlib import Path
import torch.multiprocessing
import argparse  # Import argparse

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.trainers.train import train_model

if __name__ == "__main__":
    # Basic config for logging from this script before train_model's setup_logging takes over.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    script_logger = logging.getLogger(
        __name__
    )  # Logger for this script, will be __main__

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        script_logger.info("Set PyTorch multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        script_logger.warning(
            f"Could not set multiprocessing start method (it might be already set or in use): {e}"
        )
        pass

    parser = argparse.ArgumentParser(description="Train PM2.5 model script.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/improved_model_config.yaml",
        help="Path to YAML config file (relative to project root or absolute)",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of training and validation data to use (e.g., 0.1 for 10%%). Default is 1.0 (all data).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Override model_type from config (optional)",
    )
    script_args = parser.parse_args()

    try:
        config_path_input = Path(script_args.config)
        if not config_path_input.is_absolute():
            config_path_full = project_root / config_path_input
        else:
            config_path_full = config_path_input

        if not config_path_full.exists():
            script_logger.error(f"Configuration file not found: {config_path_full}")
            sys.exit(f"Error: Configuration file not found at {config_path_full}")

        if not (0.0 < script_args.data_fraction <= 1.0):
            script_logger.error(
                f"Invalid data_fraction: {script_args.data_fraction}. Must be between >0.0 and <=1.0."
            )
            sys.exit(f"Error: data_fraction must be > 0.0 and <= 1.0.")

        script_logger.info(f"Starting training with configuration: {config_path_full}")
        if script_args.data_fraction < 1.0:
            script_logger.info(
                f"Using data_fraction: {script_args.data_fraction:.2f} ({script_args.data_fraction*100:.0f}%)"
            )

        # Correctly pass all arguments to train_model
        train_model(
            config_path=str(config_path_full),
            model_type_override=script_args.model_type,
            data_fraction=script_args.data_fraction,
        )

        script_logger.info("Training completed successfully.")

    except FileNotFoundError as fnf_error:
        script_logger.error(f"File/Path related error: {fnf_error}", exc_info=True)
        sys.exit(f"Error: A file or path was not found. {fnf_error}")
    except Exception as e:
        script_logger.error(
            f"Error during training script execution: {e}", exc_info=True
        )
        sys.exit(f"train_improved_model.py: An critical error occurred: {e}")
