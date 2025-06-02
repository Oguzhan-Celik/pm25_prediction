import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import warnings
from datetime import datetime
import sys
import traceback
import pandas as pd
import gc
import argparse
import torch.multiprocessing as mp

project_root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root_path))

from src.models.model import PM25Model
from src.models.loss_functions import calculate_metrics

from torch.cuda.amp import GradScaler, autocast


class WindowsCompatibleMultiStationDataset(Dataset):
    """Windows-compatible version that avoids complex objects in __init__"""

    def __init__(
        self,
        X_data_np: np.ndarray,
        Y_data_np: np.ndarray,
        num_stations_to_tile_to: int,
        time_steps: int,
    ):
        # Store direct references to NumPy arrays.
        # Pickling by DataLoader for worker processes will handle copying.
        self.X_data_np = X_data_np
        self.Y_data_np = Y_data_np
        self.num_stations_to_tile_to = num_stations_to_tile_to
        self.time_steps = time_steps

        if X_data_np.ndim != 3:
            print(
                f"CRITICAL DATASET ERROR: X_data_np must be 3D (N, T, F_original), got {X_data_np.shape}"
            )
            raise ValueError(
                f"X_data_np must be 3D (N, T, F_original), got {X_data_np.shape}"
            )
        if X_data_np.shape[1] != time_steps:
            print(
                f"CRITICAL DATASET ERROR: X_data_np time_steps ({X_data_np.shape[1]}) != configured ({time_steps})"
            )
            raise ValueError(
                f"X_data_np time_steps ({X_data_np.shape[1]}) != configured ({time_steps})"
            )
        if Y_data_np.ndim != 3:
            print(
                f"CRITICAL DATASET ERROR: Y_data_np must be 3D (N, S, 1), got {Y_data_np.shape}"
            )
            raise ValueError(f"Y_data_np must be 3D (N, S, 1), got {Y_data_np.shape}")
        if Y_data_np.shape[1] != num_stations_to_tile_to:
            _init_logger = logging.getLogger(
                f"{__name__}.WindowsCompatibleMultiStationDataset"
            )  # Get logger if needed for warnings
            _init_logger.warning(
                f"Y_data_np num_stations ({Y_data_np.shape[1]}) != num_stations_to_tile_to ({num_stations_to_tile_to}). Check Y data prep."
            )
        if Y_data_np.shape[2] != 1:
            _init_logger = logging.getLogger(
                f"{__name__}.WindowsCompatibleMultiStationDataset"
            )
            _init_logger.warning(
                f"Y_data_np final dimension ({Y_data_np.shape[2]}) != 1. Check Y data prep."
            )

    def __len__(self):
        return len(self.X_data_np)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sample_np = self.X_data_np[idx]
        y_sample_np = self.Y_data_np[idx]

        x_sample_tensor = torch.as_tensor(x_sample_np)

        if x_sample_tensor.ndim == 2:
            x_expanded = x_sample_tensor.unsqueeze(1)
            x_tiled = x_expanded.repeat(1, self.num_stations_to_tile_to, 1)
        else:
            print(
                f"ERROR [Dataset.__getitem__]: x_sample_tensor at index {idx} has unexpected ndim: {x_sample_tensor.ndim}. Expected 2D (T,F_original). Original np shape: {x_sample_np.shape}"
            )
            raise ValueError(
                f"x_sample_tensor at index {idx} has incompatible shape {x_sample_tensor.shape} for tiling."
            )

        return x_tiled.float(), torch.as_tensor(y_sample_np).float()


def setup_logging(output_dir=None, log_file_name="training_run.log"):
    log_level = logging.DEBUG
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        try:
            file_handler = logging.FileHandler(
                os.path.join(output_dir, log_file_name), mode="a"
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler for logging: {e}")

    logging.basicConfig(
        level=log_level, format=log_format, handlers=handlers, force=True
    )
    logger_instance = logging.getLogger(__name__)
    logger_instance.info(
        f"Logging setup. Level: {log_level}. Output: {output_dir}/{log_file_name if output_dir and any(isinstance(h, logging.FileHandler) for h in handlers) else 'Console only'}"
    )
    return logger_instance


logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = "min"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter, self.best_score, self.early_stop = 0, None, False
        self._es_logger = logging.getLogger(f"{__name__}.EarlyStopping")
        self._es_logger.debug(
            f"EarlyStopping init: patience={patience}, min_delta={min_delta}, mode={mode}"
        )

    def __call__(self, val_loss: float, model: nn.Module, path: str) -> bool:
        self._es_logger.debug(
            f"ES check: val_loss={val_loss:.4f}, best_score={self.best_score}"
        )
        score = -val_loss if self.mode == "min" else val_loss
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self._es_logger.info(
                f"ES: Model improved. New best val_loss={val_loss:.4f}. Saving to {path}"
            )
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        else:
            self.counter += 1
            self._es_logger.debug(
                f"ES: No improvement. Counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                self._es_logger.info("ES: Triggered.")
        return self.early_stop

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "timestamp": datetime.now().isoformat(),
                    "model_config": (
                        model.get_model_complexity()
                        if hasattr(model, "get_model_complexity")
                        else {}
                    ),
                },
                path,
            )
            self._es_logger.debug(f"Checkpoint saved: {path}")
        except Exception as e:
            self._es_logger.error(
                f"Failed to save checkpoint to {path}: {e}", exc_info=True
            )


def load_config(config_path: str) -> Dict[str, Any]:
    logger.info(f"Loading config from {config_path}")
    abs_path = (
        Path(config_path)
        if Path(config_path).is_absolute()
        else get_project_root() / config_path
    )
    try:
        with open(abs_path, "r") as f:
            config = yaml.safe_load(f)
        logger.debug(f"Config loaded: {config}")
        return config
    except Exception as e:
        logger.error(f"Error loading config {abs_path}: {e}", exc_info=True)
        raise


def load_processed_data(
    data_dir_str: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    np.ndarray,
    Dict,
    Optional[list],
]:
    logger.info(f"Loading processed data from: {data_dir_str}")
    data_dir = (
        Path(data_dir_str)
        if Path(data_dir_str).is_absolute()
        else get_project_root() / data_dir_str
    )
    logger.info(f"Absolute data dir: {data_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    logger.debug(
        f"Files in data dir '{data_dir}': {[f.name for f in data_dir.iterdir()]}"
    )
    paths = {
        "train": data_dir / "train_enhanced.pkl",
        "valid": data_dir / "valid_enhanced.pkl",
        "test": data_dir / "test_enhanced.pkl",
        "scalers": data_dir / "scalers_enhanced.pkl",
        "adj_matrix": data_dir / "adj_matrix_enhanced.npy",
        "feature_cols": data_dir / "feature_cols.pkl",
    }
    for name, path_obj in paths.items():
        if not path_obj.exists():
            raise FileNotFoundError(f"Required file '{name}' not found: {path_obj}")
    data = {
        name: (
            pickle.load(open(path_obj, "rb"))
            if path_obj.suffix == ".pkl"
            else np.load(path_obj)
        )
        for name, path_obj in paths.items()
    }
    f_cols = data["feature_cols"]
    if isinstance(f_cols, dict) and "feature_cols" in f_cols:
        f_cols = f_cols["feature_cols"]
    if not isinstance(f_cols, list):
        raise ValueError(f"feature_cols is not a list: {type(f_cols)}")

    def extract_xy(d_content, feat_cols, target_col_name="PM2.5"):
        if isinstance(d_content, dict):
            if "X" in d_content and "Y" in d_content:
                return np.array(d_content["X"], dtype=np.float32), np.array(
                    d_content["Y"], dtype=np.float32
                )
            else:
                d_df = pd.DataFrame(d_content)
                return d_df[feat_cols].values.astype(np.float32), d_df[
                    [target_col_name]
                ].values.astype(np.float32)
        if isinstance(d_content, pd.DataFrame):
            return d_content[feat_cols].values.astype(np.float32), d_content[
                [target_col_name]
            ].values.astype(np.float32)
        if isinstance(d_content, tuple) and len(d_content) == 2:
            return np.array(d_content[0], dtype=np.float32), np.array(
                d_content[1], dtype=np.float32
            )
        raise ValueError(f"Unknown data format for X,Y: {type(d_content)}")

    train_X, train_Y = extract_xy(data["train"], f_cols)
    valid_X, valid_Y = extract_xy(data["valid"], f_cols)
    test_X, test_Y = extract_xy(data["test"], f_cols)
    logger.info(f"Loaded train - X:{train_X.shape}, Y:{train_Y.shape}")
    logger.info(f"Loaded valid - X:{valid_X.shape}, Y:{valid_Y.shape}")
    logger.info(
        f"Adj matrix: {data['adj_matrix'].shape}, Scalers: {len(data['scalers'])} keys, Feature cols: {len(f_cols)}"
    )
    return (
        (train_X, train_Y),
        (valid_X, valid_Y),
        (test_X, test_Y),
        data["adj_matrix"],
        data["scalers"],
        f_cols,
    )


def prepare_time_series_data(
    X: np.ndarray, Y: np.ndarray, sequence_length: int, num_stations: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(
        f"Preparing time series: seq_len={sequence_length}, num_stations_for_Y={num_stations}"
    )
    logger.debug(f"Input X:{X.shape}, Y:{Y.shape}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    # Check against original length before slicing for sequence creation
    if len(X) < sequence_length:
        logger.warning(
            f"Not enough samples ({len(X)}) for seq_len ({sequence_length}) to form even one sequence for X."
        )
        # Return empty arrays that match the expected dimensionality for subsequent steps
        return np.empty(
            (0, sequence_length, X.shape[1] if X.ndim > 1 else 0), dtype=np.float32
        ), np.empty((0, num_stations, 1), dtype=np.float32)

    X_seq_list, Y_seq_list = [], []
    num_possible_sequences = len(X) - sequence_length + 1

    if num_possible_sequences <= 0:
        logger.warning(
            f"Not enough data to form any sequences. X length: {len(X)}, seq_len: {sequence_length}"
        )
        return np.empty(
            (0, sequence_length, X.shape[1] if X.ndim > 1 else 0), dtype=np.float32
        ), np.empty((0, num_stations, 1), dtype=np.float32)

    for i in range(num_possible_sequences):
        X_seq_list.append(X[i : i + sequence_length])
        Y_seq_list.append(Y[i + sequence_length - 1])

    X_seq_np = np.array(X_seq_list, dtype=np.float32)
    Y_seq_np = np.array(Y_seq_list, dtype=np.float32)

    logger.info(f"X_seq_np shape before passing to Dataset: {X_seq_np.shape}")

    if Y_seq_np.ndim == 2 and Y_seq_np.shape[1] == 1:
        Y_seq_np = np.repeat(Y_seq_np[:, np.newaxis, :], num_stations, axis=1)
    elif Y_seq_np.ndim == 2 and Y_seq_np.shape[1] == num_stations:
        Y_seq_np = Y_seq_np[:, :, np.newaxis]
    elif (
        Y_seq_np.ndim != 3
        or Y_seq_np.shape[1] != num_stations
        or Y_seq_np.shape[2] != 1
    ):
        if (
            Y_seq_np.size == 0 and num_possible_sequences > 0
        ):  # If X_seq_np was created but Y is problematic
            logger.error(
                f"Y_seq_np became empty or has incompatible shape {Y_seq_np.shape} despite having {num_possible_sequences} X sequences. This indicates a problem with Y data or its alignment with X."
            )
            Y_seq_np = np.full(
                (X_seq_np.shape[0], num_stations, 1), np.nan, dtype=np.float32
            )  # Create NaN array of expected shape
        elif Y_seq_np.size > 0:  # If not empty but wrong shape
            logger.warning(
                f"Y_seq_np shape {Y_seq_np.shape} not ideal for target (N_sequences, S, 1). Attempting to force reshape."
            )
            if Y_seq_np.ndim == 1:
                Y_seq_np = Y_seq_np.reshape(-1, 1)
            if Y_seq_np.ndim == 2 and Y_seq_np.shape[1] == 1:
                Y_seq_np = np.repeat(Y_seq_np[:, np.newaxis, :], num_stations, axis=1)
            elif Y_seq_np.ndim == 2 and Y_seq_np.shape[1] != num_stations:
                logger.error(
                    f"Y_seq_np has {Y_seq_np.shape[1]} features/stations for target, expected {num_stations}. Taking first and repeating for all stations."
                )
                Y_seq_np = Y_seq_np[:, :1]  # Take first column only
                Y_seq_np = np.repeat(
                    Y_seq_np[:, np.newaxis, :], num_stations, axis=1
                )  # Repeat to match num_stations
        # If Y_seq_np was empty because num_possible_sequences was 0, it's already handled by returning empty arrays earlier.

    logger.info(
        f"Created sequences for Dataset - X: {X_seq_np.shape}, Y: {Y_seq_np.shape}"
    )
    if X_seq_np.size > 0 and (np.isnan(X_seq_np).any() or np.isinf(X_seq_np).any()):
        logger.warning("NaN/Inf in X_seq!")
    if Y_seq_np.size > 0 and (np.isnan(Y_seq_np).any() or np.isinf(Y_seq_np).any()):
        logger.warning("NaN/Inf in Y_seq!")
    return X_seq_np, Y_seq_np


def clear_gpu_cache(current_logger: logging.Logger):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        current_logger.debug("GPU Cache cleared and synchronized.")


def log_memory_stats(
    logger_instance: logging.Logger, force_log=False, batch_idx_info=None
):
    log_condition = force_log
    if batch_idx_info is not None and batch_idx_info % 1000 == 0:
        log_condition = True

    if log_condition:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger_instance.debug(
                f"GPU Memory (Batch {batch_idx_info if batch_idx_info is not None else 'N/A'}) - Alloc: {allocated:.2f}GB, Reserv: {reserved:.2f}GB"
            )


def create_improved_scheduler(optimizer, config, steps_per_epoch_for_onecycle=None):
    train_cfg = config["training"]
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "cosine").lower()
    min_lr = float(train_cfg.get("min_lr", 1e-6))

    if scheduler_type == "cosine":
        T_0 = int(scheduler_cfg.get("T_0", 10))
        T_mult = int(scheduler_cfg.get("T_mult", 2))
        logger.info(
            f"Using CosineAnnealingWarmRestarts scheduler (T_0={T_0}, T_mult={T_mult}, eta_min={min_lr})."
        )
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr
        )
    elif scheduler_type == "onecycle":
        epochs = int(train_cfg.get("epochs", 50))
        if steps_per_epoch_for_onecycle is None:  # Should be provided if > 0
            steps_per_epoch_for_onecycle = int(scheduler_cfg.get("steps_per_epoch", 0))

        if steps_per_epoch_for_onecycle == 0 and epochs > 0:
            logger.error(
                "steps_per_epoch_for_onecycle must be > 0 for OneCycleLR if epochs > 0."
            )
            logger.warning(
                "Falling back to CosineAnnealingWarmRestarts due to OneCycleLR config issue."
            )
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=min_lr
            )
        elif (
            steps_per_epoch_for_onecycle == 0 and epochs == 0
        ):  # total_steps must be > 0 for OneCycleLR
            logger.error(
                "epochs and steps_per_epoch_for_onecycle are both 0. OneCycleLR requires positive total steps."
            )
            logger.warning(
                "Falling back to CosineAnnealingWarmRestarts due to OneCycleLR config issue."
            )
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=min_lr
            )

        logger.info(
            f"Using OneCycleLR scheduler (max_lr={train_cfg['learning_rate']}, total_steps={epochs*steps_per_epoch_for_onecycle})."
        )
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(train_cfg["learning_rate"]),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch_for_onecycle,
        )
    else:
        patience = int(train_cfg.get("patience", 10))
        factor = float(scheduler_cfg.get("factor", 0.7))
        plateau_patience = int(scheduler_cfg.get("plateau_patience", patience // 3))
        logger.info(
            f"Using ReduceLROnPlateau scheduler (factor={factor}, patience={plateau_patience})."
        )
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=plateau_patience,
            min_lr=min_lr,
            verbose=True,
        )


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    epoch_num: int,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    batch_count = 0
    total_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}
    loss_cfg = config.get("loss", {})

    use_mixed_precision_config = config.get("performance", {}).get(
        "mixed_precision", False
    )
    use_mixed_precision = use_mixed_precision_config and scaler is not None
    if sys.platform == "win32" and use_mixed_precision_config:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
            use_mixed_precision = False
        elif not torch.cuda.is_available() and use_mixed_precision_config:
            use_mixed_precision = False

    grad_clip_val = config["training"].get("grad_clip", 0.0)
    metric_compute_interval = config["training"].get("metric_compute_interval", 50)
    log_batch_interval = config["training"].get(
        "log_batch_interval",
        max(1, len(train_loader) // 10 if len(train_loader) > 0 else 1),
    )

    metrics_computed_count = 0
    accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)

    progress_bar = tqdm(
        train_loader,
        desc=f"Train Ep {epoch_num+1}",
        leave=False,
        file=sys.stdout,
        ncols=100,
        disable=len(train_loader) == 0,
    )

    for batch_idx, (X_batch, Y_batch) in enumerate(progress_bar):
        batch_count += 1
        X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(
            device, non_blocking=True
        )

        if batch_idx % log_batch_interval == 0:
            logger.debug(
                f"Train Ep {epoch_num+1}, Batch {batch_idx}/{len(train_loader)}, X:{X_batch.shape}, Y:{Y_batch.shape}"
            )

        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            scaled_preds = model(X_batch, return_scaled_for_loss=True)
            if isinstance(scaled_preds, tuple):
                scaled_preds = scaled_preds[0]
            loss = model.compute_loss(
                y_true=Y_batch, y_pred_scaled=scaled_preds, **loss_cfg
            )
            loss = loss / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"Invalid loss {loss.item() * accumulation_steps:.4f} at batch {batch_idx}. Skipping."
            )
            continue

        if use_mixed_precision and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if use_mixed_precision and scaler:
                if grad_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps

        if (
            batch_idx % max(1, log_batch_interval // 2 if log_batch_interval > 0 else 1)
            == 0
        ):
            progress_bar.set_description(
                f"Train Ep {epoch_num+1} Loss: {loss.item() * accumulation_steps:.4f}"
            )

        if batch_idx % metric_compute_interval == 0:
            with torch.no_grad():
                metrics = calculate_metrics(Y_batch, scaled_preds[:, :, 0:1])
                for k, v in metrics.items():
                    if not np.isnan(v):
                        total_metrics[k] += v
                metrics_computed_count += 1

    progress_bar.close()

    avg_loss = total_loss / max(1, batch_count) if batch_count > 0 else float("nan")
    avg_metrics = {
        k: (
            v / max(1, metrics_computed_count)
            if metrics_computed_count > 0
            else float("nan")
        )
        for k, v in total_metrics.items()
    }
    logger.info(
        f"Ep {epoch_num+1} Train Loss:{avg_loss:.4f} MAE:{avg_metrics.get('mae',np.nan):.4f} RMSE:{avg_metrics.get('rmse',np.nan):.4f}"
    )
    return avg_loss, avg_metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    epoch_num: int,
    use_mixed_precision: bool = False,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss, batch_count = 0, 0
    total_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}
    loss_cfg = config.get("loss", {})
    metric_compute_interval = config["training"].get("metric_compute_interval_val", 20)
    log_batch_interval_val = config["training"].get(
        "log_batch_interval_val",
        max(1, len(val_loader) // 10 if len(val_loader) > 0 else 1),
    )

    metrics_batches_count = 0

    progress_bar = tqdm(
        val_loader,
        desc=f"Valid Ep {epoch_num+1}",
        leave=False,
        file=sys.stdout,
        ncols=100,
        disable=len(val_loader) == 0,
    )

    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(progress_bar):
            batch_count += 1
            X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(
                device, non_blocking=True
            )

            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                scaled_preds = model(X_batch, return_scaled_for_loss=True)
                if isinstance(scaled_preds, tuple):
                    scaled_preds = scaled_preds[0]
                loss = model.compute_loss(
                    y_true=Y_batch, y_pred_scaled=scaled_preds, **loss_cfg
                )

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid val loss {loss.item()} B{batch_idx}. Skip.")
                continue

            total_loss += loss.item()

            if (
                batch_idx
                % max(
                    1, log_batch_interval_val // 2 if log_batch_interval_val > 0 else 1
                )
                == 0
            ):
                progress_bar.set_description(
                    f"Valid Ep {epoch_num+1} Loss: {loss.item():.4f}"
                )

            if batch_idx % metric_compute_interval == 0:
                metrics = calculate_metrics(Y_batch, scaled_preds[:, :, 0:1])
                for k, v in metrics.items():
                    if not np.isnan(v):
                        total_metrics[k] += v
                metrics_batches_count += 1
    progress_bar.close()

    avg_loss = total_loss / max(1, batch_count) if batch_count > 0 else float("nan")
    avg_metrics = {
        k: (
            v / max(1, metrics_batches_count)
            if metrics_batches_count > 0
            else float("nan")
        )
        for k, v in total_metrics.items()
    }
    logger.info(
        f"Ep {epoch_num+1} Valid Loss:{avg_loss:.4f} MAE:{avg_metrics.get('mae',np.nan):.4f} RMSE:{avg_metrics.get('rmse',np.nan):.4f}"
    )
    return avg_loss, avg_metrics


def create_optimized_dataloaders(train_dataset, valid_dataset, config, use_cuda):
    train_cfg = config["training"]
    batch_sz = int(train_cfg.get("batch_size", 128))

    num_w = int(train_cfg.get("num_workers", 0))
    # Ensure pin_memory config is read from training section, default to False
    pin_mem_cfg = train_cfg.get("pin_memory", False)

    actual_pin_memory = pin_mem_cfg and use_cuda

    mp_context = None
    if num_w > 0 and sys.platform == "win32":
        try:
            current_context_name = mp.get_start_method(allow_none=True)
            # Only try to get 'spawn' context if it's not already 'spawn'
            # The main script `train_improved_model.py` should have already called set_start_method('spawn', force=True)
            if current_context_name == "spawn":
                mp_context = mp.get_context("spawn")
                logger.info(
                    "Successfully got 'spawn' multiprocessing context for DataLoader on Windows."
                )
            elif (
                current_context_name is None
            ):  # Not set yet by main script, this is fallback (less ideal)
                logger.warning(
                    "Multiprocessing start method not set by main script. Attempting to set to 'spawn' for DataLoader."
                )
                mp.set_start_method("spawn", force=True)
                mp_context = mp.get_context("spawn")
                logger.info(
                    "Set and got 'spawn' multiprocessing context for DataLoader on Windows."
                )
            else:  # Set to something else, which is unexpected if main script worked
                logger.warning(
                    f"Multiprocessing start method is '{current_context_name}'. DataLoader will use this context if compatible, or PyTorch default. 'spawn' is recommended for Windows."
                )
                # We might not want to override if it was explicitly set to something else by user for a reason.
                # However, for DataLoader on Windows, 'spawn' is usually required.
                # mp_context = mp.get_context('spawn') # Cautious about overriding an explicit different setting.

        except RuntimeError as e:
            logger.warning(
                f"Could not get 'spawn' context ('{e}'), DataLoader will use default. Ensure 'torch.multiprocessing.set_start_method(\"spawn\", force=True)' is in 'if __name__ == \"__main__\":' block of main script."
            )

    logger.info(
        f"DataLoader: batch_size={batch_sz}, num_workers={num_w}, pin_memory_config={pin_mem_cfg}, actual_pin_memory_used={actual_pin_memory}"
    )

    if sys.platform == "win32":
        if num_w > 0:
            logger.info(
                f"Using num_workers={num_w} on Windows. Main script MUST be guarded by 'if __name__ == \"__main__\":'."
            )
        if num_w == 0 and actual_pin_memory:  # If pin_memory true and workers 0
            logger.warning(
                "Using pin_memory=True with num_workers=0 on Windows. This is often not optimal."
            )
    elif num_w == 0 and actual_pin_memory:  # Non-windows, pin_memory true and workers 0
        logger.info(
            f"num_workers=0 but pin_memory=True. Consider increasing num_workers."
        )

    dataloader_common_args = {"num_workers": num_w, "pin_memory": actual_pin_memory}
    if mp_context:
        dataloader_common_args["multiprocessing_context"] = mp_context

    if num_w > 0:
        persistent_workers_cfg = train_cfg.get(
            "persistent_workers", False if sys.platform == "win32" else True
        )
        dataloader_common_args["persistent_workers"] = persistent_workers_cfg
        logger.info(f"DataLoader: persistent_workers={persistent_workers_cfg}")

        data_cfg = config.get("data", {})
        dataloader_common_args["prefetch_factor"] = data_cfg.get(
            "prefetch_factor", 2 if num_w > 0 else None
        )  # prefetch_factor only if num_w > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_sz,
        shuffle=True,
        drop_last=True,
        **dataloader_common_args,
    )

    # Handle case where valid_dataset might be None (e.g., if data_fraction resulted in empty validation set)
    valid_loader = None
    if valid_dataset and len(valid_dataset) > 0:
        val_dataloader_args = dataloader_common_args.copy()
        val_dataloader_args["drop_last"] = False
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_sz, shuffle=False, **val_dataloader_args
        )
    else:
        logger.info(
            "Validation dataset is empty or None, validation loader will not be created."
        )

    return train_loader, valid_loader


def optimize_model_for_windows(model, device, config, current_logger: logging.Logger):
    perf_cfg = config.get("performance", {})
    model = model.to(device)
    current_logger.info(f"Model moved to {device}.")

    if sys.platform == "win32" and perf_cfg.get("compile_model", False):
        current_logger.warning(
            "Model compilation (torch.compile) is configured but will be SKIPPED on Windows for stability."
        )
    elif perf_cfg.get("compile_model", False) and hasattr(torch, "compile"):
        compile_mode = perf_cfg.get("compile_mode", "reduce-overhead")
        try:
            current_logger.info(
                f"Attempting model compilation with torch.compile(mode='{compile_mode}')..."
            )
            model = torch.compile(model, mode=compile_mode)
            current_logger.info("Model compiled successfully.")
        except Exception as e:
            current_logger.warning(f"Compilation failed: {e}. Using uncompiled model.")
    else:
        current_logger.info(
            "Model compilation not configured or torch.compile not available."
        )
    return model


def train_model(
    config_path: str, model_type_override: str = None, data_fraction: float = 1.0
):
    global logger

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    writer = None

    try:
        config = load_config(config_path)
        output_dir_str = config["data"]["output_dir"]
        output_dir = (
            Path(output_dir_str)
            if Path(output_dir_str).is_absolute()
            else get_project_root() / output_dir_str
        )
        logger = setup_logging(
            output_dir=str(output_dir), log_file_name="training_run.log"
        )

        if torch.cuda.is_available():
            logger.info("Optimizing CUDA settings.")
            torch.cuda.empty_cache()
            if sys.platform == "win32":
                logger.info(
                    "Using conservative CUDA settings for Windows: cudnn.benchmark=False, cudnn.deterministic=True."
                )
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            else:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
        else:
            logger.info("CUDA not available, using CPU.")

        logger.info(
            f"Starting PM2.5 model training: {config_path}, Output: {output_dir}"
        )
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(
            f"Device: {device}{' (' + torch.cuda.get_device_name(0) + ')' if use_cuda else ''}"
        )

        tensorboard_log_epoch_interval = config["training"].get(
            "tensorboard_log_epoch_interval", 5
        )

        tb_log_dir = (
            output_dir / "tensorboard_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard: {tb_log_dir}")

        logger.info("Step 1: Loading processed data...")
        (
            (X_tr_raw_full, Y_tr_raw_full),
            (X_v_raw_full, Y_v_raw_full),
            _,
            adj_mx_np,
            scalers,
            f_cols,
        ) = load_processed_data(config["data"]["data_dir"])

        if not (0.0 < data_fraction <= 1.0):  # Validate data_fraction here
            logger.warning(
                f"Invalid data_fraction: {data_fraction}. Using full dataset (1.0)."
            )
            data_fraction = 1.0

        if data_fraction < 1.0:
            logger.info(
                f"Using {data_fraction*100:.0f}% of the training and validation data."
            )

            train_len_full = len(X_tr_raw_full)
            valid_len_full = len(X_v_raw_full)

            train_subset_len = int(train_len_full * data_fraction)
            valid_subset_len = int(valid_len_full * data_fraction)

            # Ensure at least 1 sample if original data exists and fraction is very small
            if train_subset_len == 0 and train_len_full > 0:
                train_subset_len = 1
                logger.warning(
                    f"data_fraction resulted in 0 train samples, using 1 sample instead."
                )
            if valid_subset_len == 0 and valid_len_full > 0:
                valid_subset_len = 1
                logger.warning(
                    f"data_fraction resulted in 0 validation samples, using 1 sample instead."
                )

            X_tr_raw, Y_tr_raw = (
                X_tr_raw_full[:train_subset_len],
                Y_tr_raw_full[:train_subset_len],
            )
            X_v_raw, Y_v_raw = (
                X_v_raw_full[:valid_subset_len],
                Y_v_raw_full[:valid_subset_len],
            )

            logger.info(f"Reduced train data to X:{X_tr_raw.shape}, Y:{Y_tr_raw.shape}")
            logger.info(f"Reduced valid data to X:{X_v_raw.shape}, Y:{Y_v_raw.shape}")
        else:
            X_tr_raw, Y_tr_raw = X_tr_raw_full, Y_tr_raw_full
            X_v_raw, Y_v_raw = X_v_raw_full, Y_v_raw_full
            logger.info("Using full dataset.")

        seq_len = config["model"]["time_steps"]
        station_names_cfg_list = config.get("data", {}).get("station_names")
        if station_names_cfg_list and isinstance(station_names_cfg_list, list):
            num_stations_for_y_prep = len(station_names_cfg_list)
        else:
            num_stations_for_y_prep = config["model"]["num_stations"]
        logger.info(f"Using num_stations = {num_stations_for_y_prep} (for Y prep)")

        logger.info(f"Step 1.5: Preparing time series sequences (len {seq_len})...")
        X_train, Y_train = prepare_time_series_data(
            X_tr_raw, Y_tr_raw, seq_len, num_stations_for_y_prep
        )
        X_valid, Y_valid = prepare_time_series_data(
            X_v_raw, Y_v_raw, seq_len, num_stations_for_y_prep
        )

        if X_train.size == 0:  # Check after sequence preparation
            logger.error(
                "Training data is empty after sequence preparation (e.g., data_fraction too small or seq_len too large for subset). Aborting training."
            )
            if writer:
                writer.close()
            logging.shutdown()
            return

        logger.info(
            f"Data shapes after seq prep for Dataset - Train X:{X_train.shape} Y:{Y_train.shape}; Valid X:{X_valid.shape} Y:{Y_valid.shape}"
        )

        adj_mx_tensor = torch.FloatTensor(adj_mx_np)

        logger.info("Step 3: Creating data loaders...")
        model_expects_num_stations = config["model"]["num_stations"]
        if num_stations_for_y_prep != model_expects_num_stations:
            logger.warning(
                f"num_stations used for Y preparation ({num_stations_for_y_prep}) "
                f"differs from model's expected num_stations ({model_expects_num_stations}). "
                f"Dataset will tile X to {model_expects_num_stations}. Ensure Y target matches model output structure."
            )

        train_dataset = WindowsCompatibleMultiStationDataset(
            X_train,
            Y_train,
            num_stations_to_tile_to=model_expects_num_stations,
            time_steps=seq_len,
        )

        valid_dataset = None  # Initialize to None
        if (
            X_valid.size > 0 and Y_valid.size > 0
        ):  # Check if validation data is actually present
            valid_dataset = WindowsCompatibleMultiStationDataset(
                X_valid,
                Y_valid,
                num_stations_to_tile_to=model_expects_num_stations,
                time_steps=seq_len,
            )
        else:
            logger.info(
                "Validation data is empty after sequence preparation. No validation loader will be created."
            )

        train_loader, valid_loader = create_optimized_dataloaders(
            train_dataset, valid_dataset, config, use_cuda
        )
        logger.info(
            f"DataLoaders created successfully - Train batches: {len(train_loader)}, Valid batches: {len(valid_loader) if valid_loader else 0}"
        )

        logger.info("Step 4: Initializing model...")
        model_params = config["model"].copy()
        if model_type_override:
            model_params["model_type"] = model_type_override

        model_params["num_stations"] = model_expects_num_stations

        perf_cfg_model_init = config.get("performance", {})
        model_params["use_mixed_precision"] = perf_cfg_model_init.get(
            "mixed_precision", False
        )

        model = PM25Model(
            adj_matrix=adj_mx_tensor,
            scalers=scalers,
            station_names=config.get("data", {}).get("station_names"),
            **model_params,
        )

        model = optimize_model_for_windows(model, device, config, logger)

        logger.info(
            f"Params: Total {sum(p.numel() for p in model.parameters()):,}, Trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        train_cfg = config["training"]
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
        )

        # Ensure len(train_loader) > 0 before division
        steps_per_epoch_for_scheduler = 0
        if len(train_loader) > 0:
            steps_per_epoch_for_scheduler = len(train_loader) // train_cfg.get(
                "gradient_accumulation_steps", 1
            )

        scheduler = None  # Initialize scheduler to None
        if steps_per_epoch_for_scheduler > 0:
            scheduler = create_improved_scheduler(
                optimizer,
                config,
                steps_per_epoch_for_onecycle=steps_per_epoch_for_scheduler,
            )
        else:
            logger.warning(
                "No batches in train_loader (steps_per_epoch_for_scheduler is 0), scheduler will not be effectively used or might be set to None. Learning rate will remain constant unless ReduceLROnPlateau is used with validation."
            )
            # For ReduceLROnPlateau, it can still work if validation happens
            if (
                config.get("training", {})
                .get("scheduler", {})
                .get("type", "cosine")
                .lower()
                == "reduce_on_plateau"
            ):
                scheduler = create_improved_scheduler(
                    optimizer, config
                )  # Create it anyway for ReduceLROnPlateau
            else:
                scheduler = None

        early_stopping = EarlyStopping(
            patience=int(train_cfg["patience"]), min_delta=float(train_cfg["min_delta"])
        )
        logger.info("Optimizer, Scheduler, EarlyStopping initialized.")

        use_mixed_precision_runtime = (
            perf_cfg_model_init.get("mixed_precision", False) and use_cuda
        )
        if sys.platform == "win32" and use_mixed_precision_runtime:
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
                logger.warning(
                    "Mixed precision for runtime disabled on Windows for this GPU (capability < 7.0)."
                )
                use_mixed_precision_runtime = False
            elif not torch.cuda.is_available():
                use_mixed_precision_runtime = False

        grad_scaler = GradScaler() if use_mixed_precision_runtime else None

        if grad_scaler:
            logger.info("Mixed precision training enabled with GradScaler for runtime.")
        else:
            logger.info(
                f"Mixed precision for runtime disabled (use_mixed_precision_runtime: {use_mixed_precision_runtime})."
            )

        logger.info("Step 7: Starting training loop...")
        best_val_loss = float("inf")
        epochs_to_run = int(train_cfg["epochs"])

        if len(train_loader) == 0:  # Final check
            logger.error("Train loader is empty. Cannot start training loop. Aborting.")
            if writer:
                writer.close()
            logging.shutdown()
            return

        for epoch_idx in range(epochs_to_run):
            logger.info(
                f"====== Epoch {epoch_idx+1}/{epochs_to_run} ===== LR: {optimizer.param_groups[0]['lr']:.1e} ====="
            )
            try:
                train_loss, train_metrics = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    config,
                    epoch_idx,
                    grad_scaler,
                )

                val_loss, val_metrics = float("nan"), {
                    k: float("nan") for k in train_metrics.keys()
                }
                if (
                    valid_loader and len(valid_loader) > 0
                ):  # Check if valid_loader exists and is not empty
                    val_loss, val_metrics = validate(
                        model,
                        valid_loader,
                        device,
                        config,
                        epoch_idx,
                        use_mixed_precision=use_mixed_precision_runtime,
                    )
                else:
                    logger.info(
                        f"Epoch {epoch_idx+1}: No validation data/loader, skipping validation step."
                    )

                if scheduler:  # Check if scheduler exists
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if not np.isnan(val_loss):
                            scheduler.step(val_loss)
                        else:
                            logger.warning(
                                f"Epoch {epoch_idx+1}: val_loss is NaN, not stepping ReduceLROnPlateau scheduler."
                            )
                    elif not isinstance(
                        scheduler, optim.lr_scheduler.OneCycleLR
                    ):  # OneCycleLR is stepped per batch
                        scheduler.step()

                if (epoch_idx + 1) % tensorboard_log_epoch_interval == 0:
                    writer.add_scalar("Loss/train", train_loss, epoch_idx)
                    if valid_loader and not np.isnan(val_loss):
                        writer.add_scalar("Loss/val", val_loss, epoch_idx)
                    writer.add_scalar(
                        "LearningRate", optimizer.param_groups[0]["lr"], epoch_idx
                    )
                    for m, v_train in train_metrics.items():
                        if not np.isnan(v_train):
                            writer.add_scalar(f"Train/{m}", v_train, epoch_idx)
                    if (
                        valid_loader
                    ):  # Check if valid_loader exists before iterating val_metrics
                        for m, v_val in val_metrics.items():
                            if not np.isnan(v_val):
                                writer.add_scalar(f"Val/{m}", v_val, epoch_idx)

                logger.info(
                    f"Epoch {epoch_idx+1} Summary - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Use train_loss for early stopping if no validation is performed
                current_loss_for_es = (
                    val_loss if valid_loader and not np.isnan(val_loss) else train_loss
                )

                best_mdl_path = output_dir / "model_best.pt"
                # Update best model based on validation loss if available, else on training loss (less ideal but provides a checkpoint)
                current_best_score_ref = (
                    best_val_loss if valid_loader else (-train_loss)
                )  # Invert train_loss for early stopping's 'min' mode assumption for loss

                if (
                    valid_loader and not np.isnan(val_loss) and val_loss < best_val_loss
                ) or (
                    not valid_loader
                    and not np.isnan(train_loss)
                    and train_loss
                    < (-current_best_score_ref if not valid_loader else float("inf"))
                ):

                    if valid_loader:
                        best_val_loss = val_loss
                    # if no validation, best_val_loss doesn't truly represent validation, but we save the model

                    logger.info(
                        f"New best score/loss. Saving model to {best_mdl_path} (Val Loss: {val_loss if valid_loader else 'N/A'}, Train Loss: {train_loss:.4f})"
                    )
                    torch.save(model.state_dict(), str(best_mdl_path))

                if valid_loader and early_stopping(
                    current_loss_for_es, model, str(output_dir / "model_checkpoint.pt")
                ):  # Only use ES if validation is happening
                    logger.info("Early stopping triggered based on validation loss.")
                    if best_mdl_path.exists():
                        try:
                            model.load_state_dict(
                                torch.load(str(best_mdl_path), map_location=device)
                            )
                            logger.info(
                                f"Successfully loaded best model from {best_mdl_path}"
                            )
                        except Exception as e_load:
                            logger.error(
                                f"Error loading best model state: {e_load}",
                                exc_info=True,
                            )
                    else:
                        logger.warning(
                            f"Best model path {best_mdl_path} not found for early stopping reload."
                        )
                    break
            except Exception as epoch_error:
                logger.error(
                    f"Error in epoch {epoch_idx+1}: {epoch_error}", exc_info=True
                )
                logger.info(
                    f"Attempting to continue to the next epoch after error in epoch {epoch_idx+1}."
                )
                if torch.cuda.is_available():
                    clear_gpu_cache(logger)
                continue

        torch.save(model.state_dict(), str(output_dir / "model_final.pt"))
        logger.info(
            f"Final model saved. Best Val Loss: {best_val_loss if valid_loader and best_val_loss != float('inf') else 'N/A (no validation or no improvement)'}"
        )

        if f_cols:
            with open(output_dir / "feature_cols.pkl", "wb") as f:
                pickle.dump(f_cols, f)
        if scalers:
            with open(output_dir / "scalers_final.pkl", "wb") as f:
                pickle.dump(scalers, f)

        logger.info("Training complete.")

    except Exception as e:
        logger.error(f"Fatal error in training process: {e}", exc_info=True)
        # No raise here if finally block needs to run for cleanup
    finally:
        if torch.cuda.is_available():
            if (
                "logger" in locals() and logger is not None
            ):  # Ensure logger is available
                clear_gpu_cache(logger)
            else:  # Fallback print if logger not available
                print("Clearing GPU cache (logger not available).")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if writer is not None:
            try:
                writer.close()
                if "logger" in locals() and logger is not None:
                    logger.info("TensorBoard writer closed.")
                else:
                    print("TensorBoard writer closed (logger not available).")
            except Exception as e_writer:
                if "logger" in locals() and logger is not None:
                    logger.error(f"Error closing TensorBoard writer: {e_writer}")
                else:
                    print(
                        f"Error closing TensorBoard writer (logger not available): {e_writer}"
                    )

        # Ensure logging is shutdown cleanly.
        # This is important as logging.shutdown() itself can sometimes access attributes
        # that might be problematic if the logger object itself is in a weird state.
        try:
            if (
                "logger" in locals() and logger is not None
            ):  # Check if logger was ever assigned.
                logging.shutdown()
        except Exception as e_log_shutdown:
            print(f"Exception during logging.shutdown(): {e_log_shutdown}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PM2.5 model.")
    default_cfg_path = "configs/improved_model_config.yaml"
    parser.add_argument(
        "--config", type=str, default=default_cfg_path, help="Path to YAML config file"
    )
    parser.add_argument(
        "--model_type", type=str, default=None, help="Override model_type from config"
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use (0.0 to 1.0). Default is 1.0 (all data).",
    )
    args = parser.parse_args()

    # Basic print for early errors if logger isn't setup by train_model yet.
    if not (0.0 < args.data_fraction <= 1.0):
        print(
            f"Error: data_fraction must be between 0.0 (exclusive) and 1.0 (inclusive). Got {args.data_fraction}"
        )
        sys.exit(1)

    cfg_file_path = Path(args.config)
    if not cfg_file_path.is_absolute():
        # Construct path relative to the project root (parent of 'scripts' directory)
        # This assumes the script is run from 'scripts' or project root.
        # More robustly, get_project_root() could be called here if defined globally
        # or if train_improved_model.py also defines/imports it.
        # For now, relying on the structure from the previous train_improved_model.py.
        _project_root_for_config = (
            Path(__file__).resolve().parents[1]
        )  # Assuming this script is in 'scripts'
        cfg_file_path = _project_root_for_config / cfg_file_path

    if not cfg_file_path.exists():
        print(f"Config file not found: {cfg_file_path}")
        sys.exit(1)

    train_model(str(cfg_file_path), args.model_type, args.data_fraction)
