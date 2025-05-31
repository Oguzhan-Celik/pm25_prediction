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
import argparse  # Added for __main__

project_root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root_path))

from src.models.model import PM25Model, create_enhanced_model_configs
from src.models.loss_functions import calculate_metrics


class MultiStationDataset(Dataset):
    def __init__(
        self, X_data: np.ndarray, Y_data: np.ndarray, num_stations: int, time_steps: int
    ):
        self.X_data = X_data
        self.Y_data = Y_data
        self.num_stations_model_expects = num_stations
        self.time_steps = time_steps
        self.logger = logging.getLogger(f"{__name__}.MultiStationDataset")
        self.logger.debug(
            f"Init MultiStationDataset: X:{self.X_data.shape}, Y:{self.Y_data.shape}, num_stations_model_expects:{self.num_stations_model_expects}"
        )
        if self.X_data.shape[1] != self.time_steps:
            raise ValueError(f"X_data time_steps error")

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sample_seq = self.X_data[idx]
        y_sample_stations = self.Y_data[idx]
        x_tiled_for_stations = x_sample_seq  # Default if already correct shape
        if x_sample_seq.ndim == 2:
            x_expanded = np.expand_dims(x_sample_seq, axis=1)
            x_tiled_for_stations = np.tile(
                x_expanded, (1, self.num_stations_model_expects, 1)
            )
        elif x_sample_seq.ndim == 3 and x_sample_seq.shape[1] == 1:
            x_tiled_for_stations = np.tile(
                x_sample_seq, (1, self.num_stations_model_expects, 1)
            )
        elif (
            x_sample_seq.ndim == 3
            and x_sample_seq.shape[1] != self.num_stations_model_expects
        ):
            self.logger.warning(
                f"Unexpected x_sample_seq shape {x_sample_seq.shape} for tiling at index {idx}. Using slice/tile."
            )
            x_tiled_for_stations = np.tile(
                x_sample_seq[:, :1, :], (1, self.num_stations_model_expects, 1)
            )

        return torch.FloatTensor(x_tiled_for_stations), torch.FloatTensor(
            y_sample_stations
        )


def setup_logging(output_dir=None, log_file_name="training_debug.log"):
    log_level = logging.DEBUG
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, log_file_name), mode="a"
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    logging.basicConfig(
        level=log_level, format=log_format, handlers=handlers, force=True
    )
    logger = logging.getLogger(__name__)  # Get logger for current module
    logger.info(
        f"Logging setup. Level: {log_level}. Output: {output_dir}/{log_file_name if output_dir else 'Console'}"
    )
    return logger


logger = logging.getLogger(__name__)  # Default logger


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = "min"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter, self.best_score, self.early_stop = 0, None, False
        self.logger = logging.getLogger(f"{__name__}.EarlyStopping")
        self.logger.debug(
            f"EarlyStopping init: patience={patience}, min_delta={min_delta}, mode={mode}"
        )

    def __call__(self, val_loss: float, model: nn.Module, path: str) -> bool:
        self.logger.debug(
            f"ES check: val_loss={val_loss:.4f}, best_score={self.best_score}"
        )
        score = -val_loss if self.mode == "min" else val_loss
        if (
            self.best_score is None or score > self.best_score + self.min_delta
        ):  # Condition for improvement
            self.best_score = score
            self.logger.info(
                f"ES: Model improved. New best val_loss={val_loss:.4f}. Saving to {path}"
            )
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.debug(
                f"ES: No improvement. Counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info("ES: Triggered.")
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
            self.logger.debug(f"Checkpoint saved: {path}")
        except Exception as e:
            self.logger.error(
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
        n: data_dir / f"{n}_enhanced.{'npy' if n=='adj_matrix' else 'pkl'}"
        for n in ["train", "valid", "test", "scalers", "adj_matrix", "feature_cols"]
    }
    if not paths["feature_cols"].exists():
        paths["feature_cols"] = data_dir / "feature_cols.pkl"  # common name
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required file '{name}' not found: {path}")

    data = {
        name: pickle.load(open(path, "rb")) if path.suffix == ".pkl" else np.load(path)
        for name, path in paths.items()
    }

    # feature_cols might be nested if saved differently, ensure it's a list
    f_cols = data["feature_cols"]
    if isinstance(f_cols, dict) and "feature_cols" in f_cols:
        f_cols = f_cols["feature_cols"]  # Example handling
    if not isinstance(f_cols, list):
        raise ValueError(f"feature_cols is not a list: {type(f_cols)}")

    def extract_xy(d, fc, tc="PM2.5"):
        if isinstance(d, dict):
            return np.array(d["X"]), np.array(d["Y"])
        if isinstance(d, pd.DataFrame):
            if not all(c in d.columns for c in fc):
                raise ValueError("Missing feature columns in DataFrame.")
            return d[fc].values, d[[tc]].values
        if isinstance(d, tuple) and len(d) == 2:
            return np.array(d[0]), np.array(d[1])
        raise ValueError(f"Unknown data format for X,Y: {type(d)}")

    train_X, train_Y = extract_xy(data["train"], f_cols)
    valid_X, valid_Y = extract_xy(data["valid"], f_cols)
    test_X, test_Y = extract_xy(data["test"], f_cols)

    logger.info(f"Loaded train - X:{train_X.shape}, Y:{train_Y.shape}")
    # ... other logs ...
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
        f"Preparing time series: seq_len={sequence_length}, num_stations={num_stations}"
    )
    logger.debug(f"Input X:{X.shape}, Y:{Y.shape}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(X) <= sequence_length:
        raise ValueError(
            f"Not enough samples ({len(X)}) for seq_len ({sequence_length})"
        )

    X_seq, Y_seq_list = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i : i + sequence_length])
        Y_seq_list.append(Y[i + sequence_length - 1])

    X_seq_np, Y_seq_np = np.array(X_seq), np.array(Y_seq_list)

    if Y_seq_np.ndim == 2 and Y_seq_np.shape[1] == 1:
        Y_seq_np = np.repeat(Y_seq_np[:, np.newaxis, :], num_stations, axis=1)
    elif Y_seq_np.ndim == 2 and Y_seq_np.shape[1] == num_stations:
        Y_seq_np = Y_seq_np[:, :, np.newaxis]
    elif Y_seq_np.ndim != 3 or Y_seq_np.shape[1] != num_stations:
        logger.warning(
            f"Y_seq_np shape {Y_seq_np.shape} not ideal. Forcing to (N, S, 1)"
        )
        if Y_seq_np.ndim == 2 and Y_seq_np.shape[1] != 1:
            Y_seq_np = Y_seq_np[:, :1]
        if Y_seq_np.ndim == 1:
            Y_seq_np = Y_seq_np.reshape(-1, 1)
        if Y_seq_np.ndim == 2 and Y_seq_np.shape[1] == 1:
            Y_seq_np = np.repeat(Y_seq_np[:, np.newaxis, :], num_stations, axis=1)

    logger.info(f"Created sequences - X: {X_seq_np.shape}, Y: {Y_seq_np.shape}")
    if np.isnan(X_seq_np).any() or np.isinf(X_seq_np).any():
        logger.warning("NaN/Inf in X_seq!")
    if np.isnan(Y_seq_np).any() or np.isinf(Y_seq_np).any():
        logger.warning("NaN/Inf in Y_seq!")
    return X_seq_np, Y_seq_np


def clear_gpu_cache(): ...
def log_memory_stats(logger_instance): ...


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    epoch: int,
    current_logger: logging.Logger,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss, batch_count = 0, 0
    total_metrics = {"mae": 0, "rmse": 0, "mape": 0, "r2": 0}
    loss_cfg = config.get("loss", {})

    for batch_idx, (X_batch, Y_batch) in enumerate(
        tqdm(train_loader, desc=f"Train Ep {epoch+1}", leave=False)
    ):
        batch_count += 1
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        if batch_idx % 200 == 0:  # Log less
            current_logger.debug(
                f"B {batch_idx} X:{X_batch.shape} Y:{Y_batch.shape} X_minmax:{X_batch.min():.1f}/{X_batch.max():.1f}"
            )
            # log_memory_stats(current_logger) # Can be verbose

        optimizer.zero_grad()

        # Get SCALED predictions from model for loss calculation
        # Pass return_scaled_for_loss=True to model's forward method
        scaled_preds = model(X_batch, return_scaled_for_loss=True)

        if isinstance(scaled_preds, tuple):
            scaled_preds = scaled_preds[0]

        # Loss is computed using y_true (scaled) and scaled_preds (first step)
        loss = model.compute_loss(
            y_true=Y_batch, y_pred_scaled=scaled_preds, **loss_cfg
        )

        if torch.isnan(loss) or torch.isinf(loss):
            current_logger.error(
                f"Invalid loss {loss.item()} at batch {batch_idx}. Skipping."
            )
            continue

        loss.backward()
        grad_clip = config["training"].get("grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        # Metrics on SCALED data: Y_batch vs first step of scaled_preds
        metrics = calculate_metrics(Y_batch, scaled_preds[:, :, 0:1])
        for k, v in metrics.items():
            total_metrics[k] += v
        if hasattr(model, "reset_attention_weights"):
            model.reset_attention_weights()

    if batch_count == 0:
        current_logger.warning("Train loader empty!")
        return 0.0, {k: 0.0 for k in total_metrics}
    avg_loss = total_loss / batch_count
    avg_metrics = {k: v / batch_count for k, v in total_metrics.items()}
    current_logger.info(
        f"Ep {epoch+1} Train Loss:{avg_loss:.4f} MAE:{avg_metrics['mae']:.4f} RMSE:{avg_metrics['rmse']:.4f}"
    )
    return avg_loss, avg_metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    epoch: int,
    current_logger: logging.Logger,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss, batch_count = 0, 0
    total_metrics = {"mae": 0, "rmse": 0, "mape": 0, "r2": 0}
    loss_cfg = config.get("loss", {})

    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(
            tqdm(val_loader, desc=f"Valid Ep {epoch+1}", leave=False)
        ):
            batch_count += 1
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            scaled_preds = model(X_batch, return_scaled_for_loss=True)
            if isinstance(scaled_preds, tuple):
                scaled_preds = scaled_preds[0]

            loss = model.compute_loss(
                y_true=Y_batch, y_pred_scaled=scaled_preds, **loss_cfg
            )
            if torch.isnan(loss) or torch.isinf(loss):
                current_logger.warning(
                    f"Invalid val loss {loss.item()} B{batch_idx}. Skip."
                )
                continue

            total_loss += loss.item()
            metrics = calculate_metrics(
                Y_batch, scaled_preds[:, :, 0:1]
            )  # Compare with 1st step
            for k, v in metrics.items():
                total_metrics[k] += v
            if hasattr(model, "reset_attention_weights"):
                model.reset_attention_weights()

    if batch_count == 0:
        current_logger.warning("Valid loader empty!")
        return 0.0, {k: 0.0 for k in total_metrics}
    avg_loss = total_loss / batch_count
    avg_metrics = {k: v / batch_count for k, v in total_metrics.items()}
    current_logger.info(
        f"Ep {epoch+1} Valid Loss:{avg_loss:.4f} MAE:{avg_metrics['mae']:.4f} RMSE:{avg_metrics['rmse']:.4f}"
    )
    return avg_loss, avg_metrics


def train_model(config_path: str, model_type_override: str = None):
    global logger
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

        logger.info(
            f"Starting PM2.5 model training: {config_path}, Output: {output_dir}"
        )
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(
            f"Device: {device}{' (' + torch.cuda.get_device_name(0) + ')' if use_cuda else ''}"
        )

        tb_log_dir = (
            output_dir / "tensorboard_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard: {tb_log_dir}")

        logger.info("Step 1: Loading processed data...")
        ((X_tr_raw, Y_tr_raw), (X_v_raw, Y_v_raw), _, adj_mx_np, scalers, f_cols) = (
            load_processed_data(config["data"]["data_dir"])
        )

        seq_len, num_stations = (
            config["model"]["time_steps"],
            config["model"]["num_stations"],
        )

        logger.info(f"Step 1.5: Preparing time series sequences (len {seq_len})...")
        if X_tr_raw.ndim == 2:
            X_train, Y_train = prepare_time_series_data(
                X_tr_raw, Y_tr_raw, seq_len, num_stations
            )
            X_valid, Y_valid = prepare_time_series_data(
                X_v_raw, Y_v_raw, seq_len, num_stations
            )
        elif X_tr_raw.ndim == 3 and X_tr_raw.shape[1] == seq_len:
            X_train, Y_train, X_valid, Y_valid = X_tr_raw, Y_tr_raw, X_v_raw, Y_v_raw
            if Y_train.ndim == 2:
                Y_train = np.repeat(Y_train[:, np.newaxis, :], num_stations, axis=1)
            if Y_valid.ndim == 2:
                Y_valid = np.repeat(Y_valid[:, np.newaxis, :], num_stations, axis=1)
        else:
            raise ValueError(f"Unexpected X_train_raw dims: {X_tr_raw.ndim}D")
        logger.info(
            f"Data shapes after seq prep - Train X:{X_train.shape} Y:{Y_train.shape}; Valid X:{X_valid.shape} Y:{Y_valid.shape}"
        )

        adj_mx_tensor = torch.FloatTensor(adj_mx_np)

        logger.info("Step 3: Creating data loaders...")
        batch_sz = int(config["training"]["batch_size"])
        num_w = int(config["training"].get("num_workers", 0))
        if sys.platform == "win32" and num_w > 0:
            logger.warning("num_workers>0 on Win. Setting to 0.")
            num_w = 0
        logger.debug(f"Batch size: {batch_sz}, Num workers: {num_w}")

        pin_mem = use_cuda and (num_w > 0)  # Pin memory only if using CUDA and workers
        train_dataset = MultiStationDataset(
            X_train, Y_train, num_stations=num_stations, time_steps=seq_len
        )
        valid_dataset = MultiStationDataset(
            X_valid, Y_valid, num_stations=num_stations, time_steps=seq_len
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_sz,
            shuffle=True,
            num_workers=num_w,
            pin_memory=pin_mem,
            persistent_workers=(num_w > 0),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_sz,
            shuffle=False,
            num_workers=num_w,
            pin_memory=pin_mem,
            persistent_workers=(num_w > 0),
        )
        logger.debug(
            f"Loaders created: Train batches={len(train_loader)}, Valid batches={len(valid_loader)}"
        )

        logger.info("Step 4: Initializing model...")
        model_params = config["model"].copy()
        if model_type_override:
            model_params["model_type"] = model_type_override
        model_params["use_stca_features"] = config["model"].get(
            "use_stca_features", True
        )
        model_params["use_wavelet_denoising"] = config["model"].get(
            "use_wavelet_denoising", True
        )

        model = PM25Model(
            adj_matrix=adj_mx_tensor,
            scalers=scalers,
            station_names=config.get("data", {}).get("station_names"),
            **model_params,
        ).to(device)
        logger.info(f"Model architecture:\n{model}")
        logger.info(
            f"Params: Total {sum(p.numel() for p in model.parameters()):,}, Trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        opt_cfg = config["training"]
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(opt_cfg["learning_rate"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=int(opt_cfg["patience"]) // 2,
            min_lr=float(opt_cfg["min_lr"]),
            verbose=True,
        )
        early_stopping = EarlyStopping(
            patience=int(opt_cfg["patience"]), min_delta=float(opt_cfg["min_delta"])
        )
        logger.info("Optimizer, Scheduler, EarlyStopping initialized.")

        logger.info("Step 7: Starting training loop...")
        best_val_loss = float("inf")
        for epoch in range(int(opt_cfg["epochs"])):
            logger.info(
                f"====== Epoch {epoch+1}/{opt_cfg['epochs']} ===== LR: {optimizer.param_groups[0]['lr']:.1e} ====="
            )
            # log_memory_stats(logger) # Redundant if train_epoch logs it

            train_loss, train_metrics = train_epoch(
                model, train_loader, optimizer, device, config, epoch, logger
            )
            val_loss, val_metrics = validate(
                model, valid_loader, device, config, epoch, logger
            )
            scheduler.step(val_loss)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
            for m, v in train_metrics.items():
                writer.add_scalar(f"Train/{m}", v, epoch)
            for m, v in val_metrics.items():
                writer.add_scalar(f"Val/{m}", v, epoch)
            logger.info(
                f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            best_mdl_path = output_dir / "model_best.pt"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(
                    f"New best val_loss: {best_val_loss:.4f}. Saving to {best_mdl_path}"
                )
                torch.save(model.state_dict(), str(best_mdl_path))

            if early_stopping(
                val_loss, model, str(output_dir / "model_checkpoint.pt")
            ):  # ES saves its own checkpoint
                logger.info("Early stopping. Loading best model state.")
                if best_mdl_path.exists():
                    model.load_state_dict(torch.load(str(best_mdl_path)))
                break

        torch.save(model.state_dict(), str(output_dir / "model_final.pt"))
        logger.info(f"Final model saved. Best val_loss: {best_val_loss:.4f}")
        if f_cols:
            with open(output_dir / "feature_cols.pkl", "wb") as f:
                pickle.dump(f_cols, f)
        with open(output_dir / "scalers_final.pkl", "wb") as f:
            pickle.dump(scalers, f)
        writer.close()
        logger.info("Training complete.")
    except Exception as e:
        logger.error(f"Error in training: {e}", exc_info=True)
        raise
    finally:
        clear_gpu_cache()
        logging.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PM2.5 model.")
    default_cfg_path = "configs/improved_model_config.yaml"  # Relative to project root
    parser.add_argument(
        "--config", type=str, default=default_cfg_path, help="Path to YAML config file"
    )
    parser.add_argument(
        "--model_type", type=str, default=None, help="Override model_type from config"
    )
    args = parser.parse_args()

    cfg_file_path = Path(args.config)
    if not cfg_file_path.is_absolute():
        cfg_file_path = get_project_root() / cfg_file_path
    if not cfg_file_path.exists():
        print(f"Config file not found: {cfg_file_path}")
        sys.exit(1)
    train_model(str(cfg_file_path), args.model_type)
