import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, Tuple, Optional, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from sklearn.preprocessing import StandardScaler, MinMaxScaler # For type checking, not strictly needed for execution
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
from tqdm import tqdm  # Import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
# plt.style.use("seaborn-v0_8") # Using a more common style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Add the src directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Ensure PM25Model is imported correctly
# Assuming create_enhanced_model_configs is defined elsewhere or not strictly needed for this update
from src.models.model import PM25Model  # , create_enhanced_model_configs


class PM25ModelEvaluator:
    """Comprehensive PM2.5 model evaluation and forecasting system."""

    def __init__(
        self, model_dir: str, config_path: str, model_type_override: str = None
    ):  # Renamed model_type to model_type_override
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)
        self.model_type_override = model_type_override  # Use the override
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.config = self._load_config()
        self.scalers = self._load_scalers()

        self.station_names = self.config.get("data", {}).get("station_names")
        if not self.station_names:
            if (
                self.scalers
                and isinstance(self.scalers, dict)
                and all(isinstance(k, str) for k in self.scalers.keys())
            ):
                self.station_names = sorted(list(self.scalers.keys()))
                self.logger.info(
                    f"Inferred station names from scaler keys: {self.station_names}"
                )
            else:
                # Fallback to num_stations from model config if station_names is not in data config
                num_s_model_cfg = self.config.get("model", {}).get(
                    "num_stations", 0
                )  # Default to 0 if not found
                if (
                    num_s_model_cfg <= 0
                ):  # If still not found or invalid, try to infer from adj_matrix if possible later or error
                    self.logger.warning(
                        "num_stations not found in model config and station_names not in data config. This might lead to issues."
                    )
                    # Attempt to infer from adj_matrix later if needed, or it will likely fail in model init
                self.station_names = [
                    f"Station_{i+1}"
                    for i in range(num_s_model_cfg if num_s_model_cfg > 0 else 12)
                ]  # Default to 12 if all else fails
                self.logger.info(
                    f"Using default station names for {len(self.station_names)} stations based on model_config.num_stations or default."
                )

        # Ensure num_stations in model config matches the derived station_names length for consistency
        # This will be used when instantiating the model.
        self.model_num_stations_to_use = len(self.station_names)
        if self.model_num_stations_to_use == 0:
            self.logger.error(
                "Could not determine station names or number of stations. Evaluation cannot proceed."
            )
            raise ValueError(
                "Station names and number of stations could not be determined."
            )

        # Update model config's num_stations if it differs, for clarity during model init
        model_cfg_num_stations = self.config.get("model", {}).get("num_stations")
        if (
            model_cfg_num_stations is not None
            and model_cfg_num_stations != self.model_num_stations_to_use
        ):
            self.logger.warning(
                f"Model config num_stations ({model_cfg_num_stations}) differs from derived "
                f"station_names length ({self.model_num_stations_to_use}). "
                f"Will use {self.model_num_stations_to_use} for model instantiation."
            )
        # Ensure the config used for model loading reflects this
        if "model" not in self.config:
            self.config["model"] = {}
        self.config["model"]["num_stations"] = self.model_num_stations_to_use

        self.model = self._load_model()

        self.output_dir = self.model_dir / "evaluation_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Model evaluator initialized. Using device: {self.device}")
        self.logger.info(f"Results will be saved to: {self.output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        try:
            config_file_path = (
                project_root / self.config_path
                if not self.config_path.is_absolute()
                else self.config_path
            )
            with open(config_file_path, "r") as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_file_path}")
            return config
        except Exception as e:
            self.logger.error(
                f"Error loading configuration from {self.config_path}: {e}"
            )
            raise

    def _load_scalers(self) -> Dict:
        scaler_path = self.model_dir / "scalers_final.pkl"
        try:
            with open(scaler_path, "rb") as f:
                scalers = pickle.load(f)
            self.logger.info(
                f"Scalers loaded from {scaler_path}. Keys: {list(scalers.keys()) if isinstance(scalers, dict) else 'Not a dict'}"
            )
            if not isinstance(scalers, dict):
                self.logger.error(
                    f"Loaded scalers are not a dictionary. Type: {type(scalers)}"
                )
                # Attempt to load old format if it's a list of scalers (legacy)
                if isinstance(scalers, list) and all(
                    hasattr(s, "mean_") for s in scalers
                ):  # Basic check for list of scalers
                    self.logger.warning(
                        "Loaded scalers is a list, attempting to convert to dict using default station names."
                    )
                    num_s = self.config.get("model", {}).get(
                        "num_stations", len(scalers)
                    )
                    s_names = [f"Station_{i+1}" for i in range(num_s)]
                    if len(s_names) == len(scalers):
                        scalers = dict(zip(s_names, scalers))
                        self.logger.info(
                            f"Converted list of scalers to dict with keys: {list(scalers.keys())}"
                        )
                    else:
                        raise ValueError(
                            "Scaler list length mismatch with num_stations for dict conversion."
                        )
                else:
                    raise ValueError("Scalers should be a dictionary.")
            return scalers
        except FileNotFoundError:
            self.logger.error(f"Scaler file not found at {scaler_path}.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading scalers from {scaler_path}: {e}")
            raise

    def _load_model(self) -> nn.Module:
        try:
            data_dir_path_str = self.config["data"]["data_dir"]
            data_dir = Path(data_dir_path_str)
            if not data_dir.is_absolute():
                data_dir = project_root / data_dir_path_str
            adj_matrix_path = data_dir / "adj_matrix_enhanced.npy"
            if not adj_matrix_path.exists():
                # Try to find adj_matrix in model_dir as a fallback (if saved there by older training script)
                adj_matrix_path_fallback = self.model_dir / "adj_matrix_enhanced.npy"
                if adj_matrix_path_fallback.exists():
                    adj_matrix_path = adj_matrix_path_fallback
                    self.logger.info(
                        f"Found adjacency matrix in model directory: {adj_matrix_path}"
                    )
                else:
                    raise FileNotFoundError(
                        f"Adjacency matrix not found in {data_dir} or {self.model_dir}"
                    )
            adj_matrix = torch.FloatTensor(np.load(adj_matrix_path)).to(self.device)

            # Use a copy of the model config from the loaded YAML for instantiation
            model_config_for_init = self.config["model"].copy()

            # Override model_type if provided
            current_model_type = (
                self.model_type_override
                if self.model_type_override
                else model_config_for_init.get("model_type", "hybrid_lstm_cnn_v2")
            )

            # Ensure num_stations for model instantiation matches the derived station_names length
            model_config_for_init["num_stations"] = self.model_num_stations_to_use

            pm25_model_args = {
                "time_steps": int(model_config_for_init["time_steps"]),
                "num_stations": int(model_config_for_init["num_stations"]),
                "input_features": int(model_config_for_init["input_features"]),
                "adj_matrix": adj_matrix,
                "forecast_horizon": int(model_config_for_init["forecast_horizon"]),
                "hidden_dims": int(model_config_for_init.get("hidden_dims", 128)),
                "num_heads": int(model_config_for_init.get("num_heads", 4)),
                "num_layers": int(model_config_for_init.get("num_layers", 2)),
                "dropout_rate": float(model_config_for_init.get("dropout_rate", 0.15)),
                "wavelet_type": self.config.get("wavelet", {}).get(
                    "wavelet_type", "db4"
                ),
                "wavelet_level": int(self.config.get("wavelet", {}).get("level", 3)),
                "scalers": self.scalers,
                "station_names": self.station_names,
                "model_type": current_model_type,
                "use_stca_features": model_config_for_init.get(
                    "use_stca_features", True
                ),
                "use_wavelet_denoising": model_config_for_init.get(
                    "use_wavelet_denoising", True
                ),
                "use_mixed_precision": False,  # False for inference
                "kernel_regularization": float(
                    self.config.get("loss", {}).get("lambda_reg", 0.0)
                ),
            }

            self.logger.info(
                f"Instantiating PM25Model with args: { {k: v.shape if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k,v in pm25_model_args.items() if k != 'scalers'} }"
            )

            model = PM25Model(**pm25_model_args).to(self.device)

            model_path = self.model_dir / "model_best.pt"
            if not model_path.exists():
                self.logger.info(f"{model_path} not found, trying model_final.pt")
                model_path = self.model_dir / "model_final.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint (model_best.pt or model_final.pt) not found in {self.model_dir}"
                )

            state_dict = torch.load(model_path, map_location=self.device)

            if "positional_encoding" in state_dict and hasattr(
                model, "positional_encoding"
            ):
                pe_ckpt_shape = state_dict["positional_encoding"].shape
                pe_expected_shape = model.positional_encoding.shape
                self.logger.info(
                    f"PE from checkpoint: {pe_ckpt_shape}, current model PE expected: {pe_expected_shape}"
                )
                if pe_ckpt_shape != pe_expected_shape:
                    self.logger.warning(
                        f"Positional encoding shape mismatch. Checkpoint: {pe_ckpt_shape}, Model: {pe_expected_shape}. Attempting to adapt if possible."
                    )
                    if len(pe_ckpt_shape) == 2 and len(pe_expected_shape) == 4:
                        if (
                            pe_ckpt_shape[0] == pe_expected_shape[1]
                            and pe_ckpt_shape[1] == pe_expected_shape[3]
                        ):
                            state_dict["positional_encoding"] = (
                                state_dict["positional_encoding"]
                                .unsqueeze(0)
                                .unsqueeze(2)
                            )
                            self.logger.info(
                                f"Reshaped PE in state_dict to: {state_dict['positional_encoding'].shape}"
                            )

            try:
                model.load_state_dict(state_dict, strict=False)
                self.logger.info(
                    f"Model state_dict loaded from {model_path} (strict=False)"
                )
            except RuntimeError as e:
                self.logger.error(
                    f"RuntimeError loading state_dict (strict=False): {e}. This might indicate significant architecture mismatches."
                )
                try:
                    self.logger.info(
                        "Retrying load_state_dict with strict=True for detailed error..."
                    )
                    model.load_state_dict(state_dict, strict=True)
                except RuntimeError as e_strict:
                    self.logger.error(
                        f"Error loading state_dict (strict=True): {e_strict}"
                    )
                    raise e_strict
                raise e

            model.eval()
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
        try:
            data_dir_path_str = self.config["data"]["data_dir"]
            data_dir = Path(data_dir_path_str)
            if not data_dir.is_absolute():
                data_dir = project_root / data_dir_path_str
            test_data_path = data_dir / "test_enhanced.pkl"
            if not test_data_path.exists():
                raise FileNotFoundError(f"Test data file not found: {test_data_path}")

            with open(test_data_path, "rb") as f:
                test_data_loaded = pickle.load(f)
            full_df = None
            if isinstance(test_data_loaded, dict):
                if "X" not in test_data_loaded or "Y" not in test_data_loaded:
                    raise ValueError("Test data dictionary missing 'X' or 'Y' keys.")
                test_X, test_Y = test_data_loaded["X"], test_data_loaded["Y"]
                if "df" in test_data_loaded:
                    full_df = test_data_loaded["df"]
            elif isinstance(test_data_loaded, pd.DataFrame):
                full_df = test_data_loaded
                target_col = "PM2.5"
                if target_col not in full_df.columns:
                    raise ValueError(
                        f"Target column '{target_col}' not in test DataFrame."
                    )

                feature_cols_path = self.model_dir / "feature_cols.pkl"
                if feature_cols_path.exists():
                    with open(feature_cols_path, "rb") as f_cols_file:
                        feature_cols = pickle.load(f_cols_file)
                    self.logger.info(
                        f"Loaded feature_cols from {feature_cols_path}: {feature_cols}"
                    )
                else:
                    feature_cols = [
                        c
                        for c in full_df.columns
                        if c not in [target_col, "datetime", "station_id", "station"]
                    ]
                    self.logger.warning(
                        f"feature_cols.pkl not found in {self.model_dir}. Inferred {len(feature_cols)} features. This might be incorrect."
                    )

                if not all(fc in full_df.columns for fc in feature_cols):
                    missing_fc = [
                        fc for fc in feature_cols if fc not in full_df.columns
                    ]
                    raise ValueError(
                        f"Test DataFrame missing required feature columns: {missing_fc}"
                    )

                test_X, test_Y = (
                    full_df[feature_cols].values,
                    full_df[[target_col]].values,
                )
            elif isinstance(test_data_loaded, tuple) and len(test_data_loaded) == 2:
                test_X, test_Y = test_data_loaded
            else:
                raise ValueError(
                    f"Test data format unrecognized: {type(test_data_loaded)}"
                )

            test_X, test_Y = np.array(test_X, dtype=np.float32), np.array(
                test_Y, dtype=np.float32
            )
            self.logger.info(f"Test data loaded - X: {test_X.shape}, Y: {test_Y.shape}")
            return test_X, test_Y, full_df
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}", exc_info=True)
            raise

    def prepare_sequences(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.config["model"]["time_steps"]
        num_stations_for_Y_output = self.model_num_stations_to_use
        num_stations_model_expects_X = self.model_num_stations_to_use

        if X.shape[0] < seq_len:
            self.logger.error(
                f"Not enough X samples ({X.shape[0]}) to form even one sequence of length {seq_len}."
            )
            return torch.empty(0), torch.empty(0)

        X_s_list, Y_s_list = [], []
        num_possible_sequences = X.shape[0] - seq_len + 1
        if num_possible_sequences <= 0:
            self.logger.error(
                f"Cannot form any sequences. X length {X.shape[0]}, seq_len {seq_len}."
            )
            return torch.empty(0), torch.empty(0)

        for i in range(num_possible_sequences):
            X_s_list.append(X[i : i + seq_len])
            Y_s_list.append(Y[i + seq_len - 1])

        if not X_s_list:
            return torch.empty(0), torch.empty(0)

        X_s = np.array(X_s_list, dtype=np.float32)
        Y_s = np.array(Y_s_list, dtype=np.float32)

        if X_s.ndim == 3:
            X_s_expanded = np.expand_dims(X_s, axis=2)
            X_s = np.tile(X_s_expanded, (1, 1, num_stations_model_expects_X, 1))
        elif X_s.ndim == 4 and X_s.shape[2] != num_stations_model_expects_X:
            self.logger.warning(
                f"X_seq has {X_s.shape[2]} stations, model expects {num_stations_model_expects_X}. Adapting by tiling first station or truncating."
            )
            if X_s.shape[2] == 1:
                X_s = np.tile(X_s, (1, 1, num_stations_model_expects_X, 1))
            else:
                X_s = X_s[:, :, :num_stations_model_expects_X, :]

        if Y_s.ndim == 1:
            Y_s = Y_s.reshape(-1, 1)

        if Y_s.ndim == 2 and Y_s.shape[1] == 1:
            Y_s = np.repeat(Y_s[:, np.newaxis, :], num_stations_for_Y_output, axis=1)
        elif Y_s.ndim == 2 and Y_s.shape[1] == num_stations_for_Y_output:
            Y_s = np.expand_dims(Y_s, axis=2)
        elif not (
            Y_s.ndim == 3
            and Y_s.shape[1] == num_stations_for_Y_output
            and Y_s.shape[2] == 1
        ):
            self.logger.error(
                f"Y_s shape {Y_s.shape} is not compatible for target shape (N, {num_stations_for_Y_output}, 1). Attempting fallback."
            )
            if Y_s.ndim >= 2:
                Y_s = Y_s[:, :1]
            else:
                Y_s = Y_s.reshape(-1, 1)
            Y_s = np.repeat(Y_s[:, np.newaxis, :], num_stations_for_Y_output, axis=1)

        X_t, Y_t = torch.FloatTensor(X_s).to(self.device), torch.FloatTensor(Y_s).to(
            self.device
        )
        self.logger.info(
            f"Prepared sequences for evaluation/forecast - X: {X_t.shape}, Y: {Y_t.shape}"
        )
        return X_t, Y_t

    def evaluate_model(
        self, X_test_tensor: torch.Tensor, Y_test_tensor_scaled: torch.Tensor
    ) -> Dict[str, Any]:
        self.logger.info("Starting model evaluation...")
        default_metrics = {k: np.nan for k in ["MAE", "RMSE", "R2", "MAPE"]}
        if X_test_tensor.nelement() == 0:
            self.logger.warning("X_test_tensor is empty. Cannot evaluate.")
            return {
                "predictions_unscaled": np.array([]),
                "actuals_unscaled": np.array([]),
                "station_metrics": {
                    name: default_metrics.copy() for name in self.station_names
                },
                "overall_metrics": default_metrics.copy(),
            }

        predictions_unscaled_list, actuals_scaled_list = [], []
        eval_batch_size_cfg = self.config.get("evaluation", {}).get("batch_size")
        if eval_batch_size_cfg is None:
            eval_batch_size_cfg = self.config.get("training", {}).get("batch_size", 32)

        batch_size = min(int(eval_batch_size_cfg), len(X_test_tensor))
        if batch_size == 0 and len(X_test_tensor) > 0:
            batch_size = 1
        elif batch_size == 0 and len(X_test_tensor) == 0:
            self.logger.warning("X_test_tensor is empty, evaluation batch size is 0.")
            return {
                "predictions_unscaled": np.array([]),
                "actuals_unscaled": np.array([]),
                "station_metrics": {
                    name: default_metrics.copy() for name in self.station_names
                },
                "overall_metrics": default_metrics.copy(),
            }

        # Add tqdm progress bar for evaluation loop
        num_batches = (len(X_test_tensor) + batch_size - 1) // batch_size
        eval_progress_bar = tqdm(
            range(num_batches),
            desc="Evaluating Model",
            file=sys.stdout,
            ncols=100,
            disable=num_batches == 0,
        )

        with torch.no_grad():
            for i in eval_progress_bar:
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_test_tensor[start_idx:end_idx]
                Y_batch_scaled = Y_test_tensor_scaled[start_idx:end_idx]

                pred_batch_unscaled = self.model(X_batch)
                predictions_unscaled_list.append(pred_batch_unscaled.cpu().numpy())
                actuals_scaled_list.append(Y_batch_scaled.cpu().numpy())

        eval_progress_bar.close()

        if not predictions_unscaled_list:
            self.logger.warning("No predictions generated during evaluation.")
            return {
                "predictions_unscaled": np.array([]),
                "actuals_unscaled": np.array([]),
                "station_metrics": {
                    name: default_metrics.copy() for name in self.station_names
                },
                "overall_metrics": default_metrics.copy(),
            }

        predictions_unscaled_np = np.concatenate(predictions_unscaled_list, axis=0)
        actuals_scaled_np = np.concatenate(actuals_scaled_list, axis=0)

        actuals_unscaled_np = np.copy(actuals_scaled_np)
        unscaling_success_count = 0

        num_stations_in_actuals = actuals_scaled_np.shape[1]
        num_configured_stations = len(self.station_names)

        if num_stations_in_actuals != num_configured_stations:
            self.logger.warning(
                f"Mismatch: actuals data stations ({num_stations_in_actuals}) != configured station_names ({num_configured_stations}). "
                f"Will process min({num_stations_in_actuals}, {num_configured_stations}) stations for unscaling actuals."
            )

        stations_to_process_unscaling = min(
            num_stations_in_actuals, num_configured_stations
        )

        for station_idx in range(stations_to_process_unscaling):
            station_name = self.station_names[station_idx]
            scaler_obj = self.scalers.get(station_name)

            if scaler_obj and hasattr(scaler_obj, "inverse_transform"):
                try:
                    if (
                        len(scaler_obj.mean_) == 1
                        if hasattr(scaler_obj, "mean_")
                        else len(scaler_obj.data_min_) == 1
                    ):
                        current_station_pm25_scaled = actuals_scaled_np[
                            :, station_idx, 0:1
                        ]
                        unscaled_pm25 = scaler_obj.inverse_transform(
                            current_station_pm25_scaled
                        )
                        actuals_unscaled_np[:, station_idx, 0:1] = unscaled_pm25
                        unscaling_success_count += 1
                    else:
                        num_scaler_features = (
                            len(scaler_obj.mean_)
                            if hasattr(scaler_obj, "mean_")
                            else (
                                len(scaler_obj.data_min_)
                                if hasattr(scaler_obj, "data_min_")
                                else 0
                            )
                        )
                        if num_scaler_features > 0:
                            pm25_index_in_scaler = 0
                            temp_array_for_unscaling = np.zeros(
                                (actuals_scaled_np.shape[0], num_scaler_features)
                            )
                            temp_array_for_unscaling[:, pm25_index_in_scaler] = (
                                actuals_scaled_np[:, station_idx, 0]
                            )

                            unscaled_features = scaler_obj.inverse_transform(
                                temp_array_for_unscaling
                            )
                            actuals_unscaled_np[:, station_idx, 0] = unscaled_features[
                                :, pm25_index_in_scaler
                            ]
                            unscaling_success_count += 1
                        else:
                            self.logger.warning(
                                f"Scaler for '{station_name}' has no mean/min attributes, cannot determine features. Actuals remain scaled."
                            )
                except Exception as e:
                    self.logger.error(
                        f"Error unscaling actuals for '{station_name}' using scaler.inverse_transform: {e}. Actuals for this station might remain scaled."
                    )
            else:
                self.logger.warning(
                    f"Scaler for '{station_name}' not found or no inverse_transform method. Actuals for this station remain scaled."
                )

        if (
            unscaling_success_count == stations_to_process_unscaling
            and stations_to_process_unscaling > 0
        ):
            self.logger.info(
                "Successfully unscaled actual Y_test values for all processed stations."
            )
        elif unscaling_success_count > 0:
            self.logger.warning(
                f"Successfully unscaled actuals for {unscaling_success_count}/{stations_to_process_unscaling} stations. Others may remain scaled."
            )
        elif stations_to_process_unscaling > 0:
            self.logger.error(
                "Failed to unscale actual Y_test values for any station. Metrics will be based on scaled actuals vs unscaled predictions."
            )

        pred_for_metrics = predictions_unscaled_np[:, :, 0]
        actual_for_metrics = actuals_unscaled_np[:, :, 0]

        station_metrics = {}
        for st_idx in range(stations_to_process_unscaling):
            st_name = self.station_names[st_idx]
            pred_st = pred_for_metrics[:, st_idx]
            actual_st = actual_for_metrics[:, st_idx]

            valid = ~np.isnan(actual_st) & ~np.isnan(pred_st)
            if not np.any(valid):
                mae, rmse, r2, mape = np.nan, np.nan, np.nan, np.nan
            else:
                act_f, pred_f = actual_st[valid], pred_st[valid]
                if len(act_f) < 2:
                    r2 = np.nan
                else:
                    r2 = r2_score(act_f, pred_f)
                mae, rmse = mean_absolute_error(act_f, pred_f), np.sqrt(
                    mean_squared_error(act_f, pred_f)
                )

                nz_act_indices = act_f != 0
                if np.any(nz_act_indices):
                    mape = (
                        np.mean(
                            np.abs(
                                (act_f[nz_act_indices] - pred_f[nz_act_indices])
                                / (act_f[nz_act_indices] + 1e-8)
                            )
                        )
                        * 100
                    )
                else:
                    mape = 0.0 if np.all(pred_f == 0) else np.nan
            station_metrics[st_name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape,
            }

        overall_pred = pred_for_metrics[:, :stations_to_process_unscaling].reshape(-1)
        overall_actual = actual_for_metrics[:, :stations_to_process_unscaling].reshape(
            -1
        )

        valid_ovr = ~np.isnan(overall_actual) & ~np.isnan(overall_pred)
        overall_metrics = default_metrics.copy()
        if np.any(valid_ovr):
            act_o, pred_o = overall_actual[valid_ovr], overall_pred[valid_ovr]
            if len(act_o) >= 2:
                overall_metrics["R2"] = r2_score(act_o, pred_o)
            overall_metrics["MAE"] = mean_absolute_error(act_o, pred_o)
            overall_metrics["RMSE"] = np.sqrt(mean_squared_error(act_o, pred_o))

            nz_o_act_indices = act_o != 0
            if np.any(nz_o_act_indices):
                overall_metrics["MAPE"] = (
                    np.mean(
                        np.abs(
                            (act_o[nz_o_act_indices] - pred_o[nz_o_act_indices])
                            / (act_o[nz_o_act_indices] + 1e-8)
                        )
                    )
                    * 100
                )
            else:
                overall_metrics["MAPE"] = 0.0 if np.all(pred_o == 0) else np.nan

        results = {
            "predictions_unscaled": predictions_unscaled_np,
            "actuals_unscaled": actuals_unscaled_np,
            "station_metrics": station_metrics,
            "overall_metrics": overall_metrics,
        }
        self.logger.info(
            f"Overall Metrics (1st forecast step vs actuals) - R2: {overall_metrics.get('R2',np.nan):.4f}, RMSE: {overall_metrics.get('RMSE',np.nan):.4f}, MAE: {overall_metrics.get('MAE',np.nan):.4f}, MAPE: {overall_metrics.get('MAPE',np.nan):.2f}%"
        )
        return results

    def generate_72hour_forecast(
        self, recent_data_X: np.ndarray, timestamps: List[datetime] = None
    ) -> Dict[str, Any]:
        self.logger.info("Generating 72-hour forecast...")
        seq_len = self.config["model"]["time_steps"]
        num_stations_model_expects = self.model_num_stations_to_use

        if recent_data_X.shape[0] < seq_len:
            self.logger.error(
                f"Need at least {seq_len} time steps of X data for forecast, got {recent_data_X.shape[0]}."
            )
            raise ValueError(
                f"Insufficient recent_data_X for {seq_len} sequence length."
            )

        init_x_sequence_raw = recent_data_X[-seq_len:]

        if init_x_sequence_raw.ndim == 2:
            init_x_expanded = np.expand_dims(init_x_sequence_raw, axis=1)
            init_x_tiled = np.tile(init_x_expanded, (1, num_stations_model_expects, 1))
        else:
            self.logger.error(
                f"recent_data_X slice has unexpected ndim: {init_x_sequence_raw.ndim}"
            )
            raise ValueError(
                "Error preparing forecast input sequence from recent_data_X."
            )

        curr_seq_X_np = np.expand_dims(init_x_tiled, axis=0)
        curr_seq_tensor = torch.FloatTensor(curr_seq_X_np).to(self.device)

        with torch.no_grad():
            forecast_unscaled_tensor = self.model(curr_seq_tensor)

        forecast_unscaled_np = forecast_unscaled_tensor.cpu().numpy()[0]

        self.logger.info(
            f"Raw forecast (unscaled) shape from model: {forecast_unscaled_np.shape}"
        )

        model_output_stations = forecast_unscaled_np.shape[0]
        model_horizon = forecast_unscaled_np.shape[1]

        num_target_stations_output = len(self.station_names)

        if model_output_stations != num_target_stations_output:
            self.logger.warning(
                f"Model output stations ({model_output_stations}) differs from configured station_names count ({num_target_stations_output}). Forecast will be for {min(model_output_stations, num_target_stations_output)} stations."
            )
            forecast_unscaled_np = forecast_unscaled_np[:num_target_stations_output, :]

        forecast_72h = np.full((num_target_stations_output, 72), np.nan)

        horizon_to_fill = min(model_horizon, 72)
        forecast_72h[:, :horizon_to_fill] = forecast_unscaled_np[:, :horizon_to_fill]

        if model_horizon < 72 and model_horizon > 0:
            self.logger.warning(
                f"Model forecast horizon ({model_horizon}) is less than 72h. Padding forecast with the last predicted value."
            )
            padding_values = np.tile(
                forecast_72h[:, model_horizon - 1 : model_horizon],
                (1, 72 - model_horizon),
            )
            forecast_72h[:, model_horizon:] = padding_values
        elif model_horizon == 0:
            self.logger.error(
                "Model horizon is 0. Cannot generate meaningful forecast."
            )

        forecast_dict_hourly = {
            self.station_names[st_idx]: {
                f"{h+1}h": (
                    float(forecast_72h[st_idx, h])
                    if h < forecast_72h.shape[1]
                    else np.nan
                )
                for h in range(72)
            }
            for st_idx in range(num_target_stations_output)
        }

        forecast_dict_summary = {
            self.station_names[st_idx]: {
                "24h": (
                    float(forecast_72h[st_idx, 23])
                    if 23 < forecast_72h.shape[1]
                    else np.nan
                ),
                "48h": (
                    float(forecast_72h[st_idx, 47])
                    if 47 < forecast_72h.shape[1]
                    else np.nan
                ),
                "72h": (
                    float(forecast_72h[st_idx, 71])
                    if 71 < forecast_72h.shape[1]
                    else np.nan
                ),
            }
            for st_idx in range(num_target_stations_output)
        }

        self.logger.info("72-hour forecast generated and processed.")

        start_time_forecast = (
            timestamps[0] if timestamps and len(timestamps) > 0 else datetime.now()
        )
        forecast_timestamps_hourly = [
            start_time_forecast + timedelta(hours=i + 1) for i in range(72)
        ]

        return {
            "forecast_summary": forecast_dict_summary,
            "forecast_hourly": forecast_dict_hourly,
            "timestamps_hourly": forecast_timestamps_hourly,
            "raw_forecast_matrix": forecast_72h,
        }

    def plot_evaluation_results(self, results: Dict[str, Any]):
        self.logger.info("Creating evaluation plots...")
        predictions_unscaled = results["predictions_unscaled"]
        actuals_unscaled = results["actuals_unscaled"]
        overall_metrics = results["overall_metrics"]

        if predictions_unscaled.size == 0 or actuals_unscaled.size == 0:
            self.logger.warning("No data for evaluation plots.")
            return

        overall_pred_first_step = predictions_unscaled[:, :, 0].reshape(-1)
        overall_actual_vals = actuals_unscaled[:, :, 0].reshape(-1)

        valid_indices_plot = ~np.isnan(overall_actual_vals) & ~np.isnan(
            overall_pred_first_step
        )
        overall_actual_plot = overall_actual_vals[valid_indices_plot]
        overall_pred_plot = overall_pred_first_step[valid_indices_plot]

        if len(overall_actual_plot) == 0:
            self.logger.warning(
                "No valid (non-NaN) data points for overall evaluation plots after filtering."
            )
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "PM2.5 Model Evaluation Summary (Original Scale, 1st Forecast Step)",
            fontsize=16,
            fontweight="bold",
        )

        axes[0, 0].scatter(
            overall_actual_plot,
            overall_pred_plot,
            alpha=0.5,
            s=10,
            edgecolors="k",
            lw=0.5,
        )
        min_val = (
            min(overall_actual_plot.min(), overall_pred_plot.min())
            if len(overall_actual_plot) > 0
            else 0
        )
        max_val = (
            max(overall_actual_plot.max(), overall_pred_plot.max())
            if len(overall_actual_plot) > 0
            else 1
        )
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        axes[0, 0].set_xlabel("Actual PM2.5 (µg/m³)")
        axes[0, 0].set_ylabel("Predicted PM2.5 (µg/m³ - 1st Step)")
        axes[0, 0].set_title(
            f'Predicted vs Actual (R² = {overall_metrics.get("R2", np.nan):.3f})'
        )
        axes[0, 0].grid(True, alpha=0.3)

        residuals = overall_pred_plot - overall_actual_plot
        axes[0, 1].scatter(
            overall_actual_plot, residuals, alpha=0.5, s=10, edgecolors="k", lw=0.5
        )
        axes[0, 1].axhline(y=0, color="r", ls="--")
        axes[0, 1].set_xlabel("Actual PM2.5 (µg/m³)")
        axes[0, 1].set_ylabel("Residuals (µg/m³)")
        axes[0, 1].set_title("Residuals Plot")
        axes[0, 1].grid(True, alpha=0.3)

        if len(residuals) > 0:
            axes[1, 0].hist(
                residuals, bins=50, alpha=0.7, edgecolor="black", density=True
            )
            try:
                sns.kdeplot(residuals, ax=axes[1, 0], color="red", warn_singular=False)
            except Exception as e_kde:
                self.logger.warning(f"Could not plot KDE for residuals: {e_kde}")
        else:
            axes[1, 0].text(0.5, 0.5, "No residuals data", ha="center", va="center")
        axes[1, 0].set_xlabel("Residuals (µg/m³)")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Error Distribution")
        axes[1, 0].axvline(x=0, color="r", ls="--")
        axes[1, 0].grid(True, alpha=0.3)

        m_names = ["MAE", "RMSE", "MAPE"]
        m_values = [overall_metrics.get(k, np.nan) for k in m_names]
        valid_m_indices = [i for i, v in enumerate(m_values) if not np.isnan(v)]

        if valid_m_indices:
            plot_m_names = [m_names[i] for i in valid_m_indices]
            plot_m_values = [m_values[i] for i in valid_m_indices]
            axes[1, 1].bar(
                plot_m_names,
                plot_m_values,
                color=["skyblue", "lightcoral", "lightgreen"][: len(plot_m_values)],
            )
            axes[1, 1].set_ylabel("Error Value")
            axes[1, 1].set_title("Overall Metrics")
            axes[1, 1].grid(True, alpha=0.3)
            for i, val_bar in enumerate(plot_m_values):
                axes[1, 1].text(
                    i,
                    val_bar + max(plot_m_values, default=0) * 0.01,
                    f"{val_bar:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
        else:
            axes[1, 1].text(0.5, 0.5, "Metrics NaN", ha="center", va="center")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            self.output_dir / "evaluation_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        self.plot_station_performance(results.get("station_metrics", {}))
        self.plot_time_series_comparison(predictions_unscaled, actuals_unscaled)
        self.plot_advanced_analysis(predictions_unscaled, actuals_unscaled)

    def plot_station_performance(self, station_metrics: Dict[str, Dict[str, float]]):
        if not station_metrics:
            self.logger.warning("No station metrics to plot.")
            return

        stations = list(station_metrics.keys())
        if not stations:
            self.logger.warning("Station metrics provided but no station names found.")
            return

        metrics_to_plot = ["MAE", "RMSE", "R2", "MAPE"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Station-wise Performance (Original Scale, 1st Forecast Step)",
            fontsize=16,
            fontweight="bold",
        )
        axes = axes.flatten()

        for i, metric_name in enumerate(metrics_to_plot):
            if i >= len(axes):
                break

            values = [
                station_metrics.get(st, {}).get(metric_name, np.nan) for st in stations
            ]

            valid_station_data = [
                (st, val) for st, val in zip(stations, values) if not np.isnan(val)
            ]
            if not valid_station_data:
                axes[i].text(
                    0.5,
                    0.5,
                    f"{metric_name}\n(No valid data for any station)",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                axes[i].set_title(f"{metric_name} by Station")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                continue

            plot_stations, plot_values = zip(*valid_station_data)

            bars = axes[i].bar(
                range(len(plot_stations)),
                plot_values,
                color=plt.cm.viridis(np.linspace(0, 1, len(plot_stations))),
            )
            axes[i].set_title(f"{metric_name} by Station")
            axes[i].set_xlabel("Station")
            axes[i].set_ylabel(metric_name)
            axes[i].set_xticks(range(len(plot_stations)))
            axes[i].set_xticklabels(plot_stations, rotation=45, ha="right", fontsize=8)
            axes[i].grid(True, alpha=0.3, axis="y")

            max_val_for_text = max(plot_values) if plot_values else 0
            for bar_idx, (bar, val_bar) in enumerate(zip(bars, plot_values)):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    val_bar + max_val_for_text * 0.02,
                    f"{val_bar:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            self.output_dir / "station_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_time_series_comparison(
        self,
        pred_unscaled: np.ndarray,
        actual_unscaled: np.ndarray,
        sample_size: int = 200,
    ):
        if pred_unscaled.size == 0 or actual_unscaled.size == 0:
            self.logger.warning("No data for time series plot.")
            return

        n_samples_avail = pred_unscaled.shape[0]
        if n_samples_avail == 0:
            self.logger.warning("Zero samples available for time series plot.")
            return

        indices_plot = np.sort(
            np.random.choice(
                n_samples_avail, min(sample_size, n_samples_avail), replace=False
            )
        )

        pred_samp_plot = pred_unscaled[indices_plot, :, 0]
        actual_samp_plot = actual_unscaled[indices_plot, :, 0]

        n_stations_to_plot = min(4, pred_unscaled.shape[1])
        if n_stations_to_plot == 0:
            self.logger.warning("No stations available in data for time series plot.")
            return

        fig, axes = plt.subplots(
            n_stations_to_plot, 1, figsize=(15, 4 * n_stations_to_plot), sharex=True
        )
        if n_stations_to_plot == 1:
            axes = [axes]

        fig.suptitle(
            "Time Series: Predicted (1st step) vs Actual (Original Scale, Sampled)",
            fontsize=16,
            fontweight="bold",
        )

        for i in range(n_stations_to_plot):
            st_name_plot = (
                self.station_names[i]
                if i < len(self.station_names)
                else f"Station {i+1}"
            )

            axes[i].plot(
                actual_samp_plot[:, i], label="Actual", alpha=0.8, lw=1.5, color="blue"
            )
            axes[i].plot(
                pred_samp_plot[:, i],
                label="Predicted",
                alpha=0.7,
                lw=1.5,
                ls="--",
                color="red",
            )
            axes[i].set_title(f"{st_name_plot}", fontsize=12)
            axes[i].set_ylabel("PM2.5 (µg/m³)")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel(f"Time Steps (Sampled, total {len(indices_plot)} points)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            self.output_dir / "time_series_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_advanced_analysis(
        self, pred_unscaled: np.ndarray, actual_unscaled: np.ndarray
    ):
        if pred_unscaled.size == 0 or actual_unscaled.size == 0:
            self.logger.warning("No data for advanced plot.")
            return

        pred_flat_adv = pred_unscaled[:, :, 0].reshape(-1)
        actual_flat_adv = actual_unscaled[:, :, 0].reshape(-1)

        valid_idx_adv = ~np.isnan(actual_flat_adv) & ~np.isnan(pred_flat_adv)
        actual_plot_adv = actual_flat_adv[valid_idx_adv]
        pred_plot_adv = pred_flat_adv[valid_idx_adv]

        if len(actual_plot_adv) < 2:
            self.logger.warning(
                "Not enough valid (non-NaN) data points for advanced analysis plots after filtering."
            )
            return

        errors_adv = pred_plot_adv - actual_plot_adv
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Advanced Error Analysis (Original Scale, 1st Forecast Step)",
            fontsize=16,
            fontweight="bold",
        )

        axes[0, 0].scatter(
            actual_plot_adv, np.abs(errors_adv), alpha=0.5, s=10, edgecolors="k", lw=0.5
        )
        axes[0, 0].set_xlabel("Actual PM2.5 (µg/m³)")
        axes[0, 0].set_ylabel("Absolute Error (µg/m³)")
        axes[0, 0].set_title("Error vs Concentration")
        axes[0, 0].grid(True, alpha=0.3)

        try:
            percentiles = [0, 25, 50, 75, 100]
            bin_edges = np.unique(np.percentile(actual_plot_adv, percentiles))
            if len(bin_edges) < 2:
                bin_edges = np.array([actual_plot_adv.min(), actual_plot_adv.max()])
                if bin_edges[0] == bin_edges[1]:
                    bin_edges = np.array([bin_edges[0] - 0.5, bin_edges[1] + 0.5])

            if len(bin_edges) < 2:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "Not enough distinct data for bins",
                    ha="center",
                    va="center",
                )
            else:
                bin_labels_adv, bin_errors_adv = [], []
                for i in range(len(bin_edges) - 1):
                    mask_adv = (actual_plot_adv >= bin_edges[i]) & (
                        actual_plot_adv < bin_edges[i + 1]
                        if i < len(bin_edges) - 2
                        else actual_plot_adv <= bin_edges[i + 1]
                    )
                    if np.any(mask_adv):
                        bin_errors_adv.append(np.abs(errors_adv[mask_adv]).mean())
                        bin_labels_adv.append(
                            f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
                        )

                if bin_labels_adv:
                    axes[0, 1].bar(
                        bin_labels_adv,
                        bin_errors_adv,
                        color=plt.cm.coolwarm(
                            np.linspace(0.2, 0.8, len(bin_labels_adv))
                        ),
                    )
                    axes[0, 1].tick_params(axis="x", rotation=30, labelsize=8)
                else:
                    axes[0, 1].text(
                        0.5, 0.5, "No data in bins", ha="center", va="center"
                    )
            axes[0, 1].set_xlabel("Concentration Range (µg/m³)")
            axes[0, 1].set_ylabel("Mean Absolute Error (µg/m³)")
            axes[0, 1].set_title("Error by Concentration Bins")
            axes[0, 1].grid(True, alpha=0.3, axis="y")
        except Exception as e_bin:
            self.logger.warning(f"Error in binning plot: {e_bin}")
            axes[0, 1].text(0.5, 0.5, "Error in binning", ha="center", va="center")

        from scipy import stats

        if len(errors_adv) > 1:
            stats.probplot(errors_adv, dist="norm", plot=axes[1, 0])
            axes[1, 0].get_lines()[0].set_markersize(3.0)
            axes[1, 0].get_lines()[1].set_linewidth(1.5)
        else:
            axes[1, 0].text(
                0.5, 0.5, "Not enough data for Q-Q plot", ha="center", va="center"
            )
        axes[1, 0].set_title("Q-Q Plot: Error Normality")
        axes[1, 0].grid(True, alpha=0.3)

        if len(errors_adv) > 0:
            sorted_abs_err_adv = np.sort(np.abs(errors_adv))
            cum_pct_adv = (
                np.arange(1, len(sorted_abs_err_adv) + 1)
                / len(sorted_abs_err_adv)
                * 100
            )
            axes[1, 1].plot(sorted_abs_err_adv, cum_pct_adv, lw=1.5)
            axes[1, 1].set_xlabel("Absolute Error (µg/m³)")
            axes[1, 1].set_ylabel("Cumulative Percentage (%)")
            axes[1, 1].set_title("Cumulative Error Distribution")
            axes[1, 1].grid(True, alpha=0.3)
            for p_val in [50, 75, 90, 95]:
                if len(sorted_abs_err_adv) > 0:
                    err_val_adv = np.percentile(np.abs(errors_adv), p_val)
                    axes[1, 1].axvline(
                        x=err_val_adv, color="red", ls="--", alpha=0.7, lw=1
                    )
                    axes[1, 1].text(
                        err_val_adv * 1.02,
                        p_val - 5,
                        f"{p_val}th: {err_val_adv:.2f}",
                        rotation=0,
                        fontsize=8,
                        color="red",
                    )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Not enough data for cumulative error plot",
                ha="center",
                va="center",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            self.output_dir / "advanced_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_72hour_forecast(self, forecast_results: Dict[str, Any]):
        if (
            not forecast_results
            or "forecast_summary" not in forecast_results
            or not forecast_results["forecast_summary"]
        ):
            self.logger.warning("No 72h forecast data to plot.")
            return

        forecast_summary_plot = forecast_results["forecast_summary"]
        stations_plot = list(forecast_summary_plot.keys())
        if not stations_plot:
            self.logger.warning("No stations in forecast data for plotting.")
            return

        time_horizons_plot = ["24h", "48h", "72h"]
        fig, axes = plt.subplots(
            2, 1, figsize=(15, 12), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.suptitle(
            "72-Hour PM2.5 Forecast (Original Scale)", fontsize=16, fontweight="bold"
        )

        forecast_data_plot = np.array(
            [
                [
                    forecast_summary_plot.get(st_plot, {}).get(h_plot, np.nan)
                    for h_plot in time_horizons_plot
                ]
                for st_plot in stations_plot
            ]
        )

        x_pos_plot = np.arange(len(stations_plot))
        width_plot = 0.25

        for i, horizon_plot in enumerate(time_horizons_plot):
            bars_plot = axes[0].bar(
                x_pos_plot + i * width_plot - width_plot,
                forecast_data_plot[:, i],
                width_plot,
                label=horizon_plot,
                alpha=0.8,
            )
            for bar_idx_plot, bar_item in enumerate(bars_plot):
                bar_val_plot = forecast_data_plot[bar_idx_plot, i]
                if not np.isnan(bar_val_plot):
                    axes[0].text(
                        bar_item.get_x() + bar_item.get_width() / 2,
                        bar_val_plot + 0.5,
                        f"{bar_val_plot:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=45,
                    )

        axes[0].set_xlabel("Stations", fontsize=12)
        axes[0].set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=12)
        axes[0].set_title("PM2.5 Forecast (24h, 48h, 72h points)", fontsize=14)
        axes[0].set_xticks(x_pos_plot)
        axes[0].set_xticklabels(stations_plot, rotation=45, ha="right", fontsize=9)
        axes[0].legend(title="Forecast Horizon")
        axes[0].grid(True, alpha=0.3, axis="y")

        forecast_hourly_plot = forecast_results.get("forecast_hourly")
        timestamps_hourly = forecast_results.get(
            "timestamps_hourly", list(range(1, 73))
        )

        if forecast_hourly_plot:
            n_st_trend_plot = min(len(stations_plot), 5)

            for i_st_trend in range(n_st_trend_plot):
                st_name_trend = stations_plot[i_st_trend]
                if st_name_trend in forecast_hourly_plot:
                    hourly_val_trend = [
                        forecast_hourly_plot[st_name_trend].get(f"{h+1}h", np.nan)
                        for h in range(72)
                    ]
                    x_axis_data = (
                        timestamps_hourly
                        if isinstance(timestamps_hourly[0], datetime)
                        else range(1, 73)
                    )
                    axes[1].plot(
                        x_axis_data,
                        hourly_val_trend,
                        marker=".",
                        ls="-",
                        label=st_name_trend,
                        lw=1.5,
                        ms=4,
                    )
                else:
                    self.logger.warning(
                        f"Hourly data for station '{st_name_trend}' not found in forecast_hourly_plot."
                    )

            axes[1].set_xlabel(
                (
                    "Forecast Hour Ahead"
                    if not isinstance(timestamps_hourly[0], datetime)
                    else "Forecast Time"
                ),
                fontsize=12,
            )
            axes[1].set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=12)
            axes[1].set_title(
                f"Hourly Forecast Trend (First {n_st_trend_plot} Stations)", fontsize=14
            )
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
            axes[1].grid(True, alpha=0.3)
            if not isinstance(timestamps_hourly[0], datetime):
                axes[1].set_xticks(np.arange(0, 73, 6))
                axes[1].set_xlim(0, 72)
            else:
                pass

        else:
            axes[1].text(
                0.5,
                0.5,
                "Hourly forecast data not available for trend.",
                ha="center",
                va="center",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            self.output_dir / "72hour_forecast.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        self.create_interactive_forecast_plot(forecast_results)

    def create_interactive_forecast_plot(self, forecast_results: Dict[str, Any]):
        if (
            not forecast_results
            or "forecast_hourly" not in forecast_results
            or not forecast_results["forecast_hourly"]
        ):
            self.logger.warning(
                "Hourly forecast data not available for interactive plot."
            )
            return

        forecast_hourly_interactive = forecast_results["forecast_hourly"]
        stations_interactive = list(forecast_hourly_interactive.keys())
        if not stations_interactive:
            self.logger.warning(
                "No stations in hourly forecast data for interactive plot."
            )
            return

        timestamps_hourly_interactive = forecast_results.get("timestamps_hourly")
        if not timestamps_hourly_interactive or not isinstance(
            timestamps_hourly_interactive[0], datetime
        ):
            timestamps_hourly_interactive_str = [f"{h+1}h" for h in range(72)]
            x_axis_values = timestamps_hourly_interactive_str
            x_axis_title = "Forecast Hour Ahead"
            x_tick_format = None
        else:
            x_axis_values = timestamps_hourly_interactive
            x_axis_title = "Forecast Time"
            x_tick_format = "%Y-%m-%d %H:%M"

        fig_interactive = make_subplots(
            rows=1, cols=1, subplot_titles=(["Hourly PM2.5 Forecast by Station"])
        )

        for st_name_interactive in stations_interactive:
            if st_name_interactive in forecast_hourly_interactive:
                hourly_val_interactive = [
                    forecast_hourly_interactive[st_name_interactive].get(
                        f"{h+1}h", np.nan
                    )
                    for h in range(72)
                ]
                fig_interactive.add_trace(
                    go.Scatter(
                        x=x_axis_values,
                        y=hourly_val_interactive,
                        mode="lines+markers",
                        name=st_name_interactive,
                        line=dict(width=2),
                        marker=dict(size=3),
                    ),
                    row=1,
                    col=1,
                )

        who_levels_interactive = {
            "Good (Annual)": 5,
            "Good (24h)": 15,
            "Moderate (24h)": 25,
            "Unhealthy Sens. (24h)": 37.5,
            "Unhealthy (24h)": 50,
        }
        colors_interactive = ["green", "yellowgreen", "orange", "red", "purple"]

        for i, (lvl_name_interactive, lvl_val_interactive) in enumerate(
            who_levels_interactive.items()
        ):
            fig_interactive.add_hline(
                y=lvl_val_interactive,
                line_dash="dot",
                annotation_text=f"{lvl_name_interactive}: {lvl_val_interactive} µg/m³",
                annotation_position="bottom right",
                line_color=colors_interactive[i % len(colors_interactive)],
                row=1,
                col=1,
            )

        fig_interactive.update_layout(
            height=600,
            title_text="Interactive 72-Hour PM2.5 Forecast (Original Scale)",
            showlegend=True,
            xaxis_title=x_axis_title,
            yaxis_title="PM2.5 (µg/m³)",
            legend_title_text="Stations",
        )
        if x_tick_format:
            fig_interactive.update_xaxes(tickformat=x_tick_format, row=1, col=1)

        interactive_plot_path_final = self.output_dir / "interactive_forecast.html"
        try:
            fig_interactive.write_html(str(interactive_plot_path_final))
            self.logger.info(
                f"Interactive forecast plot saved: {interactive_plot_path_final}"
            )
        except Exception as e_plotly:
            self.logger.error(f"Failed to save interactive plotly: {e_plotly}")

    def generate_forecast_report(
        self, eval_res_report: Dict[str, Any], forecast_res_report: Dict[str, Any]
    ) -> str:
        self.logger.info("Generating forecast report...")
        overall_metrics_rep = eval_res_report.get("overall_metrics", {})
        station_metrics_rep = eval_res_report.get("station_metrics", {})
        forecast_summary_rep = forecast_res_report.get("forecast_summary", {})

        report_str = f"# PM2.5 Model - Evaluation & Forecast (Original Scale)\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n## Model Performance (1st Forecast Step vs Actuals)\n"
        report_str += f"- R²: {overall_metrics_rep.get('R2',np.nan):.4f}\n- RMSE: {overall_metrics_rep.get('RMSE',np.nan):.2f} µg/m³\n- MAE: {overall_metrics_rep.get('MAE',np.nan):.2f} µg/m³\n- MAPE: {overall_metrics_rep.get('MAPE',np.nan):.2f}%\n"

        r2_rep = overall_metrics_rep.get("R2", np.nan)
        quality_rep = "Undetermined"
        if not np.isnan(r2_rep):
            if r2_rep >= 0.75:
                quality_rep = "Excellent"
            elif r2_rep >= 0.6:
                quality_rep = "Good"
            elif r2_rep >= 0.4:
                quality_rep = "Fair"
            else:
                quality_rep = "Needs Improvement"
        report_str += f"**Overall Quality (R² based)**: {quality_rep}\n\n### Station Performance (1st Step vs Actual)\n| Station | R² | RMSE | MAE | MAPE (%) |\n|---|---|---|---|---|\n"

        for st_rep, met_rep in station_metrics_rep.items():
            report_str += f"| {st_rep} | {met_rep.get('R2',np.nan):.3f} | {met_rep.get('RMSE',np.nan):.2f} | {met_rep.get('MAE',np.nan):.2f} | {met_rep.get('MAPE',np.nan):.2f} |\n"

        report_str += "\n## 72-Hour PM2.5 Forecast (µg/m³)\n| Station | 24h | 48h | 72h | Max (next 72h) | Risk (based on Max) |\n|---|---|---|---|---|---|\n"

        def get_risk_rep(v_rep):
            if np.isnan(v_rep):
                return "⚪ Undet."
            if v_rep <= 15:
                return "🟢 Good"
            if v_rep <= 25:
                return "🟡 Moderate"
            if v_rep <= 37.5:
                return "🟠 Unh. Sens."
            if v_rep <= 50:
                return "🔴 Unhealthy"
            return "🟣 V. Unh."

        all_f_pts_rep = []
        forecast_hourly_rep = forecast_res_report.get("forecast_hourly", {})

        for st_name_rep in self.station_names:
            f_summary_vals = forecast_summary_rep.get(
                st_name_rep, {"24h": np.nan, "48h": np.nan, "72h": np.nan}
            )
            f24_rep, f48_rep, f72_rep = (
                f_summary_vals["24h"],
                f_summary_vals["48h"],
                f_summary_vals["72h"],
            )

            max_f_rep = np.nan
            if st_name_rep in forecast_hourly_rep:
                hourly_values = [
                    val
                    for val in forecast_hourly_rep[st_name_rep].values()
                    if not np.isnan(val)
                ]
                if hourly_values:
                    max_f_rep = max(hourly_values)
                    all_f_pts_rep.extend(hourly_values)

            risk_rep = get_risk_rep(max_f_rep)
            report_str += f"| {st_name_rep} | {f24_rep:.1f} | {f48_rep:.1f} | {f72_rep:.1f} | {max_f_rep:.1f} | {risk_rep} |\n"

        report_str += "\n### Forecast Risk Summary (across all stations & hours)\n"
        valid_all_f_rep = [v_all for v_all in all_f_pts_rep if not np.isnan(v_all)]
        if valid_all_f_rep:
            avg_f_rep, max_f_rep_overall = np.mean(valid_all_f_rep), np.max(
                valid_all_f_rep
            )
            report_str += f"- Avg PM2.5 (all station forecasts, all hours): {avg_f_rep:.1f} µg/m³\n- Max PM2.5 (overall): {max_f_rep_overall:.1f} µg/m³\n- Overall Highest Risk (based on max): {get_risk_rep(max_f_rep_overall)}\n"
        else:
            report_str += "- Forecast data insufficient for overall risk assessment.\n"

        report_path_final = self.output_dir / "forecast_report.md"
        with open(report_path_final, "w", encoding="utf-8") as f_report:
            f_report.write(report_str)
        self.logger.info(f"Report saved: {report_path_final}")
        return report_str

    def save_results(
        self, eval_res_save: Dict[str, Any], forecast_res_save: Dict[str, Any]
    ):
        self.logger.info(f"Saving results to {self.output_dir}...")
        (self.output_dir / "evaluation_results.pkl").write_bytes(
            pickle.dumps(eval_res_save)
        )
        (self.output_dir / "forecast_results.pkl").write_bytes(
            pickle.dumps(forecast_res_save)
        )

        preds1_save = eval_res_save["predictions_unscaled"][:, :, 0]
        actuals1_save = eval_res_save["actuals_unscaled"][:, :, 0]

        num_samples = preds1_save.shape[0]
        num_stations_data = preds1_save.shape[1]

        data_save_dict = {"time_index": range(num_samples)}
        for i, st_name_save in enumerate(self.station_names):
            if i < num_stations_data:
                data_save_dict[f"{st_name_save}_actual"] = (
                    actuals1_save[:, i] if i < actuals1_save.shape[1] else np.nan
                )
                data_save_dict[f"{st_name_save}_pred_step1"] = preds1_save[:, i]
            else:
                data_save_dict[f"{st_name_save}_actual"] = [np.nan] * num_samples
                data_save_dict[f"{st_name_save}_pred_step1"] = [np.nan] * num_samples

        pd.DataFrame(data_save_dict).to_csv(
            self.output_dir / "predictions_vs_actuals_step1.csv", index=False
        )

        if (
            "forecast_hourly" in forecast_res_save
            and forecast_res_save["forecast_hourly"]
        ):
            serializable_forecast_save = {
                "stations_forecast_hourly": forecast_res_save["forecast_hourly"]
            }
            if forecast_res_save.get("timestamps_hourly") and isinstance(
                forecast_res_save["timestamps_hourly"][0], datetime
            ):
                serializable_forecast_save["timestamps_hourly_iso"] = [
                    dt_save.isoformat()
                    for dt_save in forecast_res_save["timestamps_hourly"]
                ]

            (self.output_dir / "forecast_hourly.json").write_text(
                json.dumps(serializable_forecast_save, indent=2, default=str)
            )
        self.logger.info(f"Results successfully saved.")

    def run_complete_evaluation(
        self,
        recent_data_X_for_forecast: np.ndarray = None,
        forecast_start_timestamps: List[datetime] = None,
    ) -> Dict[str, Any]:
        self.logger.info("Starting complete evaluation pipeline...")
        try:
            X_test_raw, Y_test_raw_scaled, _ = self.load_test_data()
            if X_test_raw.size == 0 or Y_test_raw_scaled.size == 0:
                self.logger.error("Test data is empty. Cannot proceed with evaluation.")
                return {
                    "evaluation": {},
                    "forecast": {},
                    "report": "Error: Test data empty.",
                }

            X_test_tensor, Y_test_tensor_scaled = self.prepare_sequences(
                X_test_raw, Y_test_raw_scaled
            )
            eval_res_run = self.evaluate_model(X_test_tensor, Y_test_tensor_scaled)

            seq_len_run = self.config["model"]["time_steps"]
            forecast_res_run = {
                "forecast_summary": {},
                "forecast_hourly": {},
                "raw_forecast_matrix": np.array([]),
                "timestamps_hourly": [],
            }

            if recent_data_X_for_forecast is None:
                if X_test_raw.shape[0] >= seq_len_run:
                    # Use the most recent `seq_len_run` samples from X_test_raw for forecast input
                    recent_data_X_for_forecast = X_test_raw[-seq_len_run:]
                    self.logger.info(
                        f"Using last {seq_len_run} steps from test data for forecast input, shape: {recent_data_X_for_forecast.shape}"
                    )
                else:
                    self.logger.warning(
                        f"Not enough X_test data ({X_test_raw.shape[0]} samples) to form a sequence of length {seq_len_run} for forecast. Forecast might fail or be inaccurate."
                    )

            if (
                recent_data_X_for_forecast is not None
                and recent_data_X_for_forecast.shape[0] >= seq_len_run
            ):
                # Ensure only the required seq_len is passed if a longer array was provided
                if recent_data_X_for_forecast.shape[0] > seq_len_run:
                    recent_data_X_for_forecast = recent_data_X_for_forecast[
                        -seq_len_run:
                    ]

                forecast_res_run = self.generate_72hour_forecast(
                    recent_data_X_for_forecast, timestamps=forecast_start_timestamps
                )
            else:
                self.logger.warning(
                    "Skipping 72h forecast: insufficient recent_data_X or it was not provided and test data too short."
                )

            self.plot_evaluation_results(eval_res_run)
            if forecast_res_run and forecast_res_run.get("forecast_summary"):
                self.plot_72hour_forecast(forecast_res_run)

            report_run = self.generate_forecast_report(eval_res_run, forecast_res_run)
            self.save_results(eval_res_run, forecast_res_run)
            self.print_summary(eval_res_run, forecast_res_run)
            self.logger.info("Complete evaluation pipeline finished successfully!")
            return {
                "evaluation": eval_res_run,
                "forecast": forecast_res_run,
                "report": report_run,
            }
        except Exception as e_run:
            self.logger.error(f"Error in evaluation pipeline: {e_run}", exc_info=True)
            return {
                "evaluation": {},
                "forecast": {},
                "report": f"Error in pipeline: {e_run}",
            }

    def print_summary(
        self, eval_res_sum: Dict[str, Any], forecast_res_sum: Dict[str, Any]
    ):
        overall_metrics_sum = (
            eval_res_sum.get("overall_metrics", {}) if eval_res_sum else {}
        )

        summary_str = (
            "\n"
            + "=" * 80
            + "\nPM2.5 MODEL EVALUATION & FORECAST SUMMARY (ORIGINAL SCALE)\n"
            + "=" * 80
        )
        summary_str += "\n\n📊 MODEL PERFORMANCE (1st Forecast Step vs Actuals):\n"
        if overall_metrics_sum:
            summary_str += (
                f"   R²: {overall_metrics_sum.get('R2',np.nan):.4f}\n"
                f"   RMSE: {overall_metrics_sum.get('RMSE',np.nan):.2f} µg/m³\n"
                f"   MAE: {overall_metrics_sum.get('MAE',np.nan):.2f} µg/m³\n"
                f"   MAPE: {overall_metrics_sum.get('MAPE',np.nan):.2f}%\n"
            )
            r2_sum = overall_metrics_sum.get("R2", np.nan)
            quality_sum = "⚪ Undet."
            if not np.isnan(r2_sum):
                if r2_sum >= 0.75:
                    quality_sum = "🟢 EXCELLENT"
                elif r2_sum >= 0.6:
                    quality_sum = "🟡 GOOD"
                elif r2_sum >= 0.4:
                    quality_sum = "🟠 FAIR"
                else:
                    quality_sum = "🔴 NEEDS IMPROVEMENT"
            summary_str += f"   Model Quality (R² based): {quality_sum}\n"
        else:
            summary_str += "   Evaluation metrics not available.\n"

        forecast_summary_points = (
            forecast_res_sum.get("forecast_summary", {}) if forecast_res_sum else {}
        )
        forecast_hourly_data = (
            forecast_res_sum.get("forecast_hourly", {}) if forecast_res_sum else {}
        )

        summary_str += "\n🔮 72-HOUR FORECAST:\n"
        if forecast_summary_points:
            all_hourly_forecast_values = []
            if forecast_hourly_data:
                for station_data in forecast_hourly_data.values():
                    all_hourly_forecast_values.extend(
                        [v for v in station_data.values() if not np.isnan(v)]
                    )

            if all_hourly_forecast_values:
                avg_f_sum = np.mean(all_hourly_forecast_values)
                max_f_sum = np.max(all_hourly_forecast_values)
                summary_str += (
                    f"   Avg PM2.5 (all stations, all hours): {avg_f_sum:.1f} µg/m³\n"
                    f"   Max PM2.5 (overall): {max_f_sum:.1f} µg/m³\n"
                )
                risk_map = {
                    (0, 15): "🟢 Good",
                    (15.01, 25): "🟡 Moderate",
                    (25.01, 37.5): "🟠 Unh. Sens.",
                    (37.51, 50): "🔴 Unhealthy",
                    (50.01, float("inf")): "🟣 V. Unh.",
                }
                overall_risk_str = "⚪ Undet."
                for r, label in risk_map.items():
                    if r[0] <= max_f_sum <= r[1]:
                        overall_risk_str = label
                        break
                summary_str += (
                    f"   Overall Highest Risk (based on max): {overall_risk_str}\n"
                )
            else:
                summary_str += (
                    "   Hourly forecast data insufficient for overall summary.\n"
                )

            st_max_f_sum = {}
            if forecast_hourly_data:
                for st_s, hourly_data in forecast_hourly_data.items():
                    valid_st_f = [v for v in hourly_data.values() if not np.isnan(v)]
                    st_max_f_sum[st_s] = max(valid_st_f) if valid_st_f else np.nan

            if st_max_f_sum:
                top_risk_sum = sorted(
                    st_max_f_sum.items(),
                    key=lambda x_s: x_s[1] if not np.isnan(x_s[1]) else -float("inf"),
                    reverse=True,
                )[:3]
                summary_str += (
                    "\n⚠️  TOP RISK STATIONS (Max Hourly Forecast over 72h):\n"
                )
                for i_s, (st_s_name, val_s_risk) in enumerate(top_risk_sum):
                    station_risk_str = "⚪ Undet."
                    if not np.isnan(val_s_risk):
                        for r, label in risk_map.items():
                            if r[0] <= val_s_risk <= r[1]:
                                station_risk_str = label
                                break
                    summary_str += f"   {i_s+1}. {st_s_name}: {val_s_risk:.1f} µg/m³ ({station_risk_str})\n"
        else:
            summary_str += "   Forecast data not available.\n"
        summary_str += f"\n📁 Results saved to: {self.output_dir}\n" + "=" * 80
        print(summary_str)


def main():
    script_dir_main = Path(__file__).resolve().parent
    project_root_main_prog = script_dir_main.parents[0]

    default_model_dir_main = project_root_main_prog / "models" / "improved"
    default_config_path_main = (
        project_root_main_prog / "configs" / "improved_model_config.yaml"
    )

    parser = argparse.ArgumentParser(description="PM2.5 Model Evaluator")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(default_model_dir_main),
        help="Directory containing the trained model and scalers.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(default_config_path_main),
        help="Path to the model/training configuration YAML file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Override model_type from config (optional).",
    )
    parser.add_argument(
        "--forecast_data_path",
        type=str,
        default=None,
        help="Path to a .pkl file containing recent X data for generating a new forecast. Should be a NumPy array of shape (N, num_features).",
    )

    args = parser.parse_args()

    logger_main = logging.getLogger(__name__)
    logger_main.info(
        f"--- Running Evaluation ---\nUsing Model Dir: {args.model_dir}\nUsing Config: {args.config_path}\nUsing Model Type: {args.model_type if args.model_type else 'From Config'}"
    )

    recent_data_for_forecast_np = None
    if args.forecast_data_path:
        try:
            forecast_data_file = Path(args.forecast_data_path)
            if not forecast_data_file.is_absolute():
                forecast_data_file = project_root_main_prog / forecast_data_file

            with open(forecast_data_file, "rb") as f:
                recent_data_for_forecast_np = pickle.load(f)
            if not isinstance(recent_data_for_forecast_np, np.ndarray):
                logger_main.error(
                    f"Forecast data at {forecast_data_file} is not a NumPy array. Type: {type(recent_data_for_forecast_np)}"
                )
                recent_data_for_forecast_np = None
            else:
                logger_main.info(
                    f"Loaded recent data for forecast from {forecast_data_file}, shape: {recent_data_for_forecast_np.shape}"
                )
        except Exception as e_forecast_load:
            logger_main.error(
                f"Could not load recent data for forecast from {args.forecast_data_path}: {e_forecast_load}"
            )
            recent_data_for_forecast_np = None

    try:
        evaluator_main = PM25ModelEvaluator(
            model_dir=args.model_dir,
            config_path=args.config_path,
            model_type_override=args.model_type,
        )
        evaluator_main.run_complete_evaluation(
            recent_data_X_for_forecast=recent_data_for_forecast_np
        )
        logger_main.info(
            f"\n✅ Evaluation completed successfully!\n📊 Check results in: {evaluator_main.output_dir}"
        )
    except FileNotFoundError as fnf_main:
        logger_main.error(
            f"❌ FILE NOT FOUND: {fnf_main}\nEnsure model/scalers/data paths are correct and files exist.",
            exc_info=True,
        )
    except ValueError as ve_main:
        logger_main.error(f"❌ VALUE ERROR: {ve_main}", exc_info=True)
    except Exception as e_main:
        logger_main.error(
            f"❌ UNEXPECTED ERROR in evaluation pipeline: {e_main}", exc_info=True
        )


if __name__ == "__main__":
    main()
