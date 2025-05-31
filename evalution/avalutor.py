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
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # For type checking
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json  # Added for saving forecast as JSON

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")  # Consider updating if seaborn version causes issues
sns.set_palette("husl")

# Add the src directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.model import PM25Model, create_enhanced_model_configs


class PM25ModelEvaluator:
    """Comprehensive PM2.5 model evaluation and forecasting system."""

    def __init__(self, model_dir: str, config_path: str, model_type: str = None):
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.config = self._load_config()
        self.scalers = (
            self._load_scalers()
        )  # This now contains station-specific scalers
        # Determine station names: from config, or infer from scaler keys, or default
        self.station_names = self.config.get("data", {}).get("station_names")
        if not self.station_names:
            if (
                self.scalers
                and isinstance(self.scalers, dict)
                and all(isinstance(k, str) for k in self.scalers.keys())
            ):
                self.station_names = sorted(
                    list(self.scalers.keys())
                )  # Sort for consistency
                self.logger.info(
                    f"Inferred station names from scaler keys: {self.station_names}"
                )
            else:
                num_s = self.config.get("model", {}).get("num_stations", 12)
                self.station_names = [f"Station_{i+1}" for i in range(num_s)]
                self.logger.info(f"Using default station names for {num_s} stations.")

        num_stations_cfg = self.config.get("model", {}).get("num_stations", 12)
        if len(self.station_names) != num_stations_cfg:
            self.logger.warning(
                f"Number of station names derived ({len(self.station_names)}) "
                f"does not match num_stations in model config ({num_stations_cfg}). "
                f"This might lead to issues if model output or scaler mapping is misaligned. "
                f"Using derived station_names list of length {len(self.station_names)}."
            )

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
                raise FileNotFoundError(
                    f"Adjacency matrix not found: {adj_matrix_path}"
                )
            adj_matrix = torch.FloatTensor(np.load(adj_matrix_path)).to(self.device)

            model_config = self.config["model"].copy()
            if self.model_type:
                try:
                    enhanced_configs = create_enhanced_model_configs()
                    if self.model_type in enhanced_configs:
                        self.logger.info(
                            f"Using enhanced configuration for {self.model_type}"
                        )
                        model_config.update(enhanced_configs[self.model_type])
                except Exception as e:
                    self.logger.warning(f"Could not apply enhanced configs: {e}")

            model_num_stations = len(self.station_names)
            if model_config["num_stations"] != model_num_stations:
                self.logger.warning(
                    f"Model config num_stations ({model_config['num_stations']}) "
                    f"differs from derived station_names length ({model_num_stations}). "
                    f"Initializing model with {model_num_stations} stations based on station_names list."
                )
                model_config["num_stations"] = model_num_stations

            model = PM25Model(
                time_steps=int(model_config["time_steps"]),
                num_stations=int(model_config["num_stations"]),
                input_features=int(model_config["input_features"]),
                adj_matrix=adj_matrix,
                forecast_horizon=int(model_config["forecast_horizon"]),
                hidden_dims=int(model_config["hidden_dims"]),
                num_heads=int(model_config["num_heads"]),
                num_layers=int(model_config["num_layers"]),
                dropout_rate=float(model_config["dropout_rate"]),
                wavelet_type=self.config.get("wavelet", {}).get("wavelet_type", "db4"),
                wavelet_level=int(self.config.get("wavelet", {}).get("level", 3)),
                scalers=self.scalers,
                station_names=self.station_names,
                model_type=(
                    self.model_type
                    if self.model_type
                    else model_config.get("model_type", "hybrid")
                ),
                use_physics_guidance=model_config.get("use_physics_guidance", True),
                use_attention_regularization=model_config.get(
                    "use_attention_regularization", True
                ),
                kernel_regularization=float(
                    model_config.get("kernel_regularization", 1e-4)
                ),
                recurrent_regularization=float(
                    model_config.get("recurrent_regularization", 1e-4)
                ),
            ).to(self.device)

            model_path = self.model_dir / "model_best.pt"
            if not model_path.exists():
                model_path = self.model_dir / "model_final.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found in {self.model_dir}"
                )

            state_dict = torch.load(model_path, map_location=self.device)
            if "positional_encoding" in state_dict:
                pe_ckpt, pe_expected = (
                    state_dict["positional_encoding"],
                    model.positional_encoding.shape,
                )
                self.logger.info(
                    f"PE ckpt: {pe_ckpt.shape}, current model PE: {pe_expected}"
                )
                if pe_ckpt.ndim == 2 and len(pe_expected) == 4:
                    ts, hd = int(model_config["time_steps"]), int(
                        model_config["hidden_dims"]
                    )
                    if (
                        pe_ckpt.shape[0] == ts
                        and pe_ckpt.shape[1] == hd
                        and pe_expected[1] == ts
                        and pe_expected[3] == hd
                    ):
                        self.logger.info(
                            f"Adapting PE shape from {pe_ckpt.shape} to {pe_expected}."
                        )
                        state_dict["positional_encoding"] = pe_ckpt.unsqueeze(
                            0
                        ).unsqueeze(2)
                    else:
                        self.logger.warning("PE shape/config mismatch for reshape.")
                elif pe_ckpt.shape != pe_expected:
                    self.logger.warning("PE shape mismatch, no adaptation.")

            model.load_state_dict(state_dict, strict=False)
            model.eval()
            self.logger.info(f"Model loaded from {model_path}")
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
                test_X, test_Y = test_data_loaded["X"], test_data_loaded["Y"]
                if "df" in test_data_loaded:
                    full_df = test_data_loaded["df"]
            elif isinstance(test_data_loaded, pd.DataFrame):
                full_df, target_col = test_data_loaded, "PM2.5"
                if target_col not in full_df.columns:
                    raise ValueError(f"Target '{target_col}' not in DataFrame.")
                feature_cols_path = self.model_dir / "feature_cols.pkl"
                if feature_cols_path.exists():
                    with open(feature_cols_path, "rb") as f_cols:
                        feature_cols = pickle.load(f_cols)
                    self.logger.info(f"Loaded feature_cols from {feature_cols_path}")
                else:
                    feature_cols = [
                        c
                        for c in full_df.columns
                        if c not in [target_col, "datetime", "station"]
                    ]
                    self.logger.warning(
                        f"feature_cols.pkl not found in {self.model_dir}. Inferring: {len(feature_cols)} features."
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
        num_stations_for_Y_output = len(self.station_names)
        num_stations_model_expects_X = self.config["model"]["num_stations"]

        if len(X) < seq_len:
            self.logger.error(f"Need {seq_len} X samples, got {len(X)}")
            return torch.empty(0), torch.empty(0)
        X_s, Y_s = [X[i : i + seq_len] for i in range(len(X) - seq_len + 1)], [
            Y[i + seq_len - 1] for i in range(len(Y) - seq_len + 1)
        ]
        if not X_s:
            return torch.empty(0), torch.empty(0)
        X_s, Y_s = np.array(X_s, dtype=np.float32), np.array(Y_s, dtype=np.float32)

        if X_s.ndim == 3:
            X_s = np.tile(
                np.expand_dims(X_s, axis=2), (1, 1, num_stations_model_expects_X, 1)
            )
        elif X_s.ndim == 4 and X_s.shape[2] != num_stations_model_expects_X:
            self.logger.warning(
                f"X_seq has {X_s.shape[2]} stations, model expects {num_stations_model_expects_X}. Adapting."
            )
            if X_s.shape[2] == 1:
                X_s = np.tile(X_s, (1, 1, num_stations_model_expects_X, 1))
            else:
                X_s = X_s[:, :, :num_stations_model_expects_X, :]

        if Y_s.ndim == 1:
            Y_s = Y_s.reshape(-1, 1)
        if Y_s.ndim == 2:
            if Y_s.shape[1] == 1:
                Y_s = np.repeat(
                    Y_s[:, np.newaxis, :], num_stations_for_Y_output, axis=1
                )
            elif Y_s.shape[1] == num_stations_for_Y_output:
                Y_s = np.expand_dims(Y_s, axis=2)
            else:
                self.logger.error(
                    f"Y_s has shape {Y_s.shape} not compatible for num_stations_for_Y_output={num_stations_for_Y_output}"
                )
                Y_s = np.repeat(
                    Y_s[:, :1, np.newaxis], num_stations_for_Y_output, axis=1
                )

        X_t, Y_t = torch.FloatTensor(X_s).to(self.device), torch.FloatTensor(Y_s).to(
            self.device
        )
        self.logger.info(f"Prepared sequences - X: {X_t.shape}, Y: {Y_t.shape}")
        return X_t, Y_t

    def evaluate_model(
        self, X_test_tensor: torch.Tensor, Y_test_tensor_scaled: torch.Tensor
    ) -> Dict[str, Any]:
        self.logger.info("Starting model evaluation...")
        default_metrics = {k: np.nan for k in ["MAE", "RMSE", "R2", "MAPE"]}
        if X_test_tensor.nelement() == 0:
            return {
                "predictions_unscaled": np.array([]),
                "actuals_unscaled": np.array([]),
                "station_metrics": {},
                "overall_metrics": default_metrics,
            }

        predictions_unscaled_list, actuals_scaled_list = [], []
        batch_size = min(
            self.config.get("training", {}).get("batch_size", 32), len(X_test_tensor)
        )
        if batch_size == 0 and len(X_test_tensor) > 0:
            batch_size = len(X_test_tensor)

        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                unscaled_pred_batch = self.model(X_test_tensor[i : i + batch_size])
                predictions_unscaled_list.append(unscaled_pred_batch.cpu().numpy())
                actuals_scaled_list.append(
                    Y_test_tensor_scaled[i : i + batch_size].cpu().numpy()
                )

        if not predictions_unscaled_list:
            self.logger.warning("No predictions generated.")
            return {
                "predictions_unscaled": np.array([]),
                "actuals_unscaled": np.array([]),
                "station_metrics": {},
                "overall_metrics": default_metrics,
            }

        predictions_unscaled_np = np.concatenate(predictions_unscaled_list, axis=0)
        actuals_scaled_np = np.concatenate(actuals_scaled_list, axis=0)

        actuals_unscaled_np = np.copy(actuals_scaled_np)
        unscaling_success_count = 0

        num_stations_in_actuals = actuals_scaled_np.shape[1]

        if num_stations_in_actuals != len(self.station_names):
            self.logger.warning(
                f"Mismatch: actuals data stations ({num_stations_in_actuals}) != configured station_names ({len(self.station_names)})."
            )

        stations_to_process_unscaling = min(
            num_stations_in_actuals, len(self.station_names)
        )

        for station_idx in range(stations_to_process_unscaling):
            station_name = self.station_names[station_idx]
            scaler_obj = self.scalers.get(station_name)

            if scaler_obj:
                is_standard_scaler = hasattr(scaler_obj, "scale_") and hasattr(
                    scaler_obj, "mean_"
                )
                is_minmax_scaler = hasattr(scaler_obj, "data_min_") and hasattr(
                    scaler_obj, "data_max_"
                )  # Simpler check for MinMaxScaler attributes

                pm25_feature_index_in_scaler = 0

                try:
                    current_station_actuals_scaled = actuals_scaled_np[
                        :, station_idx, :
                    ]

                    if (
                        is_standard_scaler
                        and len(scaler_obj.scale_) > pm25_feature_index_in_scaler
                    ):
                        scale_pm25 = scaler_obj.scale_[pm25_feature_index_in_scaler]
                        mean_pm25 = scaler_obj.mean_[pm25_feature_index_in_scaler]
                        actuals_unscaled_np[:, station_idx, :] = (
                            current_station_actuals_scaled * scale_pm25 + mean_pm25
                        )
                        unscaling_success_count += 1
                    elif (
                        is_minmax_scaler
                        and len(scaler_obj.data_min_) > pm25_feature_index_in_scaler
                    ):  # Check data_min_ length
                        data_min_pm25 = scaler_obj.data_min_[
                            pm25_feature_index_in_scaler
                        ]
                        data_range_pm25 = (
                            scaler_obj.data_max_[pm25_feature_index_in_scaler]
                            - data_min_pm25
                        )
                        if data_range_pm25 == 0:
                            actuals_unscaled_np[:, station_idx, :] = data_min_pm25
                        else:
                            actuals_unscaled_np[:, station_idx, :] = (
                                current_station_actuals_scaled * data_range_pm25
                                + data_min_pm25
                            )
                        unscaling_success_count += 1
                    else:
                        self.logger.warning(
                            f"Scaler for '{station_name}' not recognized or PM2.5 index out of bounds. Actuals remain scaled."
                        )
                except IndexError:
                    self.logger.error(
                        f"Scaler for '{station_name}' (PM2.5 index {pm25_feature_index_in_scaler}) missing attributes. Unscaling failed."
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error manually unscaling actuals for '{station_name}': {e}."
                    )
            else:
                self.logger.warning(
                    f"Scaler for '{station_name}' not found. Actuals remain scaled."
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
                f"Successfully unscaled actuals for {unscaling_success_count}/{stations_to_process_unscaling} stations. Others remain scaled."
            )
        else:
            self.logger.error(
                "Failed to unscale actual Y_test values for any station. Metrics will be based on scaled actuals vs unscaled predictions."
            )

        self.logger.info(f"Unscaled Predictions shape: {predictions_unscaled_np.shape}")
        self.logger.info(
            f"Processed Actuals (attempted unscale) shape: {actuals_unscaled_np.shape}"
        )

        station_metrics = {}
        for st_idx in range(stations_to_process_unscaling):
            st_name = self.station_names[st_idx]
            pred_st1 = predictions_unscaled_np[:, st_idx, 0]
            actual_st = actuals_unscaled_np[:, st_idx, 0]
            valid = ~np.isnan(actual_st) & ~np.isnan(pred_st1)
            if not np.any(valid):
                mae, rmse, r2, mape = np.nan, np.nan, np.nan, np.nan
            else:
                act_f, pred_f = actual_st[valid], pred_st1[valid]
                r2 = r2_score(act_f, pred_f) if len(act_f) >= 2 else np.nan
                mae, rmse = mean_absolute_error(act_f, pred_f), np.sqrt(
                    mean_squared_error(act_f, pred_f)
                )
                nz_act = act_f != 0
                mape = (
                    np.mean(
                        np.abs(
                            (act_f[nz_act] - pred_f[nz_act]) / (act_f[nz_act] + 1e-8)
                        )
                    )
                    * 100
                    if np.any(nz_act)
                    else np.nan
                )
            station_metrics[st_name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape,
            }

        overall_pred = predictions_unscaled_np[
            :, :stations_to_process_unscaling, 0
        ].reshape(-1)
        overall_actual = actuals_unscaled_np[
            :, :stations_to_process_unscaling, 0
        ].reshape(-1)
        valid_ovr = ~np.isnan(overall_actual) & ~np.isnan(overall_pred)
        overall_metrics = default_metrics.copy()
        if np.any(valid_ovr):
            act_o, pred_o = overall_actual[valid_ovr], overall_pred[valid_ovr]
            if len(act_o) >= 2:
                overall_metrics["R2"] = r2_score(act_o, pred_o)
            overall_metrics["MAE"], overall_metrics["RMSE"] = mean_absolute_error(
                act_o, pred_o
            ), np.sqrt(mean_squared_error(act_o, pred_o))
            nz_o_act = act_o != 0
            if np.any(nz_o_act):
                overall_metrics["MAPE"] = (
                    np.mean(
                        np.abs(
                            (act_o[nz_o_act] - pred_o[nz_o_act])
                            / (act_o[nz_o_act] + 1e-8)
                        )
                    )
                    * 100
                )

        results = {
            "predictions_unscaled": predictions_unscaled_np,
            "actuals_unscaled": actuals_unscaled_np,
            "station_metrics": station_metrics,
            "overall_metrics": overall_metrics,
        }
        self.logger.info(
            f"Overall R2: {overall_metrics.get('R2',np.nan):.4f}, RMSE: {overall_metrics.get('RMSE',np.nan):.4f}"
        )
        return results

    def generate_72hour_forecast(
        self, recent_data_X: np.ndarray, timestamps: List[datetime] = None
    ) -> Dict[str, Any]:
        self.logger.info("Generating 72-hour forecast...")
        seq_len = self.config["model"]["time_steps"]
        num_stations_cfg = self.config["model"]["num_stations"]

        if recent_data_X.shape[0] < seq_len:
            raise ValueError(f"Need {seq_len} X samples for forecast.")
        init_seq_X = recent_data_X[-seq_len:]
        curr_seq_X_np = np.expand_dims(init_seq_X, axis=0)

        if curr_seq_X_np.ndim == 3:
            curr_seq_X_np = np.expand_dims(curr_seq_X_np, axis=2)
            curr_seq_X_np = np.tile(curr_seq_X_np, (1, 1, num_stations_cfg, 1))
        elif curr_seq_X_np.ndim == 4 and curr_seq_X_np.shape[2] == 1:
            curr_seq_X_np = np.tile(curr_seq_X_np, (1, 1, num_stations_cfg, 1))
        elif not (
            curr_seq_X_np.ndim == 4
            and curr_seq_X_np.shape[0] == 1
            and curr_seq_X_np.shape[2] == num_stations_cfg
        ):
            raise ValueError(
                f"recent_data_X shape {init_seq_X.shape} not compatible for forecast."
            )

        curr_seq_tensor = torch.FloatTensor(curr_seq_X_np).to(self.device)

        with torch.no_grad():
            forecast_unscaled_tensor = self.model(curr_seq_tensor)
        forecast_unscaled_np = forecast_unscaled_tensor.cpu().numpy()[0]

        self.logger.info(
            f"Raw forecast (unscaled) shape from model: {forecast_unscaled_np.shape}"
        )

        model_output_stations, model_horizon = (
            forecast_unscaled_np.shape[0],
            forecast_unscaled_np.shape[1],
        )
        num_target_stations_output = len(self.station_names)
        forecast_72h = np.full((num_target_stations_output, 72), np.nan)
        stations_to_fill, horizon_to_fill = min(
            model_output_stations, num_target_stations_output
        ), min(model_horizon, 72)
        forecast_72h[:stations_to_fill, :horizon_to_fill] = forecast_unscaled_np[
            :stations_to_fill, :horizon_to_fill
        ]

        if model_horizon < 72 and model_horizon > 0:
            self.logger.warning(f"Model horizon ({model_horizon}) < 72h. Padding.")
            padding_needed = 72 - model_horizon
            padding_values = np.tile(
                forecast_72h[:stations_to_fill, model_horizon - 1 : model_horizon],
                (1, padding_needed),
            )
            forecast_72h[:stations_to_fill, model_horizon:] = padding_values

        forecast_dict_hourly = {
            self.station_names[st_idx]: {
                f"{h+1}h": float(forecast_72h[st_idx, h]) for h in range(72)
            }
            for st_idx in range(num_target_stations_output)
        }
        forecast_dict_summary = {
            self.station_names[st_idx]: {
                "24h": float(forecast_72h[st_idx, 23]),
                "48h": float(forecast_72h[st_idx, 47]),
                "72h": float(forecast_72h[st_idx, 71]),
            }
            for st_idx in range(num_target_stations_output)
        }
        self.logger.info("72-hour forecast generated and processed.")
        forecast_timestamps_hourly = (
            [datetime.now() + timedelta(hours=i + 1) for i in range(72)]
            if timestamps is None
            else timestamps[:72]
        )

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
        station_metrics = results["station_metrics"]
        overall_metrics = results["overall_metrics"]

        if predictions_unscaled.size == 0 or actuals_unscaled.size == 0:
            self.logger.warning("No data for eval plots.")
            return

        overall_pred_first_step = predictions_unscaled[:, :, 0].reshape(-1)
        overall_actual_vals = actuals_unscaled[:, :, 0].reshape(-1)
        valid_indices_plot = ~np.isnan(overall_actual_vals) & ~np.isnan(
            overall_pred_first_step
        )
        overall_actual_plot, overall_pred_plot = (
            overall_actual_vals[valid_indices_plot],
            overall_pred_first_step[valid_indices_plot],
        )

        if len(overall_actual_plot) == 0:
            self.logger.warning("No valid data for overall eval plots.")
            plt.close()
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "PM2.5 Model Evaluation Summary (Original Scale)",
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
        min_p, max_p = (
            (
                min(overall_actual_plot.min(), overall_pred_plot.min()),
                max(overall_actual_plot.max(), overall_pred_plot.max()),
            )
            if len(overall_actual_plot) > 0
            else (0, 1)
        )
        axes[0, 0].plot([min_p, max_p], [min_p, max_p], "r--", lw=2)
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
            sns.kdeplot(residuals, ax=axes[1, 0], color="red", warn_singular=False)
        else:
            axes[1, 0].text(0.5, 0.5, "No residuals", ha="center", va="center")
        axes[1, 0].set_xlabel("Residuals (µg/m³)")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Error Distribution")
        axes[1, 0].axvline(x=0, color="r", ls="--")
        axes[1, 0].grid(True, alpha=0.3)
        m_names, m_values = ["MAE", "RMSE", "MAPE"], [
            overall_metrics.get(k, np.nan) for k in ["MAE", "RMSE", "MAPE"]
        ]
        valid_m_idx = [i for i, v in enumerate(m_values) if not np.isnan(v)]
        plot_m_n, plot_m_v = [m_names[i] for i in valid_m_idx], [
            m_values[i] for i in valid_m_idx
        ]
        if plot_m_v:
            axes[1, 1].bar(
                plot_m_n, plot_m_v, color=["skyblue", "lightcoral", "lightgreen"]
            )
            axes[1, 1].set_ylabel("Error Value")
            axes[1, 1].set_title("Overall Metrics")
            axes[1, 1].grid(True, alpha=0.3)
            for i, v_b in enumerate(plot_m_v):
                axes[1, 1].text(
                    i,
                    v_b + max(plot_m_v, default=0) * 0.01,
                    f"{v_b:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )  # Changed fw to fontweight
        else:
            axes[1, 1].text(0.5, 0.5, "Metrics NaN", ha="center", va="center")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            self.output_dir / "evaluation_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        self.plot_station_performance(station_metrics)
        self.plot_time_series_comparison(predictions_unscaled, actuals_unscaled)
        self.plot_advanced_analysis(predictions_unscaled, actuals_unscaled)

    def plot_station_performance(self, station_metrics: Dict[str, Dict[str, float]]):
        if not station_metrics:
            self.logger.warning("No station metrics to plot.")
            return
        stations, metrics_to_plot = list(station_metrics.keys()), [
            "MAE",
            "RMSE",
            "R2",
            "MAPE",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Station-wise Performance (Original Scale)", fontsize=16, fontweight="bold"
        )
        axes = axes.flatten()
        for i, metric_name in enumerate(metrics_to_plot):
            values = [station_metrics[st].get(metric_name, np.nan) for st in stations]
            valid_idx = [j for j, v_plot in enumerate(values) if not np.isnan(v_plot)]
            plot_st, plot_val = [stations[j] for j in valid_idx], [
                values[j] for j in valid_idx
            ]
            if not plot_val:
                axes[i].text(
                    0.5,
                    0.5,
                    f"{metric_name}\n(No valid data)",
                    ha="center",
                    va="center",
                )
                axes[i].set_title(f"{metric_name} by Station")
                continue
            bars = axes[i].bar(
                range(len(plot_st)),
                plot_val,
                color=plt.cm.viridis(np.linspace(0, 1, len(plot_st))),
            )
            axes[i].set_title(f"{metric_name} by Station")
            axes[i].set_xlabel("Station")
            axes[i].set_ylabel(metric_name)
            axes[i].set_xticks(range(len(plot_st)))
            axes[i].set_xticklabels(plot_st, rotation=45, ha="right", fontsize=8)
            axes[i].grid(True, alpha=0.3, axis="y")
            max_v_text_plot = max(plot_val, default=0)
            for bar_idx, (bar, val_bar) in enumerate(zip(bars, plot_val)):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    val_bar + max_v_text_plot * 0.02,
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
        indices_plot = np.sort(
            np.random.choice(
                n_samples_avail, min(sample_size, n_samples_avail), replace=False
            )
        )
        pred_samp_plot, actual_samp_plot = (
            pred_unscaled[indices_plot, :, 0],
            actual_unscaled[indices_plot, :, 0],
        )
        n_stations_plot_ts = min(4, pred_unscaled.shape[1])
        if n_stations_plot_ts == 0:
            self.logger.warning("No stations for time series plot.")
            return
        fig, axes = plt.subplots(
            n_stations_plot_ts, 1, figsize=(15, 4 * n_stations_plot_ts), sharex=True
        )
        fig.suptitle(
            "Time Series: Predicted (1st step) vs Actual (Original Scale)",
            fontsize=16,
            fontweight="bold",
        )
        if n_stations_plot_ts == 1:
            axes = [axes]
        for i in range(n_stations_plot_ts):
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
        axes[-1].set_xlabel("Time Steps (Sampled)")
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
        pred_flat_adv, actual_flat_adv = pred_unscaled[:, :, 0].reshape(
            -1
        ), actual_unscaled[:, :, 0].reshape(-1)
        valid_idx_adv = ~np.isnan(actual_flat_adv) & ~np.isnan(pred_flat_adv)
        actual_plot_adv, pred_plot_adv = (
            actual_flat_adv[valid_idx_adv],
            pred_flat_adv[valid_idx_adv],
        )
        if len(actual_plot_adv) < 2:
            self.logger.warning("Not enough valid data for advanced plot.")
            return
        errors_adv = pred_plot_adv - actual_plot_adv
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Advanced Error Analysis (Original Scale)", fontsize=16, fontweight="bold"
        )
        axes[0, 0].scatter(
            actual_plot_adv, np.abs(errors_adv), alpha=0.5, s=10, edgecolors="k", lw=0.5
        )
        axes[0, 0].set_xlabel("Actual PM2.5 (µg/m³)")
        axes[0, 0].set_ylabel("Abs Error (µg/m³)")
        axes[0, 0].set_title("Error vs Concentration")
        axes[0, 0].grid(True, alpha=0.3)
        try:
            bins_adv = np.unique(np.percentile(actual_plot_adv, [0, 25, 50, 75, 100]))
            if len(bins_adv) < 2:
                axes[0, 1].text(
                    0.5, 0.5, "Not enough data for bins", ha="center", va="center"
                )
            else:
                bin_labels_adv, bin_errors_adv = [], []
                for i in range(len(bins_adv) - 1):
                    mask_adv = (actual_plot_adv >= bins_adv[i]) & (
                        actual_plot_adv < bins_adv[i + 1]
                        if i < len(bins_adv) - 2
                        else actual_plot_adv <= bins_adv[i + 1]
                    )
                    if np.any(mask_adv):
                        bin_errors_adv.append(np.abs(errors_adv[mask_adv]).mean())
                        bin_labels_adv.append(f"{bins_adv[i]:.1f}-{bins_adv[i+1]:.1f}")
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
            axes[0, 1].set_ylabel("Mean Abs Error (µg/m³)")
            axes[0, 1].set_title("Error by Concentration")
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
                0.5, 0.5, "Not enough data for Q-Q", ha="center", va="center"
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
            axes[1, 1].set_xlabel("Abs Error (µg/m³)")
            axes[1, 1].set_ylabel("Cumulative %")
            axes[1, 1].set_title("Cumulative Error Dist")
            axes[1, 1].grid(True, alpha=0.3)
            for p_val in [50, 75, 90, 95]:
                err_val_adv = np.percentile(np.abs(errors_adv), p_val)
                axes[1, 1].axvline(x=err_val_adv, color="red", ls="--", alpha=0.7, lw=1)
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
                "Not enough data for cumulative error",
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
        forecast_summary_plot, stations_plot = forecast_results[
            "forecast_summary"
        ], list(forecast_results["forecast_summary"].keys())
        if not stations_plot:
            self.logger.warning("No stations in forecast data.")
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
                    forecast_summary_plot[st_plot].get(h_plot, np.nan)
                    for h_plot in time_horizons_plot
                ]
                for st_plot in stations_plot
            ]
        )
        x_pos_plot, width_plot = np.arange(len(stations_plot)), 0.25
        for i, horizon_plot in enumerate(time_horizons_plot):
            bars_plot = axes[0].bar(
                x_pos_plot + i * width_plot - width_plot,
                forecast_data_plot[:, i],
                width_plot,
                label=horizon_plot,
                alpha=0.8,
            )
            for bar_idx_plot in range(len(bars_plot)):
                bar_val_plot = forecast_data_plot[bar_idx_plot, i]
                if not np.isnan(bar_val_plot):
                    axes[0].text(
                        bars_plot[bar_idx_plot].get_x()
                        + bars_plot[bar_idx_plot].get_width() / 2,
                        bar_val_plot + 0.5,
                        f"{bar_val_plot:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=45,
                    )
        axes[0].set_xlabel("Stations", fontsize=12)
        axes[0].set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=12)
        axes[0].set_title("PM2.5 Forecast (24h,48h,72h points)", fontsize=14)
        axes[0].set_xticks(x_pos_plot)
        axes[0].set_xticklabels(stations_plot, rotation=45, ha="right", fontsize=9)
        axes[0].legend(title="Forecast Horizon")
        axes[0].grid(True, alpha=0.3, axis="y")
        forecast_hourly_plot = forecast_results.get("forecast_hourly")
        if forecast_hourly_plot:
            n_st_trend_plot = min(len(stations_plot), 5)
            hourly_tp_plot = [f"{h+1}h" for h in range(72)]
            for i, st_name_trend in enumerate(stations_plot[:n_st_trend_plot]):
                hourly_val_trend = [
                    forecast_hourly_plot[st_name_trend].get(tp_hr, np.nan)
                    for tp_hr in hourly_tp_plot
                ]
                axes[1].plot(
                    range(1, 73),
                    hourly_val_trend,
                    marker=".",
                    ls="-",
                    label=st_name_trend,
                    lw=1.5,
                    ms=4,
                )
            axes[1].set_xlabel("Forecast Hour Ahead", fontsize=12)
            axes[1].set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=12)
            axes[1].set_title(
                f"Hourly Forecast Trend (First {n_st_trend_plot} Stations)", fontsize=14
            )
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xticks(np.arange(0, 73, 6))
            axes[1].set_xlim(0, 72)
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
        forecast_hourly_interactive, stations_interactive = forecast_results[
            "forecast_hourly"
        ], list(forecast_results["forecast_hourly"].keys())
        if not stations_interactive:
            self.logger.warning(
                "No stations in hourly forecast data for interactive plot."
            )
            return
        timestamps_hourly_interactive = forecast_results.get(
            "timestamps_hourly", [f"{h+1}h" for h in range(72)]
        )
        fig_interactive = make_subplots(
            rows=1, cols=1, subplot_titles=(["Hourly PM2.5 Forecast by Station"])
        )
        for st_name_interactive in stations_interactive:
            hourly_val_interactive = [
                forecast_hourly_interactive[st_name_interactive].get(f"{h+1}h", np.nan)
                for h in range(72)
            ]
            fig_interactive.add_trace(
                go.Scatter(
                    x=timestamps_hourly_interactive,
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
            xaxis_title="Forecast Time",
            yaxis_title="PM2.5 (µg/m³)",
            legend_title_text="Stations",
        )
        if timestamps_hourly_interactive and isinstance(
            timestamps_hourly_interactive[0], datetime
        ):
            fig_interactive.update_xaxes(tickformat="%Y-%m-%d %H:%M", row=1, col=1)
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
        overall_metrics_rep, station_metrics_rep, forecast_summary_rep = (
            eval_res_report.get("overall_metrics", {}),
            eval_res_report.get("station_metrics", {}),
            forecast_res_report.get("forecast_summary", {}),
        )
        report_str = f"# PM2.5 Model - Evaluation & Forecast (Original Scale)\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n## Model Performance\n"
        report_str += f"- R²: {overall_metrics_rep.get('R2',np.nan):.4f}\n- RMSE: {overall_metrics_rep.get('RMSE',np.nan):.2f} µg/m³\n- MAE: {overall_metrics_rep.get('MAE',np.nan):.2f} µg/m³\n- MAPE: {overall_metrics_rep.get('MAPE',np.nan):.2f}%\n"
        r2_rep = overall_metrics_rep.get("R2", np.nan)
        quality_rep = (
            "Undetermined"
            if np.isnan(r2_rep)
            else (
                "Excellent"
                if r2_rep >= 0.75
                else (
                    "Good"
                    if r2_rep >= 0.6
                    else ("Fair" if r2_rep >= 0.4 else "Needs Improvement")
                )
            )
        )
        report_str += f"**Overall Quality (R²)**: {quality_rep}\n\n### Station Performance (1st Step vs Actual)\n| Station | R² | RMSE | MAE | MAPE |\n|---|---|---|---|---|\n"
        for st_rep, met_rep in station_metrics_rep.items():
            report_str += f"| {st_rep} | {met_rep.get('R2',np.nan):.3f} | {met_rep.get('RMSE',np.nan):.2f} | {met_rep.get('MAE',np.nan):.2f} | {met_rep.get('MAPE',np.nan):.2f} |\n"
        report_str += "\n## 72-Hour PM2.5 Forecast (µg/m³)\n| Station | 24h | 48h | 72h | Max | Risk (Max) |\n|---|---|---|---|---|---|\n"

        def get_risk_rep(v_rep):
            return (
                "⚪ Undet."
                if np.isnan(v_rep)
                else (
                    "🟢 Good"
                    if v_rep <= 15
                    else (
                        "🟡 Mod."
                        if v_rep <= 25
                        else (
                            "🟠 Unh.Sens."
                            if v_rep <= 37.5
                            else ("🔴 Unh." if v_rep <= 50 else "🟣 V.Unh.")
                        )
                    )
                )
            )

        all_f_pts_rep = []
        for st_name_rep in self.station_names:
            f_v_rep = forecast_summary_rep.get(
                st_name_rep, {"24h": np.nan, "48h": np.nan, "72h": np.nan}
            )
            f24_rep, f48_rep, f72_rep = f_v_rep["24h"], f_v_rep["48h"], f_v_rep["72h"]
            all_f_pts_rep.extend([f24_rep, f48_rep, f72_rep])
            valid_f_rep = [
                v_f for v_f in [f24_rep, f48_rep, f72_rep] if not np.isnan(v_f)
            ]
            max_f_rep = max(valid_f_rep) if valid_f_rep else np.nan
            risk_rep = get_risk_rep(max_f_rep)
            report_str += f"| {st_name_rep} | {f24_rep:.1f} | {f48_rep:.1f} | {f72_rep:.1f} | {max_f_rep:.1f} | {risk_rep} |\n"
        report_str += "\n### Forecast Risk Summary\n"
        valid_all_f_rep = [v_all for v_all in all_f_pts_rep if not np.isnan(v_all)]
        if valid_all_f_rep:
            avg_f_rep, max_f_rep_overall = np.mean(valid_all_f_rep), np.max(
                valid_all_f_rep
            )
            report_str += f"- Avg PM2.5 (all station forecasts): {avg_f_rep:.1f} µg/m³\n- Max PM2.5: {max_f_rep_overall:.1f} µg/m³\n- Overall Highest Risk: {get_risk_rep(max_f_rep_overall)}\n"
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

        preds1_save, actuals1_save = (
            eval_res_save["predictions_unscaled"][:, :, 0],
            eval_res_save["actuals_unscaled"][:, :, 0],
        )
        data_save_dict = {"time_index": range(preds1_save.shape[0])}
        for i, st_name_save in enumerate(self.station_names):
            if i < preds1_save.shape[1]:
                if i < actuals1_save.shape[1]:
                    data_save_dict[f"{st_name_save}_actual"] = actuals1_save[:, i]
                else:
                    data_save_dict[f"{st_name_save}_actual"] = np.nan
                data_save_dict[f"{st_name_save}_pred_step1"] = preds1_save[:, i]

        pd.DataFrame(data_save_dict).to_csv(
            self.output_dir / "predictions_vs_actuals_step1.csv", index=False
        )

        if "forecast_hourly" in forecast_res_save:
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
                json.dumps(serializable_forecast_save, indent=2)
            )
        self.logger.info(f"Results successfully saved.")

    def run_complete_evaluation(
        self, recent_data_X_for_forecast: np.ndarray = None
    ) -> Dict[str, Any]:
        self.logger.info("Starting complete evaluation pipeline...")
        try:
            X_test, Y_test_scaled, _ = self.load_test_data()
            if X_test.size == 0 or Y_test_scaled.size == 0:
                self.logger.error("Test data empty.")
                return {
                    "evaluation": {},
                    "forecast": {},
                    "report": "Error: Test data empty.",
                }
            X_test_tensor, Y_test_scaled_tensor = self.prepare_sequences(
                X_test, Y_test_scaled
            )
            eval_res_run = self.evaluate_model(X_test_tensor, Y_test_scaled_tensor)

            seq_len_run = self.config["model"]["time_steps"]
            if recent_data_X_for_forecast is None:
                if len(X_test) >= seq_len_run:
                    recent_data_X_for_forecast = X_test[-seq_len_run:]
                else:
                    self.logger.warning(
                        f"Not enough X_test for forecast. Forecast might fail."
                    )

            forecast_res_run = {
                "forecast_summary": {},
                "forecast_hourly": {},
                "raw_forecast_matrix": np.array([]),
            }
            if (
                recent_data_X_for_forecast is not None
                and recent_data_X_for_forecast.shape[0] >= seq_len_run
            ):
                forecast_res_run = self.generate_72hour_forecast(
                    recent_data_X_for_forecast
                )
            else:
                self.logger.warning(
                    "Skipping 72h forecast: insufficient recent_data_X."
                )

            self.plot_evaluation_results(eval_res_run)
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
            raise

    def print_summary(
        self, eval_res_sum: Dict[str, Any], forecast_res_sum: Dict[str, Any]
    ):
        print(
            "\n"
            + "=" * 80
            + "\nPM2.5 MODEL EVALUATION & FORECAST SUMMARY (ORIGINAL SCALE)\n"
            + "=" * 80
        )
        overall_metrics_sum = eval_res_sum.get("overall_metrics", {})
        print(f"\n📊 MODEL PERFORMANCE (1st Step vs Actual):")
        print(
            f"   R²: {overall_metrics_sum.get('R2',np.nan):.4f}\n   RMSE: {overall_metrics_sum.get('RMSE',np.nan):.2f} µg/m³\n   MAE: {overall_metrics_sum.get('MAE',np.nan):.2f} µg/m³\n   MAPE: {overall_metrics_sum.get('MAPE',np.nan):.2f}%"
        )
        r2_sum = overall_metrics_sum.get("R2", np.nan)
        quality_sum = (
            "⚪ Undet."
            if np.isnan(r2_sum)
            else (
                "🟢 EXCELLENT"
                if r2_sum >= 0.75
                else (
                    "🟡 GOOD"
                    if r2_sum >= 0.6
                    else ("🟠 FAIR" if r2_sum >= 0.4 else "🔴 NEEDS IMPROVEMENT")
                )
            )
        )
        print(f"   Model Quality (R²): {quality_sum}")

        forecast_summary_sum = forecast_res_sum.get("forecast_summary", {})
        if forecast_summary_sum:
            all_f_pts_sum = []
            [
                all_f_pts_sum.extend(
                    [sd_sum.get(h_sum, np.nan) for h_sum in ["24h", "48h", "72h"]]
                )
                for sd_sum in forecast_summary_sum.values()
            ]
            valid_f_sum = [v_sum for v_sum in all_f_pts_sum if not np.isnan(v_sum)]
            if valid_f_sum:
                avg_f_sum, max_f_sum = np.mean(valid_f_sum), np.max(valid_f_sum)
                print(
                    f"\n🔮 72-HOUR FORECAST (Summary Points):\n   Avg PM2.5: {avg_f_sum:.1f} µg/m³\n   Max PM2.5: {max_f_sum:.1f} µg/m³"
                )
                print(
                    f"   Overall Highest Risk: {('⚪ Undet.' if np.isnan(max_f_sum) else ('🟢 Good' if max_f_sum<=15 else ('🟡 Mod.' if max_f_sum<=25 else ('🟠 Unh.Sens.' if max_f_sum<=37.5 else ('🔴 Unh.' if max_f_sum<=50 else '🟣 V.Unh.')))))}"
                )

            st_max_f_sum = {}
            for st_s, val_s in forecast_summary_sum.items():
                valid_st_f = [
                    v_st
                    for v_st in [
                        val_s.get(h_s, np.nan) for h_s in ["24h", "48h", "72h"]
                    ]
                    if not np.isnan(v_st)
                ]
                st_max_f_sum[st_s] = max(valid_st_f) if valid_st_f else np.nan

            if st_max_f_sum:
                top_risk_sum = sorted(
                    st_max_f_sum.items(),
                    key=lambda x_s: x_s[1] if not np.isnan(x_s[1]) else -np.inf,
                    reverse=True,
                )[:3]
                print(f"\n⚠️  TOP RISK STATIONS (Max of 24/48/72h Forecast):")
                [
                    print(
                        f"   {i_s+1}. {st_s_name}: {val_s_risk:.1f} µg/m³ ({('⚪ Undet.' if np.isnan(val_s_risk) else ('🟢 Good' if val_s_risk<=15 else ('🟡 Mod.' if val_s_risk<=25 else ('🟠 Unh.Sens.' if val_s_risk<=37.5 else ('🔴 Unh.' if val_s_risk<=50 else '🟣 V.Unh.')))))})"
                    )
                    for i_s, (st_s_name, val_s_risk) in enumerate(top_risk_sum)
                ]
        else:
            print("\n🔮 72-HOUR FORECAST: Not available.")
        print(f"\n📁 Results saved to: {self.output_dir}\n" + "=" * 80)


def main():
    script_dir_main, project_root_main_prog = (
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
    )
    default_model_dir_main, default_config_path_main = (
        project_root_main_prog / "models" / "improved",
        project_root_main_prog / "configs" / "improved_model_config.yaml",
    )
    model_dir_use_main = os.environ.get("MODEL_DIR", str(default_model_dir_main))
    config_path_use_main = os.environ.get("CONFIG_PATH", str(default_config_path_main))
    model_type_use_main = os.environ.get("MODEL_TYPE", "hybrid")

    logger_main = logging.getLogger(__name__)
    logger_main.info(
        f"--- Running Evaluation ---\nUsing Model Dir: {model_dir_use_main}\nUsing Config: {config_path_use_main}\nUsing Model Type: {model_type_use_main}"
    )
    try:
        evaluator_main = PM25ModelEvaluator(
            model_dir=model_dir_use_main,
            config_path=config_path_use_main,
            model_type=model_type_use_main,
        )
        evaluator_main.run_complete_evaluation()  # Call the method
        logger_main.info(
            f"\n✅ Evaluation completed successfully!\n📊 Check results in: {evaluator_main.output_dir}"
        )  # Use attribute
    except FileNotFoundError as fnf_main:
        logger_main.error(
            f"❌ FILE NOT FOUND: {fnf_main}\nEnsure model/scalers/data are correct."
        )
    except ValueError as ve_main:
        logger_main.error(f"❌ VALUE ERROR: {ve_main}")
    except Exception as e_main:
        logger_main.error(f"❌ UNEXPECTED ERROR: {e_main}", exc_info=True)


if __name__ == "__main__":
    main()
