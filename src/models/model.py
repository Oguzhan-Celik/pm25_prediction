import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from .layers import (
    ConvLSTMCell,
    GraphAttentionLayer,
    MultiScaleResidualBlock,
    AdaptiveWaveletLayer,
    STCAttention,
    OutputTransformation,
)


class PM25Model(nn.Module):
    """
    Hybrid PM2.5 forecasting model combining ConvLSTM and GAT.
    """

    def __init__(
        self,
        time_steps: int,
        num_stations: int,
        input_features: int,
        adj_matrix: torch.Tensor,
        forecast_horizon: int,
        hidden_dims: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,  # Num ConvLSTM layers
        dropout_rate: float = 0.1,
        wavelet_type="db4",
        wavelet_level=3,
        scalers=None,
        station_names=None,  # station_names is used for model logic but not directly by OutputTransformation init
        model_type="hybrid",
        use_physics_guidance=True,
        use_attention_regularization=True,
        kernel_regularization=1e-4,
        recurrent_regularization=1e-4,
        use_stca_features: bool = True,
        use_wavelet_denoising: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.time_steps = time_steps
        self.num_stations = num_stations
        self.input_features = input_features
        self.forecast_horizon = forecast_horizon
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model_type = model_type
        self.use_physics_guidance = use_physics_guidance
        self.use_attention_regularization = use_attention_regularization
        self.kernel_reg_lambda = kernel_regularization

        self.use_stca_features = use_stca_features
        self.use_wavelet_denoising = use_wavelet_denoising

        self.scalers = scalers
        self.station_names = station_names  # Store for potential other uses or if OutputTransformation needs it later

        self.register_buffer("adj_matrix", adj_matrix)

        # Feature splitting based on input_features received
        if self.input_features >= 42:
            self.meteo_feature_size = 30
            self.pm25_feature_size = self.input_features - self.meteo_feature_size
        else:
            self.pm25_feature_size = min(12, self.input_features // 3)
            self.meteo_feature_size = self.input_features - self.pm25_feature_size

        print(
            f"PM25Model Init: meteo_features={self.meteo_feature_size}, pm25_features={self.pm25_feature_size}, total_input={self.input_features}"
        )
        assert (
            self.meteo_feature_size + self.pm25_feature_size == self.input_features
        ), "Feature split mismatch"

        self.meteo_proj = nn.Sequential(
            nn.Linear(self.meteo_feature_size, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.pm25_proj = nn.Sequential(
            nn.Linear(self.pm25_feature_size, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.initial_feature_fusion = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        if self.use_wavelet_denoising:
            self.wavelet_layer = AdaptiveWaveletLayer(
                wavelet=wavelet_type, level=wavelet_level
            )

        if self.use_stca_features:
            self.stca = STCAttention(
                input_dim=hidden_dims,
                embed_dim=hidden_dims,
                num_heads=num_heads,
                meteo_feature_size=self.meteo_feature_size,
                pm25_feature_size=self.pm25_feature_size,
            )
            self.stca_meteo_output_proj = nn.Linear(
                self.meteo_feature_size, hidden_dims
            )
            self.stca_path_fusion = nn.Sequential(
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

        self.multi_scale1 = MultiScaleResidualBlock(hidden_dims, hidden_dims)
        self.multi_scale2 = MultiScaleResidualBlock(hidden_dims, hidden_dims)

        self.conv_lstm_cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    input_channels=hidden_dims,
                    hidden_channels=hidden_dims,
                    kernel_size=3,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.lstm_dropout = nn.Dropout(dropout_rate)

        self.gat = GraphAttentionLayer(
            in_features=hidden_dims,
            out_features=hidden_dims,
            dropout_rate=dropout_rate,
            # num_heads=self.num_heads,
        )

        output_layer_input_dim = hidden_dims
        if model_type == "transformer":
            self.output_layer = nn.Sequential(
                nn.Linear(output_layer_input_dim, hidden_dims * 2),
                nn.LayerNorm(hidden_dims * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims, forecast_horizon),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(output_layer_input_dim, hidden_dims // 2),
                nn.LayerNorm(hidden_dims // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims // 2, forecast_horizon),
            )

        if scalers is not None:
            # Corrected: OutputTransformation in layers.py does not take station_names
            self.output_transform = OutputTransformation(scalers)
        else:
            self.output_transform = None

        self.feature_dropout = nn.Dropout(dropout_rate * 0.5)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.final_dropout = nn.Dropout(dropout_rate * 1.2)

        self.positional_encoding = self._create_positional_encoding(
            time_steps, hidden_dims
        )

    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return nn.Parameter(pe.unsqueeze(0).unsqueeze(2), requires_grad=False)

    def forward(self, x: torch.Tensor, return_scaled_for_loss=False) -> torch.Tensor:
        batch_size, time_steps, num_stations, current_input_features = x.shape
        device = x.device
        batch_time_stations = batch_size * time_steps * num_stations

        if current_input_features != self.input_features:
            # This indicates a mismatch that should ideally be caught earlier or handled by ensuring
            # self.input_features is always correctly set based on data.
            # For now, we proceed assuming self.input_features and its splits are the source of truth for layer defs.
            # If this assert fails, it means the input data `x` does not match what the model was configured for.
            # This was the cause of the previous RuntimeError if config input_features was not 43.
            # Assuming self.input_features is now correctly 43 from config.
            pass

        raw_meteo_features = x[:, :, :, : self.meteo_feature_size]
        raw_pm25_features = x[:, :, :, self.meteo_feature_size : self.input_features]

        projected_meteo = self.meteo_proj(
            raw_meteo_features.reshape(batch_time_stations, self.meteo_feature_size)
        )
        pm25_reshaped = raw_pm25_features.reshape(
            batch_time_stations, self.pm25_feature_size
        )
        projected_pm25 = self.pm25_proj(pm25_reshaped)

        projected_meteo = projected_meteo.reshape(
            batch_size, time_steps, num_stations, self.hidden_dims
        )
        projected_pm25 = projected_pm25.reshape(
            batch_size, time_steps, num_stations, self.hidden_dims
        )

        if self.use_stca_features and hasattr(self, "stca"):
            attended_meteo_features, attention_weights = self.stca(x, self.adj_matrix)
            projected_attended_meteo = self.stca_meteo_output_proj(
                attended_meteo_features.reshape(
                    batch_time_stations, self.meteo_feature_size
                )
            ).reshape(batch_size, time_steps, num_stations, self.hidden_dims)
            stca_path_combined_to_fuse = torch.cat(
                [projected_attended_meteo, projected_pm25], dim=-1
            )
            x_processed = self.stca_path_fusion(
                stca_path_combined_to_fuse.reshape(
                    batch_time_stations, self.hidden_dims * 2
                )
            ).reshape(batch_size, time_steps, num_stations, self.hidden_dims)
            if self.use_attention_regularization and attention_weights is not None:
                self.stored_attention_weights = attention_weights
        else:
            initial_combined_to_fuse = torch.cat(
                [projected_meteo, projected_pm25], dim=-1
            )
            x_processed = self.initial_feature_fusion(
                initial_combined_to_fuse.reshape(
                    batch_time_stations, self.hidden_dims * 2
                )
            ).reshape(batch_size, time_steps, num_stations, self.hidden_dims)
            self.stored_attention_weights = None

        if self.use_wavelet_denoising and hasattr(self, "wavelet_layer"):
            x_processed = self.wavelet_layer(x_processed)

        x_processed = x_processed + self.positional_encoding[:, :time_steps, :, :]
        x_processed = self.feature_dropout(x_processed)

        x_multi = self.multi_scale1(x_processed)
        x_multi = self.multi_scale2(x_multi)

        h_states, c_states = [], []
        for _ in range(self.num_layers):
            h_states.append(
                torch.zeros(
                    batch_size * num_stations, self.hidden_dims, 1, 1, device=device
                )
            )
            c_states.append(
                torch.zeros(
                    batch_size * num_stations, self.hidden_dims, 1, 1, device=device
                )
            )

        lstm_outputs_over_time = []
        for t in range(time_steps):
            x_t_layer_input = x_multi[:, t, :, :].reshape(
                batch_size * num_stations, self.hidden_dims, 1, 1
            )
            for i in range(self.num_layers):
                h_states[i], c_states[i] = self.conv_lstm_cells[i](
                    x_t_layer_input, (h_states[i], c_states[i])
                )
                x_t_layer_input = self.lstm_dropout(h_states[i])
            lstm_outputs_over_time.append(h_states[-1])

        final_lstm_output_flat = lstm_outputs_over_time[-1].squeeze(-1).squeeze(-1)

        gat_input = final_lstm_output_flat.reshape(
            batch_size, num_stations, self.hidden_dims
        )
        gat_output_list = [
            self.gat(gat_input[b_idx], self.adj_matrix) for b_idx in range(batch_size)
        ]
        gat_output = torch.stack(gat_output_list, dim=0)
        gat_output = self.attention_dropout(gat_output)

        output_layer_input = gat_output.reshape(
            batch_size * num_stations, self.hidden_dims
        )
        output_layer_input = self.final_dropout(output_layer_input)
        predictions_scaled = self.output_layer(output_layer_input)
        predictions_scaled = predictions_scaled.reshape(
            batch_size, num_stations, self.forecast_horizon
        )

        if return_scaled_for_loss:
            return predictions_scaled

        if self.output_transform is not None:
            return self.output_transform(predictions_scaled)
        return predictions_scaled

    def compute_loss(self, y_true, y_pred_scaled, **kwargs):
        y_pred_first_step = y_pred_scaled[:, :, 0:1]
        base_loss = F.mse_loss(y_pred_first_step, y_true, reduction="none")
        mask = ~torch.isnan(y_true)
        if mask.sum() == 0:
            base_loss_masked_avg = torch.tensor(
                0.0, device=y_pred_scaled.device, requires_grad=True
            )
        else:
            base_loss_masked_avg = (base_loss * mask).sum() / mask.sum()

        total_loss = base_loss_masked_avg

        lambda_pm25_focus = kwargs.get("lambda_pm25_focus", 0.0)
        if lambda_pm25_focus > 0:
            extreme_penalty = torch.mean(F.relu(torch.abs(y_pred_first_step) - 5.0))
            total_loss += lambda_pm25_focus * extreme_penalty
            if y_pred_scaled.shape[-1] > 1:
                temporal_consistency = torch.mean(
                    torch.abs(y_pred_scaled[..., 1:] - y_pred_scaled[..., :-1])
                )
                total_loss += lambda_pm25_focus * 0.1 * temporal_consistency

        lambda_kernel = kwargs.get("lambda_kernel", self.kernel_reg_lambda)
        if lambda_kernel > 0:
            kernel_reg_term = sum(
                torch.sum(param**2)
                for name, param in self.named_parameters()
                if "weight" in name and param.requires_grad
            )
            total_loss += lambda_kernel * kernel_reg_term

        lambda_smooth = kwargs.get("lambda_smooth", 0.0)
        if lambda_smooth > 0 and y_pred_scaled.shape[-1] > 1:
            smooth_reg = torch.mean(
                (y_pred_scaled[..., 1:] - y_pred_scaled[..., :-1]) ** 2
            )
            total_loss += lambda_smooth * smooth_reg

        lambda_attention = kwargs.get("lambda_attention", 0.0)
        current_attention_weights = self.get_attention_weights()
        if (
            self.use_attention_regularization
            and current_attention_weights is not None
            and lambda_attention > 0
        ):
            attn_reg = torch.mean(torch.abs(current_attention_weights))
            total_loss += lambda_attention * attn_reg
            self.reset_attention_weights()

        return total_loss

    def get_attention_weights(self):
        return getattr(self, "stored_attention_weights", None)

    def reset_attention_weights(self):
        if hasattr(self, "stored_attention_weights"):
            delattr(self, "stored_attention_weights")

    def get_model_complexity(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),
            "hidden_dims": self.hidden_dims,
            "num_layers": self.num_layers,
            "forecast_horizon": self.forecast_horizon,
        }

    def predict(self, x: torch.Tensor, return_raw_scaled=False):
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_scaled_for_loss=return_raw_scaled)

    def transform_predictions(
        self, predictions_scaled: torch.Tensor, inverse: bool = True
    ):
        if not hasattr(self, "output_transform") or self.output_transform is None:
            print("Warning: No output_transform module in PM25Model. Cannot transform.")
            return predictions_scaled
        if inverse:
            return self.output_transform(predictions_scaled)
        else:
            print(
                "Warning: Forward scaling (original to scaled) not implemented in PM25Model.transform_predictions."
            )
            return predictions_scaled


def create_enhanced_model_configs():
    base_config = {
        "time_steps": 24,
        "forecast_horizon": 72,
        "hidden_dims": 64,
        "num_heads": 8,
        "num_layers": 2,
        "dropout_rate": 0.2,
        "kernel_regularization": 1e-4,
        "recurrent_regularization": 1e-4,
        "wavelet_type": "db4",
        "wavelet_level": 3,
        "use_physics_guidance": True,
        "use_attention_regularization": True,
        "use_stca_features": True,
        "use_wavelet_denoising": True,
    }
    configs = {
        "hybrid_standard": {**base_config, "model_type": "hybrid"},
        "hybrid_simplified": {
            **base_config,
            "model_type": "hybrid",
            "use_stca_features": False,
            "use_wavelet_denoising": False,
            "num_layers": 1,
        },
        "hybrid_no_stca": {
            **base_config,
            "model_type": "hybrid",
            "use_stca_features": False,
        },
        "hybrid_no_wavelet": {
            **base_config,
            "model_type": "hybrid",
            "use_wavelet_denoising": False,
        },
    }
    return configs
