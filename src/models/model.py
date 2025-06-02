import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from .layers import (
    # GraphAttentionLayer, # Not in the new main forward path as per user snippet
    AdaptiveWaveletLayer,
    STCAttention,
    OutputTransformation,
)


class PM25Model(nn.Module):
    def __init__(
        self,
        time_steps: int,
        num_stations: int,
        input_features: int,
        adj_matrix: torch.Tensor,
        forecast_horizon: int,
        hidden_dims: int = 128,  # From YAML
        num_heads: int = 4,  # For STCA & Self-Attention, From YAML
        num_layers: int = 2,  # Num LSTM layers, From YAML
        dropout_rate: float = 0.15,  # From YAML
        wavelet_type="db4",
        wavelet_level=3,
        scalers=None,
        station_names=None,
        model_type="hybrid_lstm_cnn_v2",  # Reflecting new architecture
        use_stca_features: bool = True,  # From YAML
        use_wavelet_denoising: bool = True,  # From YAML
        # kernel_regularization is now handled by lambda_reg in loss_cfg for compute_loss
        use_mixed_precision: bool = False,  # Passed from train_model based on perf config
        **kwargs,
    ):
        super().__init__()

        self.time_steps = time_steps
        self.num_stations = num_stations
        self.input_features = input_features
        self.forecast_horizon = forecast_horizon
        self.hidden_dims = hidden_dims
        self.attention_num_heads = num_heads
        self.lstm_num_layers = num_layers  # Used for nn.LSTM

        self.use_stca_features = use_stca_features
        self.use_wavelet_denoising = use_wavelet_denoising
        # self.use_attention_regularization = kwargs.get('use_attention_regularization', True) # For STCA weights if needed
        self.kernel_reg_lambda_weights = kwargs.get(
            "kernel_regularization", 0.0
        )  # For L2 on model weights

        self.scalers = scalers
        self.station_names = station_names
        self.register_buffer("adj_matrix", adj_matrix)
        self.use_mixed_precision = use_mixed_precision  # For autocast hint in forward

        # Feature splitting
        if self.input_features == 43:
            self.pm25_feature_size = 12
            self.meteo_feature_size = self.input_features - self.pm25_feature_size
        elif self.input_features >= 42:
            self.meteo_feature_size = 30
            self.pm25_feature_size = self.input_features - self.meteo_feature_size
        else:
            self.pm25_feature_size = min(12, self.input_features // 3)
            self.meteo_feature_size = self.input_features - self.pm25_feature_size

        print(
            f"PM25Model Init: meteo_features={self.meteo_feature_size}, pm25_features={self.pm25_feature_size}, total_input={self.input_features}, hidden_dims={self.hidden_dims}"
        )
        assert (
            self.meteo_feature_size + self.pm25_feature_size == self.input_features
        ), "Feature split mismatch"

        # IMPROVEMENT 1: More efficient feature projection
        self.meteo_proj = nn.Sequential(
            nn.Linear(self.meteo_feature_size, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
        )
        self.pm25_proj = nn.Sequential(
            nn.Linear(self.pm25_feature_size, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
        )

        # IMPROVEMENT 2: Simplified feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
        )

        if self.use_stca_features:
            self.stca = STCAttention(
                input_dim=0,
                embed_dim=hidden_dims,  # STCA's internal embed_dim for its QKV projections from raw features
                # Its output for meteo part will be meteo_feature_size
                num_heads=self.attention_num_heads,
                meteo_feature_size=self.meteo_feature_size,
                pm25_feature_size=self.pm25_feature_size,
                dropout_rate=dropout_rate,
            )
            # No separate stca_path_fusion here; STCA output (meteo_feature_size) will use self.meteo_proj

        if self.use_wavelet_denoising:
            self.wavelet_layer = AdaptiveWaveletLayer(
                wavelet=wavelet_type, level=wavelet_level
            )

        # IMPROVEMENT 3: LSTM for temporal encoding
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dims,  # After fusion
            hidden_size=hidden_dims,
            num_layers=self.lstm_num_layers,
            dropout=dropout_rate if self.lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )

        # IMPROVEMENT 4: Spatial convolution for local patterns
        self.spatial_conv = nn.Conv1d(
            in_channels=hidden_dims,
            out_channels=hidden_dims,
            kernel_size=3,
            padding=1,
            groups=max(1, hidden_dims // 4),  # From user snippet
        )
        self.spatial_norm_relu = nn.Sequential(
            nn.BatchNorm1d(hidden_dims), nn.ReLU(inplace=True)
        )

        # IMPROVEMENT 5: Simplified multi-scale processing
        self.multi_scale_convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_dims, hidden_dims // 2, kernel_size=k, padding=k // 2)
                for k in [1, 3, 5]
            ]
        )
        # self.scale_fusion was commented out by user; using multi_scale_projection
        self.multi_scale_projection = nn.Conv1d(
            (hidden_dims // 2) * 3, hidden_dims, kernel_size=1
        )
        self.multiscale_norm_relu = nn.Sequential(
            nn.BatchNorm1d(hidden_dims), nn.ReLU(inplace=True)
        )

        # IMPROVEMENT 6: More efficient self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims,
            num_heads=self.attention_num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dims)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.LayerNorm(hidden_dims // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims // 2, forecast_horizon),
        )

        if scalers is not None:
            self.output_transform = OutputTransformation(scalers)
        else:
            self.output_transform = None

        self.dropout = nn.Dropout(dropout_rate)  # General dropout if needed
        self.positional_encoding = self._create_positional_encoding(
            time_steps, hidden_dims
        )

    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term_exp = torch.arange(0, d_model, 2).float() * (
            -np.log(10000.0) / d_model
        )
        div_term = torch.exp(div_term_exp)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Corrected slicing for odd d_model to avoid size mismatch
            pe[:, 1::2] = torch.cos(
                position * div_term_exp[: d_model // 2]
            )  # Use the exponent before exp for slicing
        return nn.Parameter(pe.unsqueeze(0).unsqueeze(2), requires_grad=False)

    # 2. OPTIMIZED forward method (from user snippet)
    def forward(self, x: torch.Tensor, return_scaled_for_loss=False) -> torch.Tensor:
        # User snippet implies autocast is handled in training loop
        # For model's internal autocast (if desired for inference too):
        # with torch.autocast(device_type=x.device.type, enabled=self.use_mixed_precision):

        batch_size, time_steps, num_stations, _ = x.shape

        # IMPROVEMENT 7: Efficient feature processing
        # Reshape for batch processing by projection layers
        x_flat = x.view(-1, self.input_features)  # (B*T*S, F_input)

        raw_meteo_flat = x_flat[:, : self.meteo_feature_size]
        raw_pm25_flat = x_flat[:, self.meteo_feature_size :]

        # Parallel projection
        pm25_proj_flat = self.pm25_proj(raw_pm25_flat)  # (B*T*S, H)

        if self.use_stca_features and hasattr(self, "stca"):
            # STCA expects raw x: (B, T, S, F_input)
            # STCA output (attended_meteo_raw) is [B, T, S, self.meteo_feature_size]
            attended_meteo_raw, attention_weights = self.stca(x, self.adj_matrix)
            # Project STCA's output using self.meteo_proj
            meteo_processed_flat = self.meteo_proj(
                attended_meteo_raw.reshape(-1, self.meteo_feature_size)
            )  # (B*T*S, H)
            if (
                hasattr(self, "use_attention_regularization")
                and self.use_attention_regularization
                and attention_weights is not None
            ):
                self.stored_attention_weights = attention_weights
        else:
            meteo_processed_flat = self.meteo_proj(raw_meteo_flat)  # (B*T*S, H)
            self.stored_attention_weights = None

        # Combine features
        combined_flat = torch.cat(
            [meteo_processed_flat, pm25_proj_flat], dim=-1
        )  # (B*T*S, 2H)
        fused_flat = self.feature_fusion(combined_flat)  # (B*T*S, H)

        x_processed = fused_flat.reshape(
            batch_size, time_steps, num_stations, self.hidden_dims
        )

        if self.use_wavelet_denoising and hasattr(self, "wavelet_layer"):
            x_processed = self.wavelet_layer(x_processed)

        x_processed = x_processed + self.positional_encoding[:, :time_steps, :, :]
        # Dropout is applied within layers or at specific points now

        # IMPROVEMENT 8: Efficient temporal processing (LSTM)
        # Process each station separately through LSTM
        # Reshape for LSTM: (batch_size * num_stations, time_steps, hidden_dims)
        lstm_input = x_processed.permute(0, 2, 1, 3).reshape(
            batch_size * num_stations, time_steps, self.hidden_dims
        )
        lstm_out_seq, _ = self.temporal_encoder(lstm_input)  # (B*S, T, H)

        # Take features from the last time step of LSTM for subsequent spatial processing
        station_features_temporal = lstm_out_seq[:, -1, :]  # (B*S, H)

        # Reshape to (B, S, H) for spatial processing
        station_features_reshaped = station_features_temporal.reshape(
            batch_size, num_stations, self.hidden_dims
        )

        # IMPROVEMENT 9: Efficient spatial processing
        # Apply spatial convolution across stations
        # Input for Conv1D: (Batch, Channels, Length) -> (B, H, S)
        spatial_input_conv = station_features_reshaped.transpose(1, 2)
        residual_for_spatial_conv = spatial_input_conv

        spatial_out_conv = self.spatial_conv(spatial_input_conv)
        spatial_out_conv = self.spatial_norm_relu(
            spatial_out_conv
        )  # Apply BatchNorm & ReLU
        spatial_out_conv = spatial_out_conv + residual_for_spatial_conv  # Add residual

        # IMPROVEMENT 10: Multi-scale feature extraction
        multi_scale_input = spatial_out_conv  # Output of previous step (B, H, S)
        multi_scale_feature_list = []
        for conv_layer_ms in self.multi_scale_convs:
            scale_feat = conv_layer_ms(multi_scale_input)
            multi_scale_feature_list.append(scale_feat)

        multi_scale_combined_cat = torch.cat(
            multi_scale_feature_list, dim=1
        )  # (B, (H/2)*3, S)
        final_features_fused = self.multi_scale_projection(
            multi_scale_combined_cat
        )  # (B, H, S)
        final_features_fused = self.multiscale_norm_relu(final_features_fused)
        final_features_after_multiscale = (
            final_features_fused + multi_scale_input
        )  # Residual

        # Transpose for self-attention: (B, S, H)
        self_attn_input = final_features_after_multiscale.transpose(1, 2)
        residual_attn_input = self_attn_input

        # IMPROVEMENT 11: Efficient attention
        attn_out, _ = self.self_attention(
            self_attn_input, self_attn_input, self_attn_input
        )
        output_features_attended = self.attention_norm(
            residual_attn_input + attn_out
        )  # Add residual and then LayerNorm

        output_layer_input = output_features_attended.reshape(
            batch_size * num_stations, self.hidden_dims
        )
        predictions_scaled = self.output_layer(output_layer_input)
        predictions_scaled = predictions_scaled.reshape(
            batch_size, num_stations, self.forecast_horizon
        )

        if return_scaled_for_loss:
            return predictions_scaled

        if self.output_transform is not None:
            return self.output_transform(predictions_scaled)
        return predictions_scaled

    # compute_improved_loss from user snippet, adapted as model.compute_loss
    def compute_loss(self, y_true, y_pred_scaled, **loss_cfg_kwargs):
        y_pred_first = y_pred_scaled[:, :, 0:1]

        huber_delta = loss_cfg_kwargs.get("huber_delta", 1.0)
        huber_loss = F.smooth_l1_loss(
            y_pred_first, y_true, reduction="none", beta=huber_delta
        )

        mask = ~torch.isnan(y_true) & ~torch.isinf(y_true)
        if mask.sum() == 0:
            base_loss = torch.tensor(
                0.0, device=y_pred_scaled.device, requires_grad=True
            )
        else:
            base_loss = (huber_loss * mask).sum() / mask.sum().clamp(min=1e-8)

        prediction_scale = torch.clamp(
            torch.abs(y_pred_first.detach()), min=0.1, max=10.0
        )
        weighted_loss = base_loss / prediction_scale.mean().clamp(min=1e-8)

        total_loss = weighted_loss

        lambda_reg = loss_cfg_kwargs.get("lambda_reg", 0.0)
        if lambda_reg > 0:
            pred_reg_loss = torch.mean(y_pred_first**2) * lambda_reg
            total_loss += pred_reg_loss

            # L2 on model weights (using self.kernel_reg_lambda from __init__)
            if hasattr(self, "kernel_reg_lambda") and self.kernel_reg_lambda > 0:
                # If lambda_reg from loss_cfg is *also* meant for weights, this could double count.
                # Assuming self.kernel_reg_lambda is the one for weights from model config.
                kernel_reg_term = sum(
                    torch.sum(param**2)
                    for name, param in self.named_parameters()
                    if "weight" in name and param.requires_grad
                )
                total_loss += self.kernel_reg_lambda * kernel_reg_term

        lambda_consistency = loss_cfg_kwargs.get("lambda_consistency", 0.0)
        consistency_loss_factor = loss_cfg_kwargs.get("consistency_loss_factor", 0.1)
        if lambda_consistency > 0 and y_pred_scaled.shape[-1] > 1:
            consistency = torch.mean(
                torch.abs(y_pred_scaled[:, :, 1:] - y_pred_scaled[:, :, :-1])
            )
            total_loss += lambda_consistency * consistency_loss_factor * consistency

        lambda_attention = loss_cfg_kwargs.get("lambda_attention", 0.0)
        current_attention_weights = self.get_attention_weights()
        if (
            hasattr(self, "use_attention_regularization")
            and self.use_attention_regularization
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
            "num_layers_lstm": self.lstm_num_layers,
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
