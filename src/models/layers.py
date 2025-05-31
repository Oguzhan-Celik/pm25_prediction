import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        # Input to hidden layer transformation
        self.ih = nn.Linear(input_channels, 4 * hidden_channels)

        # Hidden to hidden layer transformation
        self.hh = nn.Linear(hidden_channels, 4 * hidden_channels)

        # Initialize biases
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_channels))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_channels))

    def forward(self, x, hidden_state):
        """
        Forward pass of ConvLSTM cell.

        Args:
            x: Input tensor of shape [batch_size, num_stations, input_channels]
            h_prev: Previous hidden state [batch_size, num_stations, hidden_channels]
            c_prev: Previous cell state [batch_size, num_stations, hidden_channels]

        Returns:
            h_next: Next hidden state [batch_size, num_stations, hidden_channels]
            c_next: Next cell state [batch_size, num_stations, hidden_channels]
        """
        # Unpack hidden state tuple
        h_prev, c_prev = hidden_state

        # Handle different input formats
        if len(x.shape) == 4:  # [batch*stations, channels, 1, 1] format from main model
            batch_stations, channels, _, _ = x.shape
            x = x.squeeze(-1).squeeze(-1)  # [batch*stations, channels]
            h_prev = h_prev.squeeze(-1).squeeze(-1)  # [batch*stations, hidden_channels]
            c_prev = c_prev.squeeze(-1).squeeze(-1)  # [batch*stations, hidden_channels]

            # Calculate gates
            gates = self.ih(x) + self.hh(h_prev) + self.bias_ih + self.bias_hh

            # Split gates
            i, f, g, o = gates.chunk(4, dim=1)

            # Apply activations
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            # Update cell state
            c_next = f * c_prev + i * g

            # Update hidden state
            h_next = o * torch.tanh(c_next)

            # Reshape back to conv format
            h_next = h_next.unsqueeze(-1).unsqueeze(
                -1
            )  # [batch*stations, hidden_channels, 1, 1]
            c_next = c_next.unsqueeze(-1).unsqueeze(
                -1
            )  # [batch*stations, hidden_channels, 1, 1]

        else:  # [batch_size, num_stations, channels] format
            batch_size, num_stations, _ = x.shape

            # Reshape for batched matrix multiplication
            x_flat = x.reshape(-1, self.input_channels)
            h_flat = h_prev.reshape(-1, self.hidden_channels)
            c_flat = c_prev.reshape(-1, self.hidden_channels)

            # Calculate gates
            gates = self.ih(x_flat) + self.hh(h_flat) + self.bias_ih + self.bias_hh

            # Reshape gates
            gates = gates.reshape(batch_size, num_stations, 4 * self.hidden_channels)

            # Split gates
            i, f, g, o = gates.chunk(4, dim=2)

            # Apply activations
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            # Update cell state
            c_next = f * c_prev + i * g

            # Update hidden state
            h_next = o * torch.tanh(c_next)

        return h_next, c_next


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network (GAT) layer."""

    def __init__(self, in_features, out_features, dropout_rate=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.alpha = alpha

        # Feature transformation matrix
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W)

        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)

        # LeakyReLU for attention mechanism
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj_matrix=None):
        """
        Forward pass of GAT layer.

        Args:
            x: Input features [batch_size, num_nodes, in_features]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Output features [batch_size, num_nodes, out_features]
        """
        if x.dim() == 2:
            # Flattened input - need to know num_nodes to reshape
            assert (
                adj_matrix is not None
            ), "Need adj_matrix to determine num_nodes for flattened input"
            num_nodes = adj_matrix.size(0)
            batch_size = x.size(0) // num_nodes
            x = x.view(batch_size, num_nodes, self.in_features)

        # Linear transformation of input features
        h = torch.matmul(x, self.W)  # [batch_size, num_nodes, out_features]

        # If no adjacency matrix provided, create fully connected graph
        if adj_matrix is None:
            num_nodes = h.size(1)
            adj_matrix = torch.ones((num_nodes, num_nodes), device=x.device)

        # Create pairs of nodes for attention
        a_input = self._prepare_attentional_mechanism_input(h)

        # Calculate attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        e = e.view(h.size(0), h.size(1), h.size(1))

        # Mask attention coefficients using adjacency matrix
        mask = -9e15 * (1.0 - adj_matrix)
        e = e + mask.unsqueeze(0)  # Add batch dimension to mask

        # Apply softmax to get attention coefficients
        attention = F.softmax(e, dim=-1)

        # Apply dropout to attention coefficients
        attention = F.dropout(attention, self.dropout_rate, training=self.training)

        # Apply attention to features
        h_prime = torch.matmul(attention, h)

        return h_prime

    def _prepare_attentional_mechanism_input(self, h):
        """
        Prepare input for attention mechanism.

        Args:
            h: Hidden representations [batch_size, num_nodes, out_features]

        Returns:
            Prepared attention input [batch_size, num_nodes * num_nodes, 2 * out_features]
        """
        batch_size, num_nodes, out_features = h.size()

        # Repeat h for all nodes (N times)
        h_repeated = h.unsqueeze(1).expand(
            -1, num_nodes, -1, -1
        )  # [batch_size, num_nodes, num_nodes, out_features]

        # Repeat h for each node (N times)
        h_repeated_interleave = h.unsqueeze(2).expand(
            -1, -1, num_nodes, -1
        )  # [batch_size, num_nodes, num_nodes, out_features]

        # Concatenate and reshape
        a_input = torch.cat([h_repeated_interleave, h_repeated], dim=-1)
        return a_input.view(batch_size, num_nodes * num_nodes, 2 * out_features)


class MultiScaleResidualBlock(nn.Module):
    """Multi-Scale Residual Block for processing features at different spatial scales."""

    def __init__(self, in_features, out_features):
        super(MultiScaleResidualBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Define dense layers with different hidden dimensions
        self.fc1_small = nn.Linear(in_features, out_features // 4)
        self.fc1_medium = nn.Linear(in_features, out_features // 2)
        self.fc1_large = nn.Linear(in_features, out_features // 4)

        # Combine features
        self.combine = nn.Linear(out_features, out_features)

        # Normalization layers - using LayerNorm for 2D input
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(out_features)

        # Skip connection
        self.skip = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

        # Activation
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of multi-scale residual block.

        Args:
            x: Input features [batch_size, in_features]

        Returns:
            Output features [batch_size, out_features]
        """
        # Store original shape
        original_shape = x.shape

        # Flatten all dimensions except the last one
        x_flat = x.view(-1, self.in_features)

        # Store flattened input for skip connection
        identity = x_flat

        # Apply normalization
        x_norm = self.norm1(x_flat)

        # Process through parallel branches
        small = self.fc1_small(x_norm)
        medium = self.fc1_medium(x_norm)
        large = self.fc1_large(x_norm)

        # Concatenate branch outputs
        concat = torch.cat([small, medium, large], dim=-1)

        # Apply final combination
        out = self.combine(concat)
        out = self.act(out)
        out = self.norm2(out)

        # Add skip connection
        out = out + self.skip(identity)

        # Reshape back to original dimensions with new feature size
        # Calculate the new shape: keep all dimensions except last, change last to out_features
        new_shape = list(original_shape[:-1]) + [self.out_features]
        out = out.view(*new_shape)

        return out


class AdaptiveWaveletLayer(nn.Module):
    """Optimized Adaptive Wavelet Thresholding layer for time series denoising."""

    def __init__(self, wavelet="db4", level=3, threshold_factor=0.1):
        super(AdaptiveWaveletLayer, self).__init__()
        self.wavelet = wavelet
        self.level = level
        # Make threshold factor learnable but initialize conservatively
        self.threshold_factor = nn.Parameter(torch.tensor([threshold_factor]))

    def forward(self, x):
        """
        Forward pass applying wavelet denoising with efficient batch processing.

        Args:
            x: Input tensor [batch_size, time_steps, num_stations, features]

        Returns:
            Denoised tensor [batch_size, time_steps, num_stations, features]
        """
        # Store original shape and device
        original_shape = x.shape
        device = x.device

        # Check if input is too short for wavelet transform
        time_steps = original_shape[1]
        if time_steps < 2 ** (self.level + 1):
            # Return original if too short to process
            return x

        try:
            # Get threshold factor as scalar value (NOT tensor)
            threshold_factor_np = self.threshold_factor.detach().cpu().item()
            # Move to CPU and convert to numpy once for batch processing
            x_cpu = x.detach().cpu().numpy()

            # Get max value using numpy
            max_value = np.max(np.abs(x_cpu))

            # Calculate threshold as scalar
            threshold = threshold_factor_np * max_value

            # Reshape for batch processing
            batch_size = original_shape[0]
            num_stations = original_shape[2]
            features = original_shape[3]
            total_signals = batch_size * num_stations * features
            # Reshape to process all signals at once
            x_reshaped = x_cpu.reshape(total_signals, time_steps)

            # Reshape for batch processing
            # x_reshaped = x_cpu.reshape(-1, time_steps)

            # Get threshold value - compute as scalar using numpy
            # Convert threshold_factor to numpy value before using
            # threshold_factor_np = self.threshold_factor.detach().cpu().item()
            # threshold = threshold_factor_np * np.max(np.abs(x_cpu))

            # Prepare output array
            denoised = np.zeros_like(x_reshaped)

            # Process each signal
            for i in range(x_reshaped.shape[0]):
                # Calculate maximum level based on signal length
                max_level = pywt.dwt_max_level(
                    time_steps, pywt.Wavelet(self.wavelet).dec_len
                )
                actual_level = min(self.level, max_level)

                # Apply wavelet transform
                coeffs = pywt.wavedec(x_reshaped[i], self.wavelet, level=actual_level)

                # Apply soft thresholding - exclude approximation coefficients (first element)
                denoised_coeffs = [
                    coeffs[0]
                ]  # Keep approximation coefficients unchanged

                for c in coeffs[1:]:  # Apply thresholding only to detail coefficients
                    # Ensure all operations use numpy arrays
                    denoised_coeffs.append(
                        np.sign(c) * np.maximum(np.abs(c) - threshold, 0)
                    )

                # Reconstruct signal
                denoised_signal = pywt.waverec(denoised_coeffs, self.wavelet)

                # Handle potential length mismatch
                if len(denoised_signal) >= time_steps:
                    denoised[i, :] = denoised_signal[:time_steps]
                else:
                    # Pad with zeros if reconstructed signal is too short
                    denoised[i, : len(denoised_signal)] = denoised_signal
            # Reshape back to original dimensions
            denoised_reshaped = denoised.reshape(original_shape)

            # Convert back to tensor
            return torch.tensor(denoised_reshaped, device=device, dtype=x.dtype)

        except Exception as e:
            # Log the error in detail but return original without breaking flow
            print(
                f"Wavelet denoising error: {e}, input shape: {original_shape}, using original features"
            )
            return x


class STCAttention(nn.Module):
    """Spatiotemporal Cross Attention module for meteorological and emission features."""

    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads=8,
        dropout_rate=0.1,
        meteo_feature_size=30,
        pm25_feature_size=12,
    ):
        super(STCAttention, self).__init__()
        self.meteo_feature_size = meteo_feature_size  # 30
        self.pm25_feature_size = pm25_feature_size  # 12
        self.input_dim = (
            input_dim  # This should be half_dim (64) not full combined dim (128)
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Ensure embed_dim is divisible by num_heads
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # Projection layers - use actual feature sizes
        self.query_proj = nn.Linear(meteo_feature_size, embed_dim)  # 30 -> embed_dim
        self.key_proj = nn.Linear(pm25_feature_size, embed_dim)  # 12 -> embed_dim
        self.value_proj = nn.Linear(pm25_feature_size, embed_dim)  # 12 -> embed_dim

        # Output projection - project back to meteo feature size for residual connection
        self.output_proj = nn.Linear(embed_dim, meteo_feature_size)

        # Layer normalization for add & norm
        self.norm1 = nn.LayerNorm(meteo_feature_size)
        self.norm2 = nn.LayerNorm(meteo_feature_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(meteo_feature_size, 4 * meteo_feature_size),
            nn.GELU(),
            nn.Linear(4 * meteo_feature_size, meteo_feature_size),
            nn.Dropout(dropout_rate),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj_matrix=None):
        """
        Apply cross-attention between meteorological and emission features.

        Args:
            meteo_features: Meteorological features [batch_size, num_stations, input_dim]
            emission_features: Emission features [batch_size, num_stations, input_dim]

        Returns:
            Output tensor [batch_size, num_stations, input_dim]
        """
        batch_size, time_steps, num_stations, feature_dim = x.shape

        # Split raw features into meteo and PM2.5 features
        meteo_features = x[:, :, :, : self.meteo_feature_size]  # First 30 features
        pm25_features = x[:, :, :, self.meteo_feature_size :]  # Last 12 features

        # Reshape for processing: combine batch and time dimensions
        meteo_flat = meteo_features.reshape(
            batch_size * time_steps, num_stations, self.meteo_feature_size
        )
        pm25_flat = pm25_features.reshape(
            batch_size * time_steps, num_stations, self.pm25_feature_size
        )

        # Project to query, key, value
        query = self.query_proj(meteo_flat)  # [batch*time, stations, embed_dim]
        key = self.key_proj(pm25_flat)  # [batch*time, stations, embed_dim]
        value = self.value_proj(pm25_flat)  # [batch*time, stations, embed_dim]

        # Reshape for multi-head attention
        head_dim = self.embed_dim // self.num_heads
        batch_time = batch_size * time_steps

        query = query.view(
            batch_time, num_stations, self.num_heads, head_dim
        ).transpose(1, 2)
        key = key.view(batch_time, num_stations, self.num_heads, head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_time, num_stations, self.num_heads, head_dim
        ).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, value)

        # Reshape back
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_time, num_stations, self.embed_dim)
        )

        # Project to output dimension
        output = self.output_proj(context)

        # Add & norm (residual connection)
        norm_output = self.norm1(output + meteo_flat)

        # Feed-forward network
        ffn_output = self.ffn(norm_output)

        # Final add & norm
        final_output = self.norm2(ffn_output + norm_output)

        # Reshape back to original dimensions
        final_output = final_output.reshape(
            batch_size, time_steps, num_stations, self.meteo_feature_size
        )

        # Return average attention weights across heads for regularization
        avg_attention_weights = attention_weights.mean(dim=1)  # Average across heads
        avg_attention_weights = avg_attention_weights.reshape(
            batch_size, time_steps, num_stations, num_stations
        )

        return final_output, avg_attention_weights


class OutputTransformation(nn.Module):
    """
    A module to properly transform model outputs to the original PM2.5 scale.
    """

    def __init__(self, scalers=None):
        super(OutputTransformation, self).__init__()

        # Initialize as empty if no scalers
        if scalers is None:
            # Register empty tensors for compatibility
            self.register_buffer("scales", torch.tensor([]))
            self.register_buffer("means", torch.tensor([]))
            self.has_scalers = False
            return

        # Extract scales and means from scikit-learn scalers
        scales = []
        means = []

        for station, scaler in scalers.items():
            # PM2.5 is typically the first feature
            scales.append(scaler.scale_[0])
            means.append(scaler.mean_[0])

        # Register as buffers so they're saved with the model state
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float))
        self.register_buffer("means", torch.tensor(means, dtype=torch.float))
        self.has_scalers = True

    def forward(self, x):
        """Apply inverse transform to model predictions.

        Args:
            x: Model predictions [batch_size, num_stations, forecast_horizon]

        Returns:
            Inverse transformed predictions
        """
        if not self.has_scalers or len(self.scales) == 0:
            # If no scalers, return as is
            return x

        # Reshape for broadcasting
        scales = self.scales.view(1, -1, 1)  # [1, num_stations, 1]
        means = self.means.view(1, -1, 1)  # [1, num_stations, 1]

        # Apply inverse transform: x * scale + mean
        return x * scales + means
