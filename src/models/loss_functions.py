import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true_scaled: torch.Tensor, y_pred_scaled: torch.Tensor) -> dict:
    """
    Calculate regression metrics.
    Args:
        y_true_scaled (torch.Tensor): Ground truth tensor, scaled. Shape: [B, S, 1] or [B, S, H]
        y_pred_scaled (torch.Tensor): Prediction tensor, scaled. Shape: [B, S, 1] or [B, S, H]
                                      If H > 1 (multi-step forecast), this function will typically evaluate
                                      based on the first step if y_true is for a single step.
                                      Ensure y_true and y_pred match for comparison.
    Returns:
        dict: Dictionary of metrics (mae, rmse, mape, r2)
    """
    # Detach tensors, move to CPU, and convert to numpy
    # Ensure y_true and y_pred are compatible for comparison (e.g., both [B,S,1] or flattened)

    # If y_pred_scaled has more horizon steps than y_true_scaled, take the first step from y_pred_scaled
    if (
        y_pred_scaled.shape[-1] > y_true_scaled.shape[-1]
        and y_true_scaled.shape[-1] == 1
    ):
        y_pred_for_metrics = y_pred_scaled[:, :, 0:1].cpu().detach().numpy()
    elif (
        y_pred_scaled.shape == y_true_scaled.shape
    ):  # If shapes match (e.g. both are first step)
        y_pred_for_metrics = y_pred_scaled.cpu().detach().numpy()
    else:  # Incompatible shapes for direct comparison
        logger.warning(
            f"Metrics calculation: y_pred_scaled shape {y_pred_scaled.shape} and y_true_scaled shape {y_true_scaled.shape} are not directly comparable for all steps. Attempting to use first step if possible or will result in error."
        )
        # Attempt to make them [B, S, 1] for comparison if y_true is like that
        if y_true_scaled.shape[-1] == 1 and y_pred_scaled.shape[-1] >= 1:
            y_pred_for_metrics = y_pred_scaled[:, :, 0:1].cpu().detach().numpy()
        else:  # Cannot resolve, will likely error in metric funcs or give bad results
            y_pred_for_metrics = y_pred_scaled.cpu().detach().numpy()

    y_true_np = y_true_scaled.cpu().detach().numpy()

    # Flatten for sklearn metrics if they are multi-dimensional beyond batch
    # Original shapes might be (batch_size, num_stations, num_features_target=1)
    y_true_flat = y_true_np.reshape(-1)
    y_pred_flat = y_pred_for_metrics.reshape(-1)  # This should now be compatible

    # Filter out NaNs that might have resulted from padding or issues
    valid_indices = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    if not np.any(valid_indices):
        logger.warning("No valid (non-NaN) data points for metrics calculation.")
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    y_true_filt = y_true_flat[valid_indices]
    y_pred_filt = y_pred_flat[valid_indices]

    if len(y_true_filt) == 0:  # After filtering, if nothing is left
        logger.warning(
            "No data points left after NaN filtering for metrics calculation."
        )
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    try:
        mae = mean_absolute_error(y_true_filt, y_pred_filt)
        rmse = np.sqrt(mean_squared_error(y_true_filt, y_pred_filt))
        r2 = r2_score(y_true_filt, y_pred_filt) if len(y_true_filt) >= 2 else np.nan

        # MAPE calculation, avoiding division by zero
        # Use only non-zero true values for MAPE calculation
        non_zero_true_indices = y_true_filt != 0
        if np.any(non_zero_true_indices):
            mape = (
                np.mean(
                    np.abs(
                        (
                            y_true_filt[non_zero_true_indices]
                            - y_pred_filt[non_zero_true_indices]
                        )
                        / (y_true_filt[non_zero_true_indices] + 1e-8)
                    )
                )
                * 100
            )  # Added epsilon
        else:
            mape = (
                np.nan
            )  # Or 0 if all true values are zero and predictions are also zero
            if np.all(y_pred_filt == 0):
                mape = 0.0

    except Exception as e:
        logger.warning(f"Warning: Error in metrics calculation: {e}")
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}
