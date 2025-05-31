import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class PM25Predictor:
    def __init__(self, sequence_length=24, forecast_horizon=72):
        self.sequence_length = sequence_length  # 24 hours of input
        self.forecast_horizon = forecast_horizon  # 72 hours prediction
        self.model = None
        self.target_scaler = StandardScaler()
        self.feature_scalers = {}
        self.history = None

    def create_sequences(self, data, feature_cols, target_col="PM2.5"):
        """Create sequences for LSTM training from preprocessed data"""
        sequences = []
        targets = []

        # Sort by datetime to ensure proper sequence
        data = data.sort_values("datetime")

        print(f"Creating sequences for {len(data)} records...")

        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence (24 hours)
            seq_data = data.iloc[i : i + self.sequence_length][feature_cols].values

            # Target sequence (next 72 hours)
            target_data = data.iloc[
                i
                + self.sequence_length : i
                + self.sequence_length
                + self.forecast_horizon
            ][target_col].values

            # Check if we have complete sequences
            if (
                len(target_data) == self.forecast_horizon
                and not np.isnan(seq_data).any()
                and not np.isnan(target_data).any()
            ):
                sequences.append(seq_data)
                targets.append(target_data)

        return np.array(sequences), np.array(targets)

    def prepare_data_from_enhanced(
        self, train_df, valid_df, test_df, feature_cols, target="PM2.5"
    ):
        """Prepare data from enhanced preprocessed files"""
        print("Preparing data from enhanced files...")

        # Create sequences for each dataset and each station
        X_train_list, y_train_list = [], []
        X_valid_list, y_valid_list = [], []
        X_test_list, y_test_list = [], []

        # Process training data
        print("Processing training data...")
        for station in train_df["station"].unique():
            if pd.isna(station):
                continue
            station_data = train_df[train_df["station"] == station].copy()
            if len(station_data) > self.sequence_length + self.forecast_horizon:
                X_seq, y_seq = self.create_sequences(station_data, feature_cols, target)
                if len(X_seq) > 0:
                    X_train_list.append(X_seq)
                    y_train_list.append(y_seq)

        # Process validation data
        print("Processing validation data...")
        for station in valid_df["station"].unique():
            if pd.isna(station):
                continue
            station_data = valid_df[valid_df["station"] == station].copy()
            if len(station_data) > self.sequence_length + self.forecast_horizon:
                X_seq, y_seq = self.create_sequences(station_data, feature_cols, target)
                if len(X_seq) > 0:
                    X_valid_list.append(X_seq)
                    y_valid_list.append(y_seq)

        # Process test data
        print("Processing test data...")
        for station in test_df["station"].unique():
            if pd.isna(station):
                continue
            station_data = test_df[test_df["station"] == station].copy()
            if len(station_data) > self.sequence_length + self.forecast_horizon:
                X_seq, y_seq = self.create_sequences(station_data, feature_cols, target)
                if len(X_seq) > 0:
                    X_test_list.append(X_seq)
                    y_test_list.append(y_seq)

        # Concatenate all sequences
        X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
        y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
        X_valid = np.concatenate(X_valid_list, axis=0) if X_valid_list else np.array([])
        y_valid = np.concatenate(y_valid_list, axis=0) if y_valid_list else np.array([])
        X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
        y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])

        print(f"Final shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def build_model(self, input_shape):
        """Build LSTM model for 72-hour prediction"""
        model = keras.Sequential(
            [
                keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(64, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(32),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(self.forecast_horizon),  # Output 72 hours
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train the model"""
        print("Training model...")

        # Scale targets (features already scaled in enhanced data)
        y_train_scaled = self.target_scaler.fit_transform(
            y_train.reshape(-1, 1)
        ).reshape(y_train.shape)
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).reshape(
            y_val.shape
        )

        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        print("Model Architecture:")
        self.model.summary()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6),
        ]

        # Train
        self.history = self.model.fit(
            X_train,
            y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        return self.history

    def predict(self, X):
        """Make predictions"""
        y_pred_scaled = self.model.predict(X)
        y_pred = self.target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).reshape(y_pred_scaled.shape)
        return y_pred

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)

        # Calculate metrics for each hour
        metrics = {}
        for hour in range(self.forecast_horizon):
            mse = mean_squared_error(y_test[:, hour], y_pred[:, hour])
            mae = mean_absolute_error(y_test[:, hour], y_pred[:, hour])
            r2 = r2_score(y_test[:, hour], y_pred[:, hour])

            metrics[f"hour_{hour+1}"] = {"MSE": mse, "MAE": mae, "R2": r2}

        # Overall metrics
        overall_mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
        overall_mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
        overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())

        metrics["overall"] = {"MSE": overall_mse, "MAE": overall_mae, "R2": overall_r2}

        return metrics, y_pred


def load_enhanced_data():
    """Load all enhanced preprocessed data files"""
    print("Loading enhanced preprocessed data...")

    data_dir = "../../processed"  # Adjust path as needed

    try:
        # Load dataframes
        train_df = pd.read_pickle(os.path.join(data_dir, "train_enhanced.pkl"))
        valid_df = pd.read_pickle(os.path.join(data_dir, "valid_enhanced.pkl"))
        test_df = pd.read_pickle(os.path.join(data_dir, "test_enhanced.pkl"))

        # Load scalers
        with open(os.path.join(data_dir, "scalers_enhanced.pkl"), "rb") as f:
            scalers = pickle.load(f)

        # Load adjacency matrix
        adj_matrix = np.load(os.path.join(data_dir, "adj_matrix_enhanced.npy"))

        # Load feature columns
        with open(os.path.join(data_dir, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)

        print(f"Loaded data successfully!")
        print(f"Train shape: {train_df.shape}")
        print(f"Valid shape: {valid_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Feature columns: {len(feature_cols)}")
        print(f"Adjacency matrix shape: {adj_matrix.shape}")

        return train_df, valid_df, test_df, scalers, adj_matrix, feature_cols

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the processed data files exist in the correct directory.")
        return None, None, None, None, None, None


def plot_comprehensive_analysis(
    train_df, test_df, predictor, X_test, y_test, y_pred, metrics, feature_cols
):
    """Create comprehensive plots"""

    # Combine data for overall analysis
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    # Set up the plotting area
    fig = plt.figure(figsize=(24, 28))

    # 1. Data Overview - PM2.5 over time
    plt.subplot(5, 3, 1)
    if "datetime" in all_df.columns:
        daily_avg = all_df.groupby(all_df["datetime"].dt.date)["PM2.5"].mean()
        plt.plot(daily_avg.index, daily_avg.values, alpha=0.7, linewidth=1)
        plt.title(
            "Daily Average PM2.5 Levels Over Time", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Date")
        plt.ylabel("PM2.5 (μg/m³)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    # 2. Station-wise PM2.5 distribution
    plt.subplot(5, 3, 2)
    if "station" in all_df.columns:
        stations = all_df["station"].dropna().unique()[:10]  # Top 10 stations
        station_data = [
            all_df[all_df["station"] == station]["PM2.5"].dropna()
            for station in stations
        ]
        plt.boxplot(station_data, labels=[str(s)[:10] for s in stations])
        plt.title("PM2.5 Distribution by Station", fontsize=14, fontweight="bold")
        plt.xlabel("Station")
        plt.ylabel("PM2.5 (μg/m³)")
        plt.xticks(rotation=45)

    # 3. Hourly patterns
    plt.subplot(5, 3, 3)
    if "datetime" in all_df.columns:
        all_df["hour"] = all_df["datetime"].dt.hour
        hourly_avg = all_df.groupby("hour")["PM2.5"].mean()
        plt.plot(
            hourly_avg.index, hourly_avg.values, marker="o", linewidth=2, markersize=4
        )
        plt.title("Average PM2.5 by Hour of Day", fontsize=14, fontweight="bold")
        plt.xlabel("Hour")
        plt.ylabel("PM2.5 (μg/m³)")
        plt.grid(True, alpha=0.3)

    # 4. Training history
    if predictor.history:
        plt.subplot(5, 3, 4)
        plt.plot(predictor.history.history["loss"], label="Training Loss", linewidth=2)
        plt.plot(
            predictor.history.history["val_loss"], label="Validation Loss", linewidth=2
        )
        plt.title("Model Training History", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

    # 5. Feature importance (correlation with PM2.5) - FIXED
    plt.subplot(5, 3, 5)
    try:
        # Create a safe subset of columns that exist in the dataframe
        available_features = [col for col in feature_cols if col in all_df.columns]
        if available_features and "PM2.5" in all_df.columns:
            corr_data = all_df[available_features + ["PM2.5"]].corr()["PM2.5"]
            pm25_corr = corr_data.drop("PM2.5", errors="ignore").abs()

            if len(pm25_corr) > 0:
                pm25_corr = pm25_corr.sort_values(ascending=True)
                top_features = pm25_corr.tail(10)
                plt.barh(range(len(top_features)), top_features.values, alpha=0.7)
                plt.yticks(
                    range(len(top_features)), [str(f)[:15] for f in top_features.index]
                )
                plt.title(
                    "Top 10 Features Correlated with PM2.5",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xlabel("Absolute Correlation")
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No features available for correlation",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
        else:
            plt.text(
                0.5,
                0.5,
                "PM2.5 or features not available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
    except Exception as e:
        plt.text(
            0.5,
            0.5,
            f"Error in correlation plot: {str(e)[:50]}",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    # 6. Prediction vs Actual (first 24 hours)
    plt.subplot(5, 3, 6)
    sample_idx = 0
    hours = range(1, 25)
    plt.plot(
        hours, y_test[sample_idx, :24], "o-", label="Actual", linewidth=2, markersize=4
    )
    plt.plot(
        hours,
        y_pred[sample_idx, :24],
        "s-",
        label="Predicted",
        linewidth=2,
        markersize=4,
    )
    plt.title("24-Hour Prediction Example", fontsize=14, fontweight="bold")
    plt.xlabel("Hours Ahead")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 7. Prediction vs Actual (full 72 hours)
    plt.subplot(5, 3, 7)
    hours_72 = range(1, 73)
    plt.plot(
        hours_72,
        y_test[sample_idx, :],
        "o-",
        label="Actual",
        alpha=0.8,
        linewidth=1,
        markersize=2,
    )
    plt.plot(
        hours_72,
        y_pred[sample_idx, :],
        "s-",
        label="Predicted",
        alpha=0.8,
        linewidth=1,
        markersize=2,
    )
    plt.title("72-Hour Prediction Example", fontsize=14, fontweight="bold")
    plt.xlabel("Hours Ahead")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Error by forecast horizon
    plt.subplot(5, 3, 8)
    mae_by_hour = [metrics[f"hour_{i}"]["MAE"] for i in range(1, 73)]
    plt.plot(range(1, 73), mae_by_hour, linewidth=2, color="red")
    plt.title("MAE by Forecast Horizon", fontsize=14, fontweight="bold")
    plt.xlabel("Hours Ahead")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True, alpha=0.3)

    # 9. R² by forecast horizon
    plt.subplot(5, 3, 9)
    r2_by_hour = [metrics[f"hour_{i}"]["R2"] for i in range(1, 73)]
    plt.plot(range(1, 73), r2_by_hour, color="green", linewidth=2)
    plt.title("R² Score by Forecast Horizon", fontsize=14, fontweight="bold")
    plt.xlabel("Hours Ahead")
    plt.ylabel("R² Score")
    plt.grid(True, alpha=0.3)

    # 10. Scatter plot - Predicted vs Actual
    plt.subplot(5, 3, 10)
    sample_size = min(2000, len(y_test.flatten()))
    indices = np.random.choice(len(y_test.flatten()), sample_size, replace=False)
    plt.scatter(y_test.flatten()[indices], y_pred.flatten()[indices], alpha=0.5, s=10)
    min_val = min(y_test.flatten().min(), y_pred.flatten().min())
    max_val = max(y_test.flatten().max(), y_pred.flatten().max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.title("Predicted vs Actual (All Hours)", fontsize=14, fontweight="bold")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.grid(True, alpha=0.3)

    # 11. Residuals analysis
    plt.subplot(5, 3, 11)
    residuals = y_test.flatten() - y_pred.flatten()
    plt.hist(residuals, bins=50, alpha=0.7, density=True, color="skyblue")
    plt.title("Residuals Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)

    # 12. Multiple predictions comparison
    plt.subplot(5, 3, 12)
    n_examples = min(3, len(y_test))
    for i in range(n_examples):
        plt.plot(
            range(1, 73), y_test[i, :], alpha=0.7, linewidth=1, label=f"Actual {i+1}"
        )
        plt.plot(
            range(1, 73),
            y_pred[i, :],
            "--",
            alpha=0.7,
            linewidth=1,
            label=f"Pred {i+1}",
        )
    plt.title("Multiple 72-Hour Predictions", fontsize=14, fontweight="bold")
    plt.xlabel("Hours Ahead")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 13. Error distribution by time horizon
    plt.subplot(5, 3, 13)
    error_bins = [0, 12, 24, 48, 72]
    error_labels = ["0-12h", "12-24h", "24-48h", "48-72h"]
    mae_bins = []
    for i in range(len(error_bins) - 1):
        start, end = error_bins[i], error_bins[i + 1]
        mae_bin = np.mean(
            [metrics[f"hour_{j}"]["MAE"] for j in range(start + 1, end + 1)]
        )
        mae_bins.append(mae_bin)

    plt.bar(error_labels, mae_bins, alpha=0.7, color="orange")
    plt.title("MAE by Time Periods", fontsize=14, fontweight="bold")
    plt.xlabel("Time Period")
    plt.ylabel("Mean Absolute Error")

    # 14. Feature correlation heatmap (top features) - FIXED
    plt.subplot(5, 3, 14)
    try:
        available_features = [col for col in feature_cols if col in all_df.columns]
        if available_features and "PM2.5" in all_df.columns:
            # Get correlations safely
            corr_data = all_df[available_features + ["PM2.5"]].corr()["PM2.5"]
            pm25_corr = corr_data.drop("PM2.5", errors="ignore").abs()

            if len(pm25_corr) > 0:
                pm25_corr = pm25_corr.sort_values(ascending=True)
                top_feature_names = pm25_corr.tail(8).index.tolist() + ["PM2.5"]
                available_top_features = [
                    f for f in top_feature_names if f in all_df.columns
                ]

                if len(available_top_features) > 1:
                    corr_matrix = all_df[available_top_features].corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(
                        corr_matrix,
                        mask=mask,
                        annot=True,
                        cmap="coolwarm",
                        center=0,
                        fmt=".2f",
                        square=True,
                    )
                    plt.title(
                        "Feature Correlation Matrix", fontsize=14, fontweight="bold"
                    )
                else:
                    plt.text(
                        0.5,
                        0.5,
                        "Insufficient features for heatmap",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No correlation data available",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
        else:
            plt.text(
                0.5,
                0.5,
                "Required columns not available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
    except Exception as e:
        plt.text(
            0.5,
            0.5,
            f"Error in heatmap: {str(e)[:50]}",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    # 15. Prediction confidence intervals
    plt.subplot(5, 3, 15)
    # Calculate prediction intervals
    residuals_by_hour = []
    for hour in range(72):
        residuals_hour = np.abs(y_test[:, hour] - y_pred[:, hour])
        residuals_by_hour.append(residuals_hour)

    percentiles_95 = [np.percentile(res, 95) for res in residuals_by_hour]
    percentiles_5 = [np.percentile(res, 5) for res in residuals_by_hour]
    median_error = [np.median(res) for res in residuals_by_hour]

    plt.fill_between(
        range(1, 73), percentiles_5, percentiles_95, alpha=0.3, label="90% Error Range"
    )
    plt.plot(range(1, 73), median_error, linewidth=2, label="Median Error")
    plt.title("Prediction Error Confidence Intervals", fontsize=14, fontweight="bold")
    plt.xlabel("Hours Ahead")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0)
    return fig


def print_detailed_metrics(metrics, feature_cols):
    """Print detailed performance metrics"""
    print("\n" + "=" * 60)
    print("           DETAILED MODEL PERFORMANCE METRICS")
    print("=" * 60)

    # Overall performance
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Mean Squared Error (MSE): {metrics['overall']['MSE']:.3f}")
    print(f"   Mean Absolute Error (MAE): {metrics['overall']['MAE']:.3f}")
    print(f"   R² Score: {metrics['overall']['R2']:.3f}")

    # Performance by time periods
    time_periods = [(1, 12), (13, 24), (25, 48), (49, 72)]
    period_names = [
        "Short-term (1-12h)",
        "Medium-term (13-24h)",
        "Long-term (25-48h)",
        "Extended (49-72h)",
    ]

    print(f"\n⏰ PERFORMANCE BY TIME PERIODS:")
    for i, ((start, end), name) in enumerate(zip(time_periods, period_names)):
        mae_period = np.mean(
            [metrics[f"hour_{j}"]["MAE"] for j in range(start, end + 1)]
        )
        r2_period = np.mean([metrics[f"hour_{j}"]["R2"] for j in range(start, end + 1)])
        print(f"   {name:20}: MAE={mae_period:.3f}, R²={r2_period:.3f}")

    # Best and worst performing hours
    mae_by_hour = [metrics[f"hour_{i}"]["MAE"] for i in range(1, 73)]
    r2_by_hour = [metrics[f"hour_{i}"]["R2"] for i in range(1, 73)]

    best_mae_hour = np.argmin(mae_by_hour) + 1
    worst_mae_hour = np.argmax(mae_by_hour) + 1
    best_r2_hour = np.argmax(r2_by_hour) + 1
    worst_r2_hour = np.argmin(r2_by_hour) + 1

    print(f"\n🎯 BEST/WORST PERFORMING HOURS:")
    print(f"   Best MAE: Hour {best_mae_hour} (MAE={mae_by_hour[best_mae_hour-1]:.3f})")
    print(
        f"   Worst MAE: Hour {worst_mae_hour} (MAE={mae_by_hour[worst_mae_hour-1]:.3f})"
    )
    print(f"   Best R²: Hour {best_r2_hour} (R²={r2_by_hour[best_r2_hour-1]:.3f})")
    print(f"   Worst R²: Hour {worst_r2_hour} (R²={r2_by_hour[worst_r2_hour-1]:.3f})")

    print(f"\n🔧 MODEL CONFIGURATION:")
    print(f"   Features used: {len(feature_cols)}")
    print(f"   Sequence length: 24 hours")
    print(f"   Forecast horizon: 72 hours")
    print(f"   Architecture: Multi-layer LSTM")


def main():
    """Main execution function"""
    print("=== 72-Hour PM2.5 Prediction System (Enhanced Data) ===\n")

    # Load enhanced preprocessed data
    train_df, valid_df, test_df, scalers, adj_matrix, feature_cols = (
        load_enhanced_data()
    )

    if train_df is None:
        print("Failed to load data. Please check file paths and ensure files exist.")
        return None, None, None

    print(f"\n📊 Dataset Information:")
    print(
        f"   Training period: {train_df['datetime'].min()} to {train_df['datetime'].max()}"
    )
    print(
        f"   Validation period: {valid_df['datetime'].min()} to {valid_df['datetime'].max()}"
    )
    print(f"   Test period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    print(f"   Stations: {train_df['station'].nunique()}")
    print(f"   Features: {len(feature_cols)}")

    # Initialize predictor
    predictor = PM25Predictor(sequence_length=24, forecast_horizon=72)

    # Prepare data
    X_train, y_train, X_valid, y_valid, X_test, y_test = (
        predictor.prepare_data_from_enhanced(train_df, valid_df, test_df, feature_cols)
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No valid sequences created. Check data quality and parameters.")
        return None, None, None

    print(f"\n🔄 Data Preparation Complete:")
    print(f"   Training sequences: {len(X_train)}")
    print(f"   Validation sequences: {len(X_valid)}")
    print(f"   Test sequences: {len(X_test)}")

    # Train model
    print(f"\n🚀 Starting Model Training...")
    history = predictor.train(X_train, y_train, X_valid, y_valid, epochs=2)

    # Evaluate model
    print(f"\n📈 Evaluating Model Performance...")
    metrics, y_pred = predictor.evaluate(X_test, y_test)

    # Print detailed metrics
    print_detailed_metrics(metrics, feature_cols)

    # Create comprehensive plots
    print(f"\n📊 Creating Comprehensive Visualizations...")
    fig = plot_comprehensive_analysis(
        train_df, test_df, predictor, X_test, y_test, y_pred, metrics, feature_cols
    )
    fig.savefig("air_quality.png")

    # Show plots
    plt.show()

    print(f"\n✅ Analysis Complete! Check the comprehensive visualizations above.")

    return predictor, metrics, train_df, test_df


if __name__ == "__main__":
    predictor, metrics, train_df, test_df = main()
