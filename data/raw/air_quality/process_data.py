import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
import os
import pickle
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]
TARGET = "PM2.5"
FORECAST_HORIZON = 72  # hours
DISTANCE_THRESHOLD = 0.1  # degrees (roughly 10km)


def load_station_coordinates():
    """Load station coordinates from stations.csv."""
    stations_path = os.path.join("../../raw/air_quality", "stations.csv")
    stations = pd.read_csv(stations_path)
    # Clean column names: remove spaces and trailing commas
    stations.columns = stations.columns.str.strip().str.rstrip(",")
    return stations


def create_distance_adjacency_matrix(stations):
    """Create adjacency matrix based on station distances."""
    n_stations = len(stations)
    adj_matrix = np.zeros((n_stations, n_stations))

    # Calculate distances between stations
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            # Get coordinates
            lat1, lon1 = float(stations.iloc[i]["lat"]), float(stations.iloc[i]["lon"])
            lat2, lon2 = float(stations.iloc[j]["lat"]), float(stations.iloc[j]["lon"])

            # Calculate distance (approximate using Euclidean distance)
            distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

            # If stations are within threshold, consider them connected
            if distance < DISTANCE_THRESHOLD:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Make it symmetric

    return adj_matrix


def add_time_features(df):
    """Add cyclical time features similar to weather forecasting code."""
    logger.info("Adding time features...")

    # Ensure datetime column exists
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]], errors="coerce"
        )

    # Convert to timestamp for cyclical encoding
    timestamp_s = pd.Series(np.nan, index=df.index, dtype=float)
    valid_dates_mask = df["datetime"].notna()
    timestamp_s.loc[valid_dates_mask] = (
        df.loc[valid_dates_mask, "datetime"].astype(np.int64) // 10**9
    )

    # Time constants
    day = 24 * 60 * 60
    year = 365.2425 * day
    week = 7 * day

    # Add cyclical time features
    df["hour_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["hour_cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / week))
    df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / week))
    df["year_sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["year_cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    # Add derived time features
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    return df


def add_station_features(df, station_df):
    """Add and encode station-specific features."""
    logger.info("Adding station features...")

    # Merge station metadata
    df = df.merge(station_df, on="station", how="left", validate="many_to_one")

    # Normalize elevation (z-score normalization)
    if "elevation" in df.columns:
        df["elevation_norm"] = (df["elevation"] - df["elevation"].mean()) / df[
            "elevation"
        ].std()

        # Create elevation categories for additional features
        df["elevation_category"] = pd.cut(
            df["elevation"], bins=[0, 50, 75, 150], labels=["low", "medium", "high"]
        )

    # Encode land cover if it exists
    if "land_cover" in df.columns:
        # Create one-hot encoding for land cover types
        land_cover_dummies = pd.get_dummies(df["land_cover"], prefix="land_cover")
        df = pd.concat([df, land_cover_dummies], axis=1)

        # Also keep original as categorical
        df["land_cover_category"] = df["land_cover"].astype("category")

    # Calculate station-specific statistics (as features)
    station_stats = df.groupby("station")[TARGET].agg(["mean", "std"]).reset_index()
    station_stats.columns = [
        "station",
        f"{TARGET}_station_mean",
        f"{TARGET}_station_std",
    ]
    df = df.merge(station_stats, on="station", how="left")

    return df


def add_lag_features(
    df, station_col="station", target_col="PM2.5", lags=[1, 3, 6, 12, 24]
):
    """Add lagged features for time series forecasting."""
    logger.info("Adding lag features...")

    df_with_lags = []

    for station, group in df.groupby(station_col):
        group = group.sort_values("datetime").copy()

        # Add lag features for target variable
        for lag in lags:
            group[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)

        # Add rolling statistics
        for window in [3, 6, 12, 24]:
            group[f"{target_col}_roll_mean_{window}"] = (
                group[target_col].rolling(window=window).mean()
            )
            group[f"{target_col}_roll_std_{window}"] = (
                group[target_col].rolling(window=window).std()
            )

        df_with_lags.append(group)

    result_df = pd.concat(df_with_lags, ignore_index=True)

    # Drop rows where critical lag features are missing (first 24 hours per station)
    # Keep lag_1 but drop others that are more critical
    critical_lag_cols = [f"{target_col}_lag_{lag}" for lag in lags if lag > 1]
    result_df = result_df.dropna(subset=critical_lag_cols)

    return result_df


def add_weather_interaction_features(df):
    """Add weather interaction features."""
    logger.info("Adding weather interaction features...")

    # Handle wind direction if it exists
    if "wd" in df.columns:
        # Compass direction to degrees
        direction_degrees = {
            "n": 0,
            "nne": 22.5,
            "ne": 45,
            "ene": 67.5,
            "e": 90,
            "ese": 112.5,
            "se": 135,
            "sse": 157.5,
            "s": 180,
            "ssw": 202.5,
            "sw": 225,
            "wsw": 247.5,
            "w": 270,
            "wnw": 292.5,
            "nw": 315,
            "nnw": 337.5,
        }
        df["wd"] = df["wd"].str.strip().str.lower()
        df["wd"] = df["wd"].map(direction_degrees)

        # Wind components (if wind speed and direction available)
        if "WSPM" in df.columns:
            # Convert wind direction to radians
            wd_rad = df["wd"] * np.pi / 180
            df["wind_x"] = df["WSPM"] * np.cos(wd_rad)
            df["wind_y"] = df["WSPM"] * np.sin(wd_rad)
    elif "WSPM" in df.columns:
        # If no wind direction, create synthetic components
        df["wind_x"] = df["WSPM"] * 0.5  # Placeholder
        df["wind_y"] = df["WSPM"] * 0.5  # Placeholder

    # Temperature-humidity interaction
    if "TEMP" in df.columns and "DEWP" in df.columns:
        df["temp_dewp_diff"] = df["TEMP"] - df["DEWP"]
        # Safer humidity calculation with clipping
        temp_term = np.clip((17.625 * df["TEMP"]) / (243.04 + df["TEMP"]), -50, 50)
        dewp_term = np.clip((17.625 * df["DEWP"]) / (243.04 + df["DEWP"]), -50, 50)
        df["humidity_approx"] = np.exp(dewp_term - temp_term)

    # Pressure change (temporal derivative)
    if "PRES" in df.columns:
        df = df.sort_values(["station", "datetime"])
        df["pres_change"] = df.groupby("station")["PRES"].diff()

    # Station-elevation interactions with weather
    if "elevation_norm" in df.columns:
        if "TEMP" in df.columns:
            df["temp_elevation_interaction"] = df["TEMP"] * df["elevation_norm"]
        if "PRES" in df.columns:
            df["pres_elevation_interaction"] = df["PRES"] * df["elevation_norm"]

    return df


def split_data_by_time(df, train_end_year=2016, val_end_year=2017):
    """Split data by time periods similar to weather forecasting approach."""
    logger.info("Splitting data by time periods...")

    df["year"] = df["datetime"].dt.year

    train_df = df[df["year"] < train_end_year].copy()
    valid_df = df[df["year"] == train_end_year].copy()
    test_df = df[df["year"] >= val_end_year].copy()

    logger.info(
        f"Train: {len(train_df)} samples, Valid: {len(valid_df)} samples, Test: {len(test_df)} samples"
    )

    return train_df, valid_df, test_df


def make_dataset_air_quality(dataset_params, data, station_data=None):
    """
    Create tensorflow dataset for air quality forecasting.
    Adapted from weather forecasting make_dataset function.
    """
    if data.empty:
        logger.warning("Empty dataset provided")
        # Return empty dataset with correct structure
        empty_data = np.empty((0, dataset_params["lags"], len(dataset_params["xcols"])))
        empty_targets = np.empty(
            (0, dataset_params["steps_ahead"], len(dataset_params["ycols"]))
        )

        ds = tf.data.Dataset.from_tensor_slices((empty_data, empty_targets))
        return ds.batch(dataset_params["bs"])

    y_cols = dataset_params["ycols"]
    total_window_size = dataset_params["lags"] + dataset_params["steps_ahead"]

    # Use feature columns from dataset_params
    feature_cols = dataset_params["xcols"]

    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing columns in data: {missing_cols}")
        # Create dummy columns filled with zeros
        for col in missing_cols:
            data[col] = 0.0

    data_features = data[feature_cols].copy()

    # Handle any remaining missing values
    data_features = data_features.fillna(0.0)

    data_np = np.array(data_features, dtype=np.float32)

    # Check if we have enough data for the window size
    if len(data_np) < total_window_size:
        logger.warning(
            f"Not enough data for window size {total_window_size}, got {len(data_np)}"
        )
        # Return empty dataset
        empty_data = np.empty((0, dataset_params["lags"], len(feature_cols)))
        empty_targets = np.empty((0, dataset_params["steps_ahead"], len(y_cols)))
        ds = tf.data.Dataset.from_tensor_slices((empty_data, empty_targets))
        return ds.batch(dataset_params["bs"])

    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data_np,
        targets=None,
        sequence_length=total_window_size,
        sequence_stride=1,
        shuffle=dataset_params.get("shuffle", True),
        batch_size=dataset_params["bs"],
    )

    col_indices = {name: i for i, name in enumerate(data_features.columns)}
    X_slice = slice(0, dataset_params["lags"])
    y_start = total_window_size - dataset_params["steps_ahead"]
    y_slice = slice(y_start, None)

    def split_window(features):
        X = features[:, X_slice, :]
        y = features[:, y_slice, :]

        # Extract target columns
        y_indices = [col_indices[name] for name in y_cols if name in col_indices]
        if y_indices:
            y = tf.stack([y[:, :, idx] for idx in y_indices], axis=-1)
        else:
            # If target columns not found, use first column as fallback
            logger.warning(f"Target columns {y_cols} not found, using first column")
            y = y[:, :, 0:1]

        X.set_shape([None, dataset_params["lags"], len(feature_cols)])
        y.set_shape([None, dataset_params["steps_ahead"], len(y_cols)])

        return X, y

    ds = ds.map(split_window)
    return ds


def plot_air_quality_examples(data, x_var="datetime", stations=None, features=None):
    """Plot examples of air quality data similar to weather plotting."""
    # Ensure datetime column is in proper format
    data[x_var] = pd.to_datetime(data[x_var], errors="coerce")

    # Drop rows with invalid datetime
    data = data.dropna(subset=[x_var])

    # Auto-select stations if not given
    available_stations = data["station"].dropna().unique()
    available_features = data.select_dtypes(include=[np.number]).columns.tolist()

    if stations is None:
        stations = available_stations[:2]
    if features is None:
        features = [f for f in available_features if f not in ["No"]][:3]

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex="col")
    axs = axs.reshape(2, 3)

    for row, station in enumerate(stations):
        df_station = data[data["station"] == station].sort_values(by=x_var)

        for col, feature in enumerate(features):
            ax = axs[row][col]
            if feature in df_station and not df_station[feature].isna().all():
                ax.plot(
                    df_station[x_var],
                    df_station[feature],
                    label=feature,
                    color="tab:blue",
                )
            ax.set_title(f"{station} - {feature}")
            ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    return fig


def load_raw_data(data_dir: str = "../../raw/air_quality") -> pd.DataFrame:
    """
    Load and combine all raw data files.

    Args:
        data_dir: Directory containing raw data files

    Returns:
        Combined DataFrame with all raw data
    """
    logger.info("Loading raw data files...")

    # Match only air quality time series files
    time_series_files = glob(os.path.join(data_dir, "PRSA_Data_*.csv"))

    if not time_series_files:
        logger.error("No time series data files found!")
        return pd.DataFrame()

    # Combine all files
    df = pd.concat((pd.read_csv(f) for f in time_series_files), ignore_index=True)
    logger.info(
        f"Loaded {len(time_series_files)} time series files with {len(df)} total records"
    )

    return df


def preprocess_data(df: pd.DataFrame, station_df: pd.DataFrame) -> tuple:
    """
    Preprocess the raw data:
    1. Handle missing values
    2. Add station features
    3. Normalize features per station
    4. Create adjacency matrix
    5. Create datasets

    Args:
        df: Raw DataFrame
        station_df: Station metadata DataFrame

    Returns:
        Tuple of datasets and metadata
    """
    logger.info("Preprocessing data...")

    # Create datetime column
    df = add_time_features(df)

    # Add station features early in the pipeline
    df = add_station_features(df, station_df)

    # Handle missing values
    df = df.replace([-999, -999.0, "NA", "N/A", "NaN", "nan", ""], np.nan)
    df = df.dropna(subset=[TARGET])

    missing_percent = df.isnull().mean() * 100
    logger.info(f"Missing value percentages:\n{missing_percent[missing_percent > 0]}")

    # For columns with moderate missingness, use advanced imputation
    high_missing_cols = missing_percent[missing_percent > 30].index.tolist()
    moderate_missing_cols = [
        col
        for col in missing_percent[
            (missing_percent > 0) & (missing_percent <= 30)
        ].index
        if col in FEATURES
    ]

    logger.info(f"Columns with high missingness (>30%): {high_missing_cols}")
    logger.info(f"Columns with moderate missingness: {moderate_missing_cols}")

    df = df.drop(columns=high_missing_cols)

    # Add weather interaction features
    df = add_weather_interaction_features(df)

    # Add lag features
    df = add_lag_features(df)

    # Split data by time
    train_df, valid_df, test_df = split_data_by_time(df)

    # Create feature list (excluding non-predictive columns)
    exclude_cols = [
        "datetime",
        "station",
        "year",
        "month",
        "day",
        "hour",
        "No",
        "wd",
        "geometry",
        "elevation",
        "land_cover",
        "elevation_category",
        "land_cover_category",  # Exclude raw categorical
    ]
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols and not col.endswith("_lag_1")
    ]

    logger.info(f"Selected {len(feature_cols)} feature columns")
    logger.info(
        f"Feature columns include station features: {[col for col in feature_cols if 'elevation' in col or 'land_cover' in col]}"
    )

    # Imputation and normalization
    imputer = KNNImputer(n_neighbors=5)

    for station, group in train_df.groupby("station"):
        if len(group) > 0:  # Only impute if we have data
            station_idx = group.index
            imputed_data = imputer.fit_transform(group[feature_cols])
            train_df.loc[station_idx, feature_cols] = imputed_data

    # For validation and test, use the same imputer fitted on training data
    for df_split in [valid_df, test_df]:
        for station, group in df_split.groupby("station"):
            if len(group) > 0:
                station_idx = group.index
                if station in train_df["station"].unique():
                    imputed_data = imputer.transform(group[feature_cols])
                    df_split.loc[station_idx, feature_cols] = imputed_data
                else:
                    # For new stations, use simple median imputation
                    df_split.loc[station_idx, feature_cols] = group[
                        feature_cols
                    ].fillna(group[feature_cols].median())

    # Fill any remaining missing values by station
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df.groupby("station")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both").fillna(
                    value=x.mean()
                )
            )

    logger.info(f"NaNs after preprocessing: \n{df[feature_cols].isna().sum().sum()}")

    # Normalize features per station
    logger.info("Normalizing features per station...")
    scalers = {}
    normalized_dfs = []

    for split_name, split_df in [
        ("train", train_df),
        ("valid", valid_df),
        ("test", test_df),
    ]:
        normalized_stations = []

        for station, group in split_df.groupby("station"):
            if split_name == "train":
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(group[feature_cols])
                scalers[station] = scaler
            else:
                if station in scalers:
                    features_scaled = scalers[station].transform(group[feature_cols])
                else:
                    # Handle stations not seen in training
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(group[feature_cols])
                    scalers[station] = scaler

            temp_df = pd.DataFrame(
                features_scaled, columns=feature_cols, index=group.index
            )
            temp_df["datetime"] = group["datetime"]
            temp_df["station"] = station
            normalized_stations.append(temp_df)

        if normalized_stations:  # Only concat if we have data
            normalized_dfs.append(pd.concat(normalized_stations, ignore_index=True))
        else:
            # Create empty dataframe with correct columns
            empty_df = pd.DataFrame(columns=feature_cols + ["datetime", "station"])
            normalized_dfs.append(empty_df)

    train_norm, valid_norm, test_norm = normalized_dfs

    # Create adjacency matrix based on station distances
    logger.info("Creating adjacency matrix...")
    stations_df = load_station_coordinates()
    adj_matrix = create_distance_adjacency_matrix(stations_df)

    logger.info(f"Adjacency matrix shape: {adj_matrix.shape}")
    logger.info(f"Total connections: {np.sum(adj_matrix)}")
    logger.info(
        f"Average connections per station: {np.mean(np.sum(adj_matrix, axis=1)):.2f}"
    )

    # Prepare data for TensorFlow datasets
    datasets = {
        "train": train_norm,
        "valid": valid_norm,
        "test": test_norm,
        "feature_cols": feature_cols,
        "scalers": scalers,
        "adj_matrix": adj_matrix,
    }

    return datasets


def create_model_configs():
    """Create model configuration dictionaries."""
    base_config = {
        "lags": 24,  # Look back 24 hours
        "steps_ahead": 12,  # Predict 12 hours ahead
        "bs": 32,  # Batch size
        "ycols": ["PM2.5"],  # Target columns
        "xcols": None,  # Will be set dynamically
        "shuffle": True,
        "feat_maps": 64,
        "filters": 32,
        "kern_size": 3,
        "drop_out": 0.1,
        "kern_reg": 0.001,
        "recu_reg": 0.001,
        "model_type": "lstm",
    }

    return base_config


def main():
    """Main function to process raw data."""
    # Load raw data
    df = load_raw_data()
    if df.empty:
        return

    # Load station metadata
    station_df = pd.read_csv(
        os.path.join("../../processed", "stations_with_features.csv")
    )
    print("Station columns:", station_df.columns.tolist())
    print("Station data summary:")
    print(
        f"Elevation range: {station_df['elevation'].min():.1f} - {station_df['elevation'].max():.1f}"
    )
    print(f"Land cover types: {sorted(station_df['land_cover'].unique())}")

    # Verify station names match
    print(f"Stations in time series: {sorted(df['station'].unique())}")
    print(f"Stations in metadata: {sorted(station_df['station'].unique())}")

    # Preprocess data with station features
    datasets = preprocess_data(df, station_df)
    if datasets is None:
        logger.error("Failed to process data")
        return

    # Create model configuration
    config = create_model_configs()
    config["xcols"] = datasets["feature_cols"]

    logger.info(f"Final feature set includes {len(config['xcols'])} features:")
    station_features = [
        col
        for col in config["xcols"]
        if any(x in col for x in ["elevation", "land_cover", "station"])
    ]
    logger.info(f"Station-specific features: {station_features}")

    # Create TensorFlow datasets
    train_ds = make_dataset_air_quality(config, datasets["train"])
    valid_ds = make_dataset_air_quality({**config, "shuffle": False}, datasets["valid"])
    test_ds = make_dataset_air_quality({**config, "shuffle": False}, datasets["test"])

    # Plot examples
    fig = plot_air_quality_examples(
        datasets["train"],
        stations=["Aotizhongxin", "Tiantan"],
        features=["PM2.5", "TEMP", "WSPM"],
    )

    # Save processed data
    output_dir = "../../processed"
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    datasets["train"].to_pickle(os.path.join(output_dir, "train_enhanced.pkl"))
    datasets["valid"].to_pickle(os.path.join(output_dir, "valid_enhanced.pkl"))
    datasets["test"].to_pickle(os.path.join(output_dir, "test_enhanced.pkl"))

    datasets["train"].to_csv(
        os.path.join(output_dir, "train_enhanced.csv"), index=False
    )

    fig.savefig("air_quality_examples.png")

    # Save scalers and adjacency matrix
    with open(os.path.join(output_dir, "scalers_enhanced.pkl"), "wb") as f:
        pickle.dump(datasets["scalers"], f)

    np.save(os.path.join(output_dir, "adj_matrix_enhanced.npy"), datasets["adj_matrix"])

    # Save feature columns and config
    with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(datasets["feature_cols"], f)

    with open(os.path.join(output_dir, "model_config.pkl"), "wb") as f:
        pickle.dump(config, f)

    logger.info("Enhanced preprocessing complete with station features integrated!")

    return datasets, [train_ds, valid_ds, test_ds], config


if __name__ == "__main__":
    datasets, tf_datasets, config = main()
