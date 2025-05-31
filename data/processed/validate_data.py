import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")

# Enhanced feature columns
ENHANCED_FEATURES = [
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
    "lat",
    "lon",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "year_sin",
    "year_cos",
    "dayofweek",
    "is_weekend",
    "wind_x",
    "wind_y",
    "temp_dewp_diff",
    "humidity_approx",
    "pres_change",
    "PM2.5_station_mean",
    "pres_elevation_interaction",
    "elevation_norm",
    "PM2.5_station_std",
    "temp_elevation_interaction",
    "PM2.5_lag_3",
    "PM2.5_lag_6",
    "PM2.5_lag_12",
    "PM2.5_lag_24",
    "PM2.5_roll_mean_3",
    "PM2.5_roll_std_3",
    "PM2.5_roll_mean_6",
    "PM2.5_roll_std_6",
    "PM2.5_roll_mean_12",
    "PM2.5_roll_std_12",
    "PM2.5_roll_mean_24",
    "PM2.5_roll_std_24",
]


def create_log_file():
    """Create a log file for validation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"validation_log_{timestamp}.txt"
    return open(log_file, "w", encoding="utf-8"), log_file


def log_and_print(message, log_file=None):
    """Print message and write to log file."""
    print(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_enhanced_data():
    """Load all enhanced data files."""
    data = {}
    log_file, log_filename = create_log_file()

    log_and_print("=" * 80, log_file)
    log_and_print("ENHANCED DATASET VALIDATION", log_file)
    log_and_print("=" * 80, log_file)
    log_and_print(f"Validation started at: {datetime.now()}", log_file)
    log_and_print("", log_file)

    # Load stations with features
    try:
        stations_path = Path("stations_with_features.csv")
        data["stations"] = pd.read_csv(stations_path)
        log_and_print(
            f"[OK] Loaded stations data: {len(data['stations'])} stations", log_file
        )
    except Exception as e:
        log_and_print(f"[ERROR] Error loading stations data: {str(e)}", log_file)
        return None, log_file, log_filename

    # Load adjacency matrix
    try:
        data["adj_matrix"] = np.load("adj_matrix_enhanced.npy")
        log_and_print(
            f"[OK] Loaded enhanced adjacency matrix: shape {data['adj_matrix'].shape}",
            log_file,
        )
    except Exception as e:
        log_and_print(f"[ERROR] Error loading adjacency matrix: {str(e)}", log_file)
        return None, log_file, log_filename

    # Load feature columns
    try:
        with open("feature_cols.pkl", "rb") as f:
            data["feature_cols"] = pickle.load(f)
        log_and_print(
            f"[OK] Loaded feature columns: {len(data['feature_cols'])} features",
            log_file,
        )
    except Exception as e:
        log_and_print(f"[ERROR] Error loading feature columns: {str(e)}", log_file)
        return None, log_file, log_filename

    # Load enhanced scalers
    try:
        with open("scalers_enhanced.pkl", "rb") as f:
            data["scalers"] = pickle.load(f)
        log_and_print(f"[OK] Loaded enhanced scalers", log_file)
    except Exception as e:
        log_and_print(f"[ERROR] Error loading scalers: {str(e)}", log_file)
        return None, log_file, log_filename

    # Load train/validation/test datasets
    datasets = ["train_enhanced.pkl", "valid_enhanced.pkl", "test_enhanced.pkl"]
    for dataset_name in datasets:
        try:
            with open(dataset_name, "rb") as f:
                data[dataset_name.replace("_enhanced.pkl", "")] = pickle.load(f)
            log_and_print(f"[OK] Loaded {dataset_name}", log_file)
        except Exception as e:
            log_and_print(f"[ERROR] Error loading {dataset_name}: {str(e)}", log_file)
            return None, log_file, log_filename

    return data, log_file, log_filename


def validate_stations(data, log_file):
    """Validate stations data."""
    stations = data["stations"]
    log_and_print("\n" + "=" * 50, log_file)
    log_and_print("STATIONS DATA VALIDATION", log_file)
    log_and_print("=" * 50, log_file)

    # Basic info
    log_and_print(f"Number of stations: {len(stations)}", log_file)
    log_and_print(f"Columns: {list(stations.columns)}", log_file)

    # Check for missing values
    missing = stations.isnull().sum()
    if missing.any():
        log_and_print("[ERROR] Found missing values:", log_file)
        for col, count in missing[missing > 0].items():
            log_and_print(f"  {col}: {count} missing values", log_file)
    else:
        log_and_print("[OK] No missing values found", log_file)

    # Data types
    log_and_print("\nData types:", log_file)
    for col, dtype in stations.dtypes.items():
        log_and_print(f"  {col}: {dtype}", log_file)

    # Geographic coordinates validation
    if "lat" in stations.columns and "lon" in stations.columns:
        lat_range = (stations["lat"].min(), stations["lat"].max())
        lon_range = (stations["lon"].min(), stations["lon"].max())
        log_and_print(f"\nGeographic ranges:", log_file)
        log_and_print(f"  Latitude: {lat_range[0]:.6f} to {lat_range[1]:.6f}", log_file)
        log_and_print(
            f"  Longitude: {lon_range[0]:.6f} to {lon_range[1]:.6f}", log_file
        )

        # Check if coordinates are reasonable for China
        if not (15 <= lat_range[0] and lat_range[1] <= 55):
            log_and_print(
                "[WARNING] Latitude values seem outside China's range", log_file
            )
        if not (70 <= lon_range[0] and lon_range[1] <= 140):
            log_and_print(
                "[WARNING] Longitude values seem outside China's range", log_file
            )

    # Connectivity analysis
    if "connectivity" in stations.columns:
        conn_stats = stations["connectivity"].describe()
        log_and_print(f"\nConnectivity statistics:", log_file)
        for stat, value in conn_stats.items():
            log_and_print(f"  {stat}: {value:.2f}", log_file)

    # Create station location plot
    if "lat" in stations.columns and "lon" in stations.columns:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            stations["lon"],
            stations["lat"],
            c=stations.get("connectivity", range(len(stations))),
            cmap="viridis",
            alpha=0.7,
            s=100,
        )
        plt.colorbar(
            scatter,
            label=(
                "Connectivity"
                if "connectivity" in stations.columns
                else "Station Index"
            ),
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Station Locations and Connectivity")
        plt.grid(True, alpha=0.3)
        for i, row in stations.iterrows():
            plt.annotate(
                row["station"] if "station" in stations.columns else f"S{i}",
                (row["lon"], row["lat"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )
        plt.tight_layout()
        plt.savefig("station_locations.png", dpi=300, bbox_inches="tight")
        plt.close()
        log_and_print(
            "[OK] Created station locations plot: station_locations.png", log_file
        )


def validate_feature_columns(data, log_file):
    """Validate feature columns."""
    feature_cols = data["feature_cols"]
    log_and_print("\n" + "=" * 50, log_file)
    log_and_print("FEATURE COLUMNS VALIDATION", log_file)
    log_and_print("=" * 50, log_file)

    log_and_print(f"Number of features: {len(feature_cols)}", log_file)
    log_and_print(f"Expected features: {len(ENHANCED_FEATURES)}", log_file)

    # Check if all expected features are present
    missing_features = set(ENHANCED_FEATURES) - set(feature_cols)
    extra_features = set(feature_cols) - set(ENHANCED_FEATURES)

    if missing_features:
        log_and_print(
            f"[ERROR] Missing expected features: {list(missing_features)}", log_file
        )
    else:
        log_and_print("[OK] All expected features present", log_file)

    if extra_features:
        log_and_print(f"[INFO] Extra features found: {list(extra_features)}", log_file)

    # Categorize features
    feature_categories = {
        "Original Pollutants": ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"],
        "Weather": ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"],
        "Location": [
            "lat",
            "lon",
            "elevation_norm",
        ],
        "Temporal": [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "year_sin",
            "year_cos",
            "dayofweek",
            "is_weekend",
        ],
        "Engineered Weather": [
            "wind_x",
            "wind_y",
            "temp_dewp_diff",
            "humidity_approx",
            "pres_change",
            "PM2.5_station_mean",
            "temp_elevation_interaction",
            "PM2.5_station_std",
            "pres_elevation_interaction",
        ],
        "PM2.5 Lag Features": [f"PM2.5_lag_{i}" for i in [3, 6, 12, 24]],
        "PM2.5 Rolling Stats": [
            f"PM2.5_roll_{stat}_{window}"
            for stat in ["mean", "std"]
            for window in [3, 6, 12, 24]
        ],
    }

    log_and_print("\nFeature categories:", log_file)
    for category, features in feature_categories.items():
        present_features = [f for f in features if f in feature_cols]
        log_and_print(
            f"  {category}: {len(present_features)}/{len(features)} features", log_file
        )
        if len(present_features) != len(features):
            missing = set(features) - set(present_features)
            log_and_print(f"    Missing: {list(missing)}", log_file)


def validate_datasets(data, log_file):
    """Validate train/validation/test datasets."""
    log_and_print("\n" + "=" * 50, log_file)
    log_and_print("DATASETS VALIDATION", log_file)
    log_and_print("=" * 50, log_file)

    datasets = ["train", "valid", "test"]
    total_samples = 0
    total_missing = 0

    for dataset_name in datasets:
        if dataset_name in data:
            dataset = data[dataset_name]
            log_and_print(f"\n{dataset_name.upper()} Dataset:", log_file)

            # Check if it's a pandas DataFrame
            if isinstance(dataset, pd.DataFrame):
                log_and_print(f"  Format: pandas DataFrame", log_file)
                log_and_print(f"  Shape: {dataset.shape}", log_file)
                log_and_print(f"  Columns: {len(dataset.columns)}", log_file)
                total_samples += len(dataset)

                # Check for missing values with detailed analysis
                missing_count = dataset.isnull().sum().sum()
                total_missing += missing_count

                if missing_count > 0:
                    log_and_print(
                        f"    [INFO] Contains {missing_count} missing values", log_file
                    )

                    # Detailed missing value analysis
                    missing_by_column = dataset.isnull().sum()
                    cols_with_missing = missing_by_column[missing_by_column > 0]

                    if len(cols_with_missing) > 0:
                        log_and_print(f"    Missing values by column:", log_file)
                        for col, count in cols_with_missing.head(
                            10
                        ).items():  # Show top 10
                            percentage = (count / len(dataset)) * 100
                            log_and_print(
                                f"      {col}: {count} ({percentage:.2f}%)", log_file
                            )

                        if len(cols_with_missing) > 10:
                            log_and_print(
                                f"      ... and {len(cols_with_missing) - 10} more columns",
                                log_file,
                            )

                    # Check if missing values are in critical columns
                    critical_cols = ["PM2.5", "PM10", "NO2", "SO2"]
                    critical_missing = [
                        col for col in critical_cols if col in cols_with_missing
                    ]

                    if critical_missing:
                        log_and_print(
                            f"    [WARNING] Missing values in critical columns: {critical_missing}",
                            log_file,
                        )
                    else:
                        log_and_print(
                            f"    [OK] No missing values in critical pollutant columns",
                            log_file,
                        )

                else:
                    log_and_print(f"    [OK] No missing values", log_file)

                # Basic statistics for numeric columns
                numeric_cols = dataset.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    log_and_print(f"  Numeric columns: {len(numeric_cols)}", log_file)
                    # Show statistics for a few key columns if they exist
                    for col in ["PM2.5", "PM10", "TEMP", "PRES"][
                        :2
                    ]:  # Limit to avoid too much output
                        if col in dataset.columns:
                            col_data = dataset[
                                col
                            ].dropna()  # Remove NaN for statistics
                            if len(col_data) > 0:
                                log_and_print(
                                    f"    {col}: range [{col_data.min():.4f}, {col_data.max():.4f}], "
                                    f"mean {col_data.mean():.4f}, std {col_data.std():.4f}",
                                    log_file,
                                )
                            else:
                                log_and_print(
                                    f"    {col}: All values are missing", log_file
                                )

            # Check if it's a dictionary with expected keys (original expected format)
            elif isinstance(dataset, dict):
                for key, value in dataset.items():
                    if isinstance(value, np.ndarray):
                        log_and_print(
                            f"  {key}: shape {value.shape}, dtype {value.dtype}",
                            log_file,
                        )
                        total_samples += value.shape[0] if len(value.shape) > 0 else 0

                        # Check for NaN values
                        if np.isnan(value).any():
                            nan_count = np.isnan(value).sum()
                            total_missing += nan_count
                            log_and_print(
                                f"    [INFO] Contains {nan_count} NaN values ({(nan_count/value.size)*100:.2f}%)",
                                log_file,
                            )
                        else:
                            log_and_print(f"    [OK] No NaN values", log_file)

                        # Basic statistics (excluding NaN values)
                        if value.size > 0:
                            valid_data = value[~np.isnan(value)]
                            if len(valid_data) > 0:
                                log_and_print(
                                    f"    Range: [{valid_data.min():.4f}, {valid_data.max():.4f}]",
                                    log_file,
                                )
                                log_and_print(
                                    f"    Mean: {valid_data.mean():.4f}, Std: {valid_data.std():.4f}",
                                    log_file,
                                )
                            else:
                                log_and_print(f"    All values are NaN", log_file)
                    else:
                        log_and_print(
                            f"  {key}: {type(value)} - {value if not hasattr(value, '__len__') else f'length {len(value)}'}",
                            log_file,
                        )
            else:
                log_and_print(f"  Unexpected format: {type(dataset)}", log_file)

    log_and_print(f"\nTotal samples across all datasets: {total_samples}", log_file)
    log_and_print(
        f"Total missing values across all datasets: {total_missing}", log_file
    )

    if total_missing > 0:
        missing_percentage = (
            total_missing / (total_samples * len(data.get("feature_cols", [])))
        ) * 100
        log_and_print(
            f"Overall missing data percentage: {missing_percentage:.4f}%", log_file
        )

    # Create dataset size comparison plot
    if all(dataset_name in data for dataset_name in datasets):
        sizes = []
        labels = []
        for dataset_name in datasets:
            dataset = data[dataset_name]
            if isinstance(dataset, pd.DataFrame):
                sizes.append(len(dataset))
                labels.append(f"{dataset_name.capitalize()}\n({len(dataset)} samples)")
            elif isinstance(dataset, dict) and "X" in dataset:
                sizes.append(dataset["X"].shape[0])
                labels.append(
                    f"{dataset_name.capitalize()}\n({dataset['X'].shape[0]} samples)"
                )
            else:
                sizes.append(0)
                labels.append(f"{dataset_name.capitalize()}\n(0 samples)")

        plt.figure(figsize=(10, 6))
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        bars = plt.bar(labels, sizes, color=colors, alpha=0.8)
        plt.title("Dataset Size Distribution")
        plt.ylabel("Number of Samples")

        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sizes) * 0.01,
                str(size),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig("dataset_sizes.png", dpi=300, bbox_inches="tight")
        plt.close()
        log_and_print("[OK] Created dataset size plot: dataset_sizes.png", log_file)


def validate_adjacency_matrix(data, log_file):
    """Validate adjacency matrix."""
    adj_matrix = data["adj_matrix"]
    log_and_print("\n" + "=" * 50, log_file)
    log_and_print("ADJACENCY MATRIX VALIDATION", log_file)
    log_and_print("=" * 50, log_file)

    # Basic properties
    log_and_print(f"Shape: {adj_matrix.shape}", log_file)
    log_and_print(f"Data type: {adj_matrix.dtype}", log_file)
    log_and_print(f"Memory usage: {adj_matrix.nbytes / (1024**2):.2f} MB", log_file)

    # Check if square matrix
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        log_and_print("[ERROR] Matrix is not square!", log_file)
    else:
        log_and_print("[OK] Matrix is square", log_file)

    # Check symmetry
    is_symmetric = np.allclose(adj_matrix, adj_matrix.T, rtol=1e-10)
    log_and_print(f"Symmetric: {'[OK]' if is_symmetric else '[ERROR]'}", log_file)

    # Check diagonal (should be zeros for adjacency matrix)
    diagonal_sum = np.sum(np.diag(adj_matrix))
    log_and_print(f"Diagonal sum: {diagonal_sum} (should be 0)", log_file)

    # Value statistics
    log_and_print(f"\nValue statistics:", log_file)
    log_and_print(f"  Min: {adj_matrix.min():.6f}", log_file)
    log_and_print(f"  Max: {adj_matrix.max():.6f}", log_file)
    log_and_print(f"  Mean: {adj_matrix.mean():.6f}", log_file)
    log_and_print(f"  Std: {adj_matrix.std():.6f}", log_file)
    log_and_print(f"  Non-zero elements: {np.count_nonzero(adj_matrix)}", log_file)
    log_and_print(
        f"  Sparsity: {(1 - np.count_nonzero(adj_matrix) / adj_matrix.size) * 100:.2f}%",
        log_file,
    )

    # Connection statistics
    connections_per_node = np.sum(adj_matrix > 0, axis=1)
    log_and_print(f"\nConnections per station:", log_file)
    log_and_print(f"  Mean: {connections_per_node.mean():.2f}", log_file)
    log_and_print(f"  Min: {connections_per_node.min()}", log_file)
    log_and_print(f"  Max: {connections_per_node.max()}", log_file)

    # Detailed connection info
    if "stations" in data and len(data["stations"]) == adj_matrix.shape[0]:
        stations = data["stations"]
        station_names = (
            stations["station"].tolist()
            if "station" in stations.columns
            else [f"Station_{i}" for i in range(len(stations))]
        )

        log_and_print(f"\nDetailed connections per station:", log_file)
        for i, (name, conn_count) in enumerate(
            zip(station_names, connections_per_node)
        ):
            log_and_print(f"  {name}: {conn_count} connections", log_file)

    # Create adjacency matrix heatmap
    plt.figure(figsize=(12, 10))
    station_names = None
    if "stations" in data and len(data["stations"]) == adj_matrix.shape[0]:
        stations = data["stations"]
        station_names = (
            stations["station"].tolist() if "station" in stations.columns else None
        )

    sns.heatmap(
        adj_matrix,
        cmap="YlOrRd",
        xticklabels=station_names if station_names else False,
        yticklabels=station_names if station_names else False,
        cbar_kws={"label": "Connection Strength"},
    )
    plt.title("Enhanced Adjacency Matrix")
    if station_names:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("adjacency_matrix_enhanced.png", dpi=300, bbox_inches="tight")
    plt.close()
    log_and_print(
        "[OK] Created adjacency matrix heatmap: adjacency_matrix_enhanced.png", log_file
    )

    # Create connection distribution plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(
        connections_per_node,
        bins=max(1, len(set(connections_per_node))),
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Number of Connections")
    plt.ylabel("Number of Stations")
    plt.title("Distribution of Station Connections")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    non_zero_values = adj_matrix[adj_matrix > 0]
    if len(non_zero_values) > 0:
        plt.hist(
            non_zero_values, bins=30, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        plt.xlabel("Connection Strength")
        plt.ylabel("Frequency")
        plt.title("Distribution of Non-zero Connection Strengths")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("connection_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    log_and_print(
        "[OK] Created connection analysis plot: connection_analysis.png", log_file
    )


def validate_scalers(data, log_file):
    """Validate scaler objects."""
    scalers = data["scalers"]
    log_and_print("\n" + "=" * 50, log_file)
    log_and_print("SCALERS VALIDATION", log_file)
    log_and_print("=" * 50, log_file)

    if not isinstance(scalers, dict):
        log_and_print("[ERROR] Scalers should be a dictionary", log_file)
        return

    log_and_print(f"Number of scalers: {len(scalers)}", log_file)

    for name, scaler in scalers.items():
        log_and_print(f"\nScaler: {name}", log_file)
        log_and_print(f"  Type: {type(scaler).__name__}", log_file)

        # Check common scaler attributes
        if hasattr(scaler, "scale_"):
            log_and_print(
                f"  Scale shape: {scaler.scale_.shape if hasattr(scaler.scale_, 'shape') else 'scalar'}",
                log_file,
            )
            log_and_print(
                f"  Scale range: [{scaler.scale_.min():.6f}, {scaler.scale_.max():.6f}]",
                log_file,
            )

        if hasattr(scaler, "mean_"):
            log_and_print(
                f"  Mean shape: {scaler.mean_.shape if hasattr(scaler.mean_, 'shape') else 'scalar'}",
                log_file,
            )
            log_and_print(
                f"  Mean range: [{scaler.mean_.min():.6f}, {scaler.mean_.max():.6f}]",
                log_file,
            )

        if hasattr(scaler, "data_min_"):
            log_and_print(
                f"  Data min: [{scaler.data_min_.min():.6f}, {scaler.data_min_.max():.6f}]",
                log_file,
            )

        if hasattr(scaler, "data_max_"):
            log_and_print(
                f"  Data max: [{scaler.data_max_.min():.6f}, {scaler.data_max_.max():.6f}]",
                log_file,
            )


def create_comprehensive_summary(data, log_file):
    """Create a comprehensive summary of the dataset."""
    log_and_print("\n" + "=" * 80, log_file)
    log_and_print("COMPREHENSIVE DATASET SUMMARY", log_file)
    log_and_print("=" * 80, log_file)

    # Dataset overview
    total_stations = len(data["stations"]) if "stations" in data else 0
    total_features = len(data["feature_cols"]) if "feature_cols" in data else 0

    # Count total samples and missing values
    total_samples = 0
    total_missing = 0
    for dataset_name in ["train", "valid", "test"]:
        if dataset_name in data:
            dataset = data[dataset_name]
            if isinstance(dataset, pd.DataFrame):
                total_samples += len(dataset)
                total_missing += dataset.isnull().sum().sum()
            elif isinstance(dataset, dict) and "X" in dataset:
                total_samples += dataset["X"].shape[0]
                if np.isnan(dataset["X"]).any():
                    total_missing += np.isnan(dataset["X"]).sum()

    log_and_print(f"Dataset Overview:", log_file)
    log_and_print(f"  Total Stations: {total_stations}", log_file)
    log_and_print(f"  Total Features: {total_features}", log_file)
    log_and_print(f"  Total Samples: {total_samples}", log_file)
    log_and_print(f"  Total Missing Values: {total_missing}", log_file)

    if total_missing > 0 and total_samples > 0 and total_features > 0:
        missing_percentage = (total_missing / (total_samples * total_features)) * 100
        log_and_print(f"  Missing Data Percentage: {missing_percentage:.4f}%", log_file)

    if "adj_matrix" in data:
        sparsity = (
            1 - np.count_nonzero(data["adj_matrix"]) / data["adj_matrix"].size
        ) * 100
        log_and_print(f"  Network Sparsity: {sparsity:.2f}%", log_file)

    # Memory usage estimation
    total_memory = 0
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            total_memory += value.nbytes
        elif isinstance(value, pd.DataFrame):
            total_memory += value.memory_usage(deep=True).sum()
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    total_memory += subvalue.nbytes

    log_and_print(
        f"  Estimated Memory Usage: {total_memory / (1024**2):.2f} MB", log_file
    )

    # Data quality assessment
    log_and_print(f"\nData Quality Assessment:", log_file)

    # Check for potential issues
    issues = []

    # Check dataset balance - handle both formats
    if all(dataset_name in data for dataset_name in ["train", "valid", "test"]):
        sizes = []
        for dataset_name in ["train", "valid", "test"]:
            dataset = data[dataset_name]
            if isinstance(dataset, pd.DataFrame):
                sizes.append(len(dataset))
            elif isinstance(dataset, dict) and "X" in dataset:
                sizes.append(dataset["X"].shape[0])
            else:
                sizes.append(0)

        train_size, valid_size, test_size = sizes

        if train_size < valid_size or train_size < test_size:
            issues.append(
                "Training set might be too small compared to validation/test sets"
            )

        if valid_size == 0 or test_size == 0:
            issues.append("Validation or test set is empty")

    # Check feature completeness
    if "feature_cols" in data:
        expected_features = set(ENHANCED_FEATURES)
        actual_features = set(data["feature_cols"])
        if len(expected_features - actual_features) > 0:
            issues.append(
                f"Missing {len(expected_features - actual_features)} expected features"
            )

    # Check missing data levels
    if total_missing > 0:
        missing_percentage = (total_missing / (total_samples * total_features)) * 100
        if missing_percentage > 5.0:
            issues.append(f"High missing data percentage: {missing_percentage:.2f}%")
        elif missing_percentage > 1.0:
            issues.append(
                f"Moderate missing data percentage: {missing_percentage:.2f}%"
            )
        else:
            log_and_print(
                f"  [OK] Low missing data percentage: {missing_percentage:.4f}%",
                log_file,
            )

    if issues:
        log_and_print("  Potential Issues Found:", log_file)
        for issue in issues:
            log_and_print(f"    [WARNING] {issue}", log_file)
    else:
        log_and_print("  [OK] No major issues detected", log_file)

    # Data completeness recommendations
    if total_missing > 0:
        log_and_print(f"\nData Completeness Recommendations:", log_file)
        log_and_print(
            f"  - Consider imputation strategies for missing values", log_file
        )
        log_and_print(
            f"  - Check if missing values follow patterns (temporal/spatial)", log_file
        )
        log_and_print(f"  - Evaluate impact on model performance", log_file)
        log_and_print(
            f"  - Consider removing samples/features with excessive missing data",
            log_file,
        )

    log_and_print(f"\nValidation completed at: {datetime.now()}", log_file)


def main():
    """Main validation function."""
    print("Starting enhanced dataset validation...")

    # Load data
    data, log_file, log_filename = load_enhanced_data()
    if data is None:
        print("Failed to load data. Exiting.")
        if log_file:
            log_file.close()
        return

    try:
        # Run all validations
        validate_stations(data, log_file)
        validate_feature_columns(data, log_file)
        validate_datasets(data, log_file)
        validate_adjacency_matrix(data, log_file)
        validate_scalers(data, log_file)
        create_comprehensive_summary(data, log_file)

        log_and_print(f"\n{'='*80}", log_file)
        log_and_print("VALIDATION COMPLETE!", log_file)
        log_and_print(f"{'='*80}", log_file)
        log_and_print(f"Log file saved as: {log_filename}", log_file)
        log_and_print("Generated plots:", log_file)
        log_and_print("  - station_locations.png", log_file)
        log_and_print("  - dataset_sizes.png", log_file)
        log_and_print("  - adjacency_matrix_enhanced.png", log_file)
        log_and_print("  - connection_analysis.png", log_file)

    except Exception as e:
        log_and_print(f"\n[ERROR] Validation failed with error: {str(e)}", log_file)
        import traceback

        log_and_print(f"Traceback:\n{traceback.format_exc()}", log_file)

    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
