import rasterio
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import os
from pathlib import Path
import numpy as np
import re
import pickle


def get_raster_bounds(raster_path):
    """Get the coordinate bounds of a raster file from its filename."""
    filename = os.path.basename(raster_path)

    # Try DEM file pattern first (e.g., n39_e115_1arc_v3.tif)
    dem_match = re.match(r"n(\d+)_e(\d+)_", filename)
    if dem_match:
        lat, lon = map(int, dem_match.groups())
        return {"min_lat": lat, "max_lat": lat + 1, "min_lon": lon, "max_lon": lon + 1}

    # Try land cover file pattern (e.g., ESA_WorldCover_10m_2021_v200_N39E114_Map.tif)
    lc_match = re.match(r".*_N(\d+)E(\d+)_", filename)
    if lc_match:
        lat, lon = map(int, lc_match.groups())
        # Land cover files cover a larger area: from (lon,lat) to (lon+3,lat-3)
        return {"min_lat": lat, "max_lat": lat + 3, "min_lon": lon, "max_lon": lon + 3}

    return None


def is_point_in_bounds(point, bounds):
    """Check if a point is within the given bounds."""
    return (
        bounds["min_lat"] <= point.y <= bounds["max_lat"]
        and bounds["min_lon"] <= point.x <= bounds["max_lon"]
    )


def sample_raster(raster_path, gdf):
    try:
        with rasterio.open(raster_path) as src:
            gdf = gdf.to_crs(src.crs)  # reproject to raster CRS
            values = []
            bounds = get_raster_bounds(raster_path)

            if bounds:
                print(f"Raster bounds: {bounds}")

            for geom in gdf.geometry:
                try:
                    # First check if point is in the raster's coordinate range
                    if bounds and not is_point_in_bounds(geom, bounds):
                        values.append(None)
                        continue

                    row, col = src.index(geom.x, geom.y)
                    # Check if the point is within the raster bounds
                    if 0 <= row < src.height and 0 <= col < src.width:
                        value = src.read(1)[row, col]
                        values.append(float(value))
                    else:
                        print(
                            f"Warning: Point at {geom.x}, {geom.y} is outside raster bounds"
                        )
                        values.append(None)
                except (ValueError, IndexError) as e:
                    print(
                        f"Warning: Could not sample point at {geom.x}, {geom.y}: {str(e)}"
                    )
                    values.append(None)
        return values
    except Exception as e:
        print(f"Error reading raster file {raster_path}: {str(e)}")
        return [None] * len(gdf)


def load_processed_data():
    """Load processed data from the processed directory."""
    processed_dir = Path("../../processed")

    # Load scalers
    with open(processed_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Load adjacency matrix
    adj_matrix = np.load(processed_dir / "adj_matrix.npy")

    return scalers, adj_matrix


def main():
    # Get the absolute path to the data directory
    data_dir = Path("..")  # Go up one level to data/raw

    # Read station data
    stations_path = data_dir / "air_quality" / "stations.csv"
    if not stations_path.exists():
        raise FileNotFoundError(f"Stations file not found at {stations_path}")

    # Read CSV and clean column names
    stations = pd.read_csv(stations_path)
    # Clean column names: remove spaces and trailing commas
    stations.columns = stations.columns.str.strip().str.rstrip(",")
    # Clean data: remove trailing commas and spaces
    stations = stations.apply(
        lambda x: x.str.strip().str.rstrip(",") if x.dtype == "object" else x
    )

    print("Column names:", stations.columns.tolist())  # Debug print

    # Create geometry points
    stations["geometry"] = stations.apply(
        lambda row: Point(float(row["lon"]), float(row["lat"])), axis=1
    )
    gdf = gpd.GeoDataFrame(stations, geometry="geometry", crs="EPSG:4326")

    # Print station coordinates for debugging
    print("\nStation coordinates:")
    for idx, row in stations.iterrows():
        print(f"{row['station']}: {row['lat']}, {row['lon']}")

    # Extract elevation from DEM files
    dem_files = list(Path(".").glob("n*.tif"))  # Look in current directory
    if not dem_files:
        raise FileNotFoundError("No DEM files found in current directory")

    # Print DEM file bounds for debugging
    print("\nDEM file bounds:")
    for dem_file in dem_files:
        bounds = get_raster_bounds(dem_file)
        if bounds:
            print(f"{dem_file.name}: {bounds}")

    # Initialize elevation values with None
    elevations = [None] * len(stations)

    # Try each DEM file for each point
    for dem_file in dem_files:
        print(f"\nProcessing DEM file: {dem_file}")
        dem_values = sample_raster(str(dem_file), gdf)
        # Update only the None values with valid values from this DEM
        for i, val in enumerate(dem_values):
            if val is not None and elevations[i] is None:
                elevations[i] = val
                print(f"Found elevation for {stations.iloc[i]['station']}: {val}")

    stations["elevation"] = elevations

    # Extract land cover type
    lc_files = list(data_dir.glob("land_cover/ESA_WorldCover_10m_2021_v200_*_Map.tif"))
    if not lc_files:
        raise FileNotFoundError(
            "No land cover files found in data/raw/land_cover directory"
        )

    # Print land cover file bounds for debugging
    print("\nLand cover file bounds:")
    for lc_file in lc_files:
        bounds = get_raster_bounds(lc_file)
        if bounds:
            print(f"{lc_file.name}: {bounds}")

    # Initialize land cover values with None
    land_covers = [None] * len(stations)

    # Try each land cover file for each point
    for lc_file in lc_files:
        print(f"\nProcessing land cover file: {lc_file}")
        lc_values = sample_raster(str(lc_file), gdf)
        # Update only the None values with valid values from this file
        for i, val in enumerate(lc_values):
            if val is not None and land_covers[i] is None:
                land_covers[i] = val
                print(f"Found land cover for {stations.iloc[i]['station']}: {val}")

    stations["land_cover"] = land_covers

    # Load processed data
    try:
        scalers, adj_matrix = load_processed_data()

        # Create distance-based adjacency matrix
        n_stations = len(stations)
        adj_matrix = np.zeros((n_stations, n_stations))

        # Calculate distances between stations
        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                # Get coordinates
                lat1, lon1 = float(stations.iloc[i]["lat"]), float(
                    stations.iloc[i]["lon"]
                )
                lat2, lon2 = float(stations.iloc[j]["lat"]), float(
                    stations.iloc[j]["lon"]
                )

                # Calculate distance (approximate using Euclidean distance)
                # Note: This is a simplified distance calculation
                distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

                # If stations are within ~0.1 degrees (roughly 10km), consider them connected
                if distance < 0.1:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Make it symmetric

        # Debug prints for adjacency matrix
        print("\nAdjacency Matrix Info:")
        print(f"Shape: {adj_matrix.shape}")
        print(f"Type: {adj_matrix.dtype}")
        print("Sample values:")
        print(adj_matrix[:5, :5])  # Print first 5x5 values
        print(f"Total connections: {np.sum(adj_matrix)}")
        print(f"Average connections per station: {np.mean(np.sum(adj_matrix, axis=1))}")

        # Add station connectivity information
        station_names = stations["station"].tolist()
        connectivity = []
        for i, station in enumerate(station_names):
            # Count how many other stations this station is connected to
            connections = np.sum(adj_matrix[i])
            connectivity.append(connections)
            print(
                f"Station {station}: {connections} connections"
            )  # Debug print for each station
        stations["connectivity"] = connectivity

        # Add station order (based on the order in the processed data)
        station_order = {name: i for i, name in enumerate(station_names)}
        stations["station_order"] = stations["station"].map(station_order)

    except Exception as e:
        print(f"Warning: Could not load processed data: {str(e)}")

    # Save the results to processed directory
    output_dir = Path("../../processed")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "stations_with_features.csv"

    # Drop geometry column before saving
    stations = stations.drop(columns=["geometry"])
    stations.to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")
    print("\nSample of results:")
    print(stations.head())

    # Print summary of missing values
    missing_elevation = stations["elevation"].isna().sum()
    missing_land_cover = stations["land_cover"].isna().sum()
    print(f"\nMissing values:")
    print(f"Elevation: {missing_elevation} stations")
    print(f"Land cover: {missing_land_cover} stations")


if __name__ == "__main__":
    main()
