"""
optimal_vehicle_analysis.py
-----------------------------------
Helper module for evaluating the optimal number of vehicles (clusters)
based on network workload balance and priority weighting.

Integrates with the road network and clustering logic from v7.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString
from sklearn.cluster import KMeans
import pandas as pd

def load_traffic_csv(path):
    """
    Expect columns like:
      Station,Road #,Count Location,Municipality,From,To,Year,AADT
    Return DataFrame with [road_name, AADT].
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Count_Location" not in df.columns:
        raise ValueError("Traffic CSV must have 'Count Location' column")
    df["road_name"] = df["Count_Location"].astype(str).str.upper().str.strip()
    df["AADT"] = pd.to_numeric(df["AADT"], errors="coerce").fillna(0)
    return df[["road_name", "AADT"]]


def load_slope_csv(path):
    """
    Expect columns including OBJECTID and slope_category (1..4).
    Return dict: OBJECTID -> slope_category.
    """
    df = pd.read_csv(path)
    df["OBJECTID"] = pd.to_numeric(df["OBJECTID"], errors="coerce").astype("Int64")
    df["slope_category"] = pd.to_numeric(df["slope_category"], errors="coerce").fillna(1)
    return dict(zip(df["OBJECTID"], df["slope_category"]))

def evaluate_clusters(G, roads, k_range=range(2, 20)):
    """
    Evaluates clustering quality for k = 2..N based on
    total weighted road lengths per zone.

    Parameters
    ----------
    G : networkx.Graph
        Road network graph.
    roads : GeoDataFrame
        Roads shapefile containing 'AADT' and 'slope_category'.
    k_range : range
        Range of cluster numbers (vehicles) to test.

    Returns
    -------
    list[dict]
        Each dict contains: k, max_cost, mean_cost, balance_ratio.
    """
    coords = np.array([[x, y] for x, y in G.nodes])
    road_priorities = {}

    # Build per-edge priority weight (traffic × slope)
    for _, row in roads.iterrows():
        try:
            w = float(row.get("AADT", 1)) * float(row.get("slope_category", 1))
        except (ValueError, TypeError):
            w = 1.0

        geom = row.geometry
        if isinstance(geom, LineString):
            pts = list(geom.coords)
            for i in range(len(pts) - 1):
                road_priorities[(pts[i], pts[i + 1])] = w

    results = []

    for k in k_range:
        print(f"Evaluating {k} vehicles...")
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(coords)

        zone_scores = [0] * k
        node_list = list(G.nodes)

        for (u, v, data) in G.edges(data=True):
            try:
                i = labels[node_list.index(u)]
            except ValueError:
                continue

            length = data.get("length", 1.0)
            w = road_priorities.get((u, v), 1.0)
            zone_scores[i] += length * w

        results.append({
            "k": k,
            "max_cost": max(zone_scores),
            "mean_cost": np.mean(zone_scores),
            "balance_ratio": np.std(zone_scores) / (np.mean(zone_scores) + 1e-9)
        })

    return results


def plot_optimal_vehicle_curve(results, output_folder="routes"):
    """
    Plots the relationship between number of vehicles and workload metrics.

    Saves a figure named 'optimal_vehicle_curve.png' to the routes folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ks = [r["k"] for r in results]
    max_costs = [r["max_cost"] for r in results]
    balances = [r["balance_ratio"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, max_costs, "o-", color="tab:blue", label="Max Zone Cost")
    ax1.set_xlabel("Number of Vehicles (Clusters)")
    ax1.set_ylabel("Max Zone Cost", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(ks, balances, "s--", color="tab:orange", label="Balance Ratio")
    ax2.set_ylabel("Balance Ratio (std/mean)", color="tab:orange")

    plt.title("Optimal Number of Vehicles Analysis")
    fig.tight_layout()

    out_path = os.path.join(output_folder, "optimal_vehicle_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved optimal vehicle analysis figure to {out_path}")


if __name__ == "__main__":
    # Example usage (run standalone)
    import geopandas as gpd
    from v7 import build_graph  # adjust to your graph builder

    traffic_csv_path="ithaca_traffic_counts_2024_final.csv"     # e.g. "traffic_aadt.csv"
    slope_csv_path="Elevation/roads_with_slope.csv"

    traffic_df = None
    if traffic_csv_path and os.path.exists(traffic_csv_path):
        print("📊 Loading traffic (AADT) data...")
        traffic_df = load_traffic_csv(traffic_csv_path)
        print(f"✅ Loaded {len(traffic_df)} traffic rows")

    slope_lookup = None
    if slope_csv_path and os.path.exists(slope_csv_path):
        print("⛰️  Loading slope priority data...")
        slope_lookup = load_slope_csv(slope_csv_path)
        print(f"✅ Loaded {len(slope_lookup)} slope entries")

    shapefile = "Roads/Roads.shp"
    roads = gpd.read_file(shapefile)
    G = build_graph(shapefile,traffic_df=traffic_df, slope_lookup=slope_lookup)

    results = evaluate_clusters(G, roads, k_range=range(3, 15))
    plot_optimal_vehicle_curve(results)
