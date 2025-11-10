import os
import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString
from sklearn.cluster import KMeans
import pandas as pd

def load_traffic_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Count_Location" not in df.columns:
        raise ValueError("Traffic CSV must have 'Count Location' column")
    df["road_name"] = df["Count_Location"].astype(str).str.upper().str.strip()
    df["AADT"] = pd.to_numeric(df["AADT"], errors="coerce").fillna(0)
    return df[["road_name", "AADT"]]


def load_slope_csv(path):
    df = pd.read_csv(path)
    df["OBJECTID"] = pd.to_numeric(df["OBJECTID"], errors="coerce").astype("Int64")
    df["slope_category"] = pd.to_numeric(df["slope_category"], errors="coerce").fillna(1)
    return dict(zip(df["OBJECTID"], df["slope_category"]))

def evaluate_clusters(G, roads, k_range=range(2, 20)):
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
    times = []
    for k in k_range:
        print(f"Evaluating {k} vehicles...")
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        zone_scores = [0] * k
        node_list = list(G.nodes)
        time = [0] * k
        for (u, v, data) in G.edges(data=True):
            try:
                i = labels[node_list.index(u)]
            except ValueError:
                continue
            

            name = data.get("name",1.0)
            width = data.get("width",1.0)
            length = data.get("length", 1.0)
            a = length / (15*5280) + max(0, int(round(width/10) - 1)) * length/(15*5280)
            w = road_priorities.get((u, v), 1.0)
            zone_scores[i] += length * w
            time[i] += a

        times.append(max(time))
        results.append({
            "k": k,
            "max_cost": max(zone_scores),
            "mean_cost": np.mean(zone_scores),
            "balance_ratio": np.std(zone_scores) / (np.mean(zone_scores) + 1e-9)
        
        })

    return results, times


def plot_optimal_vehicle_curve(results, output_folder="routes"):
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
    print(f"Saved optimal vehicle analysis figure to {out_path}")
