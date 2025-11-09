"""
fast_routes_v4.py

Multi-vehicle road coverage heuristic with:
- 25 mph constant speed
- Turn-aware traversal (discourages U-turns where possible)
- Time-weighted clustering so heavy/long areas get reasonable coverage
- Road names from shapefile (NAME / ALTNAME / Fromstreet / ToStreet)
- Optional AADT traffic weighting from CSV
- Clean exports into ./routes/ (cleared each run)
"""

import os
import shutil
import math
import json
import csv

import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, MultiLineString, Point
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================
# 0. ROUTES FOLDER MANAGEMENT
# =====================================================

def prepare_routes_folder(folder="routes"):
    """Delete and recreate the routes folder for fresh exports."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder


# =====================================================
# 1. PREPROCESS ROADS
# =====================================================

def preprocess_roads(roads):
    """
    Remove obvious private/service/driveway roads.
    """

    return roads[~roads["OWNERSHIP"].isin(["PRIVATE", "CU"])]

def load_traffic_csv(path):
    """Load and normalize traffic AADT CSV from NYSDOT-style export."""
    df = pd.read_csv(path)
    # standardize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Count_Location" in df.columns:
        df["road_name"] = df["Count_Location"].astype(str).str.upper().str.strip()
    elif "Road" in df.columns:
        df["road_name"] = df["Road"].astype(str).str.upper().str.strip()
    else:
        raise ValueError("No road name column found in traffic CSV")
    df["AADT"] = pd.to_numeric(df["AADT"], errors="coerce").fillna(0)
    return df[["road_name", "AADT"]]

# =====================================================
# 2. BUILD GRAPH (25 mph, names, optional AADT)
# =====================================================

def build_graph(roads, traffic_df=None):
    """
    Build a NetworkX graph with:
      - length: geometric distance between consecutive points
      - time: travel time assuming 25 mph (11.176 m/s), adjusted by AADT if given
      - name: street name from NAME/ALTNAME/Fromstreet/ToStreet
      - aadt_weight: factor >=1 (1 means no boost, >1 means prioritized)
    """
    G = nx.Graph()
    SPEED_MPS = 25 * 0.44704  # 25 mph

    traffic_lookup = None
    if traffic_df is not None and "road_name" in traffic_df.columns and "AADT" in traffic_df.columns:
        # normalize by max AADT
        max_aadt = traffic_df["AADT"].max()
        # build simple dict: uppercased name -> normalized factor [1, 2]
        traffic_lookup = {}
        for _, r in traffic_df.iterrows():
            rn = str(r["road_name"]).strip().upper()
            aadt_val = float(r["AADT"])
            factor = 1 + (aadt_val / max_aadt)  # between 1 and 2
            traffic_lookup[rn] = factor

    for _, row in roads.iterrows():
        geom = row.geometry

        # choose a name from known columns in order of preference
        candidates = [
            row.get("NAME"),
            row.get("ALTNAME"),
            row.get("Fromstreet"),
            row.get("ToStreet")
        ]
        road_name = next(
            (str(c).strip() for c in candidates if c is not None and str(c).strip() not in ["", "None", "nan"]),
            "Unnamed Road"
        )
        # nicer formatting: title case
        road_name = road_name.title()

        # traffic weighting if available
        aadt_weight = 1.0
        if traffic_lookup is not None:
            key = road_name.upper()
            if key in traffic_lookup:
                aadt_weight = traffic_lookup[key]

        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                u, v = coords[i], coords[i + 1]
                dist = Point(u).distance(Point(v))
                if dist <= 0:
                    continue
                travel_time = dist / SPEED_MPS / aadt_weight
                if G.has_edge(u, v):
                    # keep shorter edge if duplicate
                    if dist < G[u][v]["length"]:
                        G[u][v]["length"] = dist
                        G[u][v]["time"] = travel_time
                        G[u][v]["name"] = road_name
                        G[u][v]["aadt_weight"] = aadt_weight
                else:
                    G.add_edge(
                        u, v,
                        length=dist,
                        time=travel_time,
                        name=road_name,
                        aadt_weight=aadt_weight
                    )

    return G


# =====================================================
# 3. TURN-AWARE EDGE COVER DFS
# =====================================================

def turn_alignment(prev, u, v):
    """
    Compute cosine of turn angle at u: prev -> u -> v.
    Higher cosine ~ straighter.
    """
    if prev is None:
        return 0.0
    ux, uy = u
    px, py = prev
    vx, vy = v

    v_in = np.array([ux - px, uy - py])
    v_out = np.array([vx - ux, vy - uy])
    n1 = np.linalg.norm(v_in)
    n2 = np.linalg.norm(v_out)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = float(np.dot(v_in, v_out) / (n1 * n2))
    return max(-1.0, min(1.0, cosang))


def edge_cover_dfs_turn_aware(G_sub, start):
    """
    Turn-aware DFS covering all edges in G_sub at least once.
    Returns list of (u, v) edges in traversal order.
    """
    visited_edges = set()
    route_edges = []

    def edge_key(a, b):
        return (a, b) if a <= b else (b, a)

    def dfs(u, prev=None):
        neighbors = list(G_sub.neighbors(u))
        candidates = []
        for v in neighbors:
            ek = edge_key(u, v)
            if ek in visited_edges:
                continue
            turn_score = turn_alignment(prev, u, v)
            # bonus for going toward nodes with more unvisited edges
            unvisited_bonus = sum(
                1 for w in G_sub.neighbors(v)
                if edge_key(v, w) not in visited_edges
            )
            score = turn_score + 0.1 * unvisited_bonus

            # discourage immediate backtracking (U-turn)
            if prev is not None and v == prev:
                score -= 1.0

            candidates.append((score, v))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for score, v in candidates:
            ek = edge_key(u, v)
            if ek in visited_edges:
                continue
            visited_edges.add(ek)
            route_edges.append((u, v))
            dfs(v, prev=u)

    dfs(start)

    # safety: ensure all edges were touched at least once
    for (u, v) in G_sub.edges:
        ek = (u, v) if u <= v else (v, u)
        if ek not in visited_edges:
            route_edges.append((u, v))

    return route_edges


# =====================================================
# 4. CLUSTER EDGES & BUILD EXACTLY N VEHICLE ROUTES
# =====================================================

def cluster_and_build_routes_connected(G, n_vehicles=10):
    edges = list(G.edges(data=True))
    if not edges:
        return []

    # Step 1: Spatial clustering
    pts = np.array([[(u[0]+v[0])/2, (u[1]+v[1])/2] for (u,v,_) in edges])
    weights = np.array([d.get("time", 1.0) for (_,_,d) in edges])
    kmeans = KMeans(n_clusters=n_vehicles, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pts, sample_weight=weights)

    # Step 2: Build initial subgraphs
    clusters = {i: nx.Graph() for i in range(n_vehicles)}
    for (u,v,d), lab in zip(edges, labels):
        clusters[lab].add_edge(u,v,**d)

    # Step 3: Ensure connectivity
    for i in range(n_vehicles):
        comps = list(nx.connected_components(clusters[i]))
        if len(comps) <= 1:
            continue
        print(f"⚠️ Vehicle {i}: {len(comps)} disconnected areas, fixing...")
        largest = max(comps, key=len)
        keep_nodes = set(largest)
        for comp in comps:
            if comp is largest:
                continue
            # find closest node in comp to any node in keep_nodes
            u = min(comp, key=lambda x: min(Point(x).distance(Point(y)) for y in keep_nodes))
            # find nearest other cluster that can reach u
            best_j, best_dist = None, float("inf")
            for j in range(n_vehicles):
                if j == i:
                    continue
                if not clusters[j].nodes:
                    continue
                v = min(clusters[j].nodes, key=lambda y: Point(u).distance(Point(y)))
                d = Point(u).distance(Point(v))
                if d < best_dist:
                    best_j, best_dist = j, d
            if best_j is not None:
                sub = clusters[i].subgraph(comp).copy()
                for (a,b,d) in sub.edges(data=True):
                    clusters[best_j].add_edge(a,b,**d)
        clusters[i] = clusters[i].subgraph(keep_nodes).copy()

    # Step 4: Build routes for each connected cluster
    routes = []
    for i in range(n_vehicles):
        H = clusters[i]
        if not H.edges:
            routes.append([])
            continue
        start = next(iter(H.nodes))
        route = edge_cover_dfs_turn_aware(H, start)
        routes.append(route)
    return routes



# =====================================================
# 5. VISUALIZATION
# =====================================================

def plot_routes(roads, routes):
    fig, ax = plt.subplots(figsize=(10, 10))
    roads.plot(ax=ax, color="lightgray", linewidth=0.5)
    cmap = plt.cm.get_cmap("tab20", len(routes))
    for i, route in enumerate(routes):
        for (u, v) in route:
            xs, ys = zip(u, v)
            ax.plot(xs, ys, color=cmap(i), linewidth=2)
    ax.set_axis_off()
    plt.title(f"{len(routes)}-Vehicle Coverage (Turn-aware, Time-weighted)")
    plt.show()


def animate_routes(routes, roads, interval=50):
    """Optional simple animation; may not work well in all VS Code setups."""
    fig, ax = plt.subplots(figsize=(10, 10))
    roads.plot(ax=ax, color="lightgray", linewidth=0.5)
    cmap = plt.cm.get_cmap("tab20", len(routes))

    route_coords = []
    for route in routes:
        if not route:
            route_coords.append([(0, 0)])
            continue
        coords = [route[0][0]] + [v for (_, v) in route]
        cleaned = [coords[0]]
        for pt in coords[1:]:
            if pt != cleaned[-1]:
                cleaned.append(pt)
        route_coords.append(cleaned)

    cars = []
    for i, coords in enumerate(route_coords):
        x0, y0 = coords[0]
        car, = ax.plot(x0, y0, "o", color=cmap(i), markersize=6)
        cars.append(car)

    ax.set_axis_off()
    plt.title("Vehicle Coverage Simulation")

    n_frames = max(len(c) for c in route_coords)

    def update(frame):
        for i, car in enumerate(cars):
            coords = route_coords[i]
            idx = frame % len(coords)
            x, y = coords[idx]
            car.set_data(x, y)
        return cars

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    plt.show()
    return ani


# =====================================================
# 6. EXPORT: ROUTES + DIRECTIONS
# =====================================================

def export_routes(routes, filename_prefix="routes/route"):
    """Export each route's coordinates as CSV + one GeoJSON for all."""
    geojson_features = []

    for i, route in enumerate(routes):
        if not route:
            continue
        coords = [route[0][0]] + [v for (_, v) in route]
        cleaned = [coords[0]]
        for pt in coords[1:]:
            if pt != cleaned[-1]:
                cleaned.append(pt)

        geojson_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": cleaned},
            "properties": {"vehicle_id": i}
        })

        with open(f"{filename_prefix}_{i}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["x", "y"])
            w.writerows(cleaned)

    geojson = {"type": "FeatureCollection", "features": geojson_features}
    with open(f"{filename_prefix}.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    print(f"✅ Exported {len(geojson_features)} non-empty routes to CSV + GeoJSON.")


def generate_local_directions(G, routes, filename_prefix="routes/directions"):
    """Generate human-readable turn-by-turn directions using local road names."""
    print("🧭 Creating local turn-by-turn directions...")

    def turn_type(p1, p2, p3, threshold=30):
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        ang = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        ang = (ang + 360) % 360
        if 150 <= ang <= 210:
            return "u-turn"
        elif ang > threshold and ang < 180:
            return "left"
        elif ang < 360 - threshold and ang > 180:
            return "right"
        return "straight"

    for i, route in enumerate(routes):
        if not route:
            continue
        if not isinstance(route[0], (tuple, list)) or len(route[0]) != 2:
            print(f"⚠️ Skipping invalid route {i}")
            continue

        coords = [route[0][0]] + [v for (_, v) in route]
        cleaned = [coords[0]]
        for pt in coords[1:]:
            if pt != cleaned[-1]:
                cleaned.append(pt)

        steps = []
        last_name = None

        # initial road name
        if len(cleaned) > 1 and G.has_edge(cleaned[0], cleaned[1]):
            initial_name = G[cleaned[0]][cleaned[1]].get("name", "Unnamed Road")
        else:
            initial_name = "Road"

        steps.append(f"Start on {initial_name}")

        for j in range(len(cleaned) - 2):
            u, v, w = cleaned[j], cleaned[j + 1], cleaned[j + 2]
            if not G.has_edge(u, v):
                continue

            road_name = G[u][v].get("name", "Unnamed Road")
            if road_name == "Unnamed Road" and last_name:
                road_name = last_name  # inherit previous name

            t = turn_type(u, v, w)

            # skip tiny detours
            if G[u][v]["length"] < 10:
                continue

            # merge consecutive U-turns
            if steps and "U-turn" in steps[-1] and t == "u-turn":
                continue

            if t == "u-turn":
                steps.append(f"Make a U-turn on {road_name}")
            elif t == "left":
                steps.append(f"Turn left onto {road_name}")
            elif t == "right":
                steps.append(f"Turn right onto {road_name}")

            last_name = road_name

        steps.append("Arrive at destination")

        outpath = f"{filename_prefix}_{i}.txt"
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(f"Route {i+1} Directions:\n\n")
            for s in steps:
                f.write(s + "\n")
        print(f"✅ Directions written for vehicle {i+1} ({len(steps)} steps)")


# =====================================================
# 7. MAIN DRIVER
# =====================================================

def run_fast_coverage(road_shp_path, traffic_csv_path=None, n_vehicles=10):
    print("📂 Loading roads shapefile...")
    roads = gpd.read_file(road_shp_path)
    # if needed, project to metric CRS; your data seems already in a projected CRS
    # roads = roads.to_crs(epsg=3857)

    roads = preprocess_roads(roads)
    print(f"✅ Roads after filtering: {len(roads)} segments")

    traffic_df = None
    if traffic_csv_path and os.path.exists(traffic_csv_path):
        print("📊 Loading traffic (AADT) data...")
        traffic_df = load_traffic_csv(traffic_csv_path)
        print(f"✅ Loaded {len(traffic_df)} traffic rows")


    print("🔗 Building graph...")
    G = build_graph(roads, traffic_df=traffic_df)
    if not nx.is_connected(G):
        G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len).copy()
    print(f"✅ Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges (largest component)")

    print("🚘 Clustering edges & building routes...")
    routes = cluster_and_build_routes_connected(G, n_vehicles=n_vehicles)
    print(f"✅ Built {len(routes)} vehicle routes")

    # coverage check
    covered_edges = set()
    for route in routes:
        for (u, v) in route:
            ek = (u, v) if u <= v else (v, u)
            covered_edges.add(ek)
    all_edges = {(u, v) if u <= v else (v, u) for (u, v) in G.edges}
    print(f"🧮 Edge coverage: {len(covered_edges)}/{len(all_edges)} edges traversed at least once")

    folder = prepare_routes_folder("routes")

    print("🗺️ Plotting routes...")
    plot_routes(roads, routes)

    print("💾 Exporting routes...")
    export_routes(routes, os.path.join(folder, "route"))

    print("🧭 Creating human-readable directions...")
    generate_local_directions(G, routes, os.path.join(folder, "directions"))

    # Optional animation (may or may not behave nicely in VS Code)
    print("🎞️ Animating routes (optional)...")
    animate_routes(routes, roads, interval=50)

    return G, roads, routes


if __name__ == "__main__":
    # TODO: update this to your actual file paths
    G, roads, routes = run_fast_coverage(
        road_shp_path="Roads/Roads.shp",
        traffic_csv_path="ithaca_traffic_counts_2024_final.csv",     # e.g. "traffic_aadt.csv"
        n_vehicles=10
    )
