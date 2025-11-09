"""
fast_routes_v5.py

Multi-vehicle road coverage heuristic with:
- 25 mph constant speed
- One-way / two-way streets from DIRECT1
- AADT weighting from traffic CSV
- Time-weighted clustering into N territories
- Continuous, turn-aware traversal (no teleporting)
- Strong U-turn penalty
- Human-readable directions
- Clean exports to ./routes/
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
    Remove obvious private/service/driveway roads if possible
    and drop very short segments (< 5 units length).
    """
    return roads[~roads["OWNERSHIP"].isin(["PRIVATE", "CU"])]


# =====================================================
# 2. LOAD TRAFFIC CSV (AADT)
# =====================================================

def load_traffic_csv(path):
    """
    Load traffic CSV with columns like:
      Station,Road #,Count Location,Municipality,From,To,Year,AADT
    and normalize to [road_name, AADT].
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if "Count_Location" in df.columns:
        df["road_name"] = df["Count_Location"].astype(str).str.upper().str.strip()
    else:
        raise ValueError("Could not find 'Count Location' column in traffic CSV")

    df["AADT"] = pd.to_numeric(df["AADT"], errors="coerce").fillna(0)
    return df[["road_name", "AADT"]]

def load_slope_csv(path):
    df = pd.read_csv(path)
    df["OBJECTID"] = pd.to_numeric(df["OBJECTID"], errors="coerce").astype("Int64")
    df["slope_category"] = pd.to_numeric(df["slope_category"], errors="coerce").fillna(1)
    slope_lookup = dict(zip(df["OBJECTID"], df["slope_category"]))
    return slope_lookup


# =====================================================
# 3. BUILD DIRECTED GRAPH (25 mph, names, AADT, one-way)
# =====================================================

def build_graph(roads, traffic_df=None, slope_lookup=None):
    """
    Build a DiGraph with:
      - 'length': geometric segment length
      - 'time': travel time at 25 mph, adjusted by AADT if given
      - 'name': street name from NAME / ALTNAME / Fromstreet / ToStreet
      - 'aadt_weight': >=1 (higher means more priority)
      - obeys one-way vs two-way from DIRECT1
    """
    G = nx.DiGraph()
    SPEED_MPS = 25 * 0.44704  # 25 mph

    # Build traffic lookup if provided
    traffic_lookup = None
    if traffic_df is not None and not traffic_df.empty:
        max_aadt = traffic_df["AADT"].max() or 1.0
        traffic_lookup = {}
        for _, r in traffic_df.iterrows():
            rn = str(r["road_name"]).strip().upper()
            aadt_val = float(r["AADT"])
            # factor in [1,2]
            factor = 1 + (aadt_val / max_aadt)
            traffic_lookup[rn] = factor

    for _, row in roads.iterrows():
        geom = row.geometry

        object_id = int(row.get("OBJECTID", -1))
        slope_cat = slope_lookup.get(object_id, 1) if slope_lookup else 1
        slope_factor = 1 + 0.25 * (slope_cat - 1)

        # choose name from NAME / ALTNAME / Fromstreet / ToStreet
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
        road_name = road_name.title()
        key_name = road_name.upper()

        # traffic weighting
        aadt_weight = 1.0
        if traffic_lookup is not None and key_name in traffic_lookup:
            aadt_weight = traffic_lookup[key_name]

        total_weight = aadt_weight * slope_factor

        # one-way / two-way
        direction_str = str(row.get("DIRECT1", "Two-way")).lower()
        is_oneway = "One" in direction_str

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
                travel_time = dist / SPEED_MPS / total_weight # weight of a road

                # forward direction
                G.add_edge(u, v, length=dist, time=travel_time,
                           name=road_name, aadt_weight=aadt_weight, oneway=is_oneway)

                # if two-way, add reverse edge as well
                if not is_oneway:
                    G.add_edge(v, u, length=dist, time=travel_time,
                               name=road_name, aadt_weight=aadt_weight, oneway=False)

    return G


# =====================================================
# 4. TURN & ROUTE LOGIC
# =====================================================

def turn_alignment(prev, u, v):
    """
    cosine of turn angle at u: prev -> u -> v
    +1 = straight, 0 = 90°, -1 = 180° (U-turn)
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


def build_smart_cover_route(G_sub, start, uturn_penalty=10000.0):
    """
    Continuous, directed traversal that:
      - Covers every edge in G_sub it can reach
      - Strongly penalizes U-turns
      - Prefers unvisited edges
      - Reuses edges when necessary to stay legal + continuous
    """
    visited_edges = set()
    route_edges = []
    u = start
    prev = None

    def edge_key(a, b):  # directed key
        return (a, b)

    max_iter = 10 * max(1, G_sub.number_of_edges())
    iter_count = 0

    while iter_count < max_iter:
        iter_count += 1

        # unvisited outgoing edges from u
        out_edges = [(u, v) for v in G_sub.successors(u)]
        if not out_edges:
            # try to find path to nearest node with unvisited edges
            uncovered = [(a, b) for (a, b) in G_sub.edges if edge_key(a, b) not in visited_edges]
            if not uncovered:
                break
            # endpoints of uncovered edges
            endpoints = [a for (a, b) in uncovered] + [b for (a, b) in uncovered]
            try:
                dist_map = nx.single_source_dijkstra_path_length(G_sub, u, weight="time")
            except nx.NetworkXNoPath:
                break
            reachable = [(n, dist_map[n]) for n in endpoints if n in dist_map]
            if not reachable:
                break
            next_node = min(reachable, key=lambda x: x[1])[0]
            path = nx.shortest_path(G_sub, u, next_node, weight="time")
            # move along connector path
            for a, b in zip(path[:-1], path[1:]):
                route_edges.append((a, b))
            prev = path[-2] if len(path) > 1 else prev
            u = next_node
            continue

        # Evaluate candidates (visited + unvisited) but prefer unvisited & non-U-turn
        candidates = []
        for _, v in out_edges:
            ang = turn_alignment(prev, u, v)
            # base: smaller (more negative) cost = better
            cost = 0.0
            # U-turn?
            if ang < -0.9:
                cost += uturn_penalty
            # visited penalty
            if edge_key(u, v) in visited_edges:
                cost += 50.0
            # reward straightness (subtract)
            cost -= ang * 10.0
            candidates.append((cost, v))

        candidates.sort(key=lambda x: x[0])
        _, v = candidates[0]

        route_edges.append((u, v))
        visited_edges.add(edge_key(u, v))
        prev, u = u, v

        # stop if everything reachable is visited
        uncovered = [(a, b) for (a, b) in G_sub.edges if edge_key(a, b) not in visited_edges]
        if not uncovered:
            break

    if iter_count >= max_iter:
        print(f"⚠️ Route from {start} stopped after {iter_count} steps (possible deadlock).")

    return route_edges


# =====================================================
# 5. CLUSTER EDGES & BUILD ROUTES
# =====================================================

def cluster_and_build_routes(G, n_vehicles=10):
    """
    KMeans cluster directed edges by midpoint (time-weighted),
    then for each cluster build a continuous route.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return []

    pts = []
    weights = []
    for (u, v, data) in edges:
        mx = (u[0] + v[0]) / 2.0
        my = (u[1] + v[1]) / 2.0
        pts.append([mx, my])
        weights.append(float(data.get("time", 1.0)))
    pts = np.array(pts)
    weights = np.array(weights)

    kmeans = KMeans(n_clusters=n_vehicles, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pts, sample_weight=weights)

    vehicle_edges = {k: [] for k in range(n_vehicles)}
    for (u, v, data), lab in zip(edges, labels):
        vehicle_edges[lab].append((u, v, data))

    routes = []

    for k in range(n_vehicles):
        veh_edges = vehicle_edges[k]
        if not veh_edges:
            routes.append([])
            continue

        H = nx.DiGraph()
        for (u, v, data) in veh_edges:
            H.add_edge(u, v, **data)

        # use underlying undirected connectivity to find components
        und = H.to_undirected()
        components = list(nx.connected_components(und))
        vehicle_route = []
        prev_end = None

        for comp_nodes in components:
            compG = H.subgraph(comp_nodes).copy()
            if compG.number_of_edges() == 0:
                continue

            # choose start node
            if prev_end is None:
                start = next(iter(compG.nodes))
            else:
                # find closest node in this component (ignoring direction) to prev_end
                try:
                    # shortest path on undirected view
                    dist_map = nx.single_source_dijkstra_path_length(und, prev_end, weight="time")
                    reachable = [n for n in comp_nodes if n in dist_map]
                    if reachable:
                        start = min(reachable, key=lambda n: dist_map[n])
                        path = nx.shortest_path(und, prev_end, start, weight="time")
                        for a, b in zip(path[:-1], path[1:]):
                            # choose direction if available; else skip (one-way conflict)
                            if H.has_edge(a, b):
                                vehicle_route.append((a, b))
                            elif H.has_edge(b, a):
                                vehicle_route.append((b, a))
                        prev_end = start
                    else:
                        start = next(iter(compG.nodes))
                except nx.NetworkXNoPath:
                    start = next(iter(compG.nodes))

            comp_route = build_smart_cover_route(compG, start)
            vehicle_route.extend(comp_route)
            if vehicle_route:
                prev_end = vehicle_route[-1][1]

        routes.append(vehicle_route)

    return routes


# =====================================================
# 6. VISUALIZATION
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
    plt.title(f"{len(routes)}-Vehicle Coverage (Directed, Turn-aware)")
    plt.show()


def animate_routes(routes, roads, interval=50):
    """Optional simple animation; may not work perfectly everywhere."""
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
# 7. EXPORT: ROUTES + DIRECTIONS
# =====================================================

def export_routes(routes, filename_prefix="routes/route"):
    """Export each route's coordinates as CSV + one combined GeoJSON."""
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
    """Generate human-readable directions from local road names."""
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
        # initial road name
        if len(cleaned) > 1 and G.has_edge(cleaned[0], cleaned[1]):
            initial_name = G[cleaned[0]][cleaned[1]].get("name", "Road")
        else:
            initial_name = "Road"
        steps.append(f"Start on {initial_name}")

        last_name = initial_name

        for j in range(len(cleaned) - 2):
            u, v, w = cleaned[j], cleaned[j + 1], cleaned[j + 2]
            if not G.has_edge(u, v):
                continue

            road_name = G[u][v].get("name", "Unnamed Road")
            # inherit previous if unnamed
            if road_name == "Unnamed Road" and last_name:
                road_name = last_name

            t = turn_type(u, v, w)

            # skip tiny wiggles
            if G[u][v].get("length", 999) < 10:
                last_name = road_name
                continue

            if t == "straight":
                last_name = road_name
                continue
            elif t == "u-turn":
                # avoid spamming multiple U-turns
                if steps and "U-turn" in steps[-1]:
                    continue
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
# 8. MAIN DRIVER
# =====================================================

def run_fast_coverage(road_shp_path, traffic_csv_path=None, slope_csv_path=None, n_vehicles=10):
    print("📂 Loading roads shapefile...")
    roads = gpd.read_file(road_shp_path)
    # Your data already appears projected; if not, you can project:
    # roads = roads.to_crs(epsg=3857)

    roads = preprocess_roads(roads)
    print(f"✅ Roads after filtering: {len(roads)} segments")

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

    print("🔗 Building directed graph...")
    G = build_graph(roads, traffic_df=traffic_df, slope_lookup=slope_lookup)

    # use largest weakly-connected component
    if not nx.is_weakly_connected(G):
        G = max((G.subgraph(c) for c in nx.weakly_connected_components(G)), key=len).copy()
    print(f"✅ Graph built: {len(G.nodes)} nodes, {len(G.edges)} directed edges (largest component)")

    print("🚘 Clustering edges & building routes...")
    routes = cluster_and_build_routes(G, n_vehicles=n_vehicles)
    print(f"✅ Built {len(routes)} vehicle routes")

    # coverage check (undirected)
    covered_edges = set()
    for route in routes:
        for (u, v) in route:
            ek = tuple(sorted((u, v)))
            covered_edges.add(ek)
    all_edges = {tuple(sorted((u, v))) for (u, v) in G.to_undirected().edges}
    print(f"🧮 Undirected edge coverage: {len(covered_edges)}/{len(all_edges)} edges traversed at least once")

    folder = prepare_routes_folder("routes")

    print("🗺️ Plotting routes...")
    plot_routes(roads, routes)

    print("💾 Exporting routes...")
    export_routes(routes, os.path.join(folder, "route"))

    print("🧭 Creating human-readable directions...")
    generate_local_directions(G, routes, os.path.join(folder, "directions"))

    # Optional animation:
    # print("🎞️ Animating routes (optional)...")
    animate_routes(routes, roads, interval=50)

    return G, roads, routes


if __name__ == "__main__":
    # TODO: update these paths for your environment
    G, roads, routes = run_fast_coverage(
        road_shp_path="Roads/Roads.shp",
        traffic_csv_path="ithaca_traffic_counts_2024_final.csv",     # e.g. "traffic_aadt.csv"
        slope_csv_path="Elevation/roads_with_slope.csv",
        n_vehicles=10
    )
