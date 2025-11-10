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
import matplotlib

from shapely.geometry import LineString, MultiLineString, Point
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================
# 0. ROUTES FOLDER
# =====================================================

def prepare_routes_folder(folder="routes"):
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
    and drop very short segments (< 5 units).
    """
    return roads[~roads["OWNERSHIP"].isin(["PRIVATE"])]


# =====================================================
# 2. LOAD TRAFFIC & SLOPE CSVs
# =====================================================

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


# =====================================================
# 3. BUILD DIRECTED GRAPH
# =====================================================

def build_graph(roads, traffic_df=None, slope_lookup=None):
    """
    Build a DiGraph with:
      - length
      - time (25 mph, adjusted by AADT & slope_category)
      - name
      - aadt_weight, slope_cat
      - obeys DIRECT1 one-way/two-way
    """
    G = nx.DiGraph()
    SPEED_MPS = 25 * 0.44704  # 25 mph

    traffic_lookup = None
    if traffic_df is not None and not traffic_df.empty:
        max_aadt = traffic_df["AADT"].max() or 1.0
        traffic_lookup = {}
        for _, r in traffic_df.iterrows():
            rn = str(r["road_name"]).strip().upper()
            aadt_val = float(r["AADT"])
            factor = 1 + (aadt_val / max_aadt) 
            traffic_lookup[rn] = factor

    for _, row in roads.iterrows():
        geom = row.geometry
        object_id = int(row.get("OBJECTID", -1))

        slope_cat = slope_lookup.get(object_id, 1) if slope_lookup else 1
        slope_factor = 1 + 0.25 * (slope_cat - 1)

        candidates = [
            row.get("NAME"),
            row.get("ALTNAME"),
            row.get("Fromstreet"),
            row.get("ToStreet"),
        ]
        road_name = next(
            (str(c).strip() for c in candidates
             if c is not None and str(c).strip() not in ["", "None", "nan"]),
            "Unnamed Road",
        )
        road_name = road_name.title()
        key_name = road_name.upper()

        aadt_weight = 1.0
        if traffic_lookup is not None and key_name in traffic_lookup:
            aadt_weight = traffic_lookup[key_name]

        total_weight = aadt_weight * slope_factor

        direction_str = str(row.get("DIRECT1", "Two-way")).lower()
        is_oneway = "one" in direction_str

        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            continue

        if row.get("WIDTH_C_C") is not None:
            width = row.get("WIDTH_C_C")
        elif row.get("WIDTH_APPRO") is not None:
            width = row.get("WIDTH_APPR")
        else:
            width = 1.0


        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                u, v = coords[i], coords[i + 1]
                dist = Point(u).distance(Point(v))
                if dist <= 0:
                    continue
                travel_time = dist / SPEED_MPS / total_weight

                G.add_edge(
                    u, v,
                    length=dist,
                    time=travel_time,
                    name=road_name,
                    aadt_weight=aadt_weight,
                    slope_cat=slope_cat,
                    oneway=is_oneway,
                    width = width
                )

                if not is_oneway:
                    G.add_edge(
                        v, u,
                        length=dist,
                        time=travel_time,
                        name=road_name,
                        aadt_weight=aadt_weight,
                        slope_cat=slope_cat,
                        oneway=False,
                        width = width
                    )

    return G


# =====================================================
# 4. TURN / ROUTE HELPERS
# =====================================================

def turn_alignment(prev, u, v):
    """cos(angle) at u: prev->u->v, +1 straight, -1 U-turn."""
    if prev is None:
        return 0.0
    ux, uy = u
    px, py = prev
    vx, vy = v
    v_in = np.array([ux - px, uy - py])
    v_out = np.array([vx - ux, vy - uy])
    n1, n2 = np.linalg.norm(v_in), np.linalg.norm(v_out)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = float(np.dot(v_in, v_out) / (n1 * n2))
    return max(-1.0, min(1.0, cosang))


def build_smart_cover_route_with_targets(H, undH, start, target_edges,
                                         frontier_nodes, uturn_penalty=1000.0):
    visited_edges = set()
    route_edges = []
    u = start
    prev = None

    target_edges = set(target_edges)
    max_iter = 10 * max(1, len(target_edges) or H.number_of_edges())
    iter_count = 0

    while iter_count < max_iter:
        iter_count += 1

        # Any target edges left?
        remaining_targets = [e for e in target_edges if e not in visited_edges]
        if not remaining_targets:
            break

        out_edges = list(H.out_edges(u))
        if not out_edges:
            endpoints = [a for (a, _) in remaining_targets] + \
                        [b for (_, b) in remaining_targets]
            try:
                dist_map = nx.single_source_dijkstra_path_length(
                    undH, u, weight="time"
                )
            except nx.NetworkXNoPath:
                break
            reachable = [(n, dist_map[n]) for n in endpoints if n in dist_map]
            if not reachable:
                break
            next_node = min(reachable, key=lambda x: x[1])[0]
            path = nx.shortest_path(undH, u, next_node, weight="time")
            for a, b in zip(path[:-1], path[1:]):
                if H.has_edge(a, b):
                    route_edges.append((a, b))
                    visited_edges.add((a, b))
                elif H.has_edge(b, a):
                    route_edges.append((b, a))
                    visited_edges.add((b, a))
            if len(path) > 1:
                prev = path[-2]
            u = next_node
            continue

        candidates = []
        for (_, v) in out_edges:
            ang = turn_alignment(prev, u, v)
            is_target = (u, v) in target_edges
            is_frontier_edge = (u in frontier_nodes) or (v in frontier_nodes)

            cost = 0.0
            if ang < -0.9:     # U-turn
                cost += uturn_penalty
            if (u, v) in visited_edges:
                cost += 30.0   # discourage re-using same edge
            if is_target and (u, v) not in visited_edges:
                cost -= 50.0   # strong preference to hit targets
            elif is_frontier_edge:
                cost -= 10.0   # allow crossing borders
            else:
                cost += 5.0  

            cost -= ang * 10.0  # reward straightness

            candidates.append((cost, v))

        candidates.sort(key=lambda x: x[0])
        _, v = candidates[0]

        route_edges.append((u, v))
        visited_edges.add((u, v))
        prev, u = u, v

    if iter_count >= max_iter:
        print(f"Route from {start} stopped after {iter_count} steps (possible deadlock).")

    return route_edges


# =====================================================
# 5. CLUSTER WITH FRONTIERS & BUILD ROUTES
# =====================================================

def cluster_and_build_routes_with_frontiers(G, n_vehicles=10):
    """
    Cluster edges using KMeans, detect frontier nodes between clusters,
    then build one continuous route per vehicle with frontier-aware traversal.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return []

    pts = []
    weights = []
    edge_pairs = []
    for (u, v, data) in edges:
        mx, my = (u[0] + v[0]) / 2.0, (u[1] + v[1]) / 2.0
        pts.append([mx, my])
        weights.append(float(data.get("time", 1.0)))
        edge_pairs.append((u, v))
    pts = np.array(pts)
    weights = np.array(weights)

    kmeans = KMeans(n_clusters=n_vehicles, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pts, sample_weight=weights)

    edge_label = {e: lab for e, lab in zip(edge_pairs, labels)}

    node_labels = {}
    for (u, v), lab in edge_label.items():
        node_labels.setdefault(u, set()).add(lab)
        node_labels.setdefault(v, set()).add(lab)
    frontier_nodes = {n for n, labs in node_labels.items() if len(labs) > 1}

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
        target_edges = set()
        for (u, v, data) in veh_edges:
            H.add_edge(u, v, **data)
            target_edges.add((u, v))

        for n in frontier_nodes:
            for u, v, data in G.out_edges(n, data=True):
                H.add_edge(u, v, **data)
            for u, v, data in G.in_edges(n, data=True):
                H.add_edge(u, v, **data)

        if H.number_of_edges() == 0:
            routes.append([])
            continue

        undH = H.to_undirected()
        components = list(nx.connected_components(undH))

        vehicle_route = []
        prev_end = None

        for comp_nodes in components:
            compH = H.subgraph(comp_nodes).copy()
            if compH.number_of_edges() == 0:
                continue
            undComp = undH.subgraph(comp_nodes).copy()

            comp_target_edges = [(u, v) for (u, v) in target_edges
                                 if u in comp_nodes and v in comp_nodes]
            if not comp_target_edges:
                continue

            if prev_end is None:
                start = comp_target_edges[0][0]
            else:
                try:
                    dist_map = []
                    if prev_end not in undComp:
                        start = comp_target_edges[0][0]
                    else:
                        try:
                            dist_map = nx.single_source_dijkstra_path_length(
                                undComp, prev_end, weight="time"
                            )
                        except (nx.NodeNotFound, nx.NetworkXNoPath):
                            dist_map = {}

                    endpoints = [e[0] for e in comp_target_edges] + \
                                [e[1] for e in comp_target_edges]
                    reachable = [n for n in endpoints if n in dist_map]
                    if reachable:
                        start = min(reachable, key=lambda n: dist_map[n])
                        path = nx.shortest_path(undComp, prev_end, start, weight="time")
                        for a, b in zip(path[:-1], path[1:]):
                            if H.has_edge(a, b):
                                vehicle_route.append((a, b))
                            elif H.has_edge(b, a):
                                vehicle_route.append((b, a))
                        prev_end = start
                    else:
                        start = comp_target_edges[0][0]
                except nx.NetworkXNoPath:
                    start = comp_target_edges[0][0]

            comp_route = build_smart_cover_route_with_targets(
                compH, undComp, start, comp_target_edges, frontier_nodes
            )
            vehicle_route.extend(comp_route)
            if vehicle_route:
                prev_end = vehicle_route[-1][1]

        routes.append(vehicle_route)

    return routes


# =====================================================
# 6. COVERAGE CLEANUP
# =====================================================

def cleanup_missing_edges(G, routes):
    und = G.to_undirected()

    covered = set()
    for route in routes:
        for (u, v) in route:
            covered.add(tuple(sorted((u, v))))
    all_edges = {tuple(sorted((u, v))) for (u, v) in und.edges}
    missing = list(all_edges - covered)

    if not missing:
        return routes, 1.0

    seeds = []
    for route in routes:
        if route:
            seeds.append(route[0][0])
        else:
            seeds.append(None)

    for (u, v) in missing:
        mx, my = (u[0] + v[0]) / 2.0, (u[1] + v[1]) / 2.0

        best_k, best_d = None, float("inf")
        for k, s in enumerate(seeds):
            if s is None:
                continue
            d = math.hypot(s[0] - mx, s[1] - my)
            if d < best_d:
                best_d, best_k = d, k
        if best_k is None:
            best_k = 0

        route = routes[best_k]
        if route:
            last_node = route[-1][1]
        else:
            last_node = u

        try:
            path = nx.shortest_path(und, last_node, u, weight="time")
        except nx.NetworkXNoPath:
            continue

        for a, b in zip(path[:-1], path[1:]):
            if G.has_edge(a, b):
                route.append((a, b))
            elif G.has_edge(b, a):
                route.append((b, a))

        if G.has_edge(u, v):
            route.append((u, v))
        elif G.has_edge(v, u):
            route.append((v, u))

    covered = set()
    for route in routes:
        for (u, v) in route:
            covered.add(tuple(sorted((u, v))))
    coverage_ratio = len(covered) / max(1, len(all_edges))
    return routes, coverage_ratio


# =====================================================
# 7. VISUALIZATION
# =====================================================

def plot_routes(roads, routes):
    fig, ax = plt.subplots(figsize=(10, 10))
    roads.plot(ax=ax, color="lightgray", linewidth=0.5)
    cmap = matplotlib.colormaps.get_cmap("tab20")   
    for i, route in enumerate(routes):
        for (u, v) in route:
            xs, ys = zip(u, v)
            ax.plot(xs, ys, color=cmap(i), linewidth=2)
    ax.set_axis_off()
    plt.title(f"{len(routes)}-Vehicle Coverage (v7 frontier-aware)")
    output_path = os.path.join("routes", f"coverage_map_{len(routes)}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved coverage plot to {output_path}")



def animate_routes(routes, roads, interval=50):
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

    trails = []
    for i, coords in enumerate(route_coords):
        trail, = ax.plot([], [], '-', color=cmap(i), linewidth=1.5, alpha=0.5)
        trails.append(trail)

    ax.set_axis_off()
    plt.title("Vehicle Coverage Simulation (v7)")

    n_frames = max(len(c) for c in route_coords)

    def update(frame):
        for i, car in enumerate(cars):
            coords = route_coords[i]
            if frame < len(coords):
                x, y = coords[frame]
                car.set_data(x, y)
                xs, ys = zip(*coords[:frame+1])
                trails[i].set_data(xs, ys)
            else:
                x, y = coords[-1]
                car.set_data(x, y)

            
        return cars + trails

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    plt.show()
    return ani


# =====================================================
# 8. EXPORT: ROUTES + DIRECTIONS
# =====================================================

def export_routes(routes, filename_prefix="routes/route"):
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

    print(f"Exported {len(geojson_features)} non-empty routes to CSV + GeoJSON.")


def generate_local_directions(G, routes, filename_prefix="routes/directions"):
    print("Creating local turn-by-turn directions...")

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
            print(f"Skipping invalid route :( {i}")
            continue

        coords = [route[0][0]] + [v for (_, v) in route]
        cleaned = [coords[0]]
        for pt in coords[1:]:
            if pt != cleaned[-1]:
                cleaned.append(pt)

        steps = []
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
            if road_name == "Unnamed Road" and last_name:
                road_name = last_name

            t = turn_type(u, v, w)

            if G[u][v].get("length", 999) < 10:
                last_name = road_name
                continue

            if t == "straight":
                last_name = road_name
                continue
            elif t == "u-turn":
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

        print(f"Directions written for vehicle {i+1} ({len(steps)} steps)")


# =====================================================
# 9. MAIN DRIVER
# =====================================================

def run_fast_coverage(road_shp_path,
                      traffic_csv_path=None,
                      slope_csv_path=None,
                      n_vehicles=10):
    print("Loading roads shapefile...")
    roads = gpd.read_file(road_shp_path)

    roads = preprocess_roads(roads)
    print(f"Roads after filtering: {len(roads)} segments")

    traffic_df = None
    if traffic_csv_path and os.path.exists(traffic_csv_path):
        print("Loading traffic (AADT) data...")
        traffic_df = load_traffic_csv(traffic_csv_path)
        print(f"Loaded {len(traffic_df)} traffic rows")

    slope_lookup = None
    if slope_csv_path and os.path.exists(slope_csv_path):
        print("Loading slope priority data...")
        slope_lookup = load_slope_csv(slope_csv_path)
        print(f"Loaded {len(slope_lookup)} slope entries")

    print("Building directed graph...")
    G = build_graph(roads, traffic_df=traffic_df, slope_lookup=slope_lookup)

    if not nx.is_weakly_connected(G):
        G = max(
            (G.subgraph(c) for c in nx.weakly_connected_components(G)),
            key=len
        ).copy()
    print(f"Graph built: {len(G.nodes)} nodes, {len(G.edges)} directed edges")

    print("Clustering edges & building frontier-aware routes...")
    routes = cluster_and_build_routes_with_frontiers(G, n_vehicles=n_vehicles)
    print(f"Built {len(routes)} vehicle routes")

    print("Cleaning up missing edges for high coverage...")
    routes, coverage_ratio = cleanup_missing_edges(G, routes)
    print(f"Undirected edge coverage after cleanup: {coverage_ratio:.1%}")

    folder = prepare_routes_folder("routes")

    print("Plotting routes...")
    plot_routes(roads, routes)

    print("Exporting routes...")
    export_routes(routes, os.path.join(folder, "route"))

    print("Creating human-readable directions...")
    generate_local_directions(G, routes, os.path.join(folder, "directions"))


    animate_routes(routes, roads, interval=50)

    return G, roads, routes


if __name__ == "__main__":

    from optimal_vehicle_analysis import evaluate_clusters, plot_optimal_vehicle_curve
    
    G, roads, routes = run_fast_coverage(
        road_shp_path="Roads/Roads.shp",
        traffic_csv_path="ithaca_traffic_counts_2024_final.csv",    
        slope_csv_path="Elevation/roads_with_slope.csv",        
        n_vehicles=8
    )

    results = evaluate_clusters(G, roads, k_range=range(3, 15))
    plot_optimal_vehicle_curve(results)

