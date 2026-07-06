from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import geopandas as gpd
import osmnx as ox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and save a Gangnam road network GraphML file.")
    parser.add_argument(
        "--boundary",
        default="data/boundary/gangnam_boundary.geojson",
        help="Boundary GeoJSON used to clip the road network.",
    )
    parser.add_argument(
        "--output",
        default="data/road_network/gangnam_walk.graphml",
        help="GraphML path to save the road network.",
    )
    parser.add_argument(
        "--pickle-output",
        default="data/road_network/gangnam_walk.pkl",
        help="Pickle path to save the same road network for faster app loading.",
    )
    parser.add_argument(
        "--network-type",
        default="walk",
        choices=["walk", "drive", "bike", "all"],
        help="OSMnx network type to download.",
    )
    return parser.parse_args()


def union_geometry(gdf: gpd.GeoDataFrame):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def main() -> None:
    args = parse_args()
    boundary_path = Path(args.boundary)
    output_path = Path(args.output)
    pickle_output_path = Path(args.pickle_output)

    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")

    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326", allow_override=True)
    boundary = boundary.to_crs("EPSG:4326")
    polygon = union_geometry(boundary).buffer(0)

    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.timeout = 180

    graph = ox.graph_from_polygon(
        polygon,
        network_type=args.network_type,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, output_path)
    pickle_output_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_output_path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved road network: {output_path}")
    print(f"Saved fast-load copy: {pickle_output_path}")
    print(f"Network type: {args.network_type}")
    print(f"Nodes: {len(graph.nodes):,}")
    print(f"Edges: {len(graph.edges):,}")


if __name__ == "__main__":
    main()
