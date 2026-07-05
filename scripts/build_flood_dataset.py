from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


warnings.filterwarnings(
    "ignore",
    message="One or several characters couldn't be converted.*",
    category=RuntimeWarning,
)


TARGET_CRS = "EPSG:5179"
WGS84_CRS = "EPSG:4326"

EXPECTED_STAGE_LABELS = {
    0: "none",
    1: "~0.5m",
    2: "0.5~1.0m",
    3: "1.0~1.5m",
    4: "1.5~2.0m",
    5: "2.0~3.0m",
    6: "3.0m~",
}

RAINFALL_SCENARIOS = {
    "10mm/h": {"rainfall_mm_h": 10, "max_expected_stage": 2, "risk_multiplier": 0.8},
    "30mm/h": {"rainfall_mm_h": 30, "max_expected_stage": 4, "risk_multiplier": 1.05},
    "50mm/h": {"rainfall_mm_h": 50, "max_expected_stage": 6, "risk_multiplier": 1.3},
}

# Approximate Gangnam-gu analysis extent used for the prototype grid.
GANGNAM_BOUNDS_WGS84 = {
    "min_lon": 127.013,
    "min_lat": 37.456,
    "max_lon": 127.124,
    "max_lat": 37.536,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a grid-level flood dataset from RainGuard SHP layers."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing source SHP folders.")
    parser.add_argument(
        "--output",
        default="data/processed/flood_dataset.csv",
        help="CSV path to write the generated dataset.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=500,
        help="Grid cell size in meters. Default: 500.",
    )
    parser.add_argument(
        "--region-gu",
        default="강남구",
        help="GU_NAM value used to filter flood history polygons. Default: 강남구.",
    )
    return parser.parse_args()


def find_shp_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.shp"))


def get_expected_stage(shp_path: Path) -> int | None:
    match = re.search(r"DS_FLOODING_(\d+)", shp_path.stem, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def infer_missing_crs(gdf: gpd.GeoDataFrame) -> str:
    if gdf.empty:
        return WGS84_CRS

    minx, miny, _, _ = gdf.total_bounds
    if 120 <= minx <= 140 and 30 <= miny <= 45:
        return WGS84_CRS
    if 100000 <= minx <= 300000 and 400000 <= miny <= 700000:
        return "EPSG:5186"
    if 800000 <= minx <= 1100000 and 1800000 <= miny <= 2100000:
        return TARGET_CRS
    return "EPSG:5186"


def read_shp_file(shp_path: Path) -> gpd.GeoDataFrame:
    missing = [ext for ext in (".shx", ".dbf") if not shp_path.with_suffix(ext).exists()]
    if missing:
        raise FileNotFoundError(f"{shp_path} missing required sidecar files: {', '.join(missing)}")

    read_error = None
    for options in ({}, {"encoding": "utf-8"}, {"encoding": "cp949"}, {"encoding": "euc-kr"}):
        try:
            gdf = gpd.read_file(shp_path, **options)
            break
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            read_error = exc
    else:
        raise RuntimeError(f"Failed to read {shp_path}: {read_error}")

    if "geometry" not in gdf.columns:
        raise ValueError(f"{shp_path} does not contain a geometry column")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)

    if gdf.crs is None:
        gdf = gdf.set_crs(infer_missing_crs(gdf), allow_override=True)
    if str(gdf.crs).upper() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    gdf.geometry = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def read_expected_layers(expected_dir: Path) -> gpd.GeoDataFrame:
    frames = []
    for shp_path in find_shp_files(expected_dir):
        stage = get_expected_stage(shp_path)
        if stage is None:
            continue
        gdf = read_shp_file(shp_path)
        if gdf.empty:
            continue
        gdf = gdf[["geometry"]].copy()
        gdf["expected_stage"] = stage
        frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(columns=["expected_stage", "geometry"], geometry="geometry", crs=TARGET_CRS)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=TARGET_CRS)


def read_history_layers(history_dir: Path, region_gu: str) -> gpd.GeoDataFrame:
    frames = []
    for shp_path in find_shp_files(history_dir):
        gdf = read_shp_file(shp_path)
        if gdf.empty:
            continue

        if "GU_NAM" in gdf.columns:
            gdf = gdf[gdf["GU_NAM"].astype(str).str.strip() == region_gu].copy()
        if gdf.empty:
            continue

        year_match = re.search(r"(20\d{2})", shp_path.name)
        year = int(year_match.group(1)) if year_match else None
        keep_columns = [col for col in ["F_AVR_HGT", "F_AREA", "GU_NAM", "F_YR"] if col in gdf.columns]
        gdf = gdf[keep_columns + ["geometry"]].copy()
        gdf["history_year"] = year
        frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(columns=["history_year", "geometry"], geometry="geometry", crs=TARGET_CRS)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=TARGET_CRS)


def make_analysis_area() -> gpd.GeoDataFrame:
    geom = box(
        GANGNAM_BOUNDS_WGS84["min_lon"],
        GANGNAM_BOUNDS_WGS84["min_lat"],
        GANGNAM_BOUNDS_WGS84["max_lon"],
        GANGNAM_BOUNDS_WGS84["max_lat"],
    )
    return gpd.GeoDataFrame({"name": ["gangnam_analysis_area"]}, geometry=[geom], crs=WGS84_CRS).to_crs(TARGET_CRS)


def clip_to_area(gdf: gpd.GeoDataFrame, area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    area_geom = area.geometry.iloc[0]
    return gdf[gdf.intersects(area_geom)].copy()


def make_grid(area: gpd.GeoDataFrame, cell_size: int) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = area.total_bounds
    cells = []
    grid_ids = []
    idx = 1
    for x in np.arange(minx, maxx, cell_size):
        for y in np.arange(miny, maxy, cell_size):
            cell = box(x, y, min(x + cell_size, maxx), min(y + cell_size, maxy))
            if cell.intersects(area.geometry.iloc[0]):
                cells.append(cell)
                grid_ids.append(f"G{idx:04d}")
                idx += 1
    return gpd.GeoDataFrame({"grid_id": grid_ids}, geometry=cells, crs=TARGET_CRS)


def add_expected_features(grid: gpd.GeoDataFrame, expected: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid = grid.copy()
    if expected.empty:
        grid["expected_stage"] = 0
        return grid

    joined = gpd.sjoin(
        grid[["grid_id", "geometry"]],
        expected[["expected_stage", "geometry"]],
        how="left",
        predicate="intersects",
    )
    stage_by_grid = joined.groupby("grid_id")["expected_stage"].max()
    grid["expected_stage"] = grid["grid_id"].map(stage_by_grid).fillna(0).astype(int)
    return grid


def add_history_features(grid: gpd.GeoDataFrame, history: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid = grid.copy()
    for year in (2022, 2023):
        year_history = history[history["history_year"] == year].copy()
        count_col = f"history_{year}_polygon_count"
        flag_col = f"flood_history_{year}"

        if year_history.empty:
            grid[count_col] = 0
            grid[flag_col] = 0
            continue

        joined = gpd.sjoin(
            grid[["grid_id", "geometry"]],
            year_history[["geometry"]],
            how="left",
            predicate="intersects",
        )
        counts = joined[joined["index_right"].notna()].groupby("grid_id").size()
        grid[count_col] = grid["grid_id"].map(counts).fillna(0).astype(int)
        grid[flag_col] = (grid[count_col] > 0).astype(int)

    grid["history_polygon_count"] = (
        grid["history_2022_polygon_count"] + grid["history_2023_polygon_count"]
    )
    grid["flood_history"] = (grid["history_polygon_count"] > 0).astype(int)
    return grid


def union_geometry(gdf: gpd.GeoDataFrame):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def add_distance_feature(
    grid: gpd.GeoDataFrame, expected: gpd.GeoDataFrame, history: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    grid = grid.copy()
    flood_frames = [frame[["geometry"]] for frame in (expected, history) if not frame.empty]
    if not flood_frames:
        grid["distance_to_flood_area_m"] = np.nan
        return grid

    flood_area = gpd.GeoDataFrame(pd.concat(flood_frames, ignore_index=True), geometry="geometry", crs=TARGET_CRS)
    flood_union = union_geometry(flood_area)
    grid["distance_to_flood_area_m"] = grid.geometry.centroid.distance(flood_union).round(1)
    return grid


def grade_risk(score: int) -> str:
    if score >= 75:
        return "높음"
    if score >= 45:
        return "주의"
    return "낮음"


def expand_rainfall_scenarios(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    centroid_wgs84 = grid.geometry.centroid.to_crs(WGS84_CRS)
    base = pd.DataFrame(
        {
            "grid_id": grid["grid_id"],
            "latitude": centroid_wgs84.y.round(6),
            "longitude": centroid_wgs84.x.round(6),
            "expected_stage": grid["expected_stage"].astype(int),
            "expected_depth_class": grid["expected_stage"].map(EXPECTED_STAGE_LABELS),
            "flood_expected": (grid["expected_stage"] > 0).astype(int),
            "flood_history_2022": grid["flood_history_2022"].astype(int),
            "flood_history_2023": grid["flood_history_2023"].astype(int),
            "flood_history": grid["flood_history"].astype(int),
            "history_polygon_count": grid["history_polygon_count"].astype(int),
            "distance_to_flood_area_m": grid["distance_to_flood_area_m"],
        }
    )
    base["flood_label"] = ((base["flood_expected"] == 1) | (base["flood_history"] == 1)).astype(int)

    scenario_frames = []
    for scenario_name, scenario in RAINFALL_SCENARIOS.items():
        scenario_df = base.copy()
        scenario_df["rainfall_scenario"] = scenario_name
        scenario_df["rainfall_mm_h"] = scenario["rainfall_mm_h"]
        scenario_df["display_expected_stage_max"] = scenario["max_expected_stage"]
        scenario_df["scenario_expected_visible"] = (
            (scenario_df["expected_stage"] > 0)
            & (scenario_df["expected_stage"] <= scenario["max_expected_stage"])
        ).astype(int)
        scenario_df["scenario_flood_label"] = (
            (scenario_df["scenario_expected_visible"] == 1)
            | (scenario_df["flood_history"] == 1)
        ).astype(int)

        proximity = (1 - scenario_df["distance_to_flood_area_m"].fillna(3000) / 1500).clip(0, 1)
        depth_weight = scenario_df["expected_stage"] / 6
        raw_score = (
            scenario_df["scenario_expected_visible"] * 38
            + scenario_df["flood_history"] * 35
            + depth_weight * 17
            + proximity * 10
        )
        scenario_df["scenario_risk_score"] = (
            raw_score * scenario["risk_multiplier"]
        ).clip(0, 100).round().astype(int)
        scenario_df["risk_grade"] = scenario_df["scenario_risk_score"].map(grade_risk)
        scenario_frames.append(scenario_df)

    columns = [
        "grid_id",
        "latitude",
        "longitude",
        "rainfall_scenario",
        "rainfall_mm_h",
        "display_expected_stage_max",
        "expected_stage",
        "expected_depth_class",
        "scenario_expected_visible",
        "flood_expected",
        "flood_history_2022",
        "flood_history_2023",
        "flood_history",
        "history_polygon_count",
        "distance_to_flood_area_m",
        "scenario_risk_score",
        "risk_grade",
        "flood_label",
        "scenario_flood_label",
    ]
    return pd.concat(scenario_frames, ignore_index=True)[columns]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    expected_dir = data_dir / "flood_expected"
    history_dir = data_dir / "flood_history"
    output_path = Path(args.output)

    analysis_area = make_analysis_area()
    expected = clip_to_area(read_expected_layers(expected_dir), analysis_area)
    history = clip_to_area(read_history_layers(history_dir, args.region_gu), analysis_area)

    grid = make_grid(analysis_area, args.cell_size)
    grid = add_expected_features(grid, expected)
    grid = add_history_features(grid, history)
    grid = add_distance_feature(grid, expected, history)

    dataset = expand_rainfall_scenarios(grid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")
    print(f"Rows: {len(dataset):,}")
    print(f"Grid cells: {dataset['grid_id'].nunique():,}")
    print(f"Expected polygons in area: {len(expected):,}")
    print(f"History polygons in area: {len(history):,}")
    print(dataset.groupby("rainfall_scenario")["scenario_flood_label"].sum().to_string())


if __name__ == "__main__":
    main()
