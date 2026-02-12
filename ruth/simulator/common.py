import logging
from dataclasses import asdict
from datetime import timedelta

import pandas as pd
import json
import ast

from ..data.map import BBox
from ..vehicle import Vehicle

logger = logging.getLogger(__name__)


def load_vehicles(input_path: str):
    logger.info("Loading data... %s", input_path)
    df = pd.read_parquet(input_path, engine="fastparquet")
    
    def _normalize_osm_route(osm_route):
        # Null/NaN -> empty list
        if osm_route is None or pd.isna(osm_route):
            return []
        # bytes -> decode to str
        if isinstance(osm_route, (bytes, bytearray)):
            try:
                osm_route = osm_route.decode("utf-8")
            except Exception:
                try:
                    osm_route = osm_route.decode("utf-8", errors="replace")
                except Exception:
                    return []
        # str that looks like a list -> parse
        if isinstance(osm_route, str):
            s = osm_route.strip()
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                except Exception:
                    try:
                        parsed = ast.literal_eval(s)
                    except Exception:
                        logger.debug("Failed to parse osm_route: %r", s)
                        return []
                res = []
                for el in parsed:
                    try:
                        res.append(int(el))
                    except Exception:
                        continue
                return res
            else:
                return []
        # list/tuple -> ensure ints
        if isinstance(osm_route, (list, tuple)):
            res = []
            for el in osm_route:
                try:
                    res.append(int(el))
                except Exception:
                    continue
            return res
        return []
    vehicles = [Vehicle(
        id=row["id"],
        time_offset=row["time_offset"],
        frequency=row["frequency"],
        start_index=row["start_index"],
        start_distance_offset=row["start_distance_offset"],
        origin_node=row["origin_node"],
        dest_node=row["dest_node"],
        osm_route=_normalize_osm_route(row.get("osm_route")),
        active=row["active"],
        fcd_sampling_period=row["fcd_sampling_period"],
        status=row["status"],
        vehicle_type=str(
            row.get("vehicle_type", "car").decode("utf-8", errors="ignore")
            if isinstance(row.get("vehicle_type", "car"), (bytes, bytearray))
            else row.get("vehicle_type", "car")
        ).lower()
    ) for (_, row) in df.iterrows()]

    filtered_count = 0
    for vehicle in vehicles:
        if not vehicle.osm_route or len(vehicle.osm_route) < 2:
            vehicle.active = False
            filtered_count += 1

    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} vehicles with too short routes.")
        vehicles = [v for v in vehicles if v.active]

    bbox_lat_max = df["bbox_lat_max"].iloc[0]
    bbox_lon_min = df["bbox_lon_min"].iloc[0]
    bbox_lat_min = df["bbox_lat_min"].iloc[0]
    bbox_lon_max = df["bbox_lon_max"].iloc[0]
    download_date = df["download_date"].iloc[0]
    bbox = BBox(bbox_lat_max, bbox_lon_min, bbox_lat_min, bbox_lon_max)
    return vehicles, bbox, download_date


def save_vehicles(vehicles, output_path: str):
    logger.info("Saving vehicles ... %s", output_path)

    df = pd.DataFrame([asdict(v) for v in vehicles])
    df.to_pickle(output_path)