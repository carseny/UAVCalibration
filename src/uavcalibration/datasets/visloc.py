from .dataset import UAVDataset, UAVData

import numpy as np
import cv2
from PIL import Image

import csv
from pathlib import Path
import math
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

Image.MAX_IMAGE_PIXELS = 1 << 32
LARGE_IMAGE_CACHE_SIZE = 1


@dataclass
class SatelliteInfo:
    dataset_path: Path
    partition: str

    mapname: str
    LT_lat_map: float
    LT_lon_map: float
    RB_lat_map: float
    RB_lon_map: float
    region: str

    # (west, south, east, north)
    bbox: tuple[float, float, float, float] = field(init=False)
    img_w: int = field(init=False)
    img_h: int = field(init=False)

    def __post_init__(self):
        # Calculate the bounding box of the map region in WSEN order
        self.bbox = (
            min(self.LT_lon_map, self.RB_lon_map),
            min(self.LT_lat_map, self.RB_lat_map),
            max(self.LT_lon_map, self.RB_lon_map),
            max(self.LT_lat_map, self.RB_lat_map),
        )
        # Read image's dimensions without loading the entire image into memory
        with Image.open(self.image_path) as img:
            self.img_w, self.img_h = img.size

        west, south, east, north = self.bbox
        # Check aspect ratio consistency
        assert (
            0.99 < (self.img_h / (north - south)) / (self.img_w / (east - west)) < 1.01
        ), "Aspect ratio inconsistency"

    __converters = {
        "mapname": str,
        "LT_lat_map": float,
        "LT_lon_map": float,
        "RB_lat_map": float,
        "RB_lon_map": float,
        "region": str,
    }

    @classmethod
    def convert_dict(cls, row: dict[str, str]) -> dict[str, Any]:
        return {
            k: cls.__converters[k](v) if k in cls.__converters else v
            for k, v in row.items()
        }

    @property
    def image_path(self):
        return str(self.dataset_path / self.partition / ("satellite" + self.mapname))

    @staticmethod
    @lru_cache(LARGE_IMAGE_CACHE_SIZE)
    def load_large_image(image_path):
        image = cv2.imread(image_path)
        assert image is not None, f"Failed to load image {image_path}"
        return image[..., ::-1]  # Convert BGR to RGB

    @property
    def image(self):
        return self.load_large_image(self.image_path)

    def lonlat2wh(self, lon, lat):
        west, south, east, north = self.bbox
        w = self.img_w * (lon - west) / (east - west)
        h = self.img_h * (north - lat) / (north - south)
        return w, h


@dataclass
class UAVInfo:
    dataset_path: Path
    partition: str

    num: int
    filename: str
    date: datetime
    lat: float
    lon: float
    height: float
    Omega: float
    Kappa: float
    Phi1: float
    Phi2: float

    pitch: float = field(init=False)
    roll: float = field(init=False)
    yaw: float = field(init=False)
    img_w: int = field(init=False)
    img_h: int = field(init=False)

    def __post_init__(self):
        # Convert degrees to radians
        self.pitch = math.radians(self.Omega)
        self.roll = math.radians(self.Kappa)
        self.yaw = math.radians(self.Phi1)
        # Read image's dimensions without loading the entire image into memory
        with Image.open(self.image_path) as img:
            self.img_w, self.img_h = img.size

    __converters = {
        "num": int,
        "filename": str,
        "date": datetime.fromisoformat,
        "lat": float,
        "lon": float,
        "height": float,
        "Omega": float,
        "Kappa": float,
        "Phi1": float,
        "Phi2": float,
    }

    @classmethod
    def convert_dict(cls, row: dict[str, str]) -> dict[str, Any]:
        return {
            k: cls.__converters[k](v) if k in cls.__converters else v
            for k, v in row.items()
        }

    @property
    def image_path(self):
        return str(self.dataset_path / self.partition / "drone" / self.filename)

    @property
    def image(self):
        image = cv2.imread(self.image_path)
        assert image is not None, f"Failed to load image {self.filename}"
        return image[..., ::-1]  # Convert BGR to RGB


class VisLocDataset(UAVDataset):
    def get_satellite_image(self, partition: str):
        return self.satellite_infos[partition].image

    def get_satellite_area(self, uav_info: UAVInfo):
        image = self.satellite_infos[uav_info.partition].image
        center_w, center_h = self.satellite_infos[uav_info.partition].lonlat2wh(
            uav_info.lon, uav_info.lat
        )
        scale = max(uav_info.img_h, uav_info.img_w) * uav_info.height / 1000
        h = scale
        w = scale / np.cos(np.radians(uav_info.lat))
        area = image[
            int(center_h - h / 2) : int(center_h + h / 2),
            int(center_w - w / 2) : int(center_w + w / 2),
        ]
        return cv2.resize(area, (int(scale),) * 2)

    def read_folder(self, folder: Path):
        partition = folder.name
        with (folder / (partition + ".csv")).open("r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.uav_infos.append(
                    UAVInfo(folder.parent, partition, **UAVInfo.convert_dict(row))
                )

    def __init__(self, dataset_path: str | Path):
        super().__init__()
        dataset_path = Path(dataset_path)
        self.uav_infos: list[UAVInfo] = []
        self.satellite_infos: dict[str, SatelliteInfo] = {}
        with (dataset_path / "satellite_coordinates_range.csv").open("r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                partition = row["mapname"].rsplit(".")[0]
                self.satellite_infos[partition] = SatelliteInfo(
                    dataset_path, partition, **SatelliteInfo.convert_dict(row)
                )
        for folder in dataset_path.iterdir():
            if folder.is_dir():
                self.read_folder(folder)

    def __getitem__(self, index: int):
        uav_info = self.uav_infos[index]
        return UAVData(
            uav_image=uav_info.image,
            satellite_image=self.get_satellite_area(uav_info),
            longitude=uav_info.lon,
            latitude=uav_info.lat,
            height=uav_info.height,
            pitch=uav_info.pitch,
            roll=uav_info.roll,
            yaw=uav_info.yaw,
        )

    def __len__(self) -> int:
        return len(self.uav_infos)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
