import csv
from pathlib import Path
import math
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

from .dataset import UAVDataset, UAVData

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
    focal_length: float = 4000  # coarse guess of VisLoc Dataset

    pitch: float = field(init=False)
    roll: float = field(init=False)
    yaw: float = field(init=False)

    def __post_init__(self):
        # Convert degrees to radians
        self.pitch = math.radians(self.Omega)
        self.roll = math.radians(self.Kappa)
        self.yaw = math.radians(self.Phi1)

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
        if image is None:
            raise IOError(f"Failed to load image {self.filename}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return image


class VisLocDataset(UAVDataset):
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
            longitude=uav_info.lon,
            latitude=uav_info.lat,
            height=uav_info.height,
            yaw=uav_info.yaw,
            pitch=uav_info.pitch,
            roll=uav_info.roll,
            focal_length=uav_info.focal_length,
        )

    def __len__(self) -> int:
        return len(self.uav_infos)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
