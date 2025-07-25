import numpy as np
import cv2
from PIL import Image

import math
import csv
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

Image.MAX_IMAGE_PIXELS = 1 << 32
LARGE_IMAGE_CACHE_SIZE = 2


@dataclass
class SatelliteData:
    dataset_path: Path
    partition: str

    mapname: str
    LT_lat_map: float
    LT_lon_map: float
    RB_lat_map: float
    RB_lon_map: float
    region: str

    bbox: tuple[float, float, float, float] = field(
        init=False
    )  # (west, south, east, north)
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
        return image

    @property
    def image(self):
        return self.load_large_image(self.image_path)

    def lonlat2wh(self, lon, lat):
        west, south, east, north = self.bbox
        w = self.img_w * (lon - west) / (east - west)
        h = self.img_h * (north - lat) / (north - south)
        return w, h


@dataclass
class UAVData:
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

    focal_length: float = 0

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
        # If focal length is not provided, set it to a reasonable default value
        if self.focal_length == 0:
            self.focal_length = max(self.img_w, self.img_h) * 1.5

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
        return image

    @property
    def cam_mat(self):
        # 相机内参矩阵（简化）
        K = np.array(
            [
                [self.focal_length, 0, self.img_w // 2],
                [0, self.focal_length, self.img_h // 2],
                [0, 0, 1],
            ],
        )
        return K

    @property
    def rot_mat(self):
        # Rotation matrices
        Rz = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw), 0],
                [np.sin(self.yaw), np.cos(self.yaw), 0],
                [0, 0, 1],
            ]
        )
        Ry = np.array(
            [
                [np.cos(self.pitch), 0, np.sin(self.pitch)],
                [0, 1, 0],
                [-np.sin(self.pitch), 0, np.cos(self.pitch)],
            ]
        )
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.roll), -np.sin(self.roll)],
                [0, np.sin(self.roll), np.cos(self.roll)],
            ]
        )
        # 旋转顺序待定
        return Rz @ Ry @ Rx

    @property
    def perspect_mat(self):
        K = self.cam_mat
        R = self.rot_mat
        return K @ R @ np.linalg.inv(K)

    @property
    def corrected_image(self):
        return cv2.warpPerspective(
            self.image, self.perspect_mat, (self.img_w, self.img_h)
        )


class VisLocDataset(list[UAVData]):
    def get_satellite_image(self, partition: str):
        return self.satellite_datas[partition].image

    def get_satellite_area(self, uav_data: UAVData):
        image = self.get_satellite_image(uav_data.partition)
        center_w, center_h = self.satellite_datas[uav_data.partition].lonlat2wh(
            uav_data.lon, uav_data.lat
        )
        scale = uav_data.height / 1240
        h = uav_data.img_h * scale
        w = uav_data.img_w * scale / np.cos(np.radians(uav_data.lat))
        return image[
            int(center_h - h // 2) : int(center_h + h // 2),
            int(center_w - w // 2) : int(center_w + w // 2),
        ]

    def read_folder(self, folder: Path):
        partition = folder.name
        with (folder / (partition + ".csv")).open("r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.append(
                    UAVData(folder.parent, partition, **UAVData.convert_dict(row))
                )

    def __init__(self, dataset_path: str | Path):
        super().__init__()
        dataset_path = Path(dataset_path)
        self.satellite_datas: dict[str, SatelliteData] = {}
        with (dataset_path / "satellite_coordinates_range.csv").open("r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                partition = row["mapname"].rsplit(".")[0]
                self.satellite_datas[partition] = SatelliteData(
                    dataset_path, partition, **SatelliteData.convert_dict(row)
                )
        for folder in dataset_path.iterdir():
            if folder.is_dir():
                self.read_folder(folder)


if __name__ == "__main__":
    dataset = VisLocDataset("datasets/UAV_VisLoc_example/")
    print(max(dataset, key=lambda x: x.Phi2))
    print(min(dataset, key=lambda x: x.Phi2))
    for d in dataset:
        print(
            f"lat: {d.lat:.5f}, lon: {d.lon:.5f}, yaw: {d.yaw:.2f}, pitch: {d.pitch:6.3f}, roll: {d.roll:6.3f}"
        )
        image = d.image
        corrected = cv2.warpPerspective(image, d.perspect_mat, (d.img_w, d.img_h))
        satellite = dataset.get_satellite_area(d)
        h, w = image.shape[:2]
        view_shape = (w // 5, h // 5)
        cv2.imshow("Original", cv2.resize(image, view_shape))
        cv2.imshow("Corrected", cv2.resize(corrected, view_shape))
        cv2.imshow("Satellite", cv2.resize(satellite, view_shape))
        cv2.waitKey(0)
