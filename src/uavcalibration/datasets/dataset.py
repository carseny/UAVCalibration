from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

import math
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

Image.MAX_IMAGE_PIXELS = 1 << 32
LARGE_IMAGE_CACHE_SIZE = 1


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
        return image[..., ::-1]  # Convert BGR to RGB

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


class UAVDataset(Dataset[UAVData]): ...
