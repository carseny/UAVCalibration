import numpy as np
import cv2
from PIL import Image

import math
import csv
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UAVData:
    dataset_path: Path
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
    h: int = field(init=False)
    w: int = field(init=False)

    def __post_init__(self):
        # Convert degrees to radians
        self.pitch = math.radians(self.Omega)
        self.roll = math.radians(self.Kappa)
        self.yaw = math.radians(self.Phi1)
        # Read image's dimensions without loading the entire image into memory
        with Image.open(self.image_path) as img:
            self.w, self.h = img.size
        # If focal length is not provided, set it to a reasonable default value
        if self.focal_length == 0:
            self.focal_length = max(self.w, self.h) * 1.5

    @property
    def image_path(self):
        return str(self.dataset_path / "drone" / self.filename)

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
                [self.focal_length, 0, self.w // 2],
                [0, self.focal_length, self.h // 2],
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
    def corrected_image(self):
        return self.correct_image(self.image)

    def correct_image(self, image: np.ndarray):
        K = self.cam_mat
        R = self.rot_mat
        H = K @ R @ np.linalg.inv(K)
        # 应用透视变换
        corrected = cv2.warpPerspective(image, H, (self.w, self.h))
        return corrected


class VisLocDataset(list[UAVData]):
    converters = {
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
            k: cls.converters[k](v) if k in cls.converters else v
            for k, v in row.items()
        }

    def __init__(self, dataset_path: str | Path):
        super().__init__()
        dataset_path = Path(dataset_path)
        with (dataset_path / (dataset_path.name + ".csv")).open("r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                super().append(UAVData(dataset_path, **self.convert_dict(row)))


if __name__ == "__main__":
    dataset = VisLocDataset("datasets/UAV_VisLoc_example/03")
    print(max(dataset, key=lambda x: x.Phi2))
    print(min(dataset, key=lambda x: x.Phi2))
    for d in dataset:
        print(f"lat: {d.lat:.5f}, lon: {d.lon:.5f}, height: {d.height:.2f}, Phi1: {d.Phi1:.2f}, Phi2: {d.Phi2:.2f}")
        image = d.image
        h, w = image.shape[:2]
        shape = (w // 3, h // 3)
        cv2.imshow("Image", cv2.resize(image, shape))
        cv2.imshow("Corrected", cv2.resize(d.correct_image(image), shape))
        cv2.waitKey(0)
