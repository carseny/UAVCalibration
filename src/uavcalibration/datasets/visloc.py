import csv
from pathlib import Path
import math
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from pyproj import Transformer
import rasterio
import rasterio.io
import rasterio.windows
import rasterio.warp
from rasterio.coords import BoundingBox

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

    def image_area(
        self,
        center_lon: float,
        center_lat: float,
        side: float,  # meters
        resolution=1.0,  # meters per pixel
    ):
        """
        读取卫星图像的局部区域并转换为米制坐标系

        参数:
            image_path: 卫星图像路径
            center_lon: 区域中心经度
            center_lat: 区域中心纬度
            side: 正方形区域边长 (米)
            resolution: 输出分辨率 (米/像素)

        返回:
            (裁切后的图像数组, 变换矩阵)
        """
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone  # 北半球
        if center_lat < 0:  # 南半球
            utm_epsg += 100
        dst_crs = f"EPSG:{utm_epsg}"

        # 转换中心点到UTM坐标
        to_utm = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        center_x, center_y = to_utm.transform(center_lon, center_lat)

        # 计算目标区域边界
        half_side = side / 2
        utm_bounds = BoundingBox(
            center_x - half_side,
            center_y - half_side,
            center_x + half_side,
            center_y + half_side,
        )

        with rasterio.open(self.image_path) as src:
            assert isinstance(src, rasterio.io.DatasetReader)
            # 将UTM边界转换回源坐标系
            src_bounds: BoundingBox = rasterio.warp.transform_bounds(
                dst_crs, src.crs, *utm_bounds
            )

            # 计算读取窗口
            window = rasterio.windows.from_bounds(*src_bounds, transform=src.transform)

            # 读取数据
            data = src.read(window=window, boundless=True, fill_value=0)

            # 准备重投影参数
            height = width = int(side / resolution)
            dst_transform = rasterio.Affine(
                resolution,
                0,
                utm_bounds[0],
                0,
                -resolution,
                utm_bounds[3],  # 注意Y方向取负
            )

            # 执行重投影
            reprojected = np.zeros((data.shape[0], height, width), dtype=data.dtype)

            rasterio.warp.reproject(
                source=data,
                destination=reprojected,
                src_transform=src.window_transform(window),
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=rasterio.warp.Resampling.bilinear,
            )
            return np.moveaxis(reprojected, 0, -1)


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
    _img_w: int | None = field(default=None, init=False)
    _img_h: int | None = field(default=None, init=False)

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
        assert image is not None, f"Failed to load image {self.filename}"
        self._img_h, self._img_w = image.shape[:2]
        return image[..., ::-1]  # Convert BGR to RGB

    def __read_image_meta(self):
        # Read image's dimensions without loading the entire image into memory
        with Image.open(self.image_path) as img:
            self.img_w, self.img_h = img.size

    @property
    def image_h(self) -> int:
        if self._img_h is None:
            self.__read_image_meta()
        assert isinstance(self._img_h, int)
        return self._img_h

    @property
    def image_w(self) -> int:
        if self._img_w is None:
            self.__read_image_meta()
        assert isinstance(self._img_w, int)
        return self._img_w

    @property
    def resolution(self) -> float:
        return self.height / self.focal_length

    @property
    def diagonal_pixel(self) -> float:
        return (self.image_h**2 + self.image_w**2) ** 0.5

    @property
    def diagonal_meter(self) -> float:
        return self.height / self.focal_length * self.diagonal_pixel


class VisLocDataset(UAVDataset):
    def get_satellite_area(self, uav_info: UAVInfo):
        return self.satellite_infos[uav_info.partition].image_area(
            uav_info.lon,
            uav_info.lat,
            uav_info.diagonal_meter,
            uav_info.resolution,
        )

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
            focal_length=uav_info.focal_length,
        )

    def __len__(self) -> int:
        return len(self.uav_infos)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
