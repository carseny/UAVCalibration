from pathlib import Path

import numpy as np
import cv2
import rasterio
from rasterio.warp import reproject, Resampling, transform
from rasterio.io import DatasetReader
from rasterio.crs import CRS

from .map import *

__all__ = ["GeoTiffMap"]


class GeoTiffMap(Map):
    def __init__(self, filepath: str | Path | list[str | Path]):
        super().__init__()
        self.files: list[Path] = []
        self.datasets: dict[Path, DatasetReader] = {}
        self.crs = None
        if isinstance(filepath, list):
            for path in filepath:
                self.add(path)
        else:
            self.add(filepath)

    def add(self, path: str | Path):
        path = Path(path)
        assert path.exists()
        if path.is_dir():
            for sub_path in path.iterdir():
                self.add(sub_path)
        else:
            self.files.append(path.absolute())

    async def __aenter__(self) -> Self:
        tasks: list[asyncio.Task] = []
        for file in self.files:
            if (dataset := self.datasets.get(file)) is None or dataset.closed:
                tasks.append(asyncio.Task(self._open_tiff(file)))
        await asyncio.gather(*tasks)
        return self

    async def _open_tiff(self, file: Path):
        dataset = await asyncio.to_thread(lambda: rasterio.open(file))
        if self.crs is None:
            self.crs = dataset.crs
        elif not self.crs == dataset.crs:
            raise ValueError("CRS of all dataset should be same.")
        self.datasets[file] = dataset

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()

    async def get_async(
        self,
        bounds: tuple[float, float, float, float],
        crs: str = "EPSG:4326",
        resolution: float = 1e-5,
    ) -> tuple[ImageMat, CRSTransform]:
        target_crs = CRS.from_user_input(crs)
        # transform target bounds to dataset src bounds
        xy = transform(
            target_crs, self.crs, [bounds[0], bounds[2]], [bounds[1], bounds[3]]
        )
        assert len(xy) == 2
        xy = np.asarray(xy)
        x_min, y_min = xy.min(1)
        x_max, y_max = xy.max(1)
        # find dataset that contains the bounds
        for dataset in self.datasets.values():
            left, bottom, right, top = dataset.bounds
            if left <= x_min and x_max <= right and bottom <= y_min and y_max <= top:
                break
        else:
            raise ValueError("bounds is not in any of dataset")

        # calculate transform and new shape
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        dst_transform = rasterio.Affine(
            resolution,
            0,
            bounds[0],
            0,
            -resolution,  # 注意Y方向取负
            bounds[3],
        )

        # create output array
        dst = np.zeros((dataset.count, height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(dataset, [i + 1 for i in range(dataset.count)]),
            destination=dst,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
        dst = np.moveaxis(dst, 0, -1)

        # ensure 3-channel RGB output
        if dst.shape[2] == 1:
            dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        elif dst.shape[2] > 3:
            dst = dst[:, :, :3]

        return dst, CRSTransform(dst_transform, target_crs.to_string())
