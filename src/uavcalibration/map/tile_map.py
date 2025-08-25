import math
import aiohttp
import asyncio
from enum import Enum
from pathlib import Path
import logging

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer
import cv2
from cachetools import LRUCache

from .map import *

__all__ = ["TileMap"]

LOGGER = logging.getLogger(__name__)
MAP_SIZE = 2 * 20037508.342789244  # web mercator tiles side length (meters)


class SourceType(Enum):
    WEB = "web"
    FILE = "file"


class TileMap(Map):
    def __init__(
        self,
        url: str,
        max_concurrent=10,
        cache_size=128,
        tile_size=256,
        zmin=0,
        zmax=19,
    ) -> None:
        super().__init__()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.cache: LRUCache[str, ImageMat] = LRUCache(cache_size)
        self.tile_size = tile_size
        self.zmin = zmin
        self.zmax = zmax
        self.session: aiohttp.ClientSession | None = None

        url = url.replace("\\", "/")
        base_index = url.find("/", 0, url.find("{"))
        base_url, file_url = url[:base_index], url[base_index:]
        if url.startswith("file://"):
            self.src_type = SourceType.FILE
            self.url = url[len("file://") :]
        elif (path := Path(base_url)).exists():
            self.src_type = SourceType.FILE
            self.url = path.absolute().__str__() + file_url
        else:
            self.src_type = SourceType.WEB
            self.url = url

    async def connect(self, **kwargs):
        """Connect session"""
        if self.src_type is SourceType.WEB:
            if self.session is None or self.session.closed:
                kwargs.setdefault(
                    "headers",
                    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                kwargs.setdefault("trust_env", True)  # use system proxy
                self.session = aiohttp.ClientSession(**kwargs)
                self.session = await self.session.__aenter__()

    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def get_async(
        self,
        bounds: tuple[float, float, float, float],
        crs: str = "EPSG:4326",
        resolution: float = 1e-5,
    ):
        # transform bounds to web mercator
        xx, yy = np.array(bounds).reshape(-1, 2).T
        to_merc = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
        merc_xy = to_merc.transform(xx, yy)
        # calculate resolution and zoom
        merc_resolution = (
            (merc_xy[0][1] - merc_xy[0][0]) / (bounds[2] - bounds[0]) * resolution
        )
        zoom = math.ceil(math.log(MAP_SIZE / merc_resolution))
        zoom = min(max(zoom, self.zmin), self.zmax)
        # transform web mercator to tile coordinates
        global_xy = self.merc2tile(*merc_xy, zoom=zoom).astype(np.int64)
        tile_xy, pixel_xy = np.divmod(global_xy, self.tile_size)
        xy_min, xy_max = tile_xy.min(1), tile_xy.max(1)
        (x_min, y_min), (x_max, y_max) = xy_min, xy_max

        # initialize empty canvas
        height, width = (xy_max - xy_min + 1) * self.tile_size
        canvas = np.empty((width, height, 3), np.uint8)
        # download tiles
        async with self as self:
            await self.download_tiles((x_min, y_min, x_max, y_max), zoom, dst=canvas)

        pix_min, pix_max = pixel_xy.min(1), pixel_xy.max(1)
        pix_max -= self.tile_size
        image = canvas[pix_min[1] : pix_max[1], pix_min[0] : pix_max[0]]
        # shift window
        window_trans = np.eye(3, dtype=np.float64)
        window_trans[:2, 2] = global_xy.min(1)
        # pixel -> web mercator
        window_trans = self.tile2merc_mat(zoom) @ window_trans
        return image, CRSTransform(window_trans, "EPSG:3857")

    def merc2tile(
        self,
        xx: float | NDArray[np.integer | np.floating],
        yy: float | NDArray[np.integer | np.floating],
        zoom: int | NDArray[np.integer] = 0,
    ) -> NDArray[np.floating]:
        """
        Transform from web mercator to google map tile index

        Returns
        -------
        (xx, yy) : NDArray[np.floating]
            Global coordinates (in pixel) on given zoom scale

        Examples
        --------
        >>> global_xy = tile_map.merc2tile(*merc_xy, zoom)
        >>> tile_xy, pixel_xy = np.divmod(global_xy, tile_map.tile_size)
        """
        xy1 = np.asarray((xx, yy, np.ones_like(xx)), np.float64)
        result = self.merc2tile_mat(zoom)[:2] @ xy1
        return result[:2]

    def tile2merc(
        self,
        xx: float | NDArray[np.integer | np.floating],
        yy: float | NDArray[np.integer | np.floating],
        zoom: int | NDArray[np.integer] = 0,
    ) -> NDArray[np.floating]:
        xy1 = np.asarray((xx, yy, np.ones_like(xx)), np.float64)
        result = self.tile2merc_mat(zoom) @ xy1
        return result[:2]

    def merc2tile_mat(self, zoom: int | NDArray[np.integer] = 0):
        # # reverse y axie
        # merc[1] *= -1
        # # normalize to [0,1]
        # merc /= MAP_SIZE
        # merc += 0.5
        # # scale up
        # global_xy = merc * map_size
        map_size = self.tile_size << zoom
        mat = np.array(
            [
                [map_size / MAP_SIZE, 0, map_size * 0.5],
                [0, -map_size / MAP_SIZE, map_size * 0.5],
            ],
            dtype=np.float64,
        )
        return mat

    def tile2merc_mat(self, zoom: int | NDArray[np.integer] = 0):
        map_size = self.tile_size << zoom
        mat = np.array(
            [
                [MAP_SIZE / map_size, 0, -MAP_SIZE * 0.5],
                [0, -MAP_SIZE / map_size, MAP_SIZE * 0.5],
            ],
            dtype=np.float64,
        )
        return mat

    async def download_tiles(self, bounds, zoom, dst: ImageMat) -> None:
        """Download all tiles within a bounding box to dst array"""
        x_min, y_min, x_max, y_max = bounds
        # download tiles
        tasks = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # calculate corresponding area
                x_pos = (x - x_min) * self.tile_size
                y_pos = (y - y_min) * self.tile_size
                x_slice = slice(x_pos, x_pos + self.tile_size)
                y_slice = slice(y_pos, y_pos + self.tile_size)
                # create dowanload Coroutine
                coro = self.download_tile(zoom, x, y, dst=dst[y_slice, x_slice, :])
                task = asyncio.create_task(coro)
                tasks.append(task)
        await asyncio.gather(*tasks)

    async def download_tile(
        self, zoom: int, x: int, y: int, dst: ImageMat | None = None
    ) -> ImageMat:
        url = self.url.format(z=zoom, x=x, y=y)
        async with self.semaphore:  # limit max concurrent
            try:
                img = await self.get_url(url=url)
            except Exception as e:
                img = np.empty((self.tile_size, self.tile_size, 3), np.uint8)
                img[...] = 255
                cv2.putText(
                    img=img,
                    text=f"No Image",
                    org=(10, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.005 * self.tile_size,
                    color=(0, 0, 0),
                    thickness=round(0.01 * self.tile_size),
                )
                LOGGER.warning(f"Failed to get tile ({x},{y},{zoom}): {e}")
        if dst is not None:
            dst[...] = img
        return img

    async def get_url(self, url: str):
        """Read an image from url with cache (based on self source type)"""
        if url not in self.cache:
            match self.src_type:
                case SourceType.WEB:
                    # download from internet
                    img_array = await self._get_web(url)
                case SourceType.FILE:
                    # read from disk
                    img_array = await self._get_file(url)
                case _:
                    raise NotImplementedError(f"Unexpect source type: {self.src_type}")
            self.cache[url] = img_array
        return self.cache[url]

    async def _get_web(self, url: str):
        assert self.session is not None, "Session has not created"
        async with self.session.get(url) as response:
            response.raise_for_status()
            data = await response.read()
            img_data = np.frombuffer(data, dtype=np.uint8)
            img_array = cv2.imdecode(img_data, cv2.IMREAD_COLOR_RGB)
            return img_array

    async def _get_file(self, url: str):
        img_array = await asyncio.to_thread(cv2.imread, url, cv2.IMREAD_COLOR_RGB)
        if img_array is None:
            raise IOError(f"Failed to read file: {url}")
        return img_array
