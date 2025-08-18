import math
import aiohttp
import asyncio
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer
import cv2

from .map import *

__all__ = ["TiledMap"]

MAP_SIZE = 2 * 20037508.342789244  # web mercator tiles side length (meters)


class SourceType(Enum):
    WEB = "web"
    FILE = "file"


class TiledMap(Map):
    def __init__(
        self, url: str, max_concurrent=10, tile_size=256, zmin=0, zmax=19
    ) -> None:
        super().__init__()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tile_size = tile_size
        self.zmin = zmin
        self.zmax = zmax
        self.session = None

        url = url.replace("\\", "/")
        base_index = url.find("/", 0, url.find("{"))
        base_url, file_url = url[:base_index], url[base_index:]
        if url.startswith("file://"):
            self.type = SourceType.FILE
            self.url = url[len("file://") :]
        elif (path := Path(base_url)).exists():
            self.type = SourceType.FILE
            self.url = path.absolute().__str__() + file_url
        else:
            self.type = SourceType.WEB
            self.url = url

    async def connect(self, **kwargs):
        """Connect session"""
        kwargs.setdefault(
            "headers", {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        kwargs.setdefault("trust_env", True)  # use system proxy
        if self.type is SourceType.WEB:
            if self.session is None or self.session.closed:
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
        crs: CRS | str = "EPSG:4326",
        resolution: float = 1,
    ):
        zoom = math.ceil(math.log(MAP_SIZE / resolution))
        zoom = min(max(zoom, self.zmin), self.zmax)
        # transform bounds to web mercator
        xx, yy = np.array(bounds).reshape(-1, 2).T
        to_merc = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
        merc_xy = to_merc.transform(xx, yy)
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
        >>> global_xy = tiled_map.merc2tile(*merc_xy, zoom)
        >>> tile_xy, pixel_xy = np.divmod(global_xy, tiled_map.tile_size)
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

    async def download_tiles(self, bbox, zoom, dst: NDArray[np.integer]):
        x_min, y_min, x_max, y_max = bbox
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
                task = self.download_tile(zoom, x, y, dst=dst[y_slice, x_slice, :])
                tasks.append(task)
        await asyncio.gather(*tasks)

    async def download_tile(
        self, zoom: int, x: int, y: int, dst: NDArray[np.integer] | None = None
    ) -> NDArray[np.integer]:
        async with self.semaphore:  # limit max concurrent
            if dst is None:
                dst = np.empty((self.tile_size, self.tile_size, 3), np.uint8)
            url = self.url.format(z=zoom, x=x, y=y)

            try:
                match self.type:
                    case SourceType.WEB:
                        await self.load_tile_web(url=url, dst=dst)
                    case SourceType.FILE:
                        await self.load_tile_file(url=url, dst=dst)
                    case _:
                        raise NotImplementedError(f"Unexpect source type: {self.type}")
            except Exception as e:
                dst[...] = 255
                put_text = partial(
                    cv2.putText,
                    img=dst,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.002 * self.tile_size,
                    color=(0, 0, 0),
                    thickness=round(0.004 * self.tile_size),
                )
                put_text(text=f"Failed to load tile:", org=(10, 20))
                put_text(text=f"({x},{y},{zoom})", org=(10, 60))
                put_text(text=f"{e}", org=(10, 100))
                raise e
            return dst

    async def load_tile_web(self, url: str, dst: NDArray[np.integer]):
        assert self.session is not None, "Session has not created"
        async with self.session.get(url) as response:
            response.raise_for_status()
            data = await response.read()
            img_data = np.frombuffer(data, dtype=np.uint8)
            img_array = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB, dst=dst)
            return img_array

    async def load_tile_file(self, url: str, dst: NDArray[np.integer]):
        img = await asyncio.to_thread(
            cv2.imread, filename=url, flags=cv2.IMREAD_COLOR_RGB
        )
        dst[...] = img
        return img
