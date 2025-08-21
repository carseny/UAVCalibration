import asyncio
from abc import ABC, abstractmethod
from typing import Self

from ..transform import CRSTransform
from ..types import *


class Map(ABC):
    """
    Abstract class of image map data
    """

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...

    @abstractmethod
    async def get_async(
        self, bounds: tuple[float, float, float, float], crs: str, resolution: float
    ) -> tuple[ImageMat, CRSTransform]:
        """
        Asynchronous method to get a image of given bounds

        Parameters
        ----------
        bounds
            Bounding box (x_min, y_min, x_max, y_max)
        crs: str
            CRS of bounds
        resolution: float
            CRS unit per pixel

        Returns
        -------
        image: Ndarray
            image array (h, w, 3 [rgb])
        crs_transform: CRSTransform
            perspective transform from pixel coordinate to crs coordinate
        """
        ...

    def __init__(self):
        self.runner = None

    def __enter__(self) -> Self:
        assert self.runner is None
        self.runner = asyncio.Runner().__enter__()
        return self.runner.run(self.__aenter__())

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.runner is not None
        self.runner.run(self.__aexit__(exc_type, exc_val, exc_tb))
        self.runner.__exit__(exc_type, exc_val, exc_tb)
        self.runner = None

    def get(self, *args, **kwargs):
        assert self.runner is not None
        return self.runner.run(self.get_async(*args, **kwargs))
