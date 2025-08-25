from typing import Any, TypedDict, cast
from dataclasses import asdict

from .pipeline import *
from .transform import *
from .datasets import *
from .map import *
from .types import *
from .match import *
from . import calibrate
from .calibrate import CalibrateCTX

__all__ = ["create_pipeline"]


class Stage1Param(TypedDict):
    uav_image: ImageMat
    longitude: float
    latitude: float
    focal_length: float
    yaw: float
    pitch: float
    roll: float
    height: float


class Stage1(SyncStage[UAVData, CalibrateCTX, Stage1Param, tuple[ImageMat, Transform]]):
    def preprocess(self, i):
        kwargs: Stage1Param = cast(Stage1Param, asdict(i))
        return kwargs

    @staticmethod
    def task(args):
        return calibrate.coarse_calibrate(**args)

    def postprocess(self, i, p, r):
        ctx = CalibrateCTX(i)
        ctx.rect_image, ctx.uav_transform = r
        return ctx


class Stage2Param(TypedDict):
    map: Map
    uav_shape: Shape
    uav_transform: Transform


class Stage2(
    AsyncStage[CalibrateCTX, CalibrateCTX, Stage2Param, tuple[ImageMat, CRSTransform]]
):
    def __init__(self, map: Map, input_maxsize: int = 10) -> None:
        super().__init__(input_maxsize)
        self.map = map

    def preprocess(self, i):
        kwargs: Stage2Param = {
            "map": self.map,
            "uav_shape": i.uav_shape,
            "uav_transform": i.uav_transform,
        }
        return kwargs

    @staticmethod
    async def task_async(args):
        return await calibrate.fetch_map(**args)

    def postprocess(self, i, p, r) -> CalibrateCTX:
        i.satellite_image, i.satellite_crs = r
        return i


class Stage3Param(TypedDict):
    image_src: ImageMat
    image_dst: ImageMat


class Stage3(SyncStage[CalibrateCTX, CalibrateCTX, Stage3Param, MatchResult]):
    def preprocess(self, i):
        kwargs: Stage3Param = {
            "image_src": i.rect_image,
            "image_dst": i.satellite_image,
        }
        return kwargs

    @staticmethod
    def task(args):
        return match_images(args["image_src"], args["image_dst"])

    def postprocess(self, i, p, r):
        i.kpts0 = r.kpts0
        i.kpts1 = r.kpts1
        i.match_score = r.scores
        return i


class Stage4Param(TypedDict):
    kpts0: NDArray
    kpts1: NDArray
    uav_transform: Transform
    satellite_crs: CRSTransform
    tolerance: float


class Stage4(SyncStage[CalibrateCTX, CalibrateCTX, Stage4Param, CRSTransform]):
    def __init__(self, tolerance, input_maxsize: int = 10) -> None:
        super().__init__(input_maxsize)
        self.tolerance = tolerance

    def preprocess(self, i):
        kwargs: Stage4Param = {
            "kpts0": i.kpts0,
            "kpts1": i.kpts1,
            "uav_transform": i.uav_transform,
            "satellite_crs": i.satellite_crs,
            "tolerance": self.tolerance,
        }
        return kwargs

    @staticmethod
    def task(args):
        # use ransac algorithm to remove outliers and fit homographic transform matrix
        threshold = args["tolerance"] / args["uav_transform"].crs.resolution
        homography_result = match_homography(args["kpts0"], args["kpts1"], threshold)
        # apply transform matrix
        args["uav_transform"].follow(homography_result.mat)
        args["uav_transform"].crs = args["satellite_crs"]
        return args["uav_transform"].combined

    def postprocess(self, i, p, r):
        i.final_transform = r
        return i


class Printer(SyncStage[CalibrateCTX, CalibrateCTX, Any, None]):
    def __init__(self, input_maxsize: int = 10) -> None:
        super().__init__(input_maxsize)
        self.count = 0

    def preprocess(self, i: CalibrateCTX):
        self.count += 1
        return self.count

    @staticmethod
    def task(args):
        print(args)


def create_pipeline(
    satellite_map: Map, maxsize=10, tolerance=5.0
) -> Pipeline[UAVData, CalibrateCTX]:
    stages = [
        Stage1(input_maxsize=maxsize),
        Stage2(input_maxsize=maxsize, map=satellite_map),
        Stage3(input_maxsize=maxsize),
        Stage4(input_maxsize=maxsize, tolerance=tolerance),
    ]
    return Pipeline.from_stages(stages)
