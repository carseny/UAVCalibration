from typing import TypedDict
from dataclasses import asdict

from .pipeline import *
from .transform import *
from .datasets import *
from .map import *
from .types import *
from .match import *
from . import calibrate
from .calibrate import CalibrateCTX


class Stage1Param(TypedDict):
    uav_image: ImageMat
    longitude: float
    latitude: float
    focal_length: float
    yaw: float
    pitch: float
    roll: float
    height: float


class Stage1(SyncStage[CalibrateCTX, Stage1Param, tuple[ImageMat, Transform]]):
    def preprocess(self, ctx):
        kwargs: Stage1Param = asdict(ctx.uav_data)
        return kwargs

    @staticmethod
    def task(args):
        return calibrate.coarse_calibrate(**args)

    def postprocess(self, ctx, p, r):
        ctx.rect_image, ctx.uav_transform = r
        return ctx


class Stage2Param(TypedDict):
    map: Map
    uav_shape: Shape
    uav_transform: Transform


class Stage2(AsyncStage[CalibrateCTX, Stage2Param, tuple[ImageMat, CRSTransform]]):
    def __init__(self, map: Map, input_maxize: int = 10) -> None:
        super().__init__(input_maxize)
        self.map = map

    def preprocess(self, ctx):
        kwargs: Stage2Param = {
            "map": self.map,
            "uav_shape": ctx.uav_shape,
            "uav_transform": ctx.uav_transform,
        }
        return kwargs

    @staticmethod
    async def task_async(args):
        return await calibrate.fetch_map(**args)

    def postprocess(self, ctx, p, r) -> CalibrateCTX:
        ctx.satellite_image, ctx.satellite_crs = r
        return ctx


class Stage3Param(TypedDict):
    image_src: ImageMat
    image_dst: ImageMat


class Stage3(SyncStage[CalibrateCTX, Stage3Param, MatchResult]):
    def preprocess(self, ctx):
        kwargs: Stage3Param = {
            "image_src": ctx.rect_image,
            "image_dst": ctx.satellite_image,
        }
        return kwargs

    @staticmethod
    def task(args):
        return match_images(args["image_src"], args["image_dst"])

    def postprocess(self, ctx, p, r):
        ctx.kpts0 = r.kpts0
        ctx.kpts1 = r.kpts1
        ctx.match_score = r.scores
        return ctx


class Stage4Param(TypedDict):
    kpts0: NDArray
    kpts1: NDArray
    uav_transform: Transform
    satellite_crs: CRSTransform
    tolerance: float


class Stage4(SyncStage[CalibrateCTX, Stage4Param, CRSTransform]):
    def __init__(self, tolerance, input_maxize: int = 10) -> None:
        super().__init__(input_maxize)
        self.tolerance = tolerance

    def preprocess(self, ctx):
        kwargs: Stage4Param = {
            "kpts0": ctx.kpts0,
            "kpts1": ctx.kpts1,
            "uav_transform": ctx.uav_transform,
            "satellite_crs": ctx.satellite_crs,
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

    def postprocess(self, ctx, p, r):
        ctx.final_transform = r
        return ctx


class Printer(SyncStage[CalibrateCTX, Any, None]):
    def __init__(self, input_maxize: int = 10) -> None:
        super().__init__(input_maxize)
        self.count = 0

    def preprocess(self, ctx: CalibrateCTX):
        self.count += 1
        return self.count

    @staticmethod
    def task(args):
        print(args)
