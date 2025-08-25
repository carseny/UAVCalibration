from dataclasses import dataclass, field, asdict

import numpy as np

from .datasets import UAVData
from .types import *
from .transform import *
from . import rectify as rect
from .map import *
from .match import *


def coarse_calibrate(
    uav_image: ImageMat,
    # uav crs args
    longitude: float,  # in degrees
    latitude: float,  # in degrees
    # camera args
    focal_length: float | None = None,  # in pixels
    camera_mat: np.ndarray | None = None,  # xyz -> x'y'1
    # angle args
    yaw: float | None = None,  # in radians
    pitch: float | None = None,  # in radians
    roll: float | None = None,  # in radians
    rotate_mat: np.ndarray | None = None,  # xyz -> x'y'z'
    # resolution args
    height: float = 100,  # in meters
    **_,
) -> tuple[ImageMat, Transform]:
    """CPU-intensive task"""
    uav_shape = uav_image.shape[1], uav_image.shape[0]
    if camera_mat is None:
        camera_mat = rect.camera_mat(uav_shape, focal_length)
    if rotate_mat is None:
        rotate_mat = rect.rotate_mat(yaw=yaw, pitch=pitch, roll=roll)
    rect_mat = rect.rectify_mat(camera_mat=camera_mat, rotate_mat=rotate_mat)
    crs_trans = rect.crs_trans(
        longitude=longitude,
        latitude=latitude,
        camera_mat=camera_mat,
        rotate_mat=rotate_mat,
        height=height,
    )
    transform = Transform(
        pix_mat=rect_mat,
        src_shape=uav_shape,
        crs=crs_trans,
    )
    # Adjust the transform and dst_shape to make sure all corners are inside
    transform.adjust_shape()
    rect_image = transform.warp(uav_image)
    return rect_image, transform


async def fetch_map(
    map: Map,
    uav_shape: Shape,
    uav_transform: Transform,
) -> tuple[ImageMat, CRSTransform]:
    """IO-intensive task"""
    w, h = uav_shape
    tmp_transform = uav_transform.combined
    return await map.get_async(
        tmp_transform.bounds(h=h, w=w),
        tmp_transform.crs,
        resolution=tmp_transform.resolution,
    )


@dataclass
class CalibrateCTX:
    uav_data: UAVData

    uav_image: ImageMat = field(init=False)
    uav_shape: Shape = field(init=False)

    rect_image: ImageMat = field(init=False)
    uav_transform: Transform = field(init=False)
    satellite_image: ImageMat = field(init=False)
    satellite_crs: CRSTransform = field(init=False)
    kpts0: NDArray = field(init=False)
    kpts1: NDArray = field(init=False)
    match_score: NDArray = field(init=False)
    final_transform: CRSTransform = field(init=False)

    def __post_init__(self):
        self.uav_image = self.uav_data.uav_image
        self.uav_shape = self.uav_image.shape[1], self.uav_image.shape[0]


class Calibrator:
    def __init__(
        self,
        map: Map,
        # ransac
        tolerance=5.0,  # meters
        # debug
        plot=False,
        # match args
        *args,
        **kwargs,
    ):
        self.map = map
        self.tolerance = tolerance
        self.plot = plot
        self.args = args
        self.kwargs = kwargs

    async def calibrate(
        self,
        uav_data: UAVData,
    ):
        ctx = CalibrateCTX(uav_data)

        ctx.rect_image, ctx.uav_transform = coarse_calibrate(**asdict(uav_data))

        ctx.satellite_image, ctx.satellite_crs = await fetch_map(
            self.map, ctx.uav_shape, ctx.uav_transform
        )

        # image matching
        match_result = match_images(
            ctx.rect_image, ctx.satellite_image, *self.args, **self.kwargs
        )
        if self.plot:
            plot_matches(match_result, ctx.rect_image, ctx.satellite_image)
        ctx.kpts0, ctx.kpts1, ctx.match_score = (
            match_result.kpts0,
            match_result.kpts1,
            match_result.scores,
        )

        # use ransac algorithm to remove outliers and fit homographic transform matrix
        threshold = self.tolerance / ctx.uav_transform.crs.resolution
        homography_result = match_homography(ctx.kpts0, ctx.kpts1, threshold)
        if self.plot:
            plot_matches(homography_result, ctx.rect_image, ctx.satellite_image)

        # apply transform matrix
        ctx.uav_transform.follow(homography_result.mat)
        ctx.uav_transform.crs = ctx.satellite_crs
        return ctx.uav_transform.combined
