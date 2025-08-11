from lightglue import viz2d
import numpy as np

from .matching import *
from .ransac import *


__all__ = [
    "MatchingMethod",
    "MatchResult",
    "match_homography",
    "match_images",
    "plot_matches",
]


def match_images(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    mask_src: np.ndarray | None = None,
    mask_dst: np.ndarray | None = None,
    method: MatchingMethod = MatchingMethod.LIGHTGLUE,
    **kwargs,
) -> MatchResult:
    if mask_src is None:
        mask_src = np.ones_like(image_src[..., 0])
    if mask_dst is None:
        mask_dst = np.ones_like(image_src[..., 0])
    match method:
        case MatchingMethod.SIFT:
            from . import sift

            return sift.match_images(image_src, image_dst, mask_src, mask_dst, **kwargs)

        case MatchingMethod.LIGHTGLUE:
            from . import lightglue

            return lightglue.match_images(image_src, image_dst, **kwargs)

        case _:
            raise ValueError(f"Unsupported matching method: {method}")


def plot_matches(
    match_result: MatchResult,
    image0: np.ndarray,
    image1: np.ndarray,
):
    match match_result.methed:
        # case MatchingMethod.SIFT:
        #     from . import sift

        case MatchingMethod.LIGHTGLUE:
            from . import lightglue

            lightglue.plot_matches(match_result, image0, image1)

        case MatchingMethod.HOMOGRAPHY:
            from . import ransac

            ransac.plot_matches(match_result, image0, image1)

        case _:
            viz2d.plot_images([image0, image1])
            color = viz2d.cm_RdGn(match_result.scores)
            viz2d.plot_matches(
                match_result.kpts0, match_result.kpts1, color=color, lw=0.2, ps=2
            )
