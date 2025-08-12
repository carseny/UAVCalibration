from dataclasses import dataclass

from lightglue import viz2d
import numpy as np
import cv2

from .matching import *

__all__ = [
    "match_homography",
]


@dataclass
class HomographyOutput(MatchResult):
    kpts_src0: np.ndarray
    kpts_src1: np.ndarray
    mat: np.ndarray
    mask: np.ndarray


def match_homography(
    kpts0: np.ndarray, kpts1: np.ndarray, ransacReprojThreshold=10.0
) -> HomographyOutput:
    # 使用单应性矩阵的RANSAC
    mat, mask = cv2.findHomography(
        kpts0,
        kpts1,
        cv2.RANSAC,
        ransacReprojThreshold=ransacReprojThreshold,
    )
    # mat: 单应矩阵
    # mask: 内点掩码（1表示内点，0表示外点）
    mask = mask.ravel() == 1

    m_kpts0 = kpts0[mask]
    m_kpts1 = kpts1[mask]
    kpts_target = cv2.perspectiveTransform(m_kpts0[:, None, :], mat)[:, 0, :]
    dist = np.linalg.norm(kpts_target - m_kpts1, axis=1)

    return HomographyOutput(
        methed=MatchingMethod.HOMOGRAPHY,
        kpts0=m_kpts0,
        kpts1=m_kpts1,
        scores=np.exp(-dist / ransacReprojThreshold),
        kpts_src0=kpts0,
        kpts_src1=kpts1,
        mat=mat,
        mask=mask,
    )


def plot_matches(
    match_result: MatchResult,
    image0: np.ndarray,
    image1: np.ndarray,
):
    viz2d.plot_images([image0, image1])
    color = viz2d.cm_RdGn(match_result.scores)
    color = color.tolist()  # viz2d bug that not recognize ndarray
    viz2d.plot_matches(
        match_result.kpts0, match_result.kpts1, color=color, lw=0.2, ps=2
    )

    match match_result.methed:
        case MatchingMethod.HOMOGRAPHY:
            plot_homography(match_result, image0, image1)


def plot_homography(
    match_result: MatchResult,
    image0: np.ndarray,
    image1: np.ndarray,
):
    assert isinstance(match_result, HomographyOutput)
    color = viz2d.cm_RdGn(match_result.mask)
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints(
        [match_result.kpts_src0, match_result.kpts_src1],
        colors=color,
        ps=4,
    )
