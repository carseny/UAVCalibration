from typing import Sequence
from dataclasses import dataclass

import numpy as np
import cv2

from .matching import *


@dataclass
class SiftMatchResult(MatchResult):
    matches: Sequence[cv2.DMatch]


def match_images(
    image0: np.ndarray,
    image1: np.ndarray,
    mask_src: np.ndarray,
    mask_dst: np.ndarray,
    threshold: float = 10,  # 距离阈值
) -> SiftMatchResult:
    # 初始化SIFT检测器
    sift = cv2.SIFT.create()
    # 检测关键点并计算描述符
    kpts0, desc0 = sift.detectAndCompute(image0, mask_src)
    kpts1, desc1 = sift.detectAndCompute(image1, mask_dst)
    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # 匹配描述符
    matches = bf.match(desc0, desc1)
    # 筛选最佳匹配
    good_matches = [m for m in matches if m.distance < threshold]

    # 构造MatchResult对象
    m_kpts0 = np.asarray([kpts0[m.queryIdx].pt for m in good_matches])
    m_kpts1 = np.asarray([kpts1[m.trainIdx].pt for m in good_matches])
    return SiftMatchResult(
        methed=MatchingMethod.SIFT,
        kpts0=m_kpts0,
        kpts1=m_kpts1,
        scores=np.array([1 - 2**m.distance for m in good_matches]),
        matches=matches,
    )
