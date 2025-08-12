from typing import overload
from dataclasses import dataclass

from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import numpy_image_to_torch
import torch
import numpy as np

from .matching import *

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint().eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)


@dataclass
class SuperPointInstance:
    keypoints: torch.Tensor  # (N, 2)
    keypoint_scores: torch.Tensor  # (N,)
    descriptors: torch.Tensor  # (N, D)
    image_size: torch.Tensor  # (2,)


@dataclass
class SuperPointOutput:
    keypoints: torch.Tensor  # (B, N, 2)
    keypoint_scores: torch.Tensor  # (B, N)
    descriptors: torch.Tensor  # (B, N, D)
    image_size: torch.Tensor  # (B, 2)

    @overload
    def __getitem__(self, idx: int) -> SuperPointInstance: ...
    @overload
    def __getitem__(self, idx: slice) -> "SuperPointOutput": ...
    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return SuperPointInstance(
                keypoints=self.keypoints[idx],
                keypoint_scores=self.keypoint_scores[idx],
                descriptors=self.descriptors[idx],
                image_size=self.image_size[idx],
            )
        elif isinstance(idx, slice):
            return SuperPointOutput(
                keypoints=self.keypoints[idx],
                keypoint_scores=self.keypoint_scores[idx],
                descriptors=self.descriptors[idx],
                image_size=self.image_size[idx],
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def instances(self):
        for i in range(self.keypoints.shape[0]):
            yield self[i]


@dataclass
class LightGlueInstance:
    matches0: torch.Tensor  # (M,)
    matches1: torch.Tensor  # (N,)
    matching_scores0: torch.Tensor  # (M,)
    matching_scores1: torch.Tensor  # (N,)
    prune0: torch.Tensor  # (M,)
    prune1: torch.Tensor  # (N,)
    stop: int
    matches: torch.Tensor  # (Si,)
    scores: torch.Tensor  # (Si,)


@dataclass
class LightGlueOutput:
    matches0: torch.Tensor  # (B, M)
    matches1: torch.Tensor  # (B, N)
    matching_scores0: torch.Tensor  # (B, M)
    matching_scores1: torch.Tensor  # (B, N)
    prune0: torch.Tensor  # (B, M)
    prune1: torch.Tensor  # (B, N)
    stop: int
    matches: list[torch.Tensor]  # [(Si,)] * B
    scores: list[torch.Tensor]  # [(Si,)] * B

    @overload
    def __getitem__(self, idx: int) -> LightGlueInstance: ...
    @overload
    def __getitem__(self, idx: slice) -> "LightGlueOutput": ...
    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return LightGlueInstance(
                matches0=self.matches0[idx],
                matches1=self.matches1[idx],
                matching_scores0=self.matching_scores0[idx],
                matching_scores1=self.matching_scores1[idx],
                prune0=self.prune0[idx],
                prune1=self.prune1[idx],
                stop=self.stop,
                matches=self.matches[idx],
                scores=self.scores[idx],
            )
        elif isinstance(idx, slice):
            return LightGlueOutput(
                matches0=self.matches0[idx],
                matches1=self.matches1[idx],
                matching_scores0=self.matching_scores0[idx],
                matching_scores1=self.matching_scores1[idx],
                prune0=self.prune0[idx],
                prune1=self.prune1[idx],
                stop=self.stop,
                matches=self.matches[idx],
                scores=self.scores[idx],
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def instances(self):
        for i in range(len(self.matches)):
            yield self[i]


@dataclass
class LightGlueMatchResult(MatchResult):
    feats0: SuperPointInstance
    feats1: SuperPointInstance
    matches: LightGlueInstance


def match_images(
    image0: np.ndarray,
    image1: np.ndarray,
    max_kpts0=2048,
    max_kpts1=2048,
) -> LightGlueMatchResult:
    extractor.conf.max_num_keypoints = max_kpts0
    feats0 = extractor.extract(numpy_image_to_torch(image0).to(device))
    extractor.conf.max_num_keypoints = max_kpts1
    feats1 = extractor.extract(numpy_image_to_torch(image1).to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0 = SuperPointOutput(**feats0)[0]
    feats1 = SuperPointOutput(**feats1)[0]
    matches01 = LightGlueOutput(**matches01)[0]

    kpts0, kpts1, matches = feats0.keypoints, feats1.keypoints, matches01.matches
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    return LightGlueMatchResult(
        methed=MatchingMethod.LIGHTGLUE,
        kpts0=m_kpts0.cpu().numpy(),
        kpts1=m_kpts1.cpu().numpy(),
        scores=matches01.scores.cpu().numpy(),
        feats0=feats0,
        feats1=feats1,
        matches=matches01,
    )


def plot_matches(
    match_result: MatchResult,
    image0: np.ndarray,
    image1: np.ndarray,
):
    assert isinstance(match_result, LightGlueMatchResult)

    viz2d.plot_images([image0, image1])
    color = viz2d.cm_RdGn(match_result.matches.scores.cpu().numpy())
    color = color.tolist()  # viz2d bug that not recognize ndarray
    viz2d.plot_matches(
        match_result.kpts0, match_result.kpts1, color=color, lw=0.2, ps=2
    )
    viz2d.add_text(0, f"Stop after {match_result.matches.stop} layers", fs=20)

    color0 = viz2d.cm_prune(match_result.matches.prune0.cpu().numpy())
    color1 = viz2d.cm_prune(match_result.matches.prune1.cpu().numpy())
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints(
        [match_result.feats0.keypoints, match_result.feats1.keypoints],
        colors=[color0, color1],
        ps=4,
    )
