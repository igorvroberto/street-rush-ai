"""ROI da pista: núcleo trapezoidal + margem soft.

Não altera o pipeline HSV. Classifica detecções existentes em
``core`` / ``soft`` / ``outside`` sem descartá-las.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence, Tuple

import cv2
import numpy as np

from src.config import (
    ROI_CORE_NORMALIZED,
    ROI_MIN_INTERSECTION_CORE,
    ROI_MIN_INTERSECTION_PLAYABLE,
    ROI_MIN_INTERSECTION_WITH_CENTER,
    ROI_SOFT_NORMALIZED,
)
from src.vision import DetectedObject, VisionResult

RegionName = Literal["core", "soft", "outside"]
PointNorm = Tuple[float, float]
BoundingBox = Tuple[int, int, int, int]


def normalized_to_pixels(
    polygon: Sequence[PointNorm],
    width: int,
    height: int,
) -> np.ndarray:
    """Converte polígono normalizado (0–1) em vértices pixel ``int32`` Nx1x2."""
    pts = [
        [int(round(x_n * (width - 1))), int(round(y_n * (height - 1)))]
        for x_n, y_n in polygon
    ]
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def build_roi_masks(
    width: int,
    height: int,
    *,
    core_norm: Sequence[PointNorm] = ROI_CORE_NORMALIZED,
    soft_norm: Sequence[PointNorm] = ROI_SOFT_NORMALIZED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna máscaras uint8 (0/255): ``core``, ``soft_only``, ``playable`` (core∪soft)."""
    core = np.zeros((height, width), dtype=np.uint8)
    soft_outer = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(core, [normalized_to_pixels(core_norm, width, height)], 255)
    cv2.fillPoly(soft_outer, [normalized_to_pixels(soft_norm, width, height)], 255)
    soft_only = cv2.bitwise_and(soft_outer, cv2.bitwise_not(core))
    playable = cv2.bitwise_or(core, soft_only)
    return core, soft_only, playable


def point_region(
    x: float,
    y: float,
    core_mask: np.ndarray,
    soft_only_mask: np.ndarray,
) -> RegionName:
    """Classifica um ponto (ex.: centro do bbox) em core / soft / outside."""
    h, w = core_mask.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    if core_mask[yi, xi] > 0:
        return "core"
    if soft_only_mask[yi, xi] > 0:
        return "soft"
    return "outside"


def bbox_intersection_ratio(bbox: BoundingBox, mask: np.ndarray) -> float:
    """Área(bbox ∩ mask) / área(bbox). Retorna 0 se bbox inválido."""
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return 0.0
    h, w = mask.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = mask[y0:y1, x0:x1]
    inter = float(cv2.countNonZero(roi))
    return inter / float(bw * bh)


def assign_track_region(
    *,
    center_region: RegionName,
    intersection_core: float,
    intersection_playable: float,
) -> RegionName:
    """Decide associação à pista usando interseção (não só o centro).

    - ``core`` se interseção com núcleo ≥ limiar OU (centro no núcleo e
      interseção jogável ≥ limiar fraco).
    - ``soft`` se não for core, mas interseção jogável ≥ limiar OU
      (centro na soft e interseção jogável ≥ limiar fraco).
    - caso contrário ``outside`` (detecção preservada, só rotulada).
    """
    if intersection_core >= ROI_MIN_INTERSECTION_CORE or (
        center_region == "core" and intersection_playable >= ROI_MIN_INTERSECTION_WITH_CENTER
    ):
        return "core"
    if intersection_playable >= ROI_MIN_INTERSECTION_PLAYABLE or (
        center_region == "soft" and intersection_playable >= ROI_MIN_INTERSECTION_WITH_CENTER
    ):
        return "soft"
    return "outside"


@dataclass(frozen=True)
class RoiTaggedDetection:
    """Detecção HSV original + métricas de ROI (nada é descartado)."""

    detection: DetectedObject
    center: Tuple[float, float]
    center_region: RegionName
    intersection_core: float
    intersection_playable: float
    track_region: RegionName


def tag_detections(
    detections: Iterable[DetectedObject],
    core_mask: np.ndarray,
    soft_only_mask: np.ndarray,
    playable_mask: np.ndarray,
) -> List[RoiTaggedDetection]:
    """Anota cada detecção com centro, região e percentuais de interseção."""
    tagged: List[RoiTaggedDetection] = []
    for det in detections:
        cx, cy = det.center
        center_region = point_region(cx, cy, core_mask, soft_only_mask)
        inter_core = bbox_intersection_ratio(det.bbox, core_mask)
        inter_play = bbox_intersection_ratio(det.bbox, playable_mask)
        track_region = assign_track_region(
            center_region=center_region,
            intersection_core=inter_core,
            intersection_playable=inter_play,
        )
        tagged.append(
            RoiTaggedDetection(
                detection=det,
                center=(cx, cy),
                center_region=center_region,
                intersection_core=inter_core,
                intersection_playable=inter_play,
                track_region=track_region,
            )
        )
    return tagged


def tag_vision_result(
    result: VisionResult,
    width: int,
    height: int,
) -> tuple[List[RoiTaggedDetection], np.ndarray, np.ndarray, np.ndarray]:
    """Aplica ROI a todas as detecções do ``VisionResult`` atual."""
    core, soft_only, playable = build_roi_masks(width, height)
    tagged = tag_detections(result.all_objects, core, soft_only, playable)
    return tagged, core, soft_only, playable


def count_by_class(
    detections: Iterable[DetectedObject] | Iterable[RoiTaggedDetection],
) -> dict[str, int]:
    """Contagem por label (DetectedObject ou RoiTaggedDetection)."""
    counts = {"character": 0, "truck": 0, "barrier": 0, "coin": 0}
    for item in detections:
        det = item.detection if isinstance(item, RoiTaggedDetection) else item
        if det.label in counts:
            counts[det.label] += 1
    return counts


def count_by_region(tagged: Sequence[RoiTaggedDetection]) -> dict[str, dict[str, int]]:
    """Contagens por região de pista e por classe."""
    out: dict[str, dict[str, int]] = {
        "core": {"character": 0, "truck": 0, "barrier": 0, "coin": 0},
        "soft": {"character": 0, "truck": 0, "barrier": 0, "coin": 0},
        "outside": {"character": 0, "truck": 0, "barrier": 0, "coin": 0},
    }
    for item in tagged:
        label = item.detection.label
        if label in out[item.track_region]:
            out[item.track_region][label] += 1
    return out


__all__ = [
    "RegionName",
    "RoiTaggedDetection",
    "assign_track_region",
    "bbox_intersection_ratio",
    "build_roi_masks",
    "count_by_class",
    "count_by_region",
    "normalized_to_pixels",
    "point_region",
    "tag_detections",
    "tag_vision_result",
]
