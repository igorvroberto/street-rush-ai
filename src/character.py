"""Detector de personagem independente (ROI-first).

Não altera a geometria da ROI nem os detectores HSV de truck/barrier/coin.
O blob HSV ``character`` legado em ``VisionAnalyzer`` permanece disponível
apenas para comparação — este módulo não o usa como verdade de referência.

Sinais visuais do avatar (frames reais + character_audit):
- aura/escudo azul-ciano frequente;
- capacete/camiseta azul-teal;
- capa/cachecol vermelho em alguns frames;
- silhueta compacta, mais alta que larga;
- pode correr, pular (entra na soft) ou mudar de faixa (não assume x=0.5 exato).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    CHARACTER_ASPECT_MAX,
    CHARACTER_ASPECT_MIN,
    CHARACTER_BLUE_HSV,
    CHARACTER_CONF_MIN,
    CHARACTER_CYAN_HSV,
    CHARACTER_MAX_WIDTH_FRAC,
    CHARACTER_MIN_AREA,
    CHARACTER_ORANGE_HSV,
    CHARACTER_RED_HSV_HIGH,
    CHARACTER_RED_HSV_LOW,
    CHARACTER_TARGET_HEIGHT_FRAC,
    CHARACTER_TARGET_WIDTH_FRAC,
    OBJECT_CHARACTER,
)
from src.roi import RegionName, build_roi_masks, point_region
from src.vision import DetectedObject

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]
PointNorm = Tuple[float, float]


@dataclass(frozen=True)
class CharacterDetection:
    """Detecção do personagem na ROI jogável."""

    bbox: BoundingBox
    center: Tuple[float, float]
    confidence: float
    region: RegionName  # core | soft (outside não é retornado como detecção válida)
    source_cue: str  # blue | orange | red
    score: float
    center_norm: PointNorm

    def as_detected_object(self) -> DetectedObject:
        return DetectedObject(
            label=OBJECT_CHARACTER,
            bbox=self.bbox,
            confidence=self.confidence,
            center=self.center,
        )


@dataclass
class CharacterDetectorConfig:
    """Parâmetros do detector ROI (independente do VisionConfig HSV legado)."""

    min_area: int = CHARACTER_MIN_AREA
    max_width_frac: float = CHARACTER_MAX_WIDTH_FRAC
    aspect_min: float = CHARACTER_ASPECT_MIN
    aspect_max: float = CHARACTER_ASPECT_MAX
    target_width_frac: float = CHARACTER_TARGET_WIDTH_FRAC
    target_height_frac: float = CHARACTER_TARGET_HEIGHT_FRAC
    conf_min: float = CHARACTER_CONF_MIN
    # Priores espaciais (normalizados): corrida e salto.
    run_y: float = 0.58
    run_y_sigma: float = 0.13
    jump_y: float = 0.38
    jump_y_sigma: float = 0.09
    lane_x: float = 0.50
    lane_sigma: float = 0.14  # permite mudança de faixa
    temporal_sigma: float = 0.12


class CharacterDetector:
    """Localiza o avatar apenas dentro da ROI (core ∪ soft)."""

    def __init__(self, config: Optional[CharacterDetectorConfig] = None) -> None:
        self.config = config or CharacterDetectorConfig()
        self._prev_center_norm: Optional[PointNorm] = None

    def reset_temporal(self) -> None:
        """Limpa continuidade entre frames (útil em batch offline)."""
        self._prev_center_norm = None

    def detect(
        self,
        frame: np.ndarray,
        *,
        use_temporal: bool = True,
    ) -> Optional[CharacterDetection]:
        """Detecta o personagem no frame BGR.

        Retorna ``None`` se não houver candidato confiável na ROI.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame inválido")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Esperado HxWx3 BGR, recebido {getattr(frame, 'shape', None)}")

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        core, soft_only, playable = build_roi_masks(w, h)

        blue = _in_range(hsv, CHARACTER_BLUE_HSV)
        cyan = _in_range(hsv, CHARACTER_CYAN_HSV)
        orange = _in_range(hsv, CHARACTER_ORANGE_HSV)
        red = cv2.bitwise_or(
            _in_range(hsv, CHARACTER_RED_HSV_LOW),
            _in_range(hsv, CHARACTER_RED_HSV_HIGH),
        )
        blue_m = cv2.bitwise_or(blue, cyan)

        prev = self._prev_center_norm if use_temporal else None
        candidates: List[dict] = []
        for source, mask in (("blue", blue_m), ("orange", orange), ("red", red)):
            candidates.extend(
                self._candidates_from_mask(
                    mask=mask,
                    playable=playable,
                    core=core,
                    soft_only=soft_only,
                    blue_m=blue_m,
                    orange=orange,
                    red=red,
                    source=source,
                    frame_shape=(h, w),
                    prev=prev,
                )
            )

        if not candidates:
            logger.debug("character: nenhum candidato na ROI")
            if not use_temporal:
                pass
            else:
                # Não zera o temporal imediatamente — miss isolado.
                pass
            return None

        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        if best["score"] < 2.5:
            logger.debug("character: melhor score baixo (%.2f)", best["score"])
            return None

        margin = best["score"] - (candidates[1]["score"] if len(candidates) > 1 else 0.0)
        conf = float(
            np.clip(0.18 + 0.12 * best["score"] + 0.08 * max(margin, 0.0), 0.0, 1.0)
        )
        if conf < self.config.conf_min:
            return None

        det = CharacterDetection(
            bbox=best["bbox"],
            center=best["center"],
            confidence=conf,
            region=best["region"],
            source_cue=best["source"],
            score=float(best["score"]),
            center_norm=best["center_norm"],
        )
        if use_temporal:
            self._prev_center_norm = det.center_norm
        logger.debug(
            "character: cue=%s region=%s conf=%.2f bbox=%s",
            det.source_cue,
            det.region,
            det.confidence,
            det.bbox,
        )
        return det

    def _candidates_from_mask(
        self,
        *,
        mask: np.ndarray,
        playable: np.ndarray,
        core: np.ndarray,
        soft_only: np.ndarray,
        blue_m: np.ndarray,
        orange: np.ndarray,
        red: np.ndarray,
        source: str,
        frame_shape: Tuple[int, int],
        prev: Optional[PointNorm],
    ) -> List[dict]:
        h, w = frame_shape
        cfg = self.config
        fa = float(h * w)
        min_a = cfg.min_area if source != "red" else max(40, cfg.min_area // 2)

        m = cv2.bitwise_and(mask, playable)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k5)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k9)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tw = max(30, int(cfg.target_width_frac * w))
        th = max(48, int(cfg.target_height_frac * h))
        out: List[dict] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_a or area > 0.08 * fa:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            windows = self._windows_for_blob(m, x, y, bw, bh, tw, th, source)
            for x2, y2, bw2, bh2, dens in windows:
                aspect = bh2 / float(max(bw2, 1))
                if not (cfg.aspect_min <= aspect <= cfg.aspect_max):
                    continue
                if bw2 > cfg.max_width_frac * w:
                    continue
                cx = x2 + bw2 / 2.0
                cy = y2 + bh2 / 2.0
                region = point_region(cx, cy, core, soft_only)
                if region == "outside":
                    continue
                nx, ny = cx / w, cy / h
                xa, ya = max(0, x2), max(0, y2)
                xb, yb = min(w, x2 + bw2), min(h, y2 + bh2)
                bf = float((blue_m[ya:yb, xa:xb] > 0).mean()) if xb > xa and yb > ya else 0.0
                of = float((orange[ya:yb, xa:xb] > 0).mean()) if xb > xa and yb > ya else 0.0
                rf = float((red[ya:yb, xa:xb] > 0).mean()) if xb > xa and yb > ya else 0.0
                score = self._score_candidate(
                    nx=nx,
                    ny=ny,
                    aspect=aspect,
                    region=region,
                    bf=bf,
                    of=of,
                    rf=rf,
                    dens=dens,
                    size_frac=(bw2 * bh2) / fa,
                    source=source,
                    prev=prev,
                )
                out.append(
                    {
                        "score": score,
                        "bbox": (int(x2), int(y2), int(bw2), int(bh2)),
                        "center": (cx, cy),
                        "region": region,
                        "source": source,
                        "center_norm": (nx, ny),
                    }
                )
        return out

    def _windows_for_blob(
        self,
        mask: np.ndarray,
        x: int,
        y: int,
        bw: int,
        bh: int,
        tw: int,
        th: int,
        source: str,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Gera janelas do tamanho do avatar; blobs grandes (aura) são varridos."""
        windows: List[Tuple[int, int, int, int, float]] = []
        if bw * bh > 2.0 * tw * th or bw > int(0.11 * mask.shape[1]):
            step_x = max(4, tw // 4)
            step_y = max(4, th // 4)
            ys = list(range(0, max(1, bh - th + 1), step_y))
            # Aura alta: ignora o topo extremo (céu/rochas).
            if source == "blue" and len(ys) > 3:
                ys = ys[len(ys) // 5 :]
            for yy in ys:
                for xx in range(0, max(1, bw - tw + 1), step_x):
                    dens = float(mask[y + yy : y + yy + th, x + xx : x + xx + tw].mean()) / 255.0
                    if dens < 0.07:
                        continue
                    windows.append((x + xx, y + yy, tw, th, dens))
        else:
            dens = float(mask[y : y + bh, x : x + bw].mean()) / 255.0
            windows.append((x, y, bw, bh, dens))
        return windows

    def _score_candidate(
        self,
        *,
        nx: float,
        ny: float,
        aspect: float,
        region: RegionName,
        bf: float,
        of: float,
        rf: float,
        dens: float,
        size_frac: float,
        source: str,
        prev: Optional[PointNorm],
    ) -> float:
        cfg = self.config
        score = 0.0
        score += 2.1 * bf + 1.3 * of + 1.5 * min(rf, 0.35)
        score += 1.15 * float(np.exp(-((nx - cfg.lane_x) ** 2) / (2 * cfg.lane_sigma**2)))
        score += 0.85 * float(np.exp(-((aspect - 1.55) ** 2) / (2 * 0.8**2)))
        run_p = float(np.exp(-((ny - cfg.run_y) ** 2) / (2 * cfg.run_y_sigma**2)))
        jump_p = float(np.exp(-((ny - cfg.jump_y) ** 2) / (2 * cfg.jump_y_sigma**2)))
        score += 1.0 * max(run_p, jump_p)
        score += 0.30 if region == "core" else 0.28
        if 0.0004 <= size_frac <= 0.022:
            score += 0.55
        if size_frac > 0.03:
            score -= 1.5
        score += 0.3 * min(float(dens), 1.0)
        if prev is not None:
            d = float(np.hypot(nx - prev[0], ny - prev[1]))
            score += 0.9 * float(np.exp(-(d**2) / (2 * cfg.temporal_sigma**2)))
        if source == "blue" and bf > 0.12:
            score += 0.2
        if source == "red" and rf > 0.08:
            score += 0.35
        return float(score)


def _in_range(hsv: np.ndarray, bounds: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> np.ndarray:
    lower, upper = bounds
    return cv2.inRange(
        hsv,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8),
    )


def center_distance_norm(
    a: PointNorm,
    b: PointNorm,
) -> float:
    """Distância euclidiana entre centros normalizados."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def detection_hits_ground_truth(
    detection: Optional[CharacterDetection],
    gt_center_norm: PointNorm,
    *,
    max_distance: float = 0.06,
) -> bool:
    """Acerto se o centro detectado está perto do centro REAL do corpo."""
    if detection is None:
        return False
    return center_distance_norm(detection.center_norm, gt_center_norm) <= max_distance


__all__ = [
    "CharacterDetection",
    "CharacterDetector",
    "CharacterDetectorConfig",
    "center_distance_norm",
    "detection_hits_ground_truth",
]
