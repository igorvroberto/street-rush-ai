"""Visão computacional para o Street Rush.

Detecta personagem, caminhões, barreiras e moedas via segmentação HSV
e contornos. Apenas objetos efetivamente encontrados no frame atual
são retornados — nunca coordenadas inventadas.

A perspectiva 3D do jogo é aproximada pela posição vertical no frame:
objetos mais próximos ao topo tendem a estar mais longe na pista;
objetos mais baixos, mais próximos do personagem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    BOX_COLORS,
    DEFAULT_VISION,
    OBJECT_BARRIER,
    OBJECT_CHARACTER,
    OBJECT_COIN,
    OBJECT_TRUCK,
    DetectionParams,
    HsvRange,
    VisionConfig,
)

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class DetectedObject:
    """Objeto detectado em um frame."""

    label: str
    bbox: BoundingBox
    confidence: float
    center: Tuple[float, float]

    @property
    def area(self) -> int:
        return int(self.bbox[2] * self.bbox[3])


@dataclass(frozen=True)
class RelativePosition:
    """Posição aproximada de um objeto em relação ao personagem.

    ``dx`` / ``dy``: pixels (objeto − personagem).
    ``depth_proxy``: 0 = mesma altura do personagem (mais perto na
    perspectiva tipica), 1 = topo do frame (mais longe).
    ``side``: ``left`` / ``center`` / ``right`` relativo ao personagem.
    """

    dx: float
    dy: float
    depth_proxy: float
    side: str


@dataclass
class VisionResult:
    """Resultado completo da análise de um frame."""

    character: Optional[DetectedObject]
    trucks: List[DetectedObject]
    barriers: List[DetectedObject]
    coins: List[DetectedObject]
    relatives: dict[int, RelativePosition]
    frame_shape: Tuple[int, int]  # h, w

    @property
    def all_objects(self) -> List[DetectedObject]:
        items: List[DetectedObject] = []
        if self.character is not None:
            items.append(self.character)
        items.extend(self.trucks)
        items.extend(self.barriers)
        items.extend(self.coins)
        return items


class VisionAnalyzer:
    """Analisa frames BGR e retorna detecções estruturadas."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or DEFAULT_VISION

    def analyze(self, frame: np.ndarray) -> VisionResult:
        """Executa o pipeline de detecção no frame BGR."""
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame inválido")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Esperado HxWx3 BGR, recebido {getattr(frame, 'shape', None)}")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        trucks = self._detect_color_blobs(
            hsv, self.config.truck_hsv, self.config.truck, OBJECT_TRUCK
        )
        barriers = self._detect_color_blobs(
            hsv, self.config.barrier_hsv, self.config.barrier, OBJECT_BARRIER
        )
        coins = self._detect_color_blobs(
            hsv, self.config.coin_hsv, self.config.coin, OBJECT_COIN
        )
        characters = self._detect_color_blobs(
            hsv,
            self.config.character_hsv,
            self.config.character,
            OBJECT_CHARACTER,
        )
        # Personagem: maior blob da classe que não coincida com obstáculo já rotulado.
        occupied = [*trucks, *barriers, *coins]
        character = self._pick_character(characters, occupied)

        relatives: dict[int, RelativePosition] = {}
        if character is not None:
            for idx, obj in enumerate([*trucks, *barriers, *coins]):
                relatives[id(obj)] = self._relative_to_character(obj, character, h)

        return VisionResult(
            character=character,
            trucks=trucks,
            barriers=barriers,
            coins=coins,
            relatives=relatives,
            frame_shape=(h, w),
        )

    def draw_detections(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """Desenha bounding boxes e rótulos sobre o frame."""
        out = frame.copy()
        for obj in result.all_objects:
            color = BOX_COLORS.get(obj.label, (255, 255, 255))
            x, y, bw, bh = obj.bbox
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            label = f"{obj.label} {obj.confidence:.2f}"
            rel = result.relatives.get(id(obj))
            if rel is not None:
                label += f" d={rel.depth_proxy:.2f} {rel.side}"
            cv2.putText(
                out,
                label,
                (x, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return out

    def _detect_color_blobs(
        self,
        hsv: np.ndarray,
        hsv_range: HsvRange,
        params: DetectionParams,
        label: str,
    ) -> List[DetectedObject]:
        """Segmenta por HSV, filtra contornos e devolve apenas detecções reais."""
        lower = np.array(hsv_range.lower, dtype=np.uint8)
        upper = np.array(hsv_range.upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Barreiras vermelhas: OpenCV divide o vermelho em duas faixas H.
        if label == OBJECT_BARRIER:
            low_red = cv2.inRange(
                hsv,
                np.array((0, 100, 80), dtype=np.uint8),
                np.array((10, 255, 255), dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask, low_red)

        k = max(1, int(params.morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[DetectedObject] = []
        frame_area = float(hsv.shape[0] * hsv.shape[1])

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < params.min_area or area > params.max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue
            # Confiança heurística: fração da bbox preenchida pelo contorno.
            fill = area / float(bw * bh)
            size_score = min(1.0, area / max(params.min_area * 4.0, 1.0))
            confidence = float(np.clip(0.35 * fill + 0.65 * size_score, 0.0, 1.0))
            # Descarta ruído com confiança muito baixa.
            if confidence < 0.25:
                continue
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            detections.append(
                DetectedObject(
                    label=label,
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    confidence=confidence,
                    center=(cx, cy),
                )
            )

        # Ordena do mais próximo (maior y) ao mais distante — perspectiva da pista.
        detections.sort(key=lambda o: o.center[1], reverse=True)
        logger.debug(
            "%s: %d detecção(ões) (área_frame=%.0f)",
            label,
            len(detections),
            frame_area,
        )
        return detections

    @staticmethod
    def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0

    def _pick_character(
        self,
        candidates: List[DetectedObject],
        occupied: List[DetectedObject],
    ) -> Optional[DetectedObject]:
        """Escolhe o personagem entre candidatos reais, evitando colisão de rótulo."""
        filtered = [
            c
            for c in candidates
            if all(self._bbox_iou(c.bbox, other.bbox) < 0.5 for other in occupied)
        ]
        pool = filtered if filtered else []
        if not pool:
            return None
        # Preferência: mais próximo da base do frame (maior y), típico do runner.
        return max(pool, key=lambda o: (o.center[1], o.area))

    @staticmethod
    def _relative_to_character(
        obj: DetectedObject,
        character: DetectedObject,
        frame_height: int,
    ) -> RelativePosition:
        dx = obj.center[0] - character.center[0]
        dy = obj.center[1] - character.center[1]
        # depth_proxy: 0 perto do personagem na vertical, 1 no topo do frame.
        if frame_height <= 1:
            depth = 0.0
        else:
            depth = float(np.clip((character.center[1] - obj.center[1]) / character.center[1], 0.0, 1.0))

        # Lado relativo com histerese simples em pixels.
        if dx < -25:
            side = "left"
        elif dx > 25:
            side = "right"
        else:
            side = "center"
        return RelativePosition(dx=dx, dy=dy, depth_proxy=depth, side=side)


def summarize_detections(result: VisionResult) -> str:
    """Resumo textual curto para log/console."""
    parts = [
        f"char={'yes' if result.character else 'no'}",
        f"trucks={len(result.trucks)}",
        f"barriers={len(result.barriers)}",
        f"coins={len(result.coins)}",
    ]
    return ", ".join(parts)


__all__ = [
    "DetectedObject",
    "RelativePosition",
    "VisionAnalyzer",
    "VisionResult",
    "summarize_detections",
]
