"""Representação estruturada do estado do Street Rush.

Esta versão não lê memória do cliente nem conversa com servidores.
Distância percorrida é estimada por fluxo óptico vertical quando
disponível; caso contrário permanece ``None`` até haver evidência.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.vision import DetectedObject, RelativePosition, VisionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObstacleRelative:
    """Obstáculo (caminhão ou barreira) com posição relativa ao personagem."""

    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    relative: RelativePosition


@dataclass(frozen=True)
class CoinOnPath:
    """Moeda detectada no percurso."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    relative: Optional[RelativePosition]


@dataclass
class GameState:
    """Estado do jogo derivado apenas de visão computacional."""

    timestamp: float
    frame_size: Tuple[int, int]  # width, height
    distance_traveled: Optional[float]
    character_position: Optional[Tuple[float, float]]
    character_bbox: Optional[Tuple[int, int, int, int]]
    detected_objects: List[DetectedObject] = field(default_factory=list)
    obstacles_relative: List[ObstacleRelative] = field(default_factory=list)
    coins_on_path: List[CoinOnPath] = field(default_factory=list)
    fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dict JSON-friendly (sem inventar campos)."""
        return {
            "timestamp": self.timestamp,
            "frame_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "distance_traveled": self.distance_traveled,
            "character_position": (
                {"x": self.character_position[0], "y": self.character_position[1]}
                if self.character_position is not None
                else None
            ),
            "character_bbox": self.character_bbox,
            "detected_objects": [
                {
                    "label": o.label,
                    "bbox": {"x": o.bbox[0], "y": o.bbox[1], "w": o.bbox[2], "h": o.bbox[3]},
                    "confidence": o.confidence,
                    "center": {"x": o.center[0], "y": o.center[1]},
                }
                for o in self.detected_objects
            ],
            "obstacles_relative": [
                {
                    "label": o.label,
                    "bbox": {"x": o.bbox[0], "y": o.bbox[1], "w": o.bbox[2], "h": o.bbox[3]},
                    "confidence": o.confidence,
                    "relative": asdict(o.relative),
                }
                for o in self.obstacles_relative
            ],
            "coins_on_path": [
                {
                    "bbox": {"x": c.bbox[0], "y": c.bbox[1], "w": c.bbox[2], "h": c.bbox[3]},
                    "confidence": c.confidence,
                    "relative": asdict(c.relative) if c.relative is not None else None,
                }
                for c in self.coins_on_path
            ],
            "fps": self.fps,
        }


class GameStateTracker:
    """Mantém estado entre frames e estima distância via fluxo óptico."""

    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None
        self._distance: float = 0.0
        self._has_distance_evidence: bool = False

    def reset(self) -> None:
        self._prev_gray = None
        self._distance = 0.0
        self._has_distance_evidence = False

    @property
    def distance_traveled(self) -> Optional[float]:
        """Distância acumulada (unidades arbitrárias de fluxo) ou None."""
        if not self._has_distance_evidence:
            return None
        return self._distance

    def update(
        self,
        frame: np.ndarray,
        vision: VisionResult,
        fps: float = 0.0,
    ) -> GameState:
        """Atualiza o estado a partir do frame e das detecções atuais."""
        h, w = vision.frame_shape
        distance = self._estimate_distance(frame)

        character_pos = vision.character.center if vision.character else None
        character_bbox = vision.character.bbox if vision.character else None

        obstacles: List[ObstacleRelative] = []
        for obj in [*vision.trucks, *vision.barriers]:
            rel = vision.relatives.get(id(obj))
            if rel is None:
                # Sem personagem não há posição relativa confiável.
                continue
            obstacles.append(
                ObstacleRelative(
                    label=obj.label,
                    bbox=obj.bbox,
                    confidence=obj.confidence,
                    relative=rel,
                )
            )

        coins: List[CoinOnPath] = []
        for coin in vision.coins:
            coins.append(
                CoinOnPath(
                    bbox=coin.bbox,
                    confidence=coin.confidence,
                    relative=vision.relatives.get(id(coin)),
                )
            )

        state = GameState(
            timestamp=time.time(),
            frame_size=(w, h),
            distance_traveled=distance,
            character_position=character_pos,
            character_bbox=character_bbox,
            detected_objects=list(vision.all_objects),
            obstacles_relative=obstacles,
            coins_on_path=coins,
            fps=fps,
        )
        logger.debug(
            "Estado: dist=%s char=%s obst=%d coins=%d",
            state.distance_traveled,
            state.character_position,
            len(state.obstacles_relative),
            len(state.coins_on_path),
        )
        return state

    def _estimate_distance(self, frame: np.ndarray) -> Optional[float]:
        """Estima avanço vertical por fluxo óptico denso (Farneback).

        Retorna ``None`` até haver dois frames válidos com evidência de fluxo.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as exc:
            logger.warning("Falha ao converter frame para cinza: %s", exc)
            return self.distance_traveled

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return self.distance_traveled

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            # Componente vertical média na faixa central (pista).
            h, w = gray.shape
            roi = flow[int(h * 0.3) : int(h * 0.8), int(w * 0.25) : int(w * 0.75), 1]
            mean_dy = float(np.mean(roi))
            # Movimento aparente para baixo => avanço na pista (valores positivos).
            advance = max(0.0, mean_dy)
            if advance > 0.05:
                self._distance += advance
                self._has_distance_evidence = True
        except cv2.error as exc:
            logger.warning("Fluxo óptico falhou: %s", exc)
        finally:
            self._prev_gray = gray

        return self.distance_traveled


__all__ = [
    "CoinOnPath",
    "GameState",
    "GameStateTracker",
    "ObstacleRelative",
]
