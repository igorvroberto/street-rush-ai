"""Testes básicos do módulo de visão."""

from __future__ import annotations

import numpy as np

from src.config import OBJECT_CHARACTER, OBJECT_COIN, OBJECT_TRUCK
from src.vision import VisionAnalyzer


def _blank(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_empty_frame_returns_no_invented_objects() -> None:
    analyzer = VisionAnalyzer()
    result = analyzer.analyze(_blank())
    assert result.character is None
    assert result.trucks == []
    assert result.barriers == []
    assert result.coins == []
    assert result.relatives == {}


def test_detects_yellow_coin_blob() -> None:
    frame = _blank()
    # BGR amarelo saturado na região esperada pelo HSV de moeda.
    frame[40:55, 100:115] = (0, 220, 240)
    analyzer = VisionAnalyzer()
    result = analyzer.analyze(frame)
    assert len(result.coins) >= 1
    coin = result.coins[0]
    assert coin.label == OBJECT_COIN
    x, y, w, h = coin.bbox
    # Bounding box deve cobrir o blob pintado (não coordenadas inventadas).
    assert x <= 100 and y <= 40
    assert x + w >= 115 and y + h >= 55


def test_detects_character_and_relative_truck() -> None:
    frame = _blank(300, 400)
    # Personagem (laranja) perto da base.
    frame[220:260, 180:220] = (0, 120, 255)
    # Caminhão (azul) acima, um pouco à esquerda.
    frame[80:130, 120:180] = (255, 80, 40)

    analyzer = VisionAnalyzer()
    result = analyzer.analyze(frame)

    assert result.character is not None
    assert result.character.label == OBJECT_CHARACTER
    assert len(result.trucks) >= 1
    assert result.trucks[0].label == OBJECT_TRUCK

    truck = result.trucks[0]
    rel = result.relatives.get(id(truck))
    assert rel is not None
    assert rel.dx < 0  # à esquerda do personagem
    assert rel.side == "left"
    assert 0.0 <= rel.depth_proxy <= 1.0


def test_draw_detections_preserves_shape() -> None:
    frame = _blank()
    frame[40:55, 100:115] = (0, 220, 240)
    analyzer = VisionAnalyzer()
    result = analyzer.analyze(frame)
    out = analyzer.draw_detections(frame, result)
    assert out.shape == frame.shape
