"""Testes do estado do jogo."""

from __future__ import annotations

import numpy as np

from src.game_state import GameStateTracker
from src.vision import VisionAnalyzer


def test_state_without_detections_has_nulls() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    vision = VisionAnalyzer().analyze(frame)
    tracker = GameStateTracker()
    state = tracker.update(frame, vision, fps=10.0)

    assert state.character_position is None
    assert state.character_bbox is None
    assert state.detected_objects == []
    assert state.obstacles_relative == []
    assert state.coins_on_path == []
    # Distância só existe após evidência de fluxo entre frames.
    assert state.distance_traveled is None
    assert state.frame_size == (300, 200)
    payload = state.to_dict()
    assert payload["character_position"] is None
    assert payload["distance_traveled"] is None


def test_state_includes_only_detected_obstacles() -> None:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[220:260, 180:220] = (0, 120, 255)  # character
    frame[80:130, 120:180] = (255, 80, 40)  # truck
    frame[50:62, 200:212] = (0, 220, 240)  # coin

    vision = VisionAnalyzer().analyze(frame)
    state = GameStateTracker().update(frame, vision)

    assert state.character_position is not None
    assert len(state.obstacles_relative) >= 1
    assert all(o.relative is not None for o in state.obstacles_relative)
    assert len(state.coins_on_path) >= 1
    assert "truck" in {o.label for o in state.obstacles_relative}
