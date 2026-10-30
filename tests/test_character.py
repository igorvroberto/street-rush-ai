"""Testes do detector de personagem (independente do HSV legado)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.character import (
    CharacterDetector,
    CharacterDetectorConfig,
    center_distance_norm,
    detection_hits_ground_truth,
    make_character_detector,
)
from src.character_debug import load_ground_truth, run_character_debug
from src.config import (
    CAPTURES_DIR,
    CHARACTER_DEBUG_DIR,
    CHARACTER_HIT_MAX_DISTANCE,
    CHARACTER_VARIANT_BASELINE,
    CHARACTER_VARIANT_EXPERIMENTAL,
    ROI_CORE_NORMALIZED,
    ROI_SOFT_NORMALIZED,
)
from src.roi import build_roi_masks, point_region
from src.vision import VisionAnalyzer


def _blank(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_empty_frame_returns_no_character() -> None:
    det = CharacterDetector()
    assert det.detect(_blank(), use_temporal=False) is None


def test_detects_blue_avatar_blob_inside_roi() -> None:
    """Blob azul compacto no núcleo da pista — não basta ser 'qualquer centro'."""
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Asfalto cinza na ROI
    core, soft_only, playable = build_roi_masks(w, h)
    frame[playable > 0] = (60, 60, 60)
    # Avatar azul (BGR) perto da base, centro horizontal
    cx, cy = 960, 720
    cv2.ellipse(frame, (cx, cy), (28, 48), 0, 0, 360, (220, 120, 40), -1)  # azul-ish BGR
    # capa vermelha
    cv2.rectangle(frame, (cx - 6, cy - 10), (cx + 6, cy + 35), (40, 40, 220), -1)

    detector = CharacterDetector()
    result = detector.detect(frame, use_temporal=False)
    assert result is not None
    assert result.region in {"core", "soft"}
    assert 0.0 < result.confidence <= 1.0
    assert result.bbox[2] > 0 and result.bbox[3] > 0
    # Não declarar sucesso só por centro da tela: exigir proximidade ao blob pintado
    assert abs(result.center[0] - cx) < 80
    assert abs(result.center[1] - cy) < 100


def test_ignores_large_side_blob_outside_character_scale() -> None:
    """Caminhão/rochedo azul largo na soft não deve vencer um avatar compacto."""
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _, _, playable = build_roi_masks(w, h)
    frame[playable > 0] = (50, 50, 50)
    # Falso positivo enorme à direita
    cv2.rectangle(frame, (1300, 400), (1700, 900), (230, 140, 50), -1)
    # Avatar compacto no centro
    cv2.ellipse(frame, (960, 700), (30, 50), 0, 0, 360, (220, 130, 45), -1)

    result = CharacterDetector().detect(frame, use_temporal=False)
    assert result is not None
    assert result.center_norm[0] < 0.65  # não gruda no blob lateral


def test_jump_position_can_be_soft() -> None:
    """Posição alta (salto) ainda dentro da soft é válida."""
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _, _, playable = build_roi_masks(w, h)
    frame[playable > 0] = (55, 55, 55)
    # Centro alto (~0.36) — típico do frame 300
    cx, cy = 960, int(0.36 * h)
    cv2.ellipse(frame, (cx, cy), (26, 42), 0, 0, 360, (230, 150, 40), -1)

    result = CharacterDetector().detect(frame, use_temporal=False)
    assert result is not None
    assert result.region in {"core", "soft"}
    assert abs(result.center_norm[1] - 0.36) < 0.08


def test_does_not_assume_exact_screen_center() -> None:
    """Mudança de faixa: avatar deslocado ainda é detectável."""
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _, _, playable = build_roi_masks(w, h)
    frame[playable > 0] = (50, 50, 50)
    cx, cy = int(0.58 * w), 720
    cv2.ellipse(frame, (cx, cy), (28, 48), 0, 0, 360, (220, 120, 40), -1)

    result = CharacterDetector().detect(frame, use_temporal=False)
    assert result is not None
    assert result.center_norm[0] > 0.52


def test_hit_metric_requires_proximity_not_just_core_label() -> None:
    from src.character import CharacterDetection

    # Detecção no core mas longe do GT → miss
    fake = CharacterDetection(
        bbox=(100, 100, 40, 60),
        center=(120.0, 130.0),
        confidence=0.9,
        region="core",
        source_cue="blue",
        score=5.0,
        center_norm=(0.20, 0.30),
    )
    assert detection_hits_ground_truth(fake, (0.50, 0.60), max_distance=0.06) is False
    assert detection_hits_ground_truth(None, (0.50, 0.60)) is False
    near = CharacterDetection(
        bbox=(900, 600, 50, 80),
        center=(925.0, 640.0),
        confidence=0.9,
        region="core",
        source_cue="blue",
        score=5.0,
        center_norm=(0.50, 0.60),
    )
    assert detection_hits_ground_truth(near, (0.50, 0.60), max_distance=0.06) is True


def test_hsv_legacy_unchanged_and_independent() -> None:
    """Truck/barrier/coin e character HSV legado continuam no VisionAnalyzer."""
    frame = _blank(300, 400)
    frame[40:55, 100:115] = (0, 220, 240)  # coin
    frame[80:130, 120:180] = (255, 80, 40)  # truck
    vision = VisionAnalyzer()
    result = vision.analyze(frame)
    assert len(result.coins) >= 1
    assert len(result.trucks) >= 1
    # Faixas HSV legadas de personagem intactas (laranja 8–28)
    assert vision.config.character_hsv.lower == (8, 100, 100)
    assert vision.config.character_hsv.upper == (28, 255, 255)
    # Detector novo é módulo separado (não substitui o legado)
    assert CharacterDetector is not VisionAnalyzer


def test_roi_geometry_constants_untouched() -> None:
    assert ROI_CORE_NORMALIZED[0] == (0.18, 0.98)
    assert ROI_SOFT_NORMALIZED[0] == (0.10, 1.00)


@pytest.mark.skipif(
    not any(CAPTURES_DIR.glob("frame_*.png")),
    reason="frames reais ausentes",
)
def test_real_frames_hit_rate_against_ground_truth() -> None:
    """Validação nos 10 frames reais: acerto por proximidade ao REAL, não por 'centro'."""
    summary = run_character_debug(
        captures_dir=CAPTURES_DIR,
        out_dir=CHARACTER_DEBUG_DIR,
        use_temporal=False,
    )
    assert summary["frames_with_gt"] == 10
    assert summary["hits"] == 10
    assert summary["false_negatives"] == 0
    # Nenhum acerto vazio: cada hit tem detecção com região core/soft
    for fr in summary["frames"]:
        assert fr["hit"] is True
        assert fr["detected"] is not None
        assert fr["detected"]["region"] in {"core", "soft"}
        assert fr["distance_to_gt"] is not None
        assert fr["distance_to_gt"] <= CHARACTER_HIT_MAX_DISTANCE


def test_ground_truth_lists_ten_frames() -> None:
    gt = load_ground_truth()
    assert len(gt["frames"]) == 10
    assert gt["frames"]["frame_000300_1793390965.png"]["expected_region"] == "soft"


def test_center_distance_norm() -> None:
    assert center_distance_norm((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0.0)
    assert center_distance_norm((0.0, 0.0), (0.3, 0.4)) == pytest.approx(0.5)


def test_default_detector_is_baseline() -> None:
    assert CharacterDetector().variant == CHARACTER_VARIANT_BASELINE
    assert make_character_detector().variant == CHARACTER_VARIANT_BASELINE


def test_make_character_detector_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError):
        make_character_detector("desconhecida")


def _paint_playable(frame: np.ndarray) -> None:
    _, _, playable = build_roi_masks(frame.shape[1], frame.shape[0])
    frame[playable > 0] = (55, 55, 55)


def test_experimental_rejects_homogeneous_single_color_panel() -> None:
    """Container/painel azul plano (FP típico do holdout) não deve passar no experimental."""
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _paint_playable(frame)
    # Painel homogêneo grande no núcleo (simulando container).
    cv2.rectangle(frame, (880, 620), (1040, 820), (210, 120, 40), -1)

    baseline = make_character_detector(CHARACTER_VARIANT_BASELINE)
    experimental = make_character_detector(CHARACTER_VARIANT_EXPERIMENTAL)
    base_det = baseline.detect(frame, use_temporal=False)
    exp_det = experimental.detect(frame, use_temporal=False)

    # Baseline pode ainda grudar no painel (comportamento conhecido).
    assert base_det is not None
    # Experimental deve rejeitar painel plano mono-cor.
    assert exp_det is None


def test_experimental_keeps_multicolor_textured_avatar() -> None:
    """Avatar azul + capa vermelha com variação de tom permanece detectável."""
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _paint_playable(frame)
    cx, cy = 960, 720
    # Corpo azul com leve variação (textura).
    for i, shade in enumerate((200, 220, 235, 210)):
        cv2.ellipse(
            frame,
            (cx, cy - i * 2),
            (26 - i, 46 - i),
            0,
            0,
            360,
            (shade, 110 + i * 5, 35 + i * 3),
            -1,
        )
    # Capa/cachecol vermelho
    cv2.rectangle(frame, (cx - 8, cy - 5), (cx + 8, cy + 40), (35, 35, 210), -1)
    cv2.rectangle(frame, (cx - 6, cy + 10), (cx + 6, cy + 38), (50, 50, 180), -1)
    # Detalhe laranja/claro no capacete
    cv2.circle(frame, (cx, cy - 28), 8, (40, 140, 230), -1)

    result = make_character_detector(CHARACTER_VARIANT_EXPERIMENTAL).detect(
        frame, use_temporal=False
    )
    assert result is not None
    assert result.variant == CHARACTER_VARIANT_EXPERIMENTAL
    assert result.color_cues >= 2
    assert abs(result.center[0] - cx) < 90
    assert abs(result.center[1] - cy) < 110


def test_baseline_unchanged_on_simple_blue_avatar() -> None:
    """Sanidade: baseline continua detectando o blob azul compacto dos testes atuais."""
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _paint_playable(frame)
    cx, cy = 960, 720
    cv2.ellipse(frame, (cx, cy), (28, 48), 0, 0, 360, (220, 120, 40), -1)
    cv2.rectangle(frame, (cx - 6, cy - 10), (cx + 6, cy + 35), (40, 40, 220), -1)
    result = CharacterDetector(CharacterDetectorConfig(variant=CHARACTER_VARIANT_BASELINE)).detect(
        frame, use_temporal=False
    )
    assert result is not None
    assert result.variant == CHARACTER_VARIANT_BASELINE
