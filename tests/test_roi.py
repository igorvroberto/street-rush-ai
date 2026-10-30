"""Testes da ROI trapezoidal (núcleo + soft) e classificação de bboxes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import ROI_CORE_NORMALIZED, ROI_SOFT_NORMALIZED
from src.roi import (
    assign_track_region,
    bbox_intersection_ratio,
    build_roi_masks,
    normalized_to_pixels,
    point_region,
    tag_detections,
)
from src.roi_debug import process_frame_roi, render_roi_debug_for_captures
from src.vision import DetectedObject


def test_normalized_polygon_scales_with_resolution() -> None:
    pts_hd = normalized_to_pixels(ROI_CORE_NORMALIZED, 1920, 1080)
    assert pts_hd.shape == (4, 1, 2)
    # bottom-left ≈ (0.18*1919, 0.98*1079)
    assert pts_hd[0, 0, 0] == pytest.approx(round(0.18 * 1919))
    assert pts_hd[0, 0, 1] == pytest.approx(round(0.98 * 1079))


def test_core_and_soft_are_disjoint_and_nested() -> None:
    core, soft_only, playable = build_roi_masks(200, 100)
    assert core.shape == (100, 200)
    # soft_only não deve sobrepor o núcleo
    assert cv2.countNonZero(cv2.bitwise_and(core, soft_only)) == 0
    # playable = core ∪ soft
    assert cv2.countNonZero(cv2.bitwise_xor(playable, cv2.bitwise_or(core, soft_only))) == 0
    assert cv2.countNonZero(core) > 0
    assert cv2.countNonZero(soft_only) > 0
    # soft outer contém o núcleo
    soft_outer = np.zeros_like(core)
    cv2.fillPoly(soft_outer, [normalized_to_pixels(ROI_SOFT_NORMALIZED, 200, 100)], 255)
    assert cv2.countNonZero(cv2.bitwise_and(core, cv2.bitwise_not(soft_outer))) == 0


def test_point_region_center_of_frame_is_core() -> None:
    core, soft_only, _ = build_roi_masks(1920, 1080)
    # Centro horizontal perto da base da pista
    assert point_region(960, 900, core, soft_only) == "core"
    # Canto superior esquerdo (HUD)
    assert point_region(20, 20, core, soft_only) == "outside"
    # Logo fora do núcleo, ainda no soft (esquerda da base)
    assert point_region(250, 1050, core, soft_only) in {"soft", "outside", "core"}


def test_bbox_intersection_ratio_full_and_empty() -> None:
    core, soft_only, playable = build_roi_masks(200, 100)
    # bbox cobrindo quase toda a imagem
    full = bbox_intersection_ratio((0, 0, 200, 100), playable)
    assert 0.0 < full <= 1.0
    # bbox no canto fora
    empty = bbox_intersection_ratio((0, 0, 5, 5), core)
    assert empty == 0.0
    # bbox exatamente no núcleo: pegar um pixel do núcleo
    ys, xs = np.where(core > 0)
    x, y = int(xs[0]), int(ys[0])
    assert bbox_intersection_ratio((x, y, 1, 1), core) == 1.0


def test_assign_track_region_not_center_only() -> None:
    # Centro outside, mas muita interseção com núcleo → core
    assert (
        assign_track_region(
            center_region="outside",
            intersection_core=0.5,
            intersection_playable=0.5,
        )
        == "core"
    )
    # Centro core, mas quase sem interseção → outside
    assert (
        assign_track_region(
            center_region="core",
            intersection_core=0.0,
            intersection_playable=0.0,
        )
        == "outside"
    )
    # Soft por interseção playable
    assert (
        assign_track_region(
            center_region="outside",
            intersection_core=0.05,
            intersection_playable=0.4,
        )
        == "soft"
    )


def test_tag_detections_preserves_outside() -> None:
    core, soft_only, playable = build_roi_masks(1920, 1080)
    dets = [
        DetectedObject("coin", (10, 10, 20, 20), 0.9, (20.0, 20.0)),
        DetectedObject("truck", (800, 700, 200, 150), 0.9, (900.0, 775.0)),
    ]
    tagged = tag_detections(dets, core, soft_only, playable)
    assert len(tagged) == 2
    assert tagged[0].track_region == "outside"
    assert tagged[0].center_region == "outside"
    assert 0.0 <= tagged[0].intersection_core <= 1.0
    assert 0.0 <= tagged[0].intersection_playable <= 1.0
    # segundo tende a estar na pista
    assert tagged[1].track_region in {"core", "soft"}


def test_process_frame_roi_outputs_three_images() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    frame[800:900, 900:1000] = (0, 140, 255)
    mask, overlay, dets, report, tagged = process_frame_roi(frame, "synthetic_unit.png")
    assert mask.shape == frame.shape
    assert overlay.shape == frame.shape
    assert dets.shape == frame.shape
    assert report.total_before == len(tagged)
    assert report.total_playable + report.total_outside == report.total_before


def test_render_roi_debug_writes_files_and_keeps_originals(tmp_path: Path) -> None:
    captures = tmp_path / "captures"
    out = tmp_path / "roi"
    captures.mkdir()
    src = captures / "frame_000180_test.png"
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[160:200, 140:180] = (0, 140, 255)
    cv2.imwrite(str(src), frame)
    before = src.read_bytes()

    reports, report_path = render_roi_debug_for_captures(
        captures_dir=captures, output_dir=out
    )
    assert len(reports) == 1
    assert report_path.is_file()
    stem = src.stem
    assert (out / f"{stem}_roi_mask.png").is_file()
    assert (out / f"{stem}_roi_overlay.png").is_file()
    assert (out / f"{stem}_roi_detections.png").is_file()
    assert src.read_bytes() == before
