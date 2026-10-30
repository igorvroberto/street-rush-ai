"""Testes do capturador de tela (sem depender de display real)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import CaptureRegion
from src.screen_capture import ScreenCapture, ScreenCaptureError, parse_region_arg


def test_parse_region_arg_ok() -> None:
    region = parse_region_arg("10, 20, 800, 600")
    assert region == CaptureRegion(left=10, top=20, width=800, height=600)


def test_parse_region_arg_invalid() -> None:
    with pytest.raises(ValueError):
        parse_region_arg("10,20")
    with pytest.raises(ValueError):
        parse_region_arg("0,0,0,100")


def test_clamp_region_to_monitor() -> None:
    region = CaptureRegion(left=-50, top=-10, width=5000, height=5000)
    clamped = region.clamp_to_monitor(0, 0, 1920, 1080)
    assert clamped.left == 0
    assert clamped.top == 0
    assert clamped.width == 1920
    assert clamped.height == 1080


def test_grab_converts_bgra_to_bgr() -> None:
    h, w = 40, 60
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[:, :, 0] = 10  # B
    bgra[:, :, 1] = 20  # G
    bgra[:, :, 2] = 30  # R
    bgra[:, :, 3] = 255

    shot = MagicMock()
    # np.asarray(shot) no código real usa o buffer do MSS; simulamos via side effect.
    mock_sct = MagicMock()
    mock_sct.monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
    ]
    mock_sct.grab.return_value = bgra

    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        capture = ScreenCapture(monitor_index=1)
        capture.open()
        # Força grab a usar array BGRA diretamente.
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            frame = capture.grab()
        capture.close()

    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8
    # Canal B permanece 10 após BGRA->BGR.
    assert int(frame[0, 0, 0]) == 10
    assert int(frame[0, 0, 2]) == 30


def test_draw_fps_overlay_keeps_shape() -> None:
    capture = ScreenCapture.__new__(ScreenCapture)
    capture.region = CaptureRegion(0, 0, 100, 80)
    capture._fps = 42.5
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    out = capture.draw_fps_overlay(frame)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_invalid_monitor_raises() -> None:
    mock_sct = MagicMock()
    mock_sct.monitors = [{"left": 0, "top": 0, "width": 100, "height": 100}]
    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        capture = ScreenCapture(monitor_index=5)
        with pytest.raises(ScreenCaptureError):
            capture.open()
