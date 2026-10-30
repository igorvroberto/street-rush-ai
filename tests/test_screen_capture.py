"""Testes do capturador de tela.

O que estes testes VALIDAM (automatizado, com MSS mockado):
- Uso da geometria dinâmica do monitor reportada pelo MSS
- Captura integral (sem ROI / sem selectROI)
- Conversão BGRA→BGR e checagem de resolução
- Métricas de FPS / erros
- Preview opcional não altera o caminho de grab
- Tratamento de monitor inválido e falhas de grab

O que NÃO é validado aqui (requer teste manual no Windows com Last War):
- Continuidade visual com o jogo em primeiro plano (Alt+Tab)
- Compatibilidade com tela cheia exclusiva / borderless / janela
- Qualidade real dos frames do cliente oficial
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import MonitorInfo
from src.screen_capture import CaptureStats, ScreenCapture, ScreenCaptureError


def _mock_sct(width: int = 1918, height: int = 1078) -> MagicMock:
    mock = MagicMock()
    mock.monitors = [
        {"left": 0, "top": 0, "width": width, "height": height},  # all
        {"left": 0, "top": 0, "width": width, "height": height},  # primary
    ]
    return mock


def _bgra(h: int, w: int) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = 10
    arr[:, :, 1] = 20
    arr[:, :, 2] = 30
    arr[:, :, 3] = 255
    return arr


def test_open_uses_mss_native_resolution() -> None:
    mock = _mock_sct(1918, 1078)
    with patch("src.screen_capture.mss.mss", return_value=mock):
        capture = ScreenCapture(monitor_index=1)
        info = capture.open()
        capture.close()

    assert isinstance(info, MonitorInfo)
    assert info.width == 1918
    assert info.height == 1078
    assert info.index == 1
    assert info.left == 0 and info.top == 0


def test_dynamic_resolution_from_mss() -> None:
    """Não usa 1918x1078 fixo: respeita o que o MSS reportar."""
    mock = _mock_sct(1600, 900)
    with patch("src.screen_capture.mss.mss", return_value=mock):
        capture = ScreenCapture(monitor_index=1)
        info = capture.open()
        assert info.resolution_label == "1600x900"
        capture.close()


def test_grab_full_monitor_no_region() -> None:
    h, w = 1078, 1918
    bgra = _bgra(h, w)
    mock = _mock_sct(w, h)
    mock.grab.return_value = bgra

    with patch("src.screen_capture.mss.mss", return_value=mock):
        with ScreenCapture(monitor_index=1) as capture:
            with patch("src.screen_capture.np.asarray", return_value=bgra):
                frame = capture.grab()

    # grab chamado com o dict integral do monitor, não ROI parcial.
    called_with = mock.grab.call_args[0][0]
    assert called_with["width"] == w
    assert called_with["height"] == h
    assert called_with["left"] == 0
    assert called_with["top"] == 0
    assert frame.shape == (h, w, 3)
    assert capture.stats.frames_processed == 1
    assert capture.stats.last_resolution == (w, h)


def test_grab_rejects_resolution_mismatch() -> None:
    mock = _mock_sct(100, 80)
    # Buffer menor que a geometria MSS.
    bgra = _bgra(40, 60)
    mock.grab.return_value = bgra

    with patch("src.screen_capture.mss.mss", return_value=mock):
        capture = ScreenCapture(monitor_index=1)
        capture.open()
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            with pytest.raises(ScreenCaptureError, match="diverge"):
                capture.grab()
        assert capture.stats.capture_errors >= 1
        capture.close()


def test_grab_error_is_counted() -> None:
    mock = _mock_sct()
    mock.grab.side_effect = RuntimeError("boom")
    with patch("src.screen_capture.mss.mss", return_value=mock):
        capture = ScreenCapture(monitor_index=1)
        capture.open()
        with pytest.raises(ScreenCaptureError, match="Falha na captura"):
            capture.grab()
        assert capture.stats.capture_errors == 1
        capture.close()


def test_invalid_monitor_raises() -> None:
    mock = MagicMock()
    mock.monitors = [{"left": 0, "top": 0, "width": 100, "height": 100}]
    with patch("src.screen_capture.mss.mss", return_value=mock):
        capture = ScreenCapture(monitor_index=1)
        with pytest.raises(ScreenCaptureError, match="inválido"):
            capture.open()


def test_capture_stats_fps_and_process_time() -> None:
    stats = CaptureStats()
    stats.note_frame(1918, 1078)
    stats.note_process_time(0.01)
    stats.note_process_time(0.03)
    assert stats.frames_processed == 1
    assert stats.last_resolution == (1918, 1078)
    assert stats.avg_process_ms == pytest.approx(20.0, rel=1e-3)
    stats.reset_process_average()
    assert stats.avg_process_ms == 0.0
    assert stats._process_samples == 0


def test_no_select_roi_api() -> None:
    """Garantia de arquitetura: seletor interativo removido."""
    assert not hasattr(ScreenCapture, "select_region_interactive")
    assert not hasattr(ScreenCapture, "set_region")


def test_preview_helpers_do_not_require_grab_region() -> None:
    capture = ScreenCapture.__new__(ScreenCapture)
    capture._monitor = MonitorInfo(1, 0, 0, 200, 100)
    capture.stats = CaptureStats()
    capture.stats.fps = 30.0
    capture.stats.frames_processed = 5
    capture.stats.avg_process_ms = 12.5
    capture.stats.last_resolution = (200, 100)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = capture.draw_overlay(frame)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_screen_capture_has_no_region_attribute_contract() -> None:
    """API pública: apenas monitor integral."""
    with patch("src.screen_capture.mss.mss", return_value=_mock_sct()):
        capture = ScreenCapture(monitor_index=1)
        capture.open()
        assert not hasattr(capture, "region") or getattr(capture, "region", None) is None
        # Contrato: grab não aceita ROI.
        import inspect

        sig = inspect.signature(capture.grab)
        assert list(sig.parameters) == []
        capture.close()
