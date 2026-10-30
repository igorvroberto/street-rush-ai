"""Testes do coordenador (main) — modo sem preview e salvamento.

Validado automaticamente:
- Preview desligado por padrão
- Flags de monitoramento / save
- Sentinela `.save_now`
- Análise offline de imagem

Não validado aqui (manual no Windows + Last War):
- Alt+Tab mantendo o processo vivo com o jogo em foco
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import SAVE_REQUEST_FILE
from src.main import build_parser, consume_save_request, main, save_frame


def test_preview_disabled_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.preview is False
    assert args.save_interval == 0.0
    assert args.save_every == 0
    assert args.monitor == 1


def test_preview_flag_opt_in() -> None:
    args = build_parser().parse_args(["--preview", "--save-every", "10"])
    assert args.preview is True
    assert args.save_every == 10


def test_consume_save_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = tmp_path / ".save_now"
    monkeypatch.setattr("src.main.SAVE_REQUEST_FILE", sentinel)
    assert consume_save_request() is False
    sentinel.write_text("1", encoding="utf-8")
    assert consume_save_request() is True
    assert not sentinel.exists()
    assert consume_save_request() is False


def test_save_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.main.CAPTURES_DIR", tmp_path)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    out = save_frame(frame, 7)
    assert out.exists()
    assert out.parent == tmp_path


def test_image_mode_no_preview(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import cv2

    img_path = tmp_path / "sample.png"
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[80:110, 60:90] = (0, 140, 255)
    cv2.imwrite(str(img_path), frame)

    code = main(["--image", str(img_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "detected_objects" in payload
    assert payload["frame_size"]["width"] == 160


def test_headless_loop_runs_without_preview() -> None:
    """Garante que o loop padrão não chama show_preview."""
    h, w = 40, 60
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[:, :, 3] = 255

    mock_sct = MagicMock()
    mock_sct.monitors = [
        {"left": 0, "top": 0, "width": w, "height": h},
        {"left": 0, "top": 0, "width": w, "height": h},
    ]
    mock_sct.grab.return_value = bgra

    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            with patch("src.main.ScreenCapture.show_preview") as preview:
                code = main(["--max-frames", "3", "--stats-interval", "0"])
                assert code == 0
                preview.assert_not_called()
