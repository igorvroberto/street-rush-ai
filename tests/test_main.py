"""Testes do coordenador (main) — salvamento, paths e interrupção.

Validado automaticamente:
- Interpretação de ``--save-every``
- Condição de salvamento
- ``cv2.imwrite`` / fallback e caminho absoluto
- Diagnóstico cwd vs PROJECT_ROOT
- Resumo em Ctrl+C
- ``--diag-capture`` com MSS mockado (frame “vindo do grab”, não synthetic_sample)

NÃO validado aqui (Windows + Last War real):
- Conteúdo visual do cliente oficial
- Alt+Tab / fullscreen

``synthetic_sample.png`` NÃO é captura real e não é usado nestes testes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.main import (
    RunSession,
    announce_path_diagnostics,
    build_parser,
    consume_save_request,
    main,
    print_run_summary,
    run_capture_loop,
    run_diag_capture,
    save_frame,
    should_save_frame,
)


def _mock_monitor_bgra(h: int = 40, w: int = 60) -> tuple[MagicMock, np.ndarray]:
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[:, :, 0] = 10
    bgra[:, :, 1] = 20
    bgra[:, :, 2] = 30
    bgra[:, :, 3] = 255
    mock_sct = MagicMock()
    mock_sct.monitors = [
        {"left": 0, "top": 0, "width": w, "height": h},
        {"left": 0, "top": 0, "width": w, "height": h},
    ]
    mock_sct.grab.return_value = bgra
    return mock_sct, bgra


def test_preview_disabled_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.preview is False
    assert args.save_every == 0
    assert args.diag_capture is False


def test_save_every_parsed_as_int() -> None:
    args = build_parser().parse_args(["--save-every", "30"])
    assert args.save_every == 30
    assert isinstance(args.save_every, int)


def test_should_save_every_modulo() -> None:
    ok, reason = should_save_frame(
        frames=30, save_every=30, save_interval=0, last_save_ts=0, now=10, sentinel=False
    )
    assert ok and "save_every" in reason
    ok, _ = should_save_frame(
        frames=29, save_every=30, save_interval=0, last_save_ts=0, now=10, sentinel=False
    )
    assert not ok
    ok, _ = should_save_frame(
        frames=30, save_every=0, save_interval=0, last_save_ts=0, now=10, sentinel=False
    )
    assert not ok


def test_consume_save_request(tmp_path: Path) -> None:
    sentinel = tmp_path / ".save_now"
    assert consume_save_request(sentinel) is False
    sentinel.write_text("1", encoding="utf-8")
    assert consume_save_request(sentinel) is True
    assert not sentinel.exists()


def test_save_frame_logs_imwrite_and_absolute_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:] = (0, 128, 255)
    out = save_frame(frame, 7, tmp_path, reason="teste")
    assert out.exists() and out.is_absolute() and out.stat().st_size > 0
    logged = capsys.readouterr().out
    assert "TENTATIVA DE SALVAMENTO" in logged
    assert "cv2.imwrite retornou:" in logged
    assert "RESULTADO OK" in logged
    assert str(out) in logged


def test_announce_path_diagnostics_mentions_cwd_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    announce_path_diagnostics(PROJECT_ROOT / "data" / "captures")
    out = capsys.readouterr().out
    assert "cwd (terminal)" in out
    assert "PROJECT_ROOT (código)" in out
    assert "CAPTURES_DIR (saída)" in out
    assert "cwd == PROJECT_ROOT?" in out


def test_image_mode_no_preview(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import cv2

    img_path = tmp_path / "sample.png"
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[80:110, 60:90] = (0, 140, 255)
    cv2.imwrite(str(img_path), frame)

    code = main(["--image", str(img_path)])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert payload["frame_size"]["width"] == 160


def test_save_every_creates_expected_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mock_sct, bgra = _mock_monitor_bgra()
    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            args = build_parser().parse_args(
                [
                    "--max-frames",
                    "60",
                    "--save-every",
                    "30",
                    "--stats-interval",
                    "0",
                    "--no-trace-frames",
                ]
            )
            code = run_capture_loop(args, captures_dir=tmp_path)
    assert code == 0
    saved = sorted(tmp_path.glob("frame_*.png"))
    assert len(saved) == 2
    out = capsys.readouterr().out
    assert "save_every=30" in out
    assert "Frame recebido" not in out  # trace desligado
    assert "cv2.imwrite retornou:" in out


def test_without_save_every_warns_and_saves_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mock_sct, bgra = _mock_monitor_bgra()
    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            args = build_parser().parse_args(
                ["--max-frames", "35", "--stats-interval", "0", "--no-trace-frames"]
            )
            code = run_capture_loop(args, captures_dir=tmp_path)
    assert code == 0
    assert list(tmp_path.glob("frame_*.png")) == []
    assert "AVISO CRÍTICO: save_every=0" in capsys.readouterr().out


def test_trace_frames_logs_received(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mock_sct, bgra = _mock_monitor_bgra()
    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            args = build_parser().parse_args(
                ["--max-frames", "2", "--stats-interval", "0"]
            )
            run_capture_loop(args, captures_dir=tmp_path)
    out = capsys.readouterr().out
    assert "Frame recebido | número=1" in out
    assert "Frame recebido | número=2" in out
    assert "Condição de salvamento" in out


def test_keyboard_interrupt_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.screen_capture import ScreenCapture

    mock_sct, bgra = _mock_monitor_bgra()
    calls = {"n": 0}
    real_grab = ScreenCapture.grab

    def grab_until_interrupt(self):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] > 12:
            raise KeyboardInterrupt()
        return real_grab(self)

    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            with patch.object(ScreenCapture, "grab", grab_until_interrupt):
                args = build_parser().parse_args(
                    ["--save-every", "5", "--stats-interval", "0", "--no-trace-frames"]
                )
                code = run_capture_loop(args, captures_dir=tmp_path)

    assert code == 0
    out = capsys.readouterr().out
    assert "RESUMO DA EXECUÇÃO" in out and "Ctrl+C" in out
    assert len(list(tmp_path.glob("frame_*.png"))) == 2


def test_diag_capture_uses_mss_grab_not_synthetic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Automatizado: diag-capture chama MSS.grab (mock) — não lê synthetic_sample."""
    mock_sct, bgra = _mock_monitor_bgra(h=50, w=80)
    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            code = run_diag_capture(1, captures_dir=tmp_path)
    assert code == 0
    mock_sct.grab.assert_called()
    saved = list(tmp_path.glob("frame_*.png"))
    assert len(saved) == 1
    out = capsys.readouterr().out
    assert "DIAG-CAPTURE" in out
    assert "Frame recebido | número=1" in out
    assert "cv2.imwrite retornou:" in out
    assert "synthetic_sample" not in out


def test_print_run_summary_lists_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = RunSession(captures_dir=tmp_path)
    session.frames_captured = 10
    session.frames_processed = 10
    session.images_saved = 1
    session.last_fps = 25.0
    fake = tmp_path / "frame_000010_1.png"
    fake.write_bytes(b"x")
    session.saved_paths.append(fake.resolve())
    print_run_summary(session, reason="teste")
    out = capsys.readouterr().out
    assert "Frames capturados      : 10" in out
    assert str(fake.resolve()) in out
