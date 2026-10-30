"""Testes da geração offline de imagens de debug.

Validado automaticamente:
- Ignora ``synthetic_sample.png``
- Cria ``data/debug`` e grava PNGs anotados
- Boxes / rótulos de classe e painel de contagens
- Não altera os originais em captures

Não valida qualidade das detecções HSV no Last War real.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import (
    OBJECT_BARRIER,
    OBJECT_CHARACTER,
    OBJECT_COIN,
    OBJECT_TRUCK,
)
from src.debug_render import (
    class_counts,
    debug_output_name,
    list_real_capture_frames,
    render_captures_to_debug,
    render_debug_image,
    run_debug_cli,
)
from src.main import build_parser, main
from src.vision import VisionAnalyzer


def _write_png(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (120, 160)) -> None:
    h, w = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    # blob laranja (character HSV) e amarelo (coin) para garantir alguma detecção
    frame[80:110, 60:90] = (0, 140, 255)
    frame[30:45, 100:115] = (0, 220, 240)
    assert cv2.imwrite(str(path), frame)


def test_list_real_capture_frames_ignores_synthetic(tmp_path: Path) -> None:
    captures = tmp_path / "captures"
    captures.mkdir()
    _write_png(captures / "frame_000030_1.png", (40, 40, 40))
    _write_png(captures / "synthetic_sample.png", (10, 10, 10))
    _write_png(captures / "synthetic_other.png", (10, 10, 10))
    listed = list_real_capture_frames(captures)
    assert [p.name for p in listed] == ["frame_000030_1.png"]


def test_debug_output_name() -> None:
    assert debug_output_name(Path("frame_000030_abc.png")) == "frame_000030_abc_debug.png"


def test_render_debug_image_draws_counts_panel() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[140:180, 120:160] = (0, 140, 255)
    frame[40:55, 150:165] = (0, 220, 240)
    annotated, result, counts = render_debug_image(frame)
    assert annotated.shape == frame.shape
    assert counts == class_counts(result)
    assert OBJECT_CHARACTER in counts and OBJECT_COIN in counts
    # Painel/contagens alteram pixels no topo.
    assert not np.array_equal(annotated[:40], frame[:40])


def test_render_captures_to_debug_preserves_originals(tmp_path: Path) -> None:
    captures = tmp_path / "captures"
    debug = tmp_path / "debug"
    captures.mkdir()
    src = captures / "frame_000060_x.png"
    _write_png(src, (30, 30, 30))
    before = src.read_bytes()

    written = render_captures_to_debug(captures_dir=captures, debug_dir=debug)
    assert len(written) == 1
    out = written[0]
    assert out.parent == debug.resolve()
    assert out.name == "frame_000060_x_debug.png"
    assert out.is_file() and out.stat().st_size > 0
    assert src.read_bytes() == before
    assert not (captures / "frame_000060_x_debug.png").exists()


def test_render_debug_cli_flag_no_live_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captures = tmp_path / "captures"
    debug = tmp_path / "debug"
    captures.mkdir()
    _write_png(captures / "frame_000090_y.png", (25, 25, 25))
    monkeypatch.setattr("src.debug_render.CAPTURES_DIR", captures)
    monkeypatch.setattr("src.debug_render.DEBUG_DIR", debug)
    # Também usados via import lazy em main → patch no módulo debug_render basta para run_debug_cli.
    args = build_parser().parse_args(["--render-debug"])
    assert args.render_debug is True

    code = main(["--render-debug"])
    assert code == 0
    assert (debug / "frame_000090_y_debug.png").is_file()


def test_run_debug_cli_returns_1_when_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captures = tmp_path / "captures"
    debug = tmp_path / "debug"
    captures.mkdir()
    bad = captures / "frame_000001_z.png"
    bad.write_bytes(b"not-a-png")
    monkeypatch.setattr("src.debug_render.CAPTURES_DIR", captures)
    monkeypatch.setattr("src.debug_render.DEBUG_DIR", debug)
    code = run_debug_cli(captures_dir=captures, debug_dir=debug)
    assert code == 1
    assert list(debug.glob("*.png")) == []
