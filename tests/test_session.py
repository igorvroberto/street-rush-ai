"""Testes de sessões exclusivas de captura (sem sobrescrita entre execuções)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.main import build_parser, run_capture_loop, run_diag_capture
from src.session import (
    COMPARISON_PLACEHOLDERS,
    allocate_session_id,
    create_capture_session,
    create_session_from_output_dir,
    finalize_metadata,
    list_legacy_capture_frames,
    list_session_dirs,
    list_session_frames,
    next_frame_path,
    resolve_session,
)


def _mock_monitor_bgra(h: int = 40, w: int = 60) -> tuple[MagicMock, np.ndarray]:
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[:, :, :3] = (10, 20, 30)
    bgra[:, :, 3] = 255
    mock_sct = MagicMock()
    mock_sct.monitors = [
        {"left": 0, "top": 0, "width": w, "height": h},
        {"left": 0, "top": 0, "width": w, "height": h},
    ]
    mock_sct.grab.return_value = bgra
    return mock_sct, bgra


def test_two_sessions_never_overwrite(tmp_path: Path) -> None:
    when = datetime(2026, 10, 30, 18, 30, 0, tzinfo=timezone.utc)
    p1, m1 = create_capture_session(captures_root=tmp_path, debug_root=tmp_path / "debug", when=when)
    p2, m2 = create_capture_session(captures_root=tmp_path, debug_root=tmp_path / "debug", when=when)
    assert p1.session_id != p2.session_id
    assert p1.session_dir != p2.session_dir
    assert p1.session_dir.is_dir() and p2.session_dir.is_dir()
    assert (p1.frames_dir).is_dir() and (p2.frames_dir).is_dir()
    assert p1.metadata_path.is_file() and p2.metadata_path.is_file()
    # Escrever frame em cada uma não afeta a outra
    f1 = next_frame_path(p1.frames_dir, 1)
    f2 = next_frame_path(p2.frames_dir, 1)
    f1.write_bytes(b"aaa")
    f2.write_bytes(b"bbb")
    assert f1.read_bytes() == b"aaa"
    assert f2.read_bytes() == b"bbb"
    assert m1.session_id.startswith("session_20261030_183000")
    assert m2.session_id.startswith("session_20261030_183000")


def test_allocate_session_id_suffix_on_conflict(tmp_path: Path) -> None:
    when = datetime(2026, 10, 30, 19, 0, 0, tzinfo=timezone.utc)
    first = allocate_session_id(tmp_path, when=when)
    (tmp_path / first).mkdir()
    second = allocate_session_id(tmp_path, when=when)
    assert second == f"{first}_1"
    (tmp_path / second).mkdir()
    third = allocate_session_id(tmp_path, when=when)
    assert third == f"{first}_2"


def test_next_frame_path_never_overwrites(tmp_path: Path) -> None:
    a = next_frame_path(tmp_path, 5)
    assert a.name == "frame_000005.png"
    a.write_bytes(b"x")
    b = next_frame_path(tmp_path, 5)
    assert b.name == "frame_000005_1.png"
    assert not b.exists()


def test_metadata_has_comparison_placeholders(tmp_path: Path) -> None:
    paths, meta = create_capture_session(
        captures_root=tmp_path,
        debug_root=tmp_path / "dbg",
        parameters={"save_every": 30},
    )
    finalize_metadata(
        meta,
        paths,
        frames_saved=3,
        frames_captured=90,
        frames_processed=90,
        resolution=(1918, 1078),
        fps_avg=28.5,
    )
    data = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert data["frames_saved"] == 3
    assert data["resolution"] == {"width": 1918, "height": 1078}
    assert data["parameters"]["save_every"] == 30
    assert data["comparison"]["character_hit_rate"] is None
    assert data["comparison"]["by_situation"]["jump"] is None
    assert set(data["comparison"]["by_situation"]) == set(
        COMPARISON_PLACEHOLDERS["by_situation"]
    )
    assert data["ended_at"] is not None


def test_manual_output_dir(tmp_path: Path) -> None:
    custom = tmp_path / "minha_partida"
    paths, meta = create_session_from_output_dir(
        custom, debug_root=tmp_path / "debug", parameters={"monitor": 1}
    )
    assert paths.session_dir == custom.resolve()
    assert paths.frames_dir == custom.resolve() / "frames"
    assert paths.frames_dir.is_dir()
    assert meta.parameters["monitor"] == 1


def test_legacy_frames_still_listed(tmp_path: Path) -> None:
    (tmp_path / "frame_000030_1.png").write_bytes(b"png")
    (tmp_path / "synthetic_sample.png").write_bytes(b"x")
    (tmp_path / "session_20261030_120000").mkdir()
    (tmp_path / "session_20261030_120000" / "frames").mkdir()
    (tmp_path / "session_20261030_120000" / "frames" / "frame_000001.png").write_bytes(b"y")
    legacy = list_legacy_capture_frames(tmp_path)
    assert len(legacy) == 1
    assert legacy[0].name == "frame_000030_1.png"
    sess = resolve_session("session_20261030_120000", captures_root=tmp_path, debug_root=tmp_path / "d")
    assert len(list_session_frames(sess)) == 1


def test_two_capture_loops_create_isolated_sessions(tmp_path: Path) -> None:
    mock_sct, bgra = _mock_monitor_bgra()
    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            args = build_parser().parse_args(
                [
                    "--max-frames",
                    "30",
                    "--save-every",
                    "15",
                    "--stats-interval",
                    "0",
                    "--no-trace-frames",
                ]
            )
            assert run_capture_loop(args, captures_dir=tmp_path) == 0
            assert run_capture_loop(args, captures_dir=tmp_path) == 0

    sessions = list_session_dirs(tmp_path)
    assert len(sessions) == 2
    files_a = list((sessions[0] / "frames").glob("frame_*.png"))
    files_b = list((sessions[1] / "frames").glob("frame_*.png"))
    assert len(files_a) == 2
    assert len(files_b) == 2
    # Metadados distintos
    meta_a = json.loads((sessions[0] / "metadata.json").read_text(encoding="utf-8"))
    meta_b = json.loads((sessions[1] / "metadata.json").read_text(encoding="utf-8"))
    assert meta_a["session_id"] != meta_b["session_id"]
    assert meta_a["frames_saved"] == 2
    assert meta_b["frames_saved"] == 2
    assert (sessions[0] / "frames").is_dir()
    assert (sessions[1] / "frames").is_dir()
    assert (tmp_path / "debug" / sessions[0].name / "character").is_dir()
    assert (tmp_path / "debug" / sessions[1].name / "roi").is_dir()


def test_diag_capture_writes_into_session_frames(tmp_path: Path) -> None:
    mock_sct, bgra = _mock_monitor_bgra(h=50, w=80)
    with patch("src.screen_capture.mss.mss", return_value=mock_sct):
        with patch("src.screen_capture.np.asarray", return_value=bgra):
            code = run_diag_capture(1, captures_dir=tmp_path)
    assert code == 0
    sessions = list_session_dirs(tmp_path)
    assert len(sessions) == 1
    saved = list((sessions[0] / "frames").glob("frame_*.png"))
    assert len(saved) == 1
    meta = json.loads((sessions[0] / "metadata.json").read_text(encoding="utf-8"))
    assert meta["frames_saved"] == 1
    assert meta["resolution"] == {"width": 80, "height": 50}
