"""Sessões exclusivas de captura e debug.

Cada execução cria ``data/captures/session_YYYYMMDD_HHMMSS/`` com ``frames/``
e ``metadata.json``. Debug espelhado em ``data/debug/session_.../``.

Não altera detectores nem a geometria da ROI. Apenas organiza dados para
comparação futura entre partidas (métricas ficam como campos reservados).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config import CAPTURE_IGNORE_NAMES, CAPTURES_DIR, DEBUG_DIR

SESSION_PREFIX = "session_"
SESSION_NAME_RE = re.compile(r"^session_\d{8}_\d{6}(?:_\d+)?$")

# Campos reservados para comparação futura entre sessões (não preenchidos aqui).
COMPARISON_PLACEHOLDERS: dict[str, Any] = {
    "character_hit_rate": None,
    "character_mean_confidence": None,
    "false_positives": None,
    "false_negatives": None,
    "frames_analyzed": None,
    "by_situation": {
        "run": None,
        "jump": None,
        "crouch": None,
        "lane_change": None,
    },
}


@dataclass
class SessionPaths:
    """Caminhos de uma sessão de captura."""

    session_id: str
    session_dir: Path
    frames_dir: Path
    metadata_path: Path
    debug_dir: Path

    def ensure(self) -> "SessionPaths":
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("hsv", "roi", "character"):
            (self.debug_dir / sub).mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class SessionMetadata:
    """Metadados persistidos em ``metadata.json``."""

    session_id: str
    created_at: str
    started_at: str
    ended_at: Optional[str] = None
    resolution: Optional[dict[str, int]] = None
    fps_avg: Optional[float] = None
    frames_saved: int = 0
    frames_captured: int = 0
    frames_processed: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    # Reservado: preenchido por análise futura, nunca pelo detector na captura.
    comparison: dict[str, Any] = field(
        default_factory=lambda: json.loads(json.dumps(COMPARISON_PLACEHOLDERS))
    )
    notes: str = (
        "comparison.* é reservado para validação futura entre sessões; "
        "não usar a saída do detector como ground truth automático."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "SessionMetadata":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            session_id=str(data.get("session_id", "")),
            created_at=str(data.get("created_at", "")),
            started_at=str(data.get("started_at", "")),
            ended_at=data.get("ended_at"),
            resolution=data.get("resolution"),
            fps_avg=data.get("fps_avg"),
            frames_saved=int(data.get("frames_saved", 0)),
            frames_captured=int(data.get("frames_captured", 0)),
            frames_processed=int(data.get("frames_processed", 0)),
            parameters=dict(data.get("parameters") or {}),
            paths=dict(data.get("paths") or {}),
            comparison=dict(data.get("comparison") or COMPARISON_PLACEHOLDERS),
            notes=str(data.get("notes", "")),
        )


def format_session_timestamp(when: Optional[datetime] = None) -> str:
    """``YYYYMMDD_HHMMSS`` no fuso local da máquina que captura."""
    dt = when or datetime.now().astimezone()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc).astimezone()
    return dt.strftime("%Y%m%d_%H%M%S")


def iso_now(when: Optional[datetime] = None) -> str:
    dt = when or datetime.now().astimezone()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def allocate_session_id(
    captures_root: Path,
    *,
    when: Optional[datetime] = None,
) -> str:
    """Gera ``session_YYYYMMDD_HHMMSS``; se existir, adiciona ``_1``, ``_2``, …"""
    base = f"{SESSION_PREFIX}{format_session_timestamp(when)}"
    candidate = base
    n = 1
    while (captures_root / candidate).exists():
        candidate = f"{base}_{n}"
        n += 1
        if n > 10_000:
            raise RuntimeError(f"Não foi possível alocar sessão sob {captures_root}")
    return candidate


def create_capture_session(
    *,
    captures_root: Path = CAPTURES_DIR,
    debug_root: Path = DEBUG_DIR,
    parameters: Optional[dict[str, Any]] = None,
    when: Optional[datetime] = None,
    session_id: Optional[str] = None,
) -> tuple[SessionPaths, SessionMetadata]:
    """Cria pasta exclusiva de sessão (nunca sobrescreve existente)."""
    captures_root = captures_root.resolve()
    debug_root = debug_root.resolve()
    captures_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    sid = session_id or allocate_session_id(captures_root, when=when)
    session_dir = captures_root / sid
    if session_dir.exists():
        # Corrida rara: realoca.
        sid = allocate_session_id(captures_root, when=when)
        session_dir = captures_root / sid

    paths = SessionPaths(
        session_id=sid,
        session_dir=session_dir,
        frames_dir=session_dir / "frames",
        metadata_path=session_dir / "metadata.json",
        debug_dir=debug_root / sid,
    ).ensure()

    stamp = iso_now(when)
    meta = SessionMetadata(
        session_id=sid,
        created_at=stamp,
        started_at=stamp,
        parameters=dict(parameters or {}),
        paths={
            "session_dir": str(paths.session_dir.resolve()),
            "frames_dir": str(paths.frames_dir.resolve()),
            "metadata": str(paths.metadata_path.resolve()),
            "debug_dir": str(paths.debug_dir.resolve()),
        },
    )
    meta.write(paths.metadata_path)
    return paths, meta


def create_session_from_output_dir(
    output_dir: Path,
    *,
    debug_root: Path = DEBUG_DIR,
    parameters: Optional[dict[str, Any]] = None,
) -> tuple[SessionPaths, SessionMetadata]:
    """Usa pasta manual como raiz da sessão (cria ``frames/`` e metadata)."""
    output_dir = output_dir.resolve()
    sid = output_dir.name if SESSION_NAME_RE.match(output_dir.name) else output_dir.name
    # Evita misturar com outra sessão: se já tem frames/metadata, recusa sobrescrita
    # de PNGs — apenas reabre para append se o usuário pediu explicitamente a pasta.
    paths = SessionPaths(
        session_id=sid,
        session_dir=output_dir,
        frames_dir=output_dir / "frames",
        metadata_path=output_dir / "metadata.json",
        debug_dir=(debug_root / sid).resolve(),
    ).ensure()
    if paths.metadata_path.is_file():
        meta = SessionMetadata.load(paths.metadata_path)
        meta.parameters = {**meta.parameters, **(parameters or {})}
        meta.paths = {
            "session_dir": str(paths.session_dir.resolve()),
            "frames_dir": str(paths.frames_dir.resolve()),
            "metadata": str(paths.metadata_path.resolve()),
            "debug_dir": str(paths.debug_dir.resolve()),
        }
    else:
        stamp = iso_now()
        meta = SessionMetadata(
            session_id=sid,
            created_at=stamp,
            started_at=stamp,
            parameters=dict(parameters or {}),
            paths={
                "session_dir": str(paths.session_dir.resolve()),
                "frames_dir": str(paths.frames_dir.resolve()),
                "metadata": str(paths.metadata_path.resolve()),
                "debug_dir": str(paths.debug_dir.resolve()),
            },
        )
    meta.write(paths.metadata_path)
    return paths, meta


def list_session_dirs(captures_root: Path = CAPTURES_DIR) -> List[Path]:
    """Lista pastas ``session_*`` sob ``captures_root`` (ordenadas)."""
    captures_root = captures_root.resolve()
    if not captures_root.is_dir():
        return []
    sessions = [
        p
        for p in captures_root.iterdir()
        if p.is_dir() and p.name.startswith(SESSION_PREFIX)
    ]
    return sorted(sessions, key=lambda p: p.name)


def resolve_session(
    session: str | Path,
    *,
    captures_root: Path = CAPTURES_DIR,
    debug_root: Path = DEBUG_DIR,
) -> SessionPaths:
    """Resolve id ou caminho para ``SessionPaths`` (sem criar)."""
    captures_root = captures_root.resolve()
    debug_root = debug_root.resolve()
    raw = Path(session)
    if raw.is_dir() and (raw / "frames").is_dir():
        session_dir = raw.resolve()
    elif raw.is_dir() and SESSION_NAME_RE.match(raw.name):
        session_dir = raw.resolve()
    else:
        name = raw.name if str(raw) == raw.name or not raw.is_absolute() else raw.name
        if not name.startswith(SESSION_PREFIX):
            name = f"{SESSION_PREFIX}{name}" if name[:1].isdigit() else name
        session_dir = (captures_root / name).resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Sessão não encontrada: {session_dir}")
    return SessionPaths(
        session_id=session_dir.name,
        session_dir=session_dir,
        frames_dir=session_dir / "frames",
        metadata_path=session_dir / "metadata.json",
        debug_dir=debug_root / session_dir.name,
    )


def list_frame_images(directory: Path) -> List[Path]:
    """PNGs reais em um diretório (sem recursão), ignorando sintéticos."""
    if not directory.is_dir():
        return []
    frames: List[Path] = []
    for path in sorted(directory.glob("*.png")):
        if path.name in CAPTURE_IGNORE_NAMES:
            continue
        if path.name.startswith("synthetic"):
            continue
        frames.append(path)
    return frames


def list_session_frames(session: SessionPaths | Path) -> List[Path]:
    """Frames em ``session/frames/``."""
    if isinstance(session, SessionPaths):
        return list_frame_images(session.frames_dir)
    frames_dir = session / "frames" if (session / "frames").is_dir() else session
    return list_frame_images(frames_dir)


def list_legacy_capture_frames(captures_root: Path = CAPTURES_DIR) -> List[Path]:
    """Frames soltos em ``data/captures/*.png`` (compatibilidade pré-sessão)."""
    return list_frame_images(captures_root.resolve())


def finalize_metadata(
    meta: SessionMetadata,
    paths: SessionPaths,
    *,
    frames_saved: int,
    frames_captured: int,
    frames_processed: int,
    resolution: Optional[tuple[int, int]] = None,
    fps_avg: Optional[float] = None,
    parameters: Optional[dict[str, Any]] = None,
) -> SessionMetadata:
    """Atualiza e grava metadata ao encerrar a execução."""
    meta.ended_at = iso_now()
    meta.frames_saved = int(frames_saved)
    meta.frames_captured = int(frames_captured)
    meta.frames_processed = int(frames_processed)
    if resolution is not None:
        meta.resolution = {"width": int(resolution[0]), "height": int(resolution[1])}
    if fps_avg is not None:
        meta.fps_avg = float(fps_avg)
    if parameters:
        meta.parameters.update(parameters)
    meta.paths = {
        "session_dir": str(paths.session_dir.resolve()),
        "frames_dir": str(paths.frames_dir.resolve()),
        "metadata": str(paths.metadata_path.resolve()),
        "debug_dir": str(paths.debug_dir.resolve()),
    }
    # Nunca preenche comparison a partir do detector aqui.
    meta.write(paths.metadata_path)
    return meta


def next_frame_path(frames_dir: Path, sequence: int) -> Path:
    """Caminho sequencial ``frame_000001.png`` sem sobrescrever existentes."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    base = frames_dir / f"frame_{sequence:06d}.png"
    if not base.exists():
        return base
    n = 1
    while True:
        alt = frames_dir / f"frame_{sequence:06d}_{n}.png"
        if not alt.exists():
            return alt
        n += 1


__all__ = [
    "COMPARISON_PLACEHOLDERS",
    "SESSION_PREFIX",
    "SessionMetadata",
    "SessionPaths",
    "allocate_session_id",
    "create_capture_session",
    "create_session_from_output_dir",
    "finalize_metadata",
    "format_session_timestamp",
    "iso_now",
    "list_frame_images",
    "list_legacy_capture_frames",
    "list_session_dirs",
    "list_session_frames",
    "next_frame_path",
    "resolve_session",
]
