"""Renderização offline de debug das detecções HSV atuais.

Lê frames reais de ``data/captures/``, aplica o pipeline de visão sem
alterar parâmetros, desenha bounding boxes + contagens e grava em
``data/debug/``. Não inicia captura ao vivo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from src.config import (
    BOX_COLORS,
    CAPTURES_DIR,
    DEBUG_DIR,
    OBJECT_BARRIER,
    OBJECT_CHARACTER,
    OBJECT_COIN,
    OBJECT_TRUCK,
)
from src.vision import VisionAnalyzer, VisionResult

logger = logging.getLogger(__name__)

CLASS_ORDER = (
    OBJECT_CHARACTER,
    OBJECT_TRUCK,
    OBJECT_BARRIER,
    OBJECT_COIN,
)


def list_real_capture_frames(captures_dir: Path = CAPTURES_DIR) -> List[Path]:
    """Lista PNGs reais para análise offline.

    Aceita:
    - pasta ``frames/`` de uma sessão;
    - pasta ``session_*`` (usa ``frames/``);
    - pasta plana legada ``data/captures/*.png`` (compatibilidade).
    """
    from src.session import list_frame_images, list_legacy_capture_frames, list_session_frames

    captures_dir = captures_dir.resolve()
    if not captures_dir.is_dir():
        return []
    if captures_dir.name == "frames" or (captures_dir / "frames").is_dir():
        return list_session_frames(captures_dir)
    # Raiz de captures: preferir legado plano; se vazio, não agrega todas as
    # sessões (análise deve escolher --session para não misturar partidas).
    legacy = list_legacy_capture_frames(captures_dir)
    if legacy:
        return legacy
    return list_frame_images(captures_dir)


def class_counts(result: VisionResult) -> dict[str, int]:
    """Contagem por classe a partir do resultado atual (sem alterar detecção)."""
    return {
        OBJECT_CHARACTER: 1 if result.character is not None else 0,
        OBJECT_TRUCK: len(result.trucks),
        OBJECT_BARRIER: len(result.barriers),
        OBJECT_COIN: len(result.coins),
    }


def draw_debug_overlay(frame: np.ndarray, result: VisionResult) -> np.ndarray:
    """Desenha boxes/rótulos (pipeline atual) + painel de contagens por classe."""
    analyzer = VisionAnalyzer()
    annotated = analyzer.draw_detections(frame, result)
    counts = class_counts(result)

    # Fundo semitransparente no topo para legibilidade.
    panel_h = 28 + 22 * len(CLASS_ORDER)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (min(420, annotated.shape[1]), panel_h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

    y = 22
    cv2.putText(
        annotated,
        "Street Rush AI — debug HSV",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y += 22
    for label in CLASS_ORDER:
        color = BOX_COLORS.get(label, (255, 255, 255))
        text = f"{label}: {counts[label]}"
        cv2.putText(
            annotated,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 22
    return annotated


def render_debug_image(
    frame: np.ndarray,
    *,
    analyzer: Optional[VisionAnalyzer] = None,
) -> tuple[np.ndarray, VisionResult, dict[str, int]]:
    """Analisa um frame com o HSV atual e devolve imagem anotada + contagens."""
    vision = analyzer or VisionAnalyzer()
    result = vision.analyze(frame)
    annotated = draw_debug_overlay(frame, result)
    return annotated, result, class_counts(result)


def save_debug_image(image: np.ndarray, destination: Path) -> Path:
    """Grava PNG de debug (confirma escrita no disco)."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(cv2.imwrite(str(destination), image))
    if not ok or not destination.is_file() or destination.stat().st_size <= 0:
        ok_enc, buf = cv2.imencode(".png", image)
        if not ok_enc:
            raise OSError(f"Falha ao gravar debug em {destination}")
        destination.write_bytes(buf.tobytes())
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise OSError(f"Arquivo de debug vazio/ausente: {destination}")
    return destination


def debug_output_name(source: Path) -> str:
    """Nome do arquivo de saída preservando o stem do frame original."""
    return f"{source.stem}_debug.png"


def render_captures_to_debug(
    *,
    captures_dir: Path = CAPTURES_DIR,
    debug_dir: Path = DEBUG_DIR,
    sources: Optional[Sequence[Path]] = None,
) -> List[Path]:
    """Gera imagens de debug para todos os frames reais (ou lista explícita).

    Não modifica os originais em ``captures_dir``.
    """
    debug_dir = debug_dir.resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    inputs = list(sources) if sources is not None else list_real_capture_frames(captures_dir)
    analyzer = VisionAnalyzer()
    written: List[Path] = []

    if not inputs:
        logger.warning("Nenhum frame real encontrado em %s", captures_dir)
        return written

    for src in inputs:
        frame = cv2.imread(str(src))
        if frame is None:
            logger.error("Não foi possível ler %s", src)
            continue
        annotated, result, counts = render_debug_image(frame, analyzer=analyzer)
        out_path = debug_dir / debug_output_name(src)
        save_debug_image(annotated, out_path)
        written.append(out_path)
        logger.info(
            "debug salvo: %s | char=%d truck=%d barrier=%d coin=%d",
            out_path,
            counts[OBJECT_CHARACTER],
            counts[OBJECT_TRUCK],
            counts[OBJECT_BARRIER],
            counts[OBJECT_COIN],
        )
        # result mantido só para side-effect de análise; evita lint unused
        _ = result
    return written


def run_debug_cli(
    *,
    captures_dir: Optional[Path] = None,
    debug_dir: Optional[Path] = None,
) -> int:
    """Ponto de entrada CLI: gera debug e imprime caminhos."""
    captures_dir = (captures_dir or CAPTURES_DIR).resolve()
    debug_dir = (debug_dir or DEBUG_DIR).resolve()
    print("[street-rush-ai] Render debug HSV", flush=True)
    print(f"[street-rush-ai] capturas: {captures_dir}", flush=True)
    print(f"[street-rush-ai] saída   : {debug_dir}", flush=True)
    frames = list_real_capture_frames(captures_dir)
    print(f"[street-rush-ai] frames reais: {len(frames)} (ignorando sintéticos)", flush=True)
    written = render_captures_to_debug(captures_dir=captures_dir, debug_dir=debug_dir)
    for path in written:
        print(f"[street-rush-ai] gerado: {path}", flush=True)
    print(f"[street-rush-ai] total gerado: {len(written)}", flush=True)
    return 0 if written or not frames else 1


__all__ = [
    "class_counts",
    "debug_output_name",
    "draw_debug_overlay",
    "list_real_capture_frames",
    "render_captures_to_debug",
    "render_debug_image",
    "run_debug_cli",
    "save_debug_image",
]
