"""Captura integral do monitor principal com MSS + OpenCV/NumPy.

Captura sempre o monitor completo usando as dimensões reais reportadas
pelo MSS. Não há seleção de ROI. O processo Python continua ativo com
qualquer janela em primeiro plano (Last War, Cursor, etc.).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mss
import numpy as np

from src.config import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    KEY_QUIT,
    KEY_SAVE_FRAME,
    MONITOR_INDEX,
    PREVIEW_MAX_WIDTH,
    PREVIEW_WINDOW_NAME,
    PREVIEW_WINDOW_X,
    PREVIEW_WINDOW_Y,
    MonitorInfo,
)

logger = logging.getLogger(__name__)


class ScreenCaptureError(Exception):
    """Erro de captura de tela."""


@dataclass
class CaptureStats:
    """Métricas de desempenho da captura/processamento."""

    frames_processed: int = 0
    capture_errors: int = 0
    fps: float = 0.0
    avg_process_ms: float = 0.0
    last_resolution: tuple[int, int] = (0, 0)  # width, height
    _fps_count: int = field(default=0, repr=False)
    _fps_window_start: float = field(default_factory=time.perf_counter, repr=False)
    _process_ms_accum: float = field(default=0.0, repr=False)
    _process_samples: int = field(default=0, repr=False)

    def note_frame(self, width: int, height: int) -> None:
        self.frames_processed += 1
        self.last_resolution = (width, height)
        self._fps_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_window_start
        if elapsed >= 0.5:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_window_start = now

    def note_process_time(self, elapsed_sec: float) -> None:
        self._process_ms_accum += elapsed_sec * 1000.0
        self._process_samples += 1
        if self._process_samples > 0:
            self.avg_process_ms = self._process_ms_accum / self._process_samples

    def note_error(self) -> None:
        self.capture_errors += 1

    def reset_process_average(self) -> None:
        """Reinicia a média de processamento (janela móvel simples)."""
        self._process_ms_accum = 0.0
        self._process_samples = 0
        self.avg_process_ms = 0.0


class ScreenCapture:
    """Capturador do monitor integral via MSS.

    Parameters
    ----------
    monitor_index:
        Índice MSS (1 = monitor principal).
    """

    def __init__(self, monitor_index: int = MONITOR_INDEX) -> None:
        self.monitor_index = monitor_index
        self._sct: Optional[mss.mss] = None
        self._monitor: Optional[MonitorInfo] = None
        self.stats = CaptureStats()

    def __enter__(self) -> "ScreenCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def monitor(self) -> MonitorInfo:
        """Geometria do monitor em uso (após ``open``)."""
        if self._monitor is None:
            raise ScreenCaptureError("Captura não iniciada. Chame open() primeiro.")
        return self._monitor

    @property
    def fps(self) -> float:
        return self.stats.fps

    def open(self) -> MonitorInfo:
        """Abre o MSS, resolve o monitor principal e registra resolução."""
        if self._sct is not None and self._monitor is not None:
            return self._monitor
        try:
            self._sct = mss.mss()
            if self.monitor_index < 1 or self.monitor_index >= len(self._sct.monitors):
                raise ScreenCaptureError(
                    f"Monitor {self.monitor_index} inválido. "
                    f"MSS reporta {len(self._sct.monitors) - 1} monitor(es) "
                    f"(use índice 1..{max(1, len(self._sct.monitors) - 1)})."
                )
            mon = self._sct.monitors[self.monitor_index]
            self._monitor = MonitorInfo(
                index=self.monitor_index,
                left=int(mon["left"]),
                top=int(mon["top"]),
                width=int(mon["width"]),
                height=int(mon["height"]),
            )
            self._log_monitor_selection()
            return self._monitor
        except ScreenCaptureError:
            self.close()
            raise
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise ScreenCaptureError(f"Falha ao iniciar MSS: {exc}") from exc

    def close(self) -> None:
        """Libera recursos MSS."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # noqa: BLE001
                logger.exception("Erro ao fechar MSS")
            finally:
                self._sct = None
        self._monitor = None

    def grab(self) -> np.ndarray:
        """Captura o monitor inteiro como frame BGR (uint8, HxWx3).

        Usa exclusivamente a geometria atual do MSS — sem coordenadas fixas.
        """
        self._ensure_open()
        assert self._sct is not None
        monitor = self.monitor

        try:
            # Reconsulta o dicionário MSS a cada grab para refletir mudanças
            # de resolução/DPI sem hardcode.
            mon = self._sct.monitors[self.monitor_index]
            info = MonitorInfo(
                index=self.monitor_index,
                left=int(mon["left"]),
                top=int(mon["top"]),
                width=int(mon["width"]),
                height=int(mon["height"]),
            )
            if (
                info.width != monitor.width
                or info.height != monitor.height
                or info.left != monitor.left
                or info.top != monitor.top
            ):
                self._monitor = info
                logger.warning(
                    "Geometria do monitor alterada em runtime: %s @ (%d,%d)",
                    info.resolution_label,
                    info.left,
                    info.top,
                )
            shot = self._sct.grab(info.as_mss_dict())
        except ScreenCaptureError:
            self.stats.note_error()
            raise
        except Exception as exc:  # noqa: BLE001
            self.stats.note_error()
            raise ScreenCaptureError(f"Falha na captura: {exc}") from exc

        frame = np.asarray(shot, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] < 3:
            self.stats.note_error()
            raise ScreenCaptureError(f"Formato de frame inesperado: {frame.shape}")

        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        h, w = bgr.shape[:2]
        if w != info.width or h != info.height:
            self.stats.note_error()
            raise ScreenCaptureError(
                f"Resolução capturada {w}x{h} diverge do MSS {info.width}x{info.height}"
            )
        self.stats.note_frame(w, h)
        return bgr

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Overlay de FPS/resolução para o preview opcional."""
        annotated = frame.copy()
        mon = self._monitor
        res = (
            mon.resolution_label
            if mon is not None
            else f"{self.stats.last_resolution[0]}x{self.stats.last_resolution[1]}"
        )
        lines = [
            f"FPS: {self.stats.fps:.1f}",
            f"Res: {res}",
            f"Frames: {self.stats.frames_processed}",
            f"Proc: {self.stats.avg_process_ms:.1f} ms",
        ]
        y = 28
        for line in lines:
            cv2.putText(
                annotated,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 26
        return annotated

    def show_preview(self, frame: np.ndarray, window_name: str = PREVIEW_WINDOW_NAME) -> int:
        """Janela opcional, pequena e deslocada — não altera a captura.

        A captura continua sendo o monitor integral; o preview é só visualização.
        """
        display = frame
        h, w = frame.shape[:2]
        if w > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / float(w)
            display = cv2.resize(
                frame,
                (PREVIEW_MAX_WIDTH, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display.shape[1], display.shape[0])
        cv2.moveWindow(window_name, PREVIEW_WINDOW_X, PREVIEW_WINDOW_Y)
        cv2.imshow(window_name, display)
        return int(cv2.waitKey(1) & 0xFF)

    @staticmethod
    def destroy_windows() -> None:
        cv2.destroyAllWindows()

    def _ensure_open(self) -> None:
        if self._sct is None or self._monitor is None:
            self.open()

    def _log_monitor_selection(self) -> None:
        assert self._monitor is not None
        assert self._sct is not None
        mon = self._monitor
        logger.info(
            "Monitor selecionado: índice=%d | origem=(%d,%d) | resolução=%s | "
            "monitores_mss=%d | esperado_ref=%dx%d",
            mon.index,
            mon.left,
            mon.top,
            mon.resolution_label,
            len(self._sct.monitors) - 1,
            EXPECTED_WIDTH,
            EXPECTED_HEIGHT,
        )
        if mon.width != EXPECTED_WIDTH or mon.height != EXPECTED_HEIGHT:
            logger.warning(
                "Resolução MSS (%s) difere da referência do projeto (%dx%d). "
                "A captura usa a resolução real do monitor.",
                mon.resolution_label,
                EXPECTED_WIDTH,
                EXPECTED_HEIGHT,
            )


__all__ = [
    "CaptureStats",
    "ScreenCapture",
    "ScreenCaptureError",
    "KEY_QUIT",
    "KEY_SAVE_FRAME",
]
