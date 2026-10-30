"""Captura contínua de tela com MSS, OpenCV e NumPy.

Permite capturar o monitor inteiro ou uma região selecionada do jogo,
com overlay de FPS em janela de visualização independente.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

from src.config import (
    CaptureRegion,
    KEY_QUIT,
    KEY_SAVE_FRAME,
    KEY_SELECT_ROI,
    MONITOR_INDEX,
    PREVIEW_WINDOW_NAME,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

logger = logging.getLogger(__name__)


class ScreenCaptureError(Exception):
    """Erro de captura de tela."""


class ScreenCapture:
    """Capturador de tela baseado em MSS.

    Parameters
    ----------
    monitor_index:
        Índice do monitor no MSS (1 = primário).
    region:
        Região opcional ``(left, top, width, height)``. Se omitida,
        captura o monitor completo.
    """

    def __init__(
        self,
        monitor_index: int = MONITOR_INDEX,
        region: Optional[CaptureRegion] = None,
    ) -> None:
        self.monitor_index = monitor_index
        self.region = region
        self._sct: Optional[mss.mss] = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_window_start = time.perf_counter()

    def __enter__(self) -> "ScreenCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        """Abre o contexto MSS."""
        if self._sct is not None:
            return
        try:
            self._sct = mss.mss()
            if self.monitor_index < 0 or self.monitor_index >= len(self._sct.monitors):
                raise ScreenCaptureError(
                    f"Monitor {self.monitor_index} inválido. "
                    f"Disponíveis: 0..{len(self._sct.monitors) - 1}"
                )
            logger.info(
                "Captura iniciada (monitor=%s, região=%s)",
                self.monitor_index,
                self.region,
            )
        except ScreenCaptureError:
            raise
        except Exception as exc:  # noqa: BLE001 — superfície de erro ao usuário
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

    @property
    def fps(self) -> float:
        """FPS médio da janela recente de medição."""
        return self._fps

    def monitor_geometry(self) -> CaptureRegion:
        """Retorna a geometria do monitor configurado."""
        self._ensure_open()
        assert self._sct is not None
        mon = self._sct.monitors[self.monitor_index]
        return CaptureRegion(
            left=int(mon["left"]),
            top=int(mon["top"]),
            width=int(mon["width"]),
            height=int(mon["height"]),
        )

    def set_region(self, region: Optional[CaptureRegion]) -> None:
        """Define ou limpa a região de captura."""
        if region is None:
            self.region = None
            return
        mon = self.monitor_geometry()
        self.region = region.clamp_to_monitor(mon.left, mon.top, mon.width, mon.height)
        logger.info("Região de captura atualizada: %s", self.region)

    def grab(self) -> np.ndarray:
        """Captura um frame BGR (uint8, HxWx3).

        Raises
        ------
        ScreenCaptureError
            Se a captura falhar.
        """
        self._ensure_open()
        assert self._sct is not None

        try:
            if self.region is not None:
                shot = self._sct.grab(self.region.as_mss_dict())
            else:
                shot = self._sct.grab(self._sct.monitors[self.monitor_index])
        except Exception as exc:  # noqa: BLE001
            raise ScreenCaptureError(f"Falha na captura: {exc}") from exc

        # MSS entrega BGRA; OpenCV trabalha em BGR.
        frame = np.asarray(shot, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ScreenCaptureError(f"Formato de frame inesperado: {frame.shape}")
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self._update_fps()
        return bgr

    def select_region_interactive(self, initial_frame: Optional[np.ndarray] = None) -> CaptureRegion:
        """Abre seletor de ROI (OpenCV) e atualiza a região de captura.

        A ROI é relativa ao frame capturado. Se já houver região ativa,
        as coordenadas são convertidas para coordenadas de tela absolutas.
        """
        frame = initial_frame if initial_frame is not None else self.grab()
        # Temporariamente captura área maior para seleção, se necessário.
        window = "Street Rush AI — Selecione a região do jogo"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            roi = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
        finally:
            cv2.destroyWindow(window)

        x, y, w, h = (int(v) for v in roi)
        if w <= 0 or h <= 0:
            raise ScreenCaptureError("Seleção de região cancelada ou inválida.")

        base = self.region if self.region is not None else self.monitor_geometry()
        absolute = CaptureRegion(
            left=base.left + x,
            top=base.top + y,
            width=w,
            height=h,
        )
        self.set_region(absolute)
        return absolute

    def draw_fps_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Desenha indicação de FPS no canto superior esquerdo."""
        annotated = frame.copy()
        text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            annotated,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        region_txt = (
            f"ROI: {self.region.width}x{self.region.height}"
            if self.region is not None
            else f"Monitor full ({TARGET_WIDTH}x{TARGET_HEIGHT} alvo)"
        )
        cv2.putText(
            annotated,
            region_txt,
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def show_preview(self, frame: np.ndarray, window_name: str = PREVIEW_WINDOW_NAME) -> int:
        """Exibe o frame em janela independente e retorna a tecla pressionada."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        return int(cv2.waitKey(1) & 0xFF)

    @staticmethod
    def destroy_windows() -> None:
        """Fecha janelas OpenCV."""
        cv2.destroyAllWindows()

    def _ensure_open(self) -> None:
        if self._sct is None:
            self.open()

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_window_start
        if elapsed >= 0.5:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_window_start = now


def parse_region_arg(value: str) -> CaptureRegion:
    """Parseia ``left,top,width,height`` a partir de string CLI."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Região deve ser left,top,width,height")
    left, top, width, height = (int(p) for p in parts)
    if width <= 0 or height <= 0:
        raise ValueError("width e height devem ser positivos")
    return CaptureRegion(left=left, top=top, width=width, height=height)


__all__ = [
    "ScreenCapture",
    "ScreenCaptureError",
    "parse_region_arg",
    "KEY_QUIT",
    "KEY_SELECT_ROI",
    "KEY_SAVE_FRAME",
]
