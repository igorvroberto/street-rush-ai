"""Configurações centralizadas do Street Rush AI.

Ajuste os intervalos HSV após capturar frames reais do jogo
e salvá-los em ``data/captures/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
DEBUG_DIR = DATA_DIR / "debug"
MODELS_DIR = PROJECT_ROOT / "models"

# Arquivos de captura a ignorar na renderização de debug offline.
CAPTURE_IGNORE_NAMES = frozenset({"synthetic_sample.png"})

# Arquivo-sentinela: se existir, o loop salva um frame e remove o arquivo.
# Funciona mesmo com o Last War em primeiro plano (sem roubar foco).
SAVE_REQUEST_FILE = CAPTURES_DIR / ".save_now"

# ---------------------------------------------------------------------------
# Display / captura
# ---------------------------------------------------------------------------
# Resolução esperada do monitor do usuário (referência; a captura usa o MSS).
EXPECTED_WIDTH = 1918
EXPECTED_HEIGHT = 1078

# Monitor MSS: 1 = monitor principal. 0 = virtual "all monitors".
MONITOR_INDEX = 1

# Preview desligado por padrão para não cobrir o jogo.
PREVIEW_ENABLED_DEFAULT = False
PREVIEW_WINDOW_NAME = "Street Rush AI — Preview"
# Preview reduzido e deslocado para interferir o mínimo possível.
PREVIEW_MAX_WIDTH = 480
PREVIEW_WINDOW_X = 0
PREVIEW_WINDOW_Y = 0

# Intervalo padrão de log no terminal (segundos).
STATS_LOG_INTERVAL_SEC = 1.0

# Salvamento periódico de frames (0 = desligado).
SAVE_INTERVAL_SEC = 0.0
SAVE_EVERY_N_FRAMES = 0

# Teclas OpenCV (somente com --preview).
KEY_QUIT = ord("q")
KEY_SAVE_FRAME = ord("s")

# ---------------------------------------------------------------------------
# Tipos de objeto detectáveis
# ---------------------------------------------------------------------------
OBJECT_CHARACTER = "character"
OBJECT_TRUCK = "truck"
OBJECT_BARRIER = "barrier"
OBJECT_COIN = "coin"

BOX_COLORS: dict[str, Tuple[int, int, int]] = {
    OBJECT_CHARACTER: (0, 255, 0),
    OBJECT_TRUCK: (0, 128, 255),
    OBJECT_BARRIER: (0, 0, 255),
    OBJECT_COIN: (0, 255, 255),
}


@dataclass(frozen=True)
class HsvRange:
    """Intervalo HSV inclusivo para segmentação por cor (OpenCV: H 0–179)."""

    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]


@dataclass
class DetectionParams:
    """Parâmetros de detecção por tipo de objeto.

    Objetos só entram no estado do jogo se forem efetivamente detectados
    no frame atual (área mínima + contorno válido).
    """

    min_area: int = 80
    max_area: int = 500_000
    morph_kernel: int = 3


@dataclass
class VisionConfig:
    """Configuração do pipeline de visão."""

    character: DetectionParams = field(default_factory=lambda: DetectionParams(min_area=400))
    truck: DetectionParams = field(default_factory=lambda: DetectionParams(min_area=300))
    barrier: DetectionParams = field(default_factory=lambda: DetectionParams(min_area=200))
    coin: DetectionParams = field(default_factory=lambda: DetectionParams(min_area=40, max_area=8_000))

    character_hsv: HsvRange = field(
        default_factory=lambda: HsvRange(lower=(8, 100, 100), upper=(28, 255, 255))
    )
    truck_hsv: HsvRange = field(
        default_factory=lambda: HsvRange(lower=(90, 60, 60), upper=(130, 255, 255))
    )
    barrier_hsv: HsvRange = field(
        default_factory=lambda: HsvRange(lower=(160, 100, 80), upper=(179, 255, 255))
    )
    coin_hsv: HsvRange = field(
        default_factory=lambda: HsvRange(lower=(18, 120, 120), upper=(35, 255, 255))
    )


@dataclass(frozen=True)
class MonitorInfo:
    """Geometria real do monitor retornada pelo MSS (sem coordenadas inventadas)."""

    index: int
    left: int
    top: int
    width: int
    height: int

    def as_mss_dict(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }

    @property
    def resolution_label(self) -> str:
        return f"{self.width}x{self.height}"


DEFAULT_VISION = VisionConfig()
