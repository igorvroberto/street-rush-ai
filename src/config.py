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
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Display / captura
# ---------------------------------------------------------------------------
# Resolução alvo do cliente Last War (Windows).
TARGET_WIDTH = 1918
TARGET_HEIGHT = 1078

# Monitor MSS (1 = primário). Use 0 apenas se souber o que está fazendo.
MONITOR_INDEX = 1

# Nome da janela de visualização independente.
PREVIEW_WINDOW_NAME = "Street Rush AI — Preview"

# Teclas (OpenCV waitKey).
KEY_QUIT = ord("q")
KEY_SELECT_ROI = ord("r")
KEY_SAVE_FRAME = ord("s")

# ---------------------------------------------------------------------------
# Tipos de objeto detectáveis
# ---------------------------------------------------------------------------
OBJECT_CHARACTER = "character"
OBJECT_TRUCK = "truck"
OBJECT_BARRIER = "barrier"
OBJECT_COIN = "coin"

# Cores BGR para desenhar bounding boxes no preview.
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

    Os intervalos HSV abaixo são *pontos de partida* e devem ser calibrados
    com capturas reais. Objetos só entram no estado do jogo se forem
    efetivamente detectados no frame atual (área mínima + contorno válido).
    """

    min_area: int = 80
    max_area: int = 500_000
    morph_kernel: int = 3
    # Perspectiva: objetos com centro_y menor (mais ao topo) = mais longe.
    # Usado apenas para ordenar/relatar profundidade relativa.


@dataclass
class VisionConfig:
    """Configuração do pipeline de visão."""

    character: DetectionParams = field(default_factory=lambda: DetectionParams(min_area=400))
    truck: DetectionParams = field(default_factory=lambda: DetectionParams(min_area=300))
    barrier: DetectionParams = field(default_factory=lambda: DetectionParams(min_area=200))
    coin: DetectionParams = field(default_factory=lambda: DetectionParams(min_area=40, max_area=8_000))

    # HSV iniciais (calibração necessária com frames reais do Street Rush).
    # Personagem: laranja (evita H≈0 para não colidir com barreiras vermelhas).
    character_hsv: HsvRange = field(
        default_factory=lambda: HsvRange(lower=(8, 100, 100), upper=(28, 255, 255))
    )
    # Caminhões: azuis / cinza-azulados.
    truck_hsv: HsvRange = field(
        default_factory=lambda: HsvRange(lower=(90, 60, 60), upper=(130, 255, 255))
    )
    # Barreiras / obstáculos: vermelho saturado (faixa alta H).
    barrier_hsv: HsvRange = field(
        default_factory=lambda: HsvRange(lower=(160, 100, 80), upper=(179, 255, 255))
    )
    # Moedas: amarelo/dourado.
    coin_hsv: HsvRange = field(
        default_factory=lambda: HsvRange(lower=(18, 120, 120), upper=(35, 255, 255))
    )


@dataclass
class CaptureRegion:
    """Região de captura em coordenadas de tela (pixels).

    Se ``None`` for passado ao capturador, usa o monitor inteiro.
    """

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

    def clamp_to_monitor(self, mon_left: int, mon_top: int, mon_w: int, mon_h: int) -> "CaptureRegion":
        """Garante que a região permanece dentro do monitor."""
        left = max(mon_left, min(self.left, mon_left + mon_w - 1))
        top = max(mon_top, min(self.top, mon_top + mon_h - 1))
        max_w = mon_left + mon_w - left
        max_h = mon_top + mon_h - top
        width = max(1, min(self.width, max_w))
        height = max(1, min(self.height, max_h))
        return CaptureRegion(left=left, top=top, width=width, height=height)


DEFAULT_VISION = VisionConfig()
