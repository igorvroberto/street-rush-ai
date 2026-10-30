"""Ponto de entrada do Street Rush AI.

Captura a tela (região ou monitor), analisa com visão computacional,
atualiza o estado do jogo e exibe preview com FPS e bounding boxes.

Restrições desta versão:
- Não envia comandos de teclado/mouse ao jogo.
- Não modifica o cliente oficial.
- Não manipula memória nem comunicação com servidores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

# Permite ``python src/main.py`` e ``python -m src.main``.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CAPTURES_DIR, MONITOR_INDEX  # noqa: E402
from src.game_state import GameStateTracker  # noqa: E402
from src.screen_capture import (  # noqa: E402
    KEY_QUIT,
    KEY_SAVE_FRAME,
    KEY_SELECT_ROI,
    ScreenCapture,
    ScreenCaptureError,
    parse_region_arg,
)
from src.vision import VisionAnalyzer, summarize_detections  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("street_rush_ai")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Street Rush AI — captura e visão computacional (sem controle do jogo)."
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=MONITOR_INDEX,
        help="Índice do monitor MSS (padrão: 1 = primário).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Região left,top,width,height. Se omitido, usa o monitor inteiro.",
    )
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="Abre seletor interativo de ROI antes do loop principal.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Não abre janela de preview (útil em CI/headless).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Para após N frames (0 = infinito).",
    )
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Imprime o estado JSON a cada segundo (aprox.).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Analisa uma imagem estática em vez da tela (validação offline).",
    )
    return parser


def analyze_image(path: Path, show_preview: bool) -> int:
    """Modo offline: analisa um arquivo de imagem."""
    frame = cv2.imread(str(path))
    if frame is None:
        logger.error("Não foi possível ler a imagem: %s", path)
        return 1

    vision = VisionAnalyzer()
    tracker = GameStateTracker()
    result = vision.analyze(frame)
    state = tracker.update(frame, result, fps=0.0)
    annotated = vision.draw_detections(frame, result)

    logger.info("Detecções: %s", summarize_detections(result))
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))

    if show_preview:
        cv2.namedWindow("Street Rush AI — Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Street Rush AI — Image", annotated)
        logger.info("Pressione qualquer tecla para fechar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_capture_loop(args: argparse.Namespace) -> int:
    """Loop contínuo de captura + visão + preview."""
    region = parse_region_arg(args.region) if args.region else None
    vision = VisionAnalyzer()
    tracker = GameStateTracker()
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with ScreenCapture(monitor_index=args.monitor, region=region) as capture:
            if args.select_region:
                try:
                    capture.select_region_interactive()
                except ScreenCaptureError as exc:
                    logger.error("Seleção de região: %s", exc)
                    return 1

            frames = 0
            last_dump = 0.0
            logger.info(
                "Loop iniciado. Teclas: q=sair, r=ROI, s=salvar frame. Preview=%s",
                not args.no_preview,
            )

            while True:
                frame = capture.grab()
                result = vision.analyze(frame)
                state = tracker.update(frame, result, fps=capture.fps)
                annotated = vision.draw_detections(frame, result)
                annotated = capture.draw_fps_overlay(annotated)

                frames += 1
                if args.dump_state and (frames == 1 or capture.fps > 0):
                    import time

                    now = time.time()
                    if now - last_dump >= 1.0:
                        print(json.dumps(state.to_dict(), ensure_ascii=False))
                        last_dump = now

                if not args.no_preview:
                    key = capture.show_preview(annotated)
                    if key == KEY_QUIT:
                        logger.info("Encerrado pelo usuário (q).")
                        break
                    if key == KEY_SELECT_ROI:
                        # Congela no frame atual para nova ROI sobre o monitor.
                        # Reinicia região para o monitor e reabre seletor.
                        capture.set_region(None)
                        try:
                            capture.select_region_interactive()
                            tracker.reset()
                        except ScreenCaptureError as exc:
                            logger.warning("%s", exc)
                    if key == KEY_SAVE_FRAME:
                        out = CAPTURES_DIR / f"frame_{frames:06d}.png"
                        cv2.imwrite(str(out), frame)
                        logger.info("Frame salvo: %s", out)
                else:
                    # Headless: cede CPU sem UI.
                    if frames % 30 == 0:
                        logger.info(
                            "frame=%d fps=%.1f %s",
                            frames,
                            capture.fps,
                            summarize_detections(result),
                        )

                if args.max_frames and frames >= args.max_frames:
                    logger.info("Limite de %d frames atingido.", args.max_frames)
                    break
    except ScreenCaptureError as exc:
        logger.error("Captura: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrompido (Ctrl+C).")
    finally:
        ScreenCapture.destroy_windows()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.image:
        return analyze_image(Path(args.image), show_preview=not args.no_preview)

    return run_capture_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
