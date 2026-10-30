"""Ponto de entrada do Street Rush AI.

Fluxo recomendado (Windows):
1. No terminal integrado do Cursor: ``python -m src.main``
2. Alt+Tab para o Last War em primeiro plano.
3. O processo Python continua capturando o monitor integral em segundo plano.

Padrão: sem janela de preview (não interfere no jogo).
Use ``--preview`` apenas para calibração.

Restrições:
- Não envia teclado/mouse ao jogo.
- Não modifica o cliente oficial.
- Não manipula memória nem comunicação com servidores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CAPTURES_DIR,
    MONITOR_INDEX,
    SAVE_EVERY_N_FRAMES,
    SAVE_INTERVAL_SEC,
    SAVE_REQUEST_FILE,
    STATS_LOG_INTERVAL_SEC,
)
from src.game_state import GameStateTracker  # noqa: E402
from src.screen_capture import (  # noqa: E402
    KEY_QUIT,
    KEY_SAVE_FRAME,
    ScreenCapture,
    ScreenCaptureError,
)
from src.vision import VisionAnalyzer, summarize_detections  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("street_rush_ai")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Street Rush AI — captura integral do monitor + visão "
            "(sem controle do jogo; preview desligado por padrão)."
        )
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=MONITOR_INDEX,
        help="Índice do monitor MSS (padrão: 1 = principal).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Abre janela de preview pequena/deslocada (opcional; off por padrão).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Para após N frames (0 = infinito).",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=STATS_LOG_INTERVAL_SEC,
        help="Intervalo em segundos para log de monitoramento no terminal.",
    )
    parser.add_argument(
        "--save-interval",
        type=float,
        default=SAVE_INTERVAL_SEC,
        help="Salva um frame a cada N segundos (0 = desligado).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY_N_FRAMES,
        help="Salva um frame a cada N frames processados (0 = desligado).",
    )
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Imprime o estado JSON junto com o log de stats.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Analisa uma imagem estática (validação offline / CI).",
    )
    return parser


def save_frame(frame, frames: int) -> Path:
    """Persiste um frame BGR em data/captures/."""
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    out = CAPTURES_DIR / f"frame_{frames:06d}_{int(time.time())}.png"
    if not cv2.imwrite(str(out), frame):
        raise OSError(f"Falha ao gravar frame em {out}")
    logger.info("Frame salvo: %s (%dx%d)", out, frame.shape[1], frame.shape[0])
    return out


def consume_save_request() -> bool:
    """Retorna True se o arquivo-sentinela `.save_now` existir (e o remove)."""
    try:
        if SAVE_REQUEST_FILE.exists():
            SAVE_REQUEST_FILE.unlink(missing_ok=True)
            return True
    except OSError as exc:
        logger.warning("Não foi possível ler/remover sentinela de save: %s", exc)
    return False


def log_stats(
    capture: ScreenCapture,
    detections_summary: str,
    *,
    dump_state: bool = False,
    state_dict: dict | None = None,
) -> None:
    mon = capture.monitor
    stats = capture.stats
    w, h = stats.last_resolution
    logger.info(
        "monitor=%d res=%s captura=%dx%d fps=%.1f frames=%d proc_ms=%.1f erros=%d | %s",
        mon.index,
        mon.resolution_label,
        w,
        h,
        stats.fps,
        stats.frames_processed,
        stats.avg_process_ms,
        stats.capture_errors,
        detections_summary,
    )
    if dump_state and state_dict is not None:
        print(json.dumps(state_dict, ensure_ascii=False))


def analyze_image(path: Path, show_preview: bool) -> int:
    """Modo offline: analisa um arquivo de imagem."""
    frame = cv2.imread(str(path))
    if frame is None:
        logger.error("Não foi possível ler a imagem: %s", path)
        return 1

    vision = VisionAnalyzer()
    tracker = GameStateTracker()
    t0 = time.perf_counter()
    result = vision.analyze(frame)
    state = tracker.update(frame, result, fps=0.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "Imagem %s | %dx%d | proc_ms=%.1f | %s",
        path,
        frame.shape[1],
        frame.shape[0],
        elapsed_ms,
        summarize_detections(result),
    )
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))

    if show_preview:
        annotated = vision.draw_detections(frame, result)
        cv2.namedWindow("Street Rush AI — Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Street Rush AI — Image", annotated)
        logger.info("Pressione qualquer tecla para fechar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_capture_loop(args: argparse.Namespace) -> int:
    """Loop contínuo: captura integral + visão + monitoramento no terminal."""
    vision = VisionAnalyzer()
    tracker = GameStateTracker()
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    preview = bool(args.preview)
    last_stats_log = 0.0
    last_save = 0.0

    logger.info(
        "Iniciando captura integral do monitor (preview=%s). "
        "Alt+Tab para o Last War — o processo continua em segundo plano. "
        "Ctrl+C no terminal para encerrar. "
        "Salvar frame com o jogo em foco: crie o arquivo %s",
        preview,
        SAVE_REQUEST_FILE,
    )

    try:
        with ScreenCapture(monitor_index=args.monitor) as capture:
            mon = capture.monitor
            print(
                f"[street-rush-ai] Monitor {mon.index} | "
                f"resolução={mon.resolution_label} | "
                f"origem=({mon.left},{mon.top}) | preview={'on' if preview else 'off'}",
                flush=True,
            )

            while True:
                try:
                    frame = capture.grab()
                except ScreenCaptureError as exc:
                    logger.error("Erro de captura: %s", exc)
                    time.sleep(0.05)
                    if args.max_frames and capture.stats.frames_processed >= args.max_frames:
                        break
                    continue

                t0 = time.perf_counter()
                result = vision.analyze(frame)
                state = tracker.update(frame, result, fps=capture.fps)
                capture.stats.note_process_time(time.perf_counter() - t0)

                frames = capture.stats.frames_processed
                now = time.time()

                should_save = False
                if args.save_interval > 0 and (now - last_save) >= args.save_interval:
                    should_save = True
                if args.save_every > 0 and frames % args.save_every == 0:
                    should_save = True
                if consume_save_request():
                    should_save = True

                if should_save:
                    try:
                        save_frame(frame, frames)
                        last_save = now
                    except OSError as exc:
                        logger.error("Falha ao salvar frame: %s", exc)

                if args.stats_interval > 0 and (now - last_stats_log) >= args.stats_interval:
                    log_stats(
                        capture,
                        summarize_detections(result),
                        dump_state=args.dump_state,
                        state_dict=state.to_dict() if args.dump_state else None,
                    )
                    # Janela móvel da média de processamento entre logs.
                    capture.stats.reset_process_average()
                    last_stats_log = now

                if preview:
                    annotated = vision.draw_detections(frame, result)
                    annotated = capture.draw_overlay(annotated)
                    key = capture.show_preview(annotated)
                    if key == KEY_QUIT:
                        logger.info("Encerrado pelo usuário (q no preview).")
                        break
                    if key == KEY_SAVE_FRAME:
                        try:
                            save_frame(frame, frames)
                            last_save = now
                        except OSError as exc:
                            logger.error("Falha ao salvar frame: %s", exc)

                if args.max_frames and frames >= args.max_frames:
                    logger.info("Limite de %d frames atingido.", args.max_frames)
                    log_stats(capture, summarize_detections(result))
                    break

    except ScreenCaptureError as exc:
        logger.error("Captura: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrompido (Ctrl+C). Captura encerrada sem alterar o jogo.")
    finally:
        if preview:
            ScreenCapture.destroy_windows()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.image:
        return analyze_image(Path(args.image), show_preview=bool(args.preview))

    return run_capture_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
